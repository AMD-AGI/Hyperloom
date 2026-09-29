# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Public AgentX launches resolve a native recipe without model-specific defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import _configure_benchmark_config, _finalize_benchmark_config
from hyperloom.inference_optimizer.cli.agentx_source import prepare_native_agentx_source


@pytest.fixture(autouse=True)
def fresh_launch_env(monkeypatch):
    original = dict(os.environ)

    def managed(name):
        return name.startswith(("AGENTX_", "HYPERLOOM_AGENTX", "HYPERLOOM_BENCHMARK_CONFIG")) or name in {
            "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR",
            "MODEL_PATH",
            "HYPERLOOM_IMAGE",
        }

    for name in tuple(os.environ):
        if managed(name):
            monkeypatch.setenv(name, "")
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    yield
    source = os.environ.get("HYPERLOOM_BENCHMARK_CONFIG", "")
    if source and Path(source).name.startswith("hyperloom-agentx-"):
        Path(source).unlink(missing_ok=True)
    for name in tuple(os.environ):
        if managed(name):
            if name in original:
                os.environ[name] = original[name]
            else:
                os.environ.pop(name)


def _args(**updates):
    values = dict(
        resume_from=None,
        benchmark_config=None,
        model="Qwen/Qwen3-32B",
        framework="vllm",
        precision="bf16",
        gpu_type="mi300x",
        conc=8,
        tp=None,
        ep=None,
    )
    values.update(updates)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "model,framework,runner,precision",
    [
        ("Qwen/Qwen3-32B", "vllm", "mi300x", "bf16"),
        ("moonshotai/Kimi-K2-Instruct", "sglang", "mi355x", "fp8"),
    ],
)
def test_fresh_switch_builds_native_source_and_resolver_owns_launcher(
    monkeypatch, tmp_path, model, framework, runner, precision
):
    args = _args(model=model, framework=framework, gpu_type=runner, precision=precision)
    assert _configure_benchmark_config(args)
    assert prepare_native_agentx_source(args)
    source = Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"])
    benchmark = yaml.safe_load(source.read_text())["benchmark"]
    assert benchmark["model"] == model
    assert benchmark["framework"] == framework
    assert benchmark["runner_type"] == runner
    assert benchmark["precision"] == precision
    assert benchmark["agentx"]["launch_overrides"] == {"version": 1}
    assert benchmark["envs"]["CONC"] == 8
    assert "benchmark_script" not in benchmark
    assert os.environ["HYPERLOOM_BENCHMARK_CONFIG_SHA256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    from hyperloom.inference_optimizer.agentx import native

    launcher = f"single_node/agentic/{framework}_resolved_model.sh"

    def resolve(benchmark, *, inferencex_path):
        assert "benchmark_script" not in benchmark
        assert benchmark["agentx"]["launch_overrides"] == {"version": 1}
        return {
            "benchmark": dict(benchmark, benchmark_script=launcher),
            "recipe": "unique-recipe",
            "config_file": "configs/amd-master.yaml",
            "entry": {"image": "image@sha256:abc"},
            "topology": {"gpu_count": 4, "ep": 4, "conc": 8, "recipe_fingerprint": "a" * 64},
        }

    monkeypatch.setattr(native, "preview_native_recipe", resolve)
    monkeypatch.setattr(
        native, "native_execution_identity", lambda **kwargs: {"static_execution_fingerprint": "b" * 64}
    )
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    assert _finalize_benchmark_config(args)
    assert os.environ["AGENTX_SERVER_SCRIPT"] == launcher
    assert args.tp == 4
    assert args.ep == 4


def test_local_checkpoint_needs_explicit_canonical_identity(monkeypatch, tmp_path):
    args = _args(model=str(tmp_path))
    with pytest.raises(ValueError, match="canonical model id"):
        prepare_native_agentx_source(args)
    monkeypatch.setenv("AGENTX_MODEL_ID", "Qwen/Qwen3-32B")
    assert prepare_native_agentx_source(args)
    benchmark = yaml.safe_load(Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text())["benchmark"]
    assert benchmark["model"] == "Qwen/Qwen3-32B"
    assert benchmark["envs"]["MODEL_PATH"] == str(tmp_path)


def test_unknown_gpu_is_not_replaced_with_example_gpu():
    with pytest.raises(ValueError, match="detected GPU"):
        prepare_native_agentx_source(_args(gpu_type=None))


def test_fresh_source_preserves_operator_file_and_selectors(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "model": "Qwen/Qwen3-32B",
                    "agentx": {"recipe": "explicit-recipe", "selector": {"tp": 4}, "concurrency": 8},
                }
            }
        )
    )
    original = source.read_bytes()
    args = _args(benchmark_config=str(source))
    _configure_benchmark_config(args)
    prepare_native_agentx_source(args)
    generated = yaml.safe_load(Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text())["benchmark"]
    assert source.read_bytes() == original
    assert generated["agentx"]["recipe"] == "explicit-recipe"
    assert generated["agentx"]["selector"] == {"tp": 4}
    assert generated["agentx"]["launch_overrides"] == {"version": 1}


def test_unsupported_recipe_does_not_fall_back_to_legacy(monkeypatch, tmp_path):
    args = _args()
    prepare_native_agentx_source(args)
    from hyperloom.inference_optimizer.agentx import native

    def ambiguous(*args, **kwargs):
        raise ValueError("multiple AgentX recipes: choose agentx.recipe")

    monkeypatch.setattr(native, "preview_native_recipe", ambiguous)
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="multiple AgentX recipes"):
        _finalize_benchmark_config(args)
    assert "aiperf_client.sh" not in Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text()


def test_resume_does_not_generate_new_source():
    assert not prepare_native_agentx_source(_args(resume_from="existing-session"))


def test_fresh_entry_clears_previous_session_before_mode_selection(monkeypatch, tmp_path):
    from hyperloom.common.agentx_mode import native_agentx_optimization_session

    (tmp_path / "state.json").write_text(json.dumps({"benchmark_mode": "agentx", "agentx_epoch": 1}))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(tmp_path))
    assert not native_agentx_optimization_session()
    assert _configure_benchmark_config(_args())
    assert "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR" not in os.environ
    assert native_agentx_optimization_session()


@pytest.mark.parametrize("epoch", [1, 3])
def test_resume_preflight_uses_saved_mode_and_restores_previous_pointer(monkeypatch, tmp_path, epoch):
    import hyperloom.inference_optimizer.cli as cli
    from hyperloom.common.agentx_mode import native_agentx_session

    session = tmp_path / "resumed"
    session.mkdir()
    (session / "state.json").write_text(json.dumps({"benchmark_mode": "agentx", "agentx_epoch": epoch}))
    previous = str(tmp_path / "previous-session")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", previous)

    def preflight(args):
        assert native_agentx_session() is (epoch >= 2)
        return {"provider": "test"}

    monkeypatch.setattr(cli, "_preflight", preflight)
    assert cli._preflight_for_session(_args(resume_from=str(session))) == {"provider": "test"}
    assert os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] == previous


def test_source_preserves_explicit_launch_request(tmp_path):
    source = tmp_path / "source.yaml"
    request = {"version": 1, "append_args": ["--mem-fraction-static", "0.8"], "env": {"SGLANG_USE_AITER": "1"}}
    source.write_text(
        yaml.safe_dump({"benchmark": {"model": "Qwen/Qwen3-32B", "agentx": {"launch_overrides": request}}})
    )
    args = _args(benchmark_config=str(source))
    _configure_benchmark_config(args)
    prepare_native_agentx_source(args)
    generated = yaml.safe_load(Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text())["benchmark"]
    assert generated["agentx"]["launch_overrides"] == request


def test_custom_model_forwards_explicit_topology_context_image_and_local_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("HYPERLOOM_IMAGE", "operator/image@sha256:abc")
    monkeypatch.setenv("AGENTX_MODEL_ID", "operator/custom-model")
    args = _args(model=str(tmp_path), tp=4, ep=2, max_model_len=32768)
    prepare_native_agentx_source(args)
    benchmark = yaml.safe_load(Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text())["benchmark"]
    assert benchmark["docker_image"] == "operator/image@sha256:abc"
    assert benchmark["model"] == "operator/custom-model"
    assert benchmark["envs"] == {
        "TP": 4,
        "EP_SIZE": 2,
        "CONC": 8,
        "MAX_MODEL_LEN": 32768,
        "AGENTX_MODEL_ID": "operator/custom-model",
        "MODEL_PATH": str(tmp_path),
    }


def test_custom_source_topology_survives_unspecified_cli_defaults(tmp_path):
    source = tmp_path / "custom.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "model": "operator/custom-model",
                    "docker_image": "operator/custom-image",
                    "agentx": {"enabled": True},
                    "envs": {"TP": 4, "EP_SIZE": 2, "MAX_MODEL_LEN": 32768},
                }
            }
        )
    )
    args = _args(benchmark_config=str(source), model=None, max_model_len=None)
    _configure_benchmark_config(args)
    prepare_native_agentx_source(args)
    benchmark = yaml.safe_load(Path(os.environ["HYPERLOOM_BENCHMARK_CONFIG"]).read_text())["benchmark"]
    assert benchmark["docker_image"] == "operator/custom-image"
    assert benchmark["envs"]["TP"] == 4
    assert benchmark["envs"]["EP_SIZE"] == 2
    assert benchmark["envs"]["MAX_MODEL_LEN"] == 32768


def test_conflicting_explicit_image_fails_before_recipe_resolution(monkeypatch, tmp_path):
    source = tmp_path / "custom.yaml"
    source.write_text(yaml.safe_dump({"benchmark": {"model": "operator/custom-model", "docker_image": "source/image"}}))
    args = _args(benchmark_config=str(source), model=None)
    _configure_benchmark_config(args)
    monkeypatch.setenv("HYPERLOOM_IMAGE", "operator/different-image")
    with pytest.raises(ValueError, match="HYPERLOOM_IMAGE conflicts"):
        prepare_native_agentx_source(args)
