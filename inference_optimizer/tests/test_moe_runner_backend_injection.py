# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""sglang ``--moe-runner-backend`` injection tests (issue #512).

MoE models served by sglang on MI300X/MI355X (aiter) route through aiter's CK
2-stage fused-MoE kernel by default (``--moe-runner-backend auto``); its
first-request JIT build is broken in some ROCm images (missing cub header ->
hipcc fail -> stale lock -> 600s warmup timeout -> baseline_failed). Hyperloom
injects ``--moe-runner-backend triton`` for MoE sglang models on AMD unless the
operator already pinned one. Exercised at both the pure-helper and
``materialize_config_with_envs`` layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from inference_optimizer import cli
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND,
    HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV,
    inject_sglang_moe_runner_backend,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)

_AMD = "mi300x"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Neutralise host GPU autodetect + env so AMD-gating is deterministic."""
    monkeypatch.delenv(HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV, raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    # Without an explicit/env GPU type, _resolve_amd_gpu_type falls back to
    # autodetect; pin it OFF so non-AMD test cases never see real hardware.
    monkeypatch.setattr(cli, "_autodetect_gpu_type", lambda: None)
    for key in (
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES", "PRECISION", "RUN_EVAL", "FRAMEWORK",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_model_config(dir_path: Path, config: dict) -> str:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(dir_path)


@pytest.fixture
def moe_model(tmp_path) -> str:
    """A Qwen3-MoE-style checkpoint dir (declares an expert count)."""
    return _write_model_config(
        tmp_path / "Qwen-Qwen3-30B-A3B",
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "num_experts": 128,
        },
    )


@pytest.fixture
def dense_model(tmp_path) -> str:
    """A dense (non-MoE) checkpoint dir."""
    return _write_model_config(
        tmp_path / "Qwen-Qwen3-8B",
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    )


# _model_is_moe detection
@pytest.mark.parametrize("config", [
    {"num_experts": 128},
    {"num_local_experts": 8},
    {"n_routed_experts": 64},
    {"moe_intermediate_size": 768},
    {"model_type": "qwen3_moe"},
    {"architectures": ["Qwen3MoeForCausalLM"]},
    {"text_config": {"num_experts": 16}},
])
def test_model_is_moe_true(tmp_path, config):
    path = _write_model_config(tmp_path / "m", config)
    assert cli._model_is_moe(path) is True


@pytest.mark.parametrize("config", [
    {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    {"num_experts": 1},        # single "expert" is not MoE
    {"num_experts": True},     # bool must not count as an int expert count
    {},
])
def test_model_is_moe_false(tmp_path, config):
    path = _write_model_config(tmp_path / "m", config)
    assert cli._model_is_moe(path) is False


def test_model_is_moe_missing_config_is_false(tmp_path):
    assert cli._model_is_moe(str(tmp_path / "does-not-exist")) is False


# inject_sglang_moe_runner_backend (pure helper)
def test_inject_appends_triton_for_moe_on_amd(moe_model):
    out = inject_sglang_moe_runner_backend("--foo bar", "sglang", moe_model, _AMD)
    assert out == "--foo bar --moe-runner-backend triton"
    assert DEFAULT_SGLANG_AMD_MOE_RUNNER_BACKEND == "triton"


def test_inject_appends_when_args_empty(moe_model):
    assert (
        inject_sglang_moe_runner_backend("", "sglang", moe_model, _AMD)
        == "--moe-runner-backend triton"
    )
    assert (
        inject_sglang_moe_runner_backend(None, "sglang", moe_model, _AMD)
        == "--moe-runner-backend triton"
    )


def test_inject_honors_env_override(moe_model, monkeypatch):
    monkeypatch.setenv(HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND_ENV, "ck")
    assert (
        inject_sglang_moe_runner_backend("", "sglang", moe_model, _AMD)
        == "--moe-runner-backend ck"
    )


@pytest.mark.parametrize("existing", [
    "--moe-runner-backend ck",
    "--moe-runner-backend=ck",
    "--foo 1 --moe-runner-backend ck --bar 2",
])
def test_inject_does_not_double_user_value(moe_model, existing):
    out = inject_sglang_moe_runner_backend(existing, "sglang", moe_model, _AMD)
    assert out == existing
    assert out.count("--moe-runner-backend") == 1
    assert "triton" not in out


def test_inject_noop_for_dense_model(dense_model):
    assert inject_sglang_moe_runner_backend("--foo", "sglang", dense_model, _AMD) == "--foo"
    assert "--moe-runner-backend" not in inject_sglang_moe_runner_backend(
        "", "sglang", dense_model, _AMD,
    )


def test_inject_noop_on_non_amd_gpu(moe_model):
    # Explicit non-AMD gpu + autodetect pinned off (fixture) -> no injection.
    assert inject_sglang_moe_runner_backend("--foo", "sglang", moe_model, "h100") == "--foo"


@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_inject_noop_for_non_sglang(moe_model, framework):
    assert inject_sglang_moe_runner_backend("--foo", framework, moe_model, _AMD) == "--foo"
    assert inject_sglang_moe_runner_backend("", framework, moe_model, _AMD) == ""


# materialize_config_with_envs (the production choke point)
def _write_yaml(path: Path, *, model: str, framework: str = "sglang") -> None:
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _materialize_envs(
    tmp_path: Path, *, model: str, framework: str = "sglang",
    extra_server_args: str = "",
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, model=model, framework=framework)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base, out, extra_server_args=extra_server_args,
    )
    return yaml.safe_load(materialized.read_text())["benchmark"]["envs"]


def test_materialize_injects_triton_for_moe_on_amd(tmp_path, moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=moe_model)
    assert "--moe-runner-backend triton" in envs["EXTRA_SGLANG_ARGS"]


def test_materialize_noop_for_dense_model_on_amd(tmp_path, dense_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(tmp_path, model=dense_model)
    assert "--moe-runner-backend" not in envs.get("EXTRA_SGLANG_ARGS", "")


def test_materialize_does_not_double_user_backend(tmp_path, moe_model, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", _AMD)
    envs = _materialize_envs(
        tmp_path, model=moe_model, extra_server_args="--moe-runner-backend ck",
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert sglang_args.count("--moe-runner-backend") == 1
    assert "ck" in sglang_args
    assert "triton" not in sglang_args
