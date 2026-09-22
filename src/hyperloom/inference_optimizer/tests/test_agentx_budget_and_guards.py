# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX budget profile, search-scope collapse, and the session-level guards."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import (
    _apply_agentx_budget_profile,
    _configure_benchmark_config,
    _finalize_benchmark_config,
    _preflight_agentx_backend,
    _restore_agentx_env_from_state,
    _restore_agentx_runtime_pins_from_state,
)
from hyperloom.inference_optimizer.cli.bootstrap import (
    AGENTX_MEASUREMENT_EPOCH,
    agentx_state_is_stale,
)
from hyperloom.inference_optimizer.cli.parser import DEFAULT_MAX_HOURS


_MAGPIE_REF = "c" * 40
_INFERENCEX_REF = "d" * 40
_RECIPE_FINGERPRINT = "a" * 64
_EXECUTION_FINGERPRINT = "b" * 64


def _budget_args(**over) -> argparse.Namespace:
    base = dict(
        # What the parser produces when ``--max-hours`` is absent: the flag
        # carries no argparse default, so the profile sees ``None``, not 2.0.
        max_hours=None,
        conc_sweep_total_budget_sec=9000,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _off(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)


def _blank_agentx_runtime_pins(monkeypatch) -> None:
    from hyperloom.inference_optimizer.agentx.native import (
        AGENTX_RUNTIME_PIN_NAMES,
    )

    for name in AGENTX_RUNTIME_PIN_NAMES:
        monkeypatch.setenv(name, "")


def _on(monkeypatch):
    _blank_agentx_runtime_pins(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_MODEL_ID", "acme/Test-Model")
    monkeypatch.setenv(
        "AGENTX_SERVER_SCRIPT",
        "single_node/agentic/test_fp4_mi300x_vllm_mtp.sh",
    )
    monkeypatch.setenv("INFERENCEX_PATH", "/tmp/inferencex")
    monkeypatch.setenv("HYPERLOOM_IMAGE", "example/agentx:test")
    monkeypatch.setenv(
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT",
        _RECIPE_FINGERPRINT,
    )
    monkeypatch.setenv(
        "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT",
        _EXECUTION_FINGERPRINT,
    )
    monkeypatch.setenv("HYPERLOOM_AGENTX_GPU_COUNT", "1")
    monkeypatch.setenv("MAGPIE_REF", _MAGPIE_REF)
    monkeypatch.setenv("INFERENCEX_REF", _INFERENCEX_REF)


def _pin_source_hash(monkeypatch, config: Path) -> None:
    monkeypatch.setenv(
        "HYPERLOOM_BENCHMARK_CONFIG_SHA256",
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )


def _stub_native_execution_identity(monkeypatch, native_agentx) -> None:
    monkeypatch.setattr(
        native_agentx,
        "native_execution_identity",
        lambda **_kwargs: {
            "static_execution_fingerprint": _EXECUTION_FINGERPRINT,
            "execution_fingerprint": _EXECUTION_FINGERPRINT,
            "inferencex_commit": _INFERENCEX_REF,
            "magpie_commit": _MAGPIE_REF,
        },
    )


# --- budget profile -----------------------------------------------------------


def test_budget_profile_is_noop_without_agentx(monkeypatch):
    _off(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert vars(args) == vars(_budget_args())


def test_budget_profile_does_not_expand_benchmark_caps(monkeypatch):
    _on(monkeypatch)
    args = _budget_args()
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_total_budget_sec == 9000
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    monkeypatch.delenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", raising=False)
    assert resolve_benchmark_timeouts()[1] == 7800


def test_budget_profile_never_touches_max_hours(monkeypatch):
    """``--max-hours`` is the operator's contract with the scheduler."""
    _on(monkeypatch)
    args = _budget_args(max_hours=2.0)
    _apply_agentx_budget_profile(args)
    assert args.max_hours == 2.0


def test_the_note_fires_when_the_operator_passed_no_budget(monkeypatch, capsys):
    _on(monkeypatch)
    _apply_agentx_budget_profile(_budget_args())
    assert "--max-hours" in capsys.readouterr().err


def test_the_note_stays_quiet_for_a_budget_the_operator_typed(monkeypatch, capsys):
    """An explicit value is a deliberate choice, even at the default's number."""
    _on(monkeypatch)
    _apply_agentx_budget_profile(_budget_args(max_hours=DEFAULT_MAX_HOURS))
    assert capsys.readouterr().err == ""


def test_budget_profile_preserves_operator_values(monkeypatch):
    """A value the operator typed is left exactly as typed."""
    _on(monkeypatch)
    args = _budget_args(
        conc_sweep_total_budget_sec=1200,
    )
    _apply_agentx_budget_profile(args)
    assert args.conc_sweep_total_budget_sec == 1200


# --- explicit-flag detection --------------------------------------------------
def test_bypass_guard_allows_magpie(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_is_inert_without_agentx(monkeypatch):
    _off(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    _preflight_agentx_backend(argparse.Namespace())  # must not raise


def test_bypass_guard_rejects_the_silent_combination(monkeypatch):
    """AgentX + bypass runs synthetic work and labels it AgentX."""
    _on(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace())
    assert ei.value.code == 2


def test_guard_rejects_explicit_ray_execution(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    with pytest.raises(SystemExit) as exc:
        _preflight_agentx_backend(argparse.Namespace(framework="sglang", nodes=1))
    assert exc.value.code == 2


def test_guard_rejects_agentx_with_a_scriptable_framework(monkeypatch):
    """The other way for the switch to no-op while every gate still fires."""
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as ei:
        _preflight_agentx_backend(argparse.Namespace(framework="xdit"))
    assert ei.value.code == 2


def test_guard_allows_agentx_with_a_serving_framework(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    for fw in ("vllm", "sglang"):
        _preflight_agentx_backend(argparse.Namespace(framework=fw))  # must not raise


def test_guard_rejects_native_agentx_multi_node(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as exc:
        _preflight_agentx_backend(argparse.Namespace(framework="sglang", nodes=2))
    assert exc.value.code == 2


def test_guard_rejects_framework_without_native_argv_bridge(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    with pytest.raises(SystemExit) as exc:
        _preflight_agentx_backend(argparse.Namespace(framework="atom", nodes=1))
    assert exc.value.code == 2


@pytest.mark.parametrize("framework", ["atom", "xdit"])
def test_guard_rejects_unsupported_framework_from_environment(monkeypatch, framework):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    monkeypatch.setenv("FRAMEWORK", framework)
    with pytest.raises(SystemExit) as exc:
        _preflight_agentx_backend(argparse.Namespace(nodes=1))
    assert exc.value.code == 2


def test_scriptable_guard_is_inert_without_agentx(monkeypatch):
    """A scriptable run on its own is perfectly normal."""
    _off(monkeypatch)
    _preflight_agentx_backend(argparse.Namespace(framework="xdit"))  # must not raise


def test_benchmark_yaml_agentx_switch_sets_session_mode_and_pins(monkeypatch, tmp_path):
    for name in (
        "HYPERLOOM_AGENTX",
        "HYPERLOOM_BENCHMARK_CONFIG",
        "AGENTX_MODEL_ID",
        "AGENTX_SERVER_SCRIPT",
        "INFERENCEX_PATH",
    ):
        # _configure_benchmark_config writes directly to os.environ. Seed a
        # tracked value so monkeypatch restores/removes every write at teardown.
        monkeypatch.setenv(name, "")
    _off(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    config = tmp_path / "agentx.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "amd/GLM-5.2-MXFP4",
                    "precision": "fp4",
                    "agentx": "enable",
                    "benchmark_script": "single_node/agentic/glm.sh",
                    "inferencex_path": str(inferencex),
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmark_config=str(config),
        resume_from=None,
        framework=None,
        model=None,
        precision=None,
    )

    assert _configure_benchmark_config(args) is True
    assert os.environ["HYPERLOOM_AGENTX"] == "1"
    assert os.environ["HYPERLOOM_BENCHMARK_CONFIG"] == str(config.resolve())
    assert os.environ["AGENTX_MODEL_ID"] == "amd/GLM-5.2-MXFP4"
    assert os.environ["AGENTX_SERVER_SCRIPT"] == "single_node/agentic/glm.sh"
    assert os.environ["INFERENCEX_PATH"] == str(inferencex.resolve())
    assert args.framework == "sglang"
    assert args.model == Path("amd/GLM-5.2-MXFP4")
    assert args.precision == "fp4"


def test_benchmark_yaml_projects_workload_before_preflight_without_resolving(
    monkeypatch,
    tmp_path,
):
    """Early configuration must not need an installed Magpie interpreter."""
    for name in (
        "HYPERLOOM_AGENTX",
        "HYPERLOOM_BENCHMARK_CONFIG",
        "AGENTX_MODEL_ID",
        "AGENTX_SERVER_SCRIPT",
        "INFERENCEX_PATH",
    ):
        monkeypatch.setenv(name, "")
    _off(monkeypatch)
    from hyperloom.inference_optimizer.agentx import native as native_agentx

    def _fail_resolver(*_args, **_kwargs):
        pytest.fail("recipe resolution must run after dependency preflight")

    monkeypatch.setattr(
        native_agentx,
        "preview_native_recipe",
        _fail_resolver,
    )
    monkeypatch.setattr(native_agentx, "_run_magpie_recipe_resolver", _fail_resolver)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    local_model = tmp_path / "models" / "glm"
    config = tmp_path / "agentx.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "agentx": "enable",
                    "framework": "sglang",
                    "runner_type": "mi355x",
                    "model": "amd/GLM-5.2-MXFP4",
                    "benchmark_script": "single_node/agentic/glm.sh",
                    "inferencex_path": str(inferencex),
                    "envs": {
                        "MODEL_PATH": str(local_model),
                        "CONC": 64,
                        "ISL": 1024,
                        "OSL": 1024,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmark_config=str(config),
        resume_from=None,
        framework=None,
        gpu_type=None,
        model=None,
        precision=None,
        tp=None,
        ep=None,
        conc=None,
        isl=None,
        osl=None,
        max_model_len=None,
    )

    assert _configure_benchmark_config(args) is True
    assert args.model == local_model
    assert args.gpu_type == "mi355x"
    assert args.conc == 64
    assert args.isl == 1024
    assert args.osl == 1024
    assert args.tp is None
    assert os.environ["AGENTX_MODEL_ID"] == "amd/GLM-5.2-MXFP4"


def test_non_agentx_benchmark_yaml_projects_physical_tp_and_concurrency(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", "")
    _off(monkeypatch)
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "agentx": "disable",
                    "framework": "sglang",
                    "model": "/models/qwen",
                    "envs": {"TP": 8, "EP": 4, "CONC": 32},
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmark_config=str(config),
        resume_from=None,
        framework=None,
        gpu_type=None,
        model=None,
        precision=None,
        tp=None,
        ep=None,
        conc=None,
        isl=None,
        osl=None,
        max_model_len=None,
    )

    assert _configure_benchmark_config(args) is False
    assert args.tp == 8
    assert args.ep == 4
    assert args.conc == 32


def test_finalize_benchmark_yaml_projects_resolved_agentx_topology(
    monkeypatch,
    tmp_path,
):
    _blank_agentx_runtime_pins(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    config = tmp_path / "agentx.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "agentx": {
                        "enabled": True,
                        "concurrency": 8,
                        "resolved": {"stale": True},
                    },
                    "precision": "fp4",
                    "envs": {"CONC": 8},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(config))
    _pin_source_hash(monkeypatch, config)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setenv("AGENTX_MODEL_ID", "amd/GLM-5.2-MXFP4")
    monkeypatch.setenv(
        "AGENTX_SERVER_SCRIPT",
        "single_node/agentic/glm.sh",
    )
    monkeypatch.setenv("HYPERLOOM_IMAGE", "rocm/agentx:accepted")
    captured = {}

    def _preview(benchmark, *, inferencex_path):
        captured["benchmark"] = benchmark
        captured["inferencex_path"] = inferencex_path
        return {
            "recipe": "glm5-agentic",
            "config_file": "configs/amd-master.yaml",
            "topology": {
                "tp": 2,
                "pp": 2,
                "pcp_size": 2,
                "ep": 4,
                "gpu_count": 8,
                "conc": 32,
                "duration_seconds": 3600,
                "recipe_fingerprint": _RECIPE_FINGERPRINT,
            },
            "entry": {"image": "rocm/agentx:accepted"},
            "magpie_execution": {
                "fingerprint": "e" * 64,
                "source_commit": _MAGPIE_REF,
            },
        }

    from hyperloom.inference_optimizer.agentx import native as native_agentx

    monkeypatch.setattr(native_agentx, "preview_native_recipe", _preview)
    _stub_native_execution_identity(monkeypatch, native_agentx)
    args = argparse.Namespace(
        resume_from=None,
        tp=None,
        ep=None,
        conc=32,
        precision="bf16",
    )

    assert _finalize_benchmark_config(args) is True
    assert args.tp == 8
    assert args.ep == 4
    assert args.conc == 32
    assert captured["inferencex_path"] == str(inferencex)
    assert captured["benchmark"]["envs"]["CONC"] == 32
    assert captured["benchmark"]["envs"]["AGENTX_MODEL_ID"] == ("amd/GLM-5.2-MXFP4")
    assert captured["benchmark"]["precision"] == "bf16"
    assert "concurrency" not in captured["benchmark"]["agentx"]
    assert "resolved" not in captured["benchmark"]["agentx"]
    assert os.environ["HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT"] == _RECIPE_FINGERPRINT
    assert os.environ["HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT"] == _EXECUTION_FINGERPRINT
    assert os.environ["HYPERLOOM_AGENTX_GPU_COUNT"] == "8"
    assert os.environ["AGENTX_MODE"] == "canonical"
    assert os.environ["AGENTX_RECIPE"] == "glm5-agentic"
    assert os.environ["AGENTX_CONFIG_FILE"] == "configs/amd-master.yaml"
    assert os.environ["AGENTX_FAILED_REQUEST_THRESHOLD"] == "0.1"


@pytest.mark.parametrize(
    ("requested_tp", "requested_ep", "message"),
    [
        (4, None, "physical GPU count 8"),
        (None, 2, "resolved AgentX EP 4"),
    ],
)
def test_finalize_benchmark_yaml_rejects_resolved_topology_mismatch(
    monkeypatch,
    tmp_path,
    requested_tp,
    requested_ep,
    message,
):
    _blank_agentx_runtime_pins(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    config = tmp_path / "agentx.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "agentx": "enable",
                    "envs": {"CONC": 64},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(config))
    _pin_source_hash(monkeypatch, config)
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setenv("HYPERLOOM_IMAGE", "rocm/agentx:accepted")
    from hyperloom.inference_optimizer.agentx import native as native_agentx

    monkeypatch.setattr(
        native_agentx,
        "preview_native_recipe",
        lambda *_args, **_kwargs: {
            "recipe": "glm5-agentic",
            "config_file": "configs/amd-master.yaml",
            "topology": {
                "tp": 2,
                "pp": 2,
                "pcp_size": 2,
                "ep": 4,
                "gpu_count": 8,
                "conc": 64,
                "duration_seconds": 3600,
                "recipe_fingerprint": _RECIPE_FINGERPRINT,
            },
            "entry": {"image": "rocm/agentx:accepted"},
            "magpie_execution": {
                "fingerprint": "e" * 64,
                "source_commit": _MAGPIE_REF,
            },
        },
    )
    _stub_native_execution_identity(monkeypatch, native_agentx)
    args = argparse.Namespace(
        resume_from=None,
        tp=requested_tp,
        ep=requested_ep,
        conc=None,
        precision=None,
    )

    with pytest.raises(ValueError, match=message):
        _finalize_benchmark_config(args)


def test_resume_ignores_inherited_benchmark_yaml(monkeypatch, tmp_path):
    foreign = tmp_path / "foreign.yaml"
    foreign.write_text(
        yaml.safe_dump({"benchmark": {"agentx": "enable", "framework": "sglang"}}),
        encoding="utf-8",
    )
    _off(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(foreign))
    args = argparse.Namespace(
        benchmark_config=None,
        resume_from=str(tmp_path / "session"),
    )

    assert _configure_benchmark_config(args) is False
    assert "HYPERLOOM_BENCHMARK_CONFIG" not in os.environ
    assert "HYPERLOOM_AGENTX" not in os.environ


def test_agentx_preflight_rejects_explicit_concurrency_sweep(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    with pytest.raises(SystemExit) as exc:
        _preflight_agentx_backend(
            argparse.Namespace(
                framework="sglang",
                nodes=1,
                enable_conc_sweep=True,
            )
        )
    assert exc.value.code == 2


# --- resume staleness ---------------------------------------------------------


class _St:
    def __init__(
        self,
        mode="",
        epoch=0,
        baseline_config_path="",
        agentx_runtime_pins=None,
        active_inferencex_path="",
    ):
        self.benchmark_mode = mode
        self.agentx_epoch = epoch
        self.baseline_config_path = baseline_config_path
        self.agentx_runtime_pins = dict(agentx_runtime_pins or {})
        self.active_inferencex_path = active_inferencex_path


def _saved_native_baseline(tmp_path):
    inferencex_path = tmp_path / "InferenceX"
    inferencex_path.mkdir()
    baseline_path = tmp_path / "accepted-agentx.yaml"
    pins = {
        "AGENTX_MODEL_ID": "amd/GLM-5.2-MXFP4",
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/glm_fp4_mi355x_sglang.sh",
        "INFERENCEX_PATH": str(inferencex_path),
        "HYPERLOOM_IMAGE": "rocm/agentx:accepted",
        "AGENTX_MODE": "canonical",
        "AGENTX_RECIPE": "glm5-agentic",
        "AGENTX_CONFIG_FILE": "configs/amd-master.yaml",
        "AGENTX_SELECTOR": '{"ep":4,"tp":4}',
        "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT": _RECIPE_FINGERPRINT,
        "HYPERLOOM_AGENTX_EXPECTED_EXECUTION_FINGERPRINT": _EXECUTION_FINGERPRINT,
        "HYPERLOOM_AGENTX_EXPECTED_MATERIALIZED_EXECUTION_FINGERPRINT": _EXECUTION_FINGERPRINT,
        "HYPERLOOM_AGENTX_GPU_COUNT": "4",
        "MAGPIE_REF": _MAGPIE_REF,
        "INFERENCEX_REF": _INFERENCEX_REF,
    }
    baseline_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "agentx": {
                        "enabled": True,
                        "mode": "canonical",
                        "recipe": pins["AGENTX_RECIPE"],
                        "config_file": pins["AGENTX_CONFIG_FILE"],
                        "selector": {"tp": 4, "ep": 4},
                        "failed_request_threshold": 0.10,
                    },
                    "model": pins["AGENTX_MODEL_ID"],
                    "benchmark_script": pins["AGENTX_SERVER_SCRIPT"],
                    "inferencex_path": pins["INFERENCEX_PATH"],
                    "docker_image": pins["HYPERLOOM_IMAGE"],
                    "envs": {
                        "AGENTX_MODEL_ID": pins["AGENTX_MODEL_ID"],
                        "AGENTX_SERVER_SCRIPT": pins["AGENTX_SERVER_SCRIPT"],
                    },
                    "workload_spec": {
                        "outer_image": pins["HYPERLOOM_IMAGE"],
                        "recipe": {
                            "name": pins["AGENTX_RECIPE"],
                            "config_file": pins["AGENTX_CONFIG_FILE"],
                            "image": pins["HYPERLOOM_IMAGE"],
                            "recipe_fingerprint": _RECIPE_FINGERPRINT,
                        },
                        "resolved_topology": {
                            "tp": 4,
                            "pp": 1,
                            "pcp_size": 1,
                            "ep": 4,
                            "gpu_count": 4,
                            "conc": 8,
                            "duration_seconds": 3600,
                            "recipe_fingerprint": _RECIPE_FINGERPRINT,
                        },
                        "execution": {
                            "static_execution_fingerprint": _EXECUTION_FINGERPRINT,
                            "execution_fingerprint": _EXECUTION_FINGERPRINT,
                            "magpie_commit": _MAGPIE_REF,
                            "inferencex_commit": _INFERENCEX_REF,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    state = _St(
        "agentx",
        AGENTX_MEASUREMENT_EPOCH,
        baseline_config_path=str(baseline_path),
    )
    return state, pins


def test_resume_accepts_matching_agentx_state(monkeypatch):
    _on(monkeypatch)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) == ""


def test_resume_accepts_matching_synthetic_state(monkeypatch):
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("synthetic", 0)) == ""


def test_resume_rejects_mode_switch(monkeypatch):
    """The KEEP ledger keys on server args alone, so the rows would collide."""
    _on(monkeypatch)
    assert "benchmark_mode" in agentx_state_is_stale(_St("synthetic", 0))
    monkeypatch.setenv("HYPERLOOM_AGENTX", "0")
    assert "benchmark_mode" in agentx_state_is_stale(_St("agentx", 1))


def test_fresh_shell_resume_inherits_persisted_agentx_mode(monkeypatch):
    """An omitted mode on resume is not an implicit request to go synthetic."""
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) == ""


@pytest.mark.parametrize("shell_value", [None, "", "   "])
def test_fresh_shell_resume_reexports_persisted_agentx_mode(monkeypatch, shell_value):
    if shell_value is None:
        monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    else:
        monkeypatch.setenv("HYPERLOOM_AGENTX", shell_value)

    try:
        assert _restore_agentx_env_from_state(_St("agentx", AGENTX_MEASUREMENT_EPOCH)) is True
        assert os.environ["HYPERLOOM_AGENTX"] == "1"
    finally:
        # The helper deliberately writes os.environ directly. When the key was
        # initially absent, monkeypatch.delenv has no undo entry for that new
        # write, so clean it explicitly to keep later tests mode-neutral.
        if shell_value is None:
            os.environ.pop("HYPERLOOM_AGENTX", None)


def test_fresh_shell_resume_restores_native_agentx_runtime_pins(monkeypatch, tmp_path):
    state, pins = _saved_native_baseline(tmp_path)
    for name in pins:
        # A blank value models an omitted fresh-shell export while preserving
        # monkeypatch's teardown record for the helper's direct env writes.
        monkeypatch.setenv(name, "")

    assert _restore_agentx_runtime_pins_from_state(state) == pins
    assert {name: os.environ[name] for name in pins} == pins


def test_prebaseline_resume_restores_seed_time_agentx_runtime_pins(monkeypatch, tmp_path):
    _blank_agentx_runtime_pins(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    pins = {
        "AGENTX_MODEL_ID": "amd/GLM-5.2-MXFP4",
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/glm.sh",
        "INFERENCEX_PATH": str(inferencex),
        "HYPERLOOM_IMAGE": "rocm/agentx:seeded",
        "AGENTX_MODE": "canonical",
        "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
    }
    state = _St(
        "agentx",
        AGENTX_MEASUREMENT_EPOCH,
        agentx_runtime_pins=pins,
        active_inferencex_path=str(inferencex),
    )
    for name in pins:
        monkeypatch.setenv(name, "")

    assert _restore_agentx_runtime_pins_from_state(state) == pins
    assert {name: os.environ[name] for name in pins} == pins


def test_resume_ignores_generic_active_framework_checkout(monkeypatch, tmp_path):
    state, pins = _saved_native_baseline(tmp_path)
    promoted = tmp_path / "InferenceX-promoted"
    promoted.mkdir()
    state.active_inferencex_path = str(promoted)
    state.agentx_runtime_pins = dict(pins)
    for name in pins:
        monkeypatch.setenv(name, "")

    restored = _restore_agentx_runtime_pins_from_state(state)

    assert restored["INFERENCEX_PATH"] == str((tmp_path / "InferenceX").resolve())
    assert os.environ["INFERENCEX_PATH"] == str((tmp_path / "InferenceX").resolve())


def test_prebaseline_resume_restores_selector_and_resolved_contract(
    monkeypatch,
    tmp_path,
):
    _blank_agentx_runtime_pins(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    pins = {
        "AGENTX_MODEL_ID": "amd/GLM-5.2-MXFP4",
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/glm.sh",
        "INFERENCEX_PATH": str(inferencex),
        "HYPERLOOM_IMAGE": "rocm/agentx:seeded",
        "AGENTX_RECIPE": "glm5-agentic",
        "AGENTX_SELECTOR": '{"ep":4,"tp":4}',
        "AGENTX_MODE": "canonical",
        "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
        "HYPERLOOM_AGENTX_EXPECTED_RECIPE_FINGERPRINT": "a" * 64,
        "HYPERLOOM_AGENTX_GPU_COUNT": "4",
    }
    state = _St(
        "agentx",
        AGENTX_MEASUREMENT_EPOCH,
        agentx_runtime_pins=pins,
    )
    for name in pins:
        monkeypatch.setenv(name, "")

    assert _restore_agentx_runtime_pins_from_state(state) == pins
    assert {name: os.environ[name] for name in pins} == pins


def test_resume_ignores_preflight_generated_path_when_operator_did_not_pin_one(
    monkeypatch,
    tmp_path,
):
    state, pins = _saved_native_baseline(tmp_path)
    auto_checkout = tmp_path / "auto-cloned-InferenceX"
    auto_checkout.mkdir()
    for name in pins:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("INFERENCEX_PATH", str(auto_checkout))

    restored = _restore_agentx_runtime_pins_from_state(
        state,
        operator_pins={name: "" for name in pins},
    )

    assert restored["INFERENCEX_PATH"] == pins["INFERENCEX_PATH"]
    assert os.environ["INFERENCEX_PATH"] == pins["INFERENCEX_PATH"]


def test_resume_accepts_equivalent_saved_inferencex_checkout(monkeypatch, tmp_path):
    state, pins = _saved_native_baseline(tmp_path)
    for name, value in pins.items():
        monkeypatch.setenv(name, value)
    checkout = tmp_path / "InferenceX"
    monkeypatch.setenv("INFERENCEX_PATH", str(checkout / ".." / checkout.name))

    expected = str(checkout.resolve())
    assert _restore_agentx_runtime_pins_from_state(state) == {
        "INFERENCEX_PATH": expected,
    }
    assert os.environ["INFERENCEX_PATH"] == expected


def test_resume_rejects_native_agentx_runtime_pin_conflict(monkeypatch, tmp_path):
    state, pins = _saved_native_baseline(tmp_path)
    for name in pins:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("HYPERLOOM_IMAGE", "rocm/agentx:different")

    with pytest.raises(ValueError, match="HYPERLOOM_IMAGE=.*conflicts"):
        _restore_agentx_runtime_pins_from_state(state)
    assert os.environ["AGENTX_MODEL_ID"] == ""
    assert os.environ["AGENTX_SERVER_SCRIPT"] == ""
    assert os.environ["INFERENCEX_PATH"] == ""


def test_resume_rejects_failed_request_threshold_drift(monkeypatch, tmp_path):
    state, pins = _saved_native_baseline(tmp_path)
    for name in pins:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("AGENTX_FAILED_REQUEST_THRESHOLD", "1")

    with pytest.raises(
        ValueError,
        match="AGENTX_FAILED_REQUEST_THRESHOLD=.*conflicts",
    ):
        _restore_agentx_runtime_pins_from_state(state)


def test_prebaseline_resume_rejects_unpinned_selector_pollution(
    monkeypatch,
    tmp_path,
):
    _blank_agentx_runtime_pins(monkeypatch)
    inferencex = tmp_path / "InferenceX"
    inferencex.mkdir()
    pins = {
        "AGENTX_MODEL_ID": "amd/GLM-5.2-MXFP4",
        "AGENTX_SERVER_SCRIPT": "single_node/agentic/glm.sh",
        "INFERENCEX_PATH": str(inferencex),
        "HYPERLOOM_IMAGE": "rocm/agentx:seeded",
        "AGENTX_MODE": "canonical",
        "AGENTX_FAILED_REQUEST_THRESHOLD": "0.1",
    }
    state = _St(
        "agentx",
        AGENTX_MEASUREMENT_EPOCH,
        agentx_runtime_pins=pins,
    )
    for name in pins:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("AGENTX_SELECTOR", '{"kv_offloading":"none"}')

    with pytest.raises(ValueError, match="AGENTX_SELECTOR=.*conflicts"):
        _restore_agentx_runtime_pins_from_state(state)


def test_synthetic_resume_does_not_read_agentx_runtime_pins(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    monkeypatch.setenv("AGENTX_MODEL_ID", "")
    state = _St("synthetic", baseline_config_path=str(missing))

    assert _restore_agentx_runtime_pins_from_state(state) == {}
    assert os.environ["AGENTX_MODEL_ID"] == ""


def test_resume_rejects_stale_agentx_epoch(monkeypatch):
    """Same knobs, different workload: the old numbers cannot anchor."""
    _on(monkeypatch)
    reason = agentx_state_is_stale(_St("agentx", AGENTX_MEASUREMENT_EPOCH - 1))
    assert "epoch" in reason


def test_resume_tolerates_sessions_predating_the_field(monkeypatch):
    """An empty mode means "not asserted", not "mismatch"."""
    _off(monkeypatch)
    assert agentx_state_is_stale(_St("", 0)) == ""


# --- submission verdict gate --------------------------------------------------


def _measurement(**over):
    # Serving shape: positive throughput plus at least one completed request.
    base = {"output_throughput": 100.0, "completed_requests": 42}
    base.update(over)
    return base


def _valid(result):
    from hyperloom.orchestrator.actions.executors.benchmark_result import (
        is_valid_measurement,
    )

    return is_valid_measurement(result)


def test_verdict_gate_rejects_a_failed_submission(monkeypatch):
    """A scenario-rejected run is not comparable and must not reach KEEP."""
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is False


def test_verdict_gate_rejects_an_unknown_verdict(monkeypatch):
    """None means no scenario, or an aiperf too old to stamp one."""
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=None)) is False


def test_verdict_gate_accepts_a_valid_submission(monkeypatch):
    _on(monkeypatch)
    assert _valid(_measurement(submission_valid=True)) is True


def test_verdict_gate_is_inert_on_the_synthetic_path(monkeypatch):
    """``is_valid_measurement`` is hot for every synthetic measurement too."""
    _off(monkeypatch)
    assert _valid(_measurement(submission_valid=False)) is True
    assert _valid(_measurement(submission_valid=None)) is True


def test_verdict_gate_spares_scriptable_runs_under_agentx(monkeypatch):
    """A scriptable framework skips the aiperf switch entirely."""
    _on(monkeypatch)
    assert _valid(_measurement(framework="xdit", quality_gate={"passed": True})) is True


# --- inner Magpie timeout follows the AgentX cap ------------------------------


def test_agentx_switch_does_not_resolve_launch_timeout(monkeypatch):
    """The flat Magpie ``timeout_seconds`` must follow the raised AgentX cap."""
    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "25200")
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == 7200
    assert bench["benchmark_script"] == "single_node/agentic/test_fp4_mi300x_vllm_mtp.sh"
    assert bench["agentx"] == "enable"


def test_agentx_switch_leaves_the_inner_timeout_alone_without_agentx(monkeypatch):
    """The default (synthetic) cap must be untouched when AgentX is off."""
    _off(monkeypatch)
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == 7200
    assert "benchmark_script" not in bench


def test_agentx_switch_skips_scriptable_inner_timeout(monkeypatch):
    """A scriptable framework returns early, so its cap is never rewritten."""
    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_BASELINE_TIMEOUT_SEC", "25200")
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    bench = {"framework": "xdit", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert bench["timeout_seconds"] == 7200


# --- the client's warmup bound must be the SCALED grace, not the raw one ------


def _switched(monkeypatch, **env):
    """Run the AgentX switch over a vllm bench and hand back its envs."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    return bench.get("envs", {})


def test_the_client_is_handed_the_conc_scaled_grace(monkeypatch):
    """One number, two layers."""
    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="3600", AGENTX_WARMUP_GRACE_CONC="8", CONC="32")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "14400"


def test_at_or_below_the_anchor_the_operators_grace_round_trips(monkeypatch):
    """Zero drift for every concurrency the old behaviour was validated at."""
    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="3600", CONC="8")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "3600"


def test_the_exported_grace_matches_what_the_cap_budgeted(monkeypatch):
    """The invariant itself, asserted directly rather than via two constants."""
    from hyperloom.orchestrator.actions.executors.baseline import (
        agentx_warmup_grace_sec,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    envs = _switched(monkeypatch, AGENTX_WARMUP_GRACE_PERIOD="1800", CONC="64")
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == str(agentx_warmup_grace_sec())


def test_nothing_is_exported_on_the_default_synthetic_path(monkeypatch):
    """AgentX off: no envs block, no grace, no benchmark_script -- untouched."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    _off(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    bench = {"framework": "vllm", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert "envs" not in bench
    assert "benchmark_script" not in bench
    assert bench["timeout_seconds"] == 7200


def test_a_scriptable_framework_gets_no_grace_either(monkeypatch):
    """The other early return: scriptable frameworks never reach the switch."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        apply_agentx_switch,
    )

    _on(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "32")
    bench = {"framework": "xdit", "model": "/models/x", "timeout_seconds": 7200}
    apply_agentx_switch(bench)
    assert "envs" not in bench


# --- a sweep variant's warmup bound must follow ITS concurrency ----------------


def _variant_envs(monkeypatch, tmp_path, *, session_conc, variant_conc, anchor="3600", grace_conc=None):
    """Materialize one grid variant and hand back the envs it will run with."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", anchor)
    monkeypatch.setenv("CONC", str(session_conc))
    if grace_conc is None:
        monkeypatch.delenv("AGENTX_WARMUP_GRACE_CONC", raising=False)
    else:
        monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", str(grace_conc))

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )
    variant = GridVariant(
        name="v0",
        extra_server_args="",
        extra_envs={"CONC": str(variant_conc)},
    )
    out = tmp_path / "v0"
    out.mkdir(exist_ok=True)
    cfg_path = _build_variant_yaml(base, "", variant, output_subdir=out)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return cfg["benchmark"]["envs"]


def test_a_sweep_variant_is_bounded_by_its_own_concurrency(monkeypatch, tmp_path):
    """A rung's grace follows the rung, not the concurrency the session started at."""
    envs = _variant_envs(monkeypatch, tmp_path, session_conc=8, variant_conc=128, grace_conc=8)
    assert envs["CONC"] == "128"
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == str(3600 * 128 // 8)


def test_a_lower_rung_is_not_given_the_sessions_larger_grace(monkeypatch, tmp_path):
    """The scaling only ever raises; at or below the anchor it is the identity."""
    envs = _variant_envs(monkeypatch, tmp_path, session_conc=32, variant_conc=2, grace_conc=8)
    assert envs["AGENTX_WARMUP_GRACE_PERIOD"] == "3600"


def test_the_inner_cap_moves_with_the_grace(monkeypatch, tmp_path):
    """The two bounds are derived from the same number and must not disagree."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _on(monkeypatch)
    monkeypatch.delenv("AGENTX_BASELINE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_CONC", "8")
    monkeypatch.setenv("CONC", "8")

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )

    def _cap_for(conc: int) -> int:
        out = tmp_path / f"v{conc}"
        out.mkdir(exist_ok=True)
        cfg_path = _build_variant_yaml(
            base,
            "",
            GridVariant(name=f"v{conc}", extra_server_args="", extra_envs={"CONC": str(conc)}),
            output_subdir=out,
        )
        return int(yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["benchmark"]["timeout_seconds"])

    assert _cap_for(64) == _cap_for(8) == 7200


def test_a_rung_that_names_no_concurrency_reads_the_session(monkeypatch):
    from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant, variant_conc

    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "16"})) == 16
    assert variant_conc(GridVariant(name="v", extra_envs={})) is None
    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "nonsense"})) is None
    assert variant_conc(GridVariant(name="v", extra_envs={"CONC": "0"})) is None
    assert variant_conc(None) is None


def test_the_default_grid_never_re_derives_a_grace(monkeypatch, tmp_path):
    """AgentX off: a synthetic variant's env must carry no warmup grace at all."""
    import yaml

    from hyperloom.orchestrator.actions.executors._grid_runner import (
        GridVariant,
        _build_variant_yaml,
    )

    _off(monkeypatch)
    monkeypatch.setenv("AGENTX_WARMUP_GRACE_PERIOD", "3600")
    monkeypatch.setenv("CONC", "8")

    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "/m/x", "timeout_seconds": 7200, "envs": {}}}),
        encoding="utf-8",
    )
    out = tmp_path / "v0"
    out.mkdir(exist_ok=True)
    cfg_path = _build_variant_yaml(
        base, "", GridVariant(name="v0", extra_server_args="", extra_envs={"CONC": "128"}), output_subdir=out
    )
    envs = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert "AGENTX_WARMUP_GRACE_PERIOD" not in envs
    assert envs["CONC"] == "128"
