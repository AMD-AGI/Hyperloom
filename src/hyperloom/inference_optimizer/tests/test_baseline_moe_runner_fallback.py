# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline one-shot fallback when a pinned MoE runner backend kills the server.

Hyperloom no longer injects a default ``--moe-runner-backend`` (sglang's own
``auto`` already picks the right one). An operator pin, a reference recipe, or
an inherited grid/variant flag can still select a scheme without a runner (e.g.
Quark MXFP4 forced onto ``triton``), crashing on the first forward pass; the
executor retries once with the flag dropped.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

from ._baseline_fakes import fake_workspace, run_baseline, write_sglang_yaml

_QUARK_TRACEBACK = (
    'File ".../quark/schemes/quark_w4a4_mxfp4_moe.py", line 295, in apply_weights\n'
    "    return self.runner.run(dispatch_output, quant_info)\n"
    "AttributeError: 'QuarkW4A4MXFp4MoE' object has no attribute 'runner'\n"
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))
    monkeypatch.delenv("HYPERLOOM_SGLANG_MOE_RUNNER_BACKEND", raising=False)


def _moe_model_dir(tmp_path: Path) -> str:
    """A MoE checkpoint dir with no quant marker."""
    d = tmp_path / "Qwen3-30B-A3B"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "num_experts": 128}),
        encoding="utf-8",
    )
    return str(d)


def _quark_mxfp4_moe_model_dir(tmp_path: Path) -> str:
    """A Quark MX-FP4 MoE checkpoint dir (the gate skips injection for it)."""
    spec = {"dtype": "fp4", "qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
    d = tmp_path / "Qwen3.5-397B-A17B-MXFP4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_experts": 128,
                "quantization_config": {
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": {**spec, "is_dynamic": False},
                        "input_tensors": {**spec, "is_dynamic": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return str(d)


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-moe-1", params=params)
    return SimpleNamespace(task=task, extra={})


# --- _is_moe_runner_rooted_failure -----------------------------------------
def test_moe_runner_failure_from_error_tail():
    result = {"status": "failed", "error": _QUARK_TRACEBACK}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


def test_moe_runner_failure_scans_logs(tmp_path: Path):
    out = tmp_path / "task"
    ws = out / "benchmark_sglang_x"
    ws.mkdir(parents=True)
    (ws / "server.log").write_text(_QUARK_TRACEBACK, encoding="utf-8")
    result = {"status": "failed", "error": "server died during startup", "output_dir": str(out)}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


def test_moe_runner_failure_negative():
    result = {"status": "failed", "error": "CUDA out of memory"}
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is False


def test_unrelated_missing_runner_attribute_is_not_a_moe_failure():
    # A missing 'runner' attribute outside the MoE path must not burn a retry.
    result = {
        "status": "failed",
        "error": (
            'File ".../bench/harness.py", line 12, in start\n'
            "    self.runner.run(request)\n"
            "AttributeError: 'BenchHarness' object has no attribute 'runner'\n"
        ),
    }
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is False


def test_moe_failure_detected_for_non_quark_scheme():
    # Detection is keyed on the MoE runner, not on Quark: sglang's int4fp8 and mxfp4 dynamic-quant MoE methods fail
    # the same way.
    result = {
        "status": "failed",
        "error": (
            'File ".../layers/quantization/quark_int4fp8_moe.py", line 446, in apply\n'
            "    return self.runner.run(dispatch_output, quant_info)\n"
            "AttributeError: 'QuarkInt4Fp8MoEMethod' object has no attribute 'runner'\n"
        ),
    }
    assert BaselineExecutor._is_moe_runner_rooted_failure(result) is True


# --- end-to-end fallback ----------------------------------------------------
def _sglang_args(cmd) -> str:
    cfg = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
    return str(cfg["benchmark"]["envs"].get("EXTRA_SGLANG_ARGS", ""))


def _crash_on_pin(sglang_args: str) -> str | None:
    return _QUARK_TRACEBACK if "--moe-runner-backend" in sglang_args else None


def _run_baseline(tmp_path, model: str, crash_stderr, **params) -> tuple[list[str], dict]:
    """Run the baseline over a Magpie that fails with ``crash_stderr(sglang_args)`` when it returns text."""
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sglang_args = _sglang_args(cmd)
        calls.append(sglang_args)
        stderr = crash_stderr(sglang_args)
        if stderr is not None:
            return subprocess.CompletedProcess(cmd, 1, "", stderr)
        fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "extra_envs": {"RUN_EVAL": "false"},
            **params,
        }
    )
    result = run_baseline(write_sglang_yaml(tmp_path / "base.yaml", model), tmp_path, ctx, fake_run)
    return calls, result


def test_moe_runner_crash_triggers_flagless_retry(tmp_path, monkeypatch):
    """A pinned backend that crashes (e.g. a stale recipe/task param, since
    Hyperloom itself no longer injects one) still gets stripped on retry."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    calls, result = _run_baseline(
        tmp_path, _moe_model_dir(tmp_path), _crash_on_pin, extra_server_args="--moe-runner-backend triton"
    )

    # Warmup crashes on the pinned triton, the retry re-runs warmup + measure flagless.
    assert len(calls) == 3
    assert "--moe-runner-backend triton" in calls[0]
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert result["status"] == "succeeded"
    assert "moe_runner_backend_fallback_dropped_flag" in result.get("nonfatal_warnings", [])


def test_operator_pinned_backend_is_also_dropped_on_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    calls, result = _run_baseline(
        tmp_path, _moe_model_dir(tmp_path), _crash_on_pin, extra_server_args="--moe-runner-backend triton --foo 1"
    )

    assert len(calls) == 3
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert "--foo 1" in calls[1]
    assert result["status"] == "succeeded"


@pytest.mark.parametrize("source", ["operator_env", "reference_recipe"])
def test_backend_from_other_arg_sources_is_dropped_on_retry(tmp_path, monkeypatch, source):
    """The flag can also arrive via $INFERENCE_OPTIMIZER_SERVER_ARGS or the reference recipe; both are merged after the task params, so the retry must strip the merged result rather than only the params."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    if source == "operator_env":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--moe-runner-backend triton")
    else:
        from hyperloom.orchestrator.state.shared_state import SharedState

        sd = tmp_path / "session"
        sd.mkdir(parents=True, exist_ok=True)
        SharedState(session_id="ref", reference_server_args="--moe-runner-backend triton").save(sd)
        monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(sd))
    calls, result = _run_baseline(tmp_path, _moe_model_dir(tmp_path), _crash_on_pin)

    assert "--moe-runner-backend triton" in calls[0]
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert result["status"] == "succeeded"


def test_quark_checkpoint_with_operator_pinned_backend_recovers(tmp_path, monkeypatch):
    """The original bug shape: on a Quark MX-FP4 checkpoint the gate skips injection, but an operator pin still reaches the server on attempt 1."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    calls, result = _run_baseline(
        tmp_path,
        _quark_mxfp4_moe_model_dir(tmp_path),
        _crash_on_pin,
        extra_server_args="--moe-runner-backend triton --mem-fraction-static 0.9",
    )

    assert "--moe-runner-backend triton" in calls[0]
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert "--mem-fraction-static 0.9" in calls[1]
    assert result["status"] == "succeeded"


def test_quark_checkpoint_without_pin_never_gets_the_flag(tmp_path, monkeypatch):
    """Without an operator/recipe pin, a Quark MX-FP4 checkpoint launches
    clean on the first attempt (Hyperloom never injects the flag) -- no
    crash, no retry."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    calls, result = _run_baseline(tmp_path, _quark_mxfp4_moe_model_dir(tmp_path), _crash_on_pin)

    assert all("--moe-runner-backend" not in c for c in calls)
    assert result["status"] == "succeeded"
    assert "moe_runner_backend_fallback_dropped_flag" not in result.get("nonfatal_warnings", [])


def test_fallback_fires_only_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    # The crash persists even without the flag: no endless retry loop.
    calls, result = _run_baseline(tmp_path, _moe_model_dir(tmp_path), lambda _args: _QUARK_TRACEBACK)

    assert len(calls) == 2
    assert result["status"] == "failed"


def test_unrelated_failure_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    calls, result = _run_baseline(tmp_path, _moe_model_dir(tmp_path), lambda _args: "CUDA out of memory\n")

    assert len(calls) == 1
    assert result["status"] == "failed"
