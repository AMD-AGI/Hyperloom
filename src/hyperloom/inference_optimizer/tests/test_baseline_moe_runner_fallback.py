# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline one-shot fallback when the injected MoE runner backend kills the server.

Hyperloom injects ``--moe-runner-backend triton`` for MoE sglang models on AMD.
Quant schemes without a triton MoE runner (e.g. Quark MXFP4) crash on the first
forward pass; the executor retries once with the flag dropped.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

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
    """A MoE checkpoint dir with no quant marker (so triton IS injected)."""
    d = tmp_path / "Qwen3-30B-A3B"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "num_experts": 128}),
        encoding="utf-8",
    )
    return str(d)


def _write_yaml(path: Path, model: str) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
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


def _fake_workspace(slot: Path, *, tput: float = 1237.0) -> Path:
    ws = slot / "benchmark_sglang_20260728_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        )
    )
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-moe-1", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


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


# --- end-to-end fallback ----------------------------------------------------
def _sglang_args(cmd) -> str:
    cfg = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
    return str(cfg["benchmark"]["envs"].get("EXTRA_SGLANG_ARGS", ""))


def test_moe_runner_crash_triggers_flagless_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sglang_args = _sglang_args(cmd)
        calls.append(sglang_args)
        if "--moe-runner-backend" in sglang_args:
            return subprocess.CompletedProcess(cmd, 1, "", _QUARK_TRACEBACK)
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    # Warmup crashes on triton, the retry re-runs warmup + measure flagless.
    assert len(calls) == 3
    assert "--moe-runner-backend triton" in calls[0]
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert result["status"] == "succeeded"
    assert "moe_runner_backend_fallback_dropped_flag" in result.get("nonfatal_warnings", [])


def test_operator_pinned_backend_is_also_dropped_on_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sglang_args = _sglang_args(cmd)
        calls.append(sglang_args)
        if "--moe-runner-backend" in sglang_args:
            return subprocess.CompletedProcess(cmd, 1, "", _QUARK_TRACEBACK)
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "disable_run_eval": True,
            "extra_server_args": "--moe-runner-backend triton --foo 1",
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert len(calls) == 3
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert "--foo 1" in calls[1]
    assert result["status"] == "succeeded"


@pytest.mark.parametrize("source", ["operator_env", "reference_recipe"])
def test_backend_from_other_arg_sources_is_dropped_on_retry(tmp_path, monkeypatch, source):
    """The flag can also arrive via $INFERENCE_OPTIMIZER_SERVER_ARGS or the
    reference recipe; both are merged after the task params, so the retry must
    strip the merged result rather than only the params."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    params = {
        "output_dir": str(tmp_path / "ws"),
        "timeout_sec": 10,
        "model_path": model,
        "gpu_type": "mi300x",
        "disable_run_eval": True,
    }
    if source == "operator_env":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--moe-runner-backend triton")
    else:
        params["reference_server_args"] = "--moe-runner-backend triton"
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        sglang_args = _sglang_args(cmd)
        calls.append(sglang_args)
        if "--moe-runner-backend" in sglang_args:
            return subprocess.CompletedProcess(cmd, 1, "", _QUARK_TRACEBACK)
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(_make_ctx(params)))

    assert "--moe-runner-backend triton" in calls[0]
    assert all("--moe-runner-backend" not in c for c in calls[1:])
    assert result["status"] == "succeeded"


def test_moe_fallback_keeps_eval_disabled_by_earlier_fallback(tmp_path, monkeypatch):
    """An eval-rooted failure turns eval off; a MoE failure on that retry must
    keep it off instead of resurrecting the eval that already broke."""
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    calls: list[tuple[str, str]] = []

    def fake_run(cmd, *args, **kwargs):
        cfg = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        sglang_args = str(cfg["benchmark"]["envs"].get("EXTRA_SGLANG_ARGS", ""))
        calls.append((run_eval, sglang_args))
        if run_eval != "false":
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")
        if "--moe-runner-backend" in sglang_args:
            return subprocess.CompletedProcess(cmd, 1, "", _QUARK_TRACEBACK)
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert [c[0] for c in calls] == ["true", "false", "false", "false"]
    assert all("--moe-runner-backend" not in args for _, args in calls[2:])
    assert result["status"] == "succeeded"
    # Both fallbacks stay visible on the salvaged result.
    assert result.get("accuracy_source") == "eval_unavailable"
    warnings = result.get("nonfatal_warnings", [])
    assert "eval_failed_fallback_no_accuracy" in warnings
    assert "moe_runner_backend_fallback_dropped_flag" in warnings


def test_fallback_fires_only_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(_sglang_args(cmd))
        # The crash persists even without the flag: no endless retry loop.
        return subprocess.CompletedProcess(cmd, 1, "", _QUARK_TRACEBACK)

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert len(calls) == 2
    assert result["status"] == "failed"


def test_unrelated_failure_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    model = _moe_model_dir(tmp_path)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(_sglang_args(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "CUDA out of memory\n")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": model,
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert len(calls) == 1
    assert result["status"] == "failed"
