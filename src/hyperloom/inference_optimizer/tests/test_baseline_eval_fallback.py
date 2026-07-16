# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Baseline accuracy-eval handling: ``disable_run_eval`` wiring + the eval-failure fallback that salvages the throughput baseline."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.common.env import is_truthy
from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _write_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/wekafs/models/Qwen-Qwen3-8B",
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


def _fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:
    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/wekafs/models/Qwen-Qwen3-8B",
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
    task = SimpleNamespace(task_id="t-eval-1", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


# --- is_truthy (baseline's disable_run_eval param interpretation) ----------
@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", False),
        (None, False),
        ("nonsense", False),
    ],
)
def test_is_truthy(value, expected):
    assert is_truthy(value) is expected


# --- _is_eval_rooted_failure ----------------------------------------------
def test_eval_rooted_failure_from_error_tail():
    result = {"status": "failed", "error": "...\nERROR: run_eval failed with exit code 1\n"}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_from_warning():
    result = {
        "status": "failed",
        "error": "boom",
        "nonfatal_warnings": ["Unknown parameter: --concurrent-requests"],
    }
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_negative():
    result = {"status": "failed", "error": "CUDA out of memory"}
    assert BaselineExecutor._is_eval_rooted_failure(result) is False


def test_eval_rooted_failure_scans_logs(tmp_path: Path):
    out = tmp_path / "task"
    ws = out / "benchmark_sglang_x"
    ws.mkdir(parents=True)
    (ws / "benchmark_stderr.log").write_text(
        "+ run_eval ...\nrun_eval failed with exit code 1\n", encoding="utf-8"
    )
    result = {"status": "failed", "error": "generic", "output_dir": str(out)}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_climbs_from_round_subdir(tmp_path: Path):
    # The result points at measure_round but the eval marker lives in the
    # sibling warmup_round; the scan must climb to the task root.
    task = tmp_path / "task"
    warm_ws = task / "warmup_round" / "benchmark_sglang_x"
    warm_ws.mkdir(parents=True)
    (warm_ws / "server.log").write_text(
        "Unknown parameter: --concurrent-requests\n", encoding="utf-8"
    )
    measure = task / "measure_round"
    measure.mkdir(parents=True)
    result = {"status": "failed", "error": "100% request failures", "output_dir": str(measure)}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


# --- disable_run_eval -> RUN_EVAL=false ------------------------------------
def test_disable_run_eval_param_forces_run_eval_false(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        captured["cfg"] = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        _fake_workspace(Path(cmd[out_idx + 1]))
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
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert str(captured["cfg"]["benchmark"]["envs"]["RUN_EVAL"]).lower() == "false"


# --- eval-failure fallback end-to-end --------------------------------------
def test_eval_failure_triggers_run_eval_false_retry(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    calls: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        slot = Path(cmd[out_idx + 1])
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        calls.append({"run_eval": run_eval})
        if run_eval != "false":
            # Simulate a broken eval that aborts the script: no valid workspace,
            # marker in stderr.
            return subprocess.CompletedProcess(
                cmd, 1, "", "ERROR: run_eval failed with exit code 1\n"
            )
        _fake_workspace(slot)
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
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    # Warmup tries eval=true, falls back to eval=false, then the measured
    # baseline reuses the eval-disabled config.
    assert [c["run_eval"] for c in calls] == ["true", "false", "false"]
    assert result["status"] == "succeeded"
    assert result.get("accuracy_source") == "eval_unavailable"
    assert "eval_failed_fallback_no_accuracy" in result.get("nonfatal_warnings", [])


def test_non_eval_failure_does_not_retry(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append("x")
        # A non-eval failure (no marker), no workspace -> failed, no retry.
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
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert len(calls) == 1  # no fallback retry
    assert result["status"] == "failed"
    assert result.get("accuracy_source") != "eval_unavailable"


def _make_baseline_ctx(params: dict, shared_state) -> SimpleNamespace:
    """A genuine ``baseline`` ctx carrying a live SharedState for stop wiring."""
    task = SimpleNamespace(task_id="t-bl-acc", kind="baseline", params=params)
    return SimpleNamespace(task=task, extra={"shared_state": shared_state})


# --- baseline accuracy missing -> stop the whole run -----------------------
def test_baseline_missing_accuracy_stops_run(tmp_path):
    """Serving baseline with eval expected but no accuracy result -> the run
    halts with ``stop_reason=baseline_accuracy_failed`` (broken setup)."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    base = tmp_path / "base.yaml"
    _write_yaml(base)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]))  # throughput only, no GSM8K
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    state = SharedState()
    ctx = _make_baseline_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        },
        state,
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert result.get("accuracy") is None
    assert state.stop_reason == "baseline_accuracy_failed"


def test_baseline_operator_disabled_eval_does_not_stop(tmp_path):
    """When the operator explicitly disables the serving eval, accuracy is
    intentionally off: a missing accuracy result must NOT stop the run."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    base = tmp_path / "base.yaml"
    _write_yaml(base)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    state = SharedState()
    ctx = _make_baseline_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        },
        state,
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert state.stop_reason == ""


def test_baseline_eval_failure_fallback_stops_run(tmp_path):
    """The eval-failure fallback still salvages the throughput baseline, but a
    genuine baseline with no accuracy result now halts the run."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    base = tmp_path / "base.yaml"
    _write_yaml(base)

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        slot = Path(cmd[out_idx + 1])
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        if run_eval != "false":
            return subprocess.CompletedProcess(
                cmd, 1, "", "ERROR: run_eval failed with exit code 1\n"
            )
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    state = SharedState()
    ctx = _make_baseline_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        },
        state,
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert result.get("accuracy_source") == "eval_unavailable"
    assert state.stop_reason == "baseline_accuracy_failed"


def test_eval_already_off_does_not_retry(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append("x")
        # Even with an eval marker, an explicit opt-out must not double-run.
        return subprocess.CompletedProcess(

            cmd, 1, "", "ERROR: run_eval failed with exit code 1\n"
        )

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/wekafs/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
            "disable_run_eval": True,
        }
    )
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert len(calls) == 1  # already off, no fallback
    assert result["status"] == "failed"
