# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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


class _StopRecorder:
    """Minimal SharedState stub capturing ``set_stop_reason`` calls."""

    def __init__(self, enablement_mode: str = "off") -> None:
        self.stop_reason = ""
        self.baseline_accuracy = 0.0
        self.enablement_mode = enablement_mode
        # Mirrors the SharedState default so ctx-backed runs keep the cold+hot pair.
        self.baseline_double_run = True

    def set_stop_reason(self, value, **_kwargs):
        self.stop_reason = value
        return value


def _write_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
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
                "model": "/path/models/Qwen-Qwen3-8B",
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


def _make_ctx(params: dict, *, enablement_mode: str = "off") -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-eval-1", params=params)
    return SimpleNamespace(task=task, extra={"shared_state": _StopRecorder(enablement_mode)})


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
    (ws / "benchmark_stderr.log").write_text("+ run_eval ...\nrun_eval failed with exit code 1\n", encoding="utf-8")
    result = {"status": "failed", "error": "generic", "output_dir": str(out)}
    assert BaselineExecutor._is_eval_rooted_failure(result) is True


def test_eval_rooted_failure_climbs_from_round_subdir(tmp_path: Path):
    # The result points at measure_round but the eval marker lives in the
    # sibling warmup_round; the scan must climb to the task root.
    task = tmp_path / "task"
    warm_ws = task / "warmup_round" / "benchmark_sglang_x"
    warm_ws.mkdir(parents=True)
    (warm_ws / "server.log").write_text("Unknown parameter: --concurrent-requests\n", encoding="utf-8")
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
            "model_path": "/path/models/Qwen-Qwen3-8B",
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


def test_no_eval_forces_run_eval_false(tmp_path):
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
            "model_path": "/path/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        }
    )
    ctx.extra["shared_state"].eval_disabled = True
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert str(captured["cfg"]["benchmark"]["envs"]["RUN_EVAL"]).lower() == "false"
    assert ctx.extra["shared_state"].stop_reason == ""


def test_eval_disabled_resolves_without_a_ctx_shared_state(tmp_path):
    # The kernel integrate lane builds the executor with no shared_state= and a
    # RunnerContext with no extra, so the flag has to come off the session dir.
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="s1")
    state.eval_disabled = True
    state.save(tmp_path)

    executor = BaselineExecutor(session_dir=tmp_path)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t", kind="baseline", params={}), extra=None)
    assert executor._eval_disabled(ctx) is True


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
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")
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
            "model_path": "/path/models/Qwen-Qwen3-8B",
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


def test_eval_crash_routes_to_enablement_no_salvage(tmp_path, monkeypatch):
    """flag on + single-node: an eval crash is stamped as an eval-failure
    contract with no RUN_EVAL=false salvage retry."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    calls: list[dict] = []

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        calls.append({"run_eval": run_eval})
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/path/models/Qwen-Qwen3-8B",
            "gpu_type": "mi300x",
        },
        enablement_mode="eval",
    )
    ctx.task.kind = "baseline"
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    # No RUN_EVAL=false salvage retry: eval stays on.
    assert all(c["run_eval"] != "false" for c in calls)
    assert result["status"] == "failed"
    assert result["baseline_eval_failed"] is True
    assert result["baseline_eval_failure_kind"] == "eval_runtime_failure"
    assert result["eval_origin"] == "eval"
    assert result["materialized_config"]
    assert result["baseline_eval_contract_fingerprint"]


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
            "model_path": "/path/models/Qwen-Qwen3-8B",
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
    state = SharedState(enablement_mode="off")
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


def test_baseline_operator_disabled_eval_still_stops(tmp_path):
    """Disabling the serving eval on a genuine baseline is not an opt-out.

    A baseline exists to establish the accuracy reference, so turning the eval
    off does not make a missing accuracy acceptable -- it only means the
    reference was never measured. The run halts either way, which is what makes
    ``disable_run_eval`` useless as a way around the accuracy gate.
    """
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
    state = SharedState(enablement_mode="off")
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
    assert state.stop_reason == "baseline_accuracy_failed"


def test_baseline_eval_failure_stops_run_without_burning_a_retry(tmp_path):
    """A genuine baseline whose eval aborted must stop the run IMMEDIATELY.

    Regression (2026-07-27 Qwen3-8B outage): the executor used to re-run the
    whole baseline with ``RUN_EVAL=false`` to "salvage the throughput
    baseline", and ``_maybe_stop_on_missing_baseline_accuracy`` then halted the
    run anyway (the fallback is tagged ``eval_unavailable``, which is NOT an
    operator opt-out). That burned a second full server boot + benchmark to
    produce a result guaranteed to be discarded. The retry is now skipped for a
    genuine ``baseline`` task and the same ``baseline_accuracy_failed`` stop is
    recorded straight away.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    base = tmp_path / "base.yaml"
    _write_yaml(base)
    run_evals: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        slot = Path(cmd[out_idx + 1])
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        run_evals.append(run_eval)
        if run_eval != "false":
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    state = SharedState(enablement_mode="off")
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

    # No RUN_EVAL=false salvage round was launched -- the wasted budget is gone.
    assert "false" not in run_evals
    assert result["status"] == "failed"
    assert result.get("accuracy_source") == "eval_unavailable"
    assert "eval_failed_no_fallback_baseline_requires_accuracy" in result.get("nonfatal_warnings", [])
    # Same operator-visible verdict as before, minutes earlier.
    assert state.stop_reason == "baseline_accuracy_failed"


def test_non_baseline_kind_still_gets_the_throughput_salvage_retry(tmp_path):
    """The fail-fast rule is scoped to genuine baselines.

    ``replay_warm_recipe`` reuses this executor but does NOT establish the
    quality reference, so a throughput-only result IS usable there and the
    one-shot ``RUN_EVAL=false`` salvage must still run.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    base = tmp_path / "base.yaml"
    _write_yaml(base)
    run_evals: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        slot = Path(cmd[out_idx + 1])
        run_eval = str(cfg["benchmark"]["envs"].get("RUN_EVAL", "true")).lower()
        run_evals.append(run_eval)
        if run_eval != "false":
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    state = SharedState()
    task = SimpleNamespace(task_id="t-warm", kind="replay_warm_recipe", params={
        "output_dir": str(tmp_path / "ws"),
        "timeout_sec": 10,
        "model_path": "/wekafs/models/Qwen-Qwen3-8B",
        "gpu_type": "mi300x",
    })
    ctx = SimpleNamespace(task=task, extra={"shared_state": state})
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert "false" in run_evals  # salvage retry ran
    assert result["status"] == "succeeded"
    assert result.get("accuracy_source") == "eval_unavailable"
    assert "eval_failed_fallback_no_accuracy" in result.get("nonfatal_warnings", [])
    # Not a genuine baseline -> the accuracy stop gate does not fire.
    assert state.stop_reason == ""


def _stop_ctx(framework: str, recorder, params: dict | None = None) -> SimpleNamespace:
    task = SimpleNamespace(
        task_id="t-bl",
        kind="baseline",
        params={"framework": framework, **(params or {})},
    )
    return SimpleNamespace(task=task, extra={"shared_state": recorder})


def _stopped(
    framework: str,
    result: dict,
    *,
    params: dict | None = None,
    eval_disabled: bool = False,
) -> str:
    """Run ``_maybe_stop_on_missing_baseline_accuracy`` and return the reason."""
    executor = BaselineExecutor()
    rec = _StopRecorder()
    rec.eval_disabled = eval_disabled
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx(framework, rec, params), result)
    return rec.stop_reason


# --- accuracy-stop decision matrix -----------------------------------------
def test_stop_scriptable_missing_gate_zero_accuracy():
    # Finding: scriptable fail-closed records accuracy=0.0 -> must still stop.
    reason = _stopped(
        "xdit",
        {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": True},
    )
    assert reason == "baseline_accuracy_failed"


def test_stop_serving_no_accuracy_eval_on():
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "run_eval_disabled": False},
    )
    assert reason == "baseline_accuracy_failed"


def test_stop_serving_zero_accuracy():
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": False},
    )
    assert reason == "baseline_accuracy_failed"


def test_stop_serving_operator_disabled_via_config():
    # A YAML/reference-env RUN_EVAL=false folds into run_eval_disabled, but a
    # genuine baseline has no opt-out: turning the eval off does not make a
    # missing accuracy reference acceptable, so the run still stops.
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "run_eval_disabled": True},
    )
    assert reason == "baseline_accuracy_failed"


def test_no_stop_when_eval_is_disabled():
    # --no-eval never asked for a reference, so the baseline anchors on
    # throughput instead of halting.
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "run_eval_disabled": True},
        eval_disabled=True,
    )
    assert reason == ""


def test_no_stop_when_quality_ref_exempt():
    # Synthetic kernel-lane re-baselines (kind="baseline" + quality_ref_exempt)
    # are throughput-only A/B probes: no accuracy, no stop.
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "run_eval_disabled": True},
        params={"quality_ref_exempt": True},
    )
    assert reason == ""


def test_stop_serving_eval_failure_fallback():
    # Fallback forces RUN_EVAL=false but eval was expected and broke -> stop.
    reason = _stopped(
        "sglang",
        {
            "status": "succeeded",
            "run_eval_disabled": True,
            "accuracy_source": "eval_unavailable",
        },
    )
    assert reason == "baseline_accuracy_failed"


def test_no_stop_valid_accuracy():
    reason = _stopped(
        "sglang",
        {"status": "succeeded", "accuracy": 0.85, "run_eval_disabled": False},
    )
    assert reason == ""


def test_no_stop_when_not_genuine_baseline():
    executor = BaselineExecutor()
    rec = _StopRecorder()
    task = SimpleNamespace(task_id="t", kind="replay_warm_recipe", params={"framework": "sglang"})
    ctx = SimpleNamespace(task=task, extra={"shared_state": rec})
    executor._maybe_stop_on_missing_baseline_accuracy(ctx, {"status": "succeeded", "run_eval_disabled": False})
    assert rec.stop_reason == ""


def test_no_stop_when_baseline_failed():
    reason = _stopped("sglang", {"status": "failed", "error": "boom"})
    assert reason == ""


# --- session-level salvage (sibling attempt already measured accuracy) -------
def _write_gsm8k_results(measure_round: Path, score: float) -> None:
    """Write a minimal lm-eval ``results*.json`` under a measure round dir."""
    d = measure_round / "benchmark_vllm_x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_x.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}),
        encoding="utf-8",
    )


def test_salvage_sibling_attempt_accuracy_prevents_stop(tmp_path):
    # A sibling attempt already produced a valid gsm8k result; the deciding
    # attempt's own RESULT_DIR is empty. The run must NOT stop -- the accuracy
    # is salvaged from the sibling and promoted onto the result.
    runs_baseline = tmp_path / "runs" / "baseline"
    good = runs_baseline / "786a793e" / "measure_round"
    _write_gsm8k_results(good, 0.9128)
    deciding = runs_baseline / "retry2_bootsafe"
    deciding.mkdir(parents=True, exist_ok=True)

    executor = BaselineExecutor()
    rec = _StopRecorder()
    result = {
        "status": "succeeded",
        "run_eval_disabled": False,
        "output_dir": str(deciding),
    }
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx("vllm", rec), result)

    assert rec.stop_reason == ""
    assert result["accuracy"] == pytest.approx(0.9128)
    assert rec.baseline_accuracy == pytest.approx(0.9128)
    assert "baseline_accuracy_salvaged_from_sibling_attempt" in result.get("nonfatal_warnings", [])


def test_salvage_uses_a_warmup_score_when_it_is_the_only_one(tmp_path):
    """A warmup-round eval is a valid accuracy source, so the run must not stop.

    What a warmup discards is throughput -- the cold-boot window inflates later
    gains. Accuracy is not timing-sensitive, and the baseline double-run now
    evaluates only in the warmup round, so this is the sole score available.
    """
    runs_baseline = tmp_path / "runs" / "baseline"
    _write_gsm8k_results(runs_baseline / "786a793e" / "warmup_round", 0.9)
    deciding = runs_baseline / "retry2_bootsafe"
    deciding.mkdir(parents=True, exist_ok=True)

    reason = _stopped(
        "vllm",
        {"status": "succeeded", "run_eval_disabled": False, "output_dir": str(deciding)},
    )
    assert reason == ""


def test_salvage_prefers_a_measured_round_over_a_warmup(tmp_path):
    """The warmup is a fallback, not a substitute: a real round still wins."""
    runs_baseline = tmp_path / "runs" / "baseline"
    _write_gsm8k_results(runs_baseline / "786a793e" / "warmup_round", 0.10)
    _write_gsm8k_results(runs_baseline / "786a793e" / "measure_round", 0.90)
    deciding = runs_baseline / "retry2_bootsafe"
    deciding.mkdir(parents=True, exist_ok=True)

    from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results

    parsed = parse_eval_results(runs_baseline, framework="vllm")
    assert parsed["accuracy"] == pytest.approx(0.90)


def test_eval_already_off_does_not_retry(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    calls: list[str] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append("x")
        # Even with an eval marker, an explicit opt-out must not double-run.
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(tmp_path / "ws"),
            "timeout_sec": 10,
            "model_path": "/path/models/Qwen-Qwen3-8B",
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


# --- eval-origin enablement routing (flag on) ------------------------------
from hyperloom.orchestrator.actions.executors._accuracy_gate import (  # noqa: E402
    BASELINE_EVAL_ACCURACY_FLOOR_KEY,
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    EVAL_KIND_ACCURACY_BELOW_FLOOR,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
    EVAL_KIND_GENERATION_PATHOLOGY,
)


def _write_minimal_route_yaml(tmp_path: Path, framework: str = "sglang") -> Path:
    """Write a minimal materialized YAML for _route tests."""
    p = tmp_path / "route_config.yaml"
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": "/path/models/test",
            "benchmark_script": f"{framework}_mi300x.sh",
            "precision": "bf16",
            "envs": {"CONC": 64, "ISL": 1024, "OSL": 1024, "TP": 8, "RUN_EVAL": "true"},
        }
    }
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _route(monkeypatch, framework, result, *, nodes=None, tmp_path=None):
    if nodes is not None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", str(nodes))
    if tmp_path is not None and "materialized_config" not in result:
        result["materialized_config"] = str(_write_minimal_route_yaml(tmp_path, framework))
    executor = BaselineExecutor()
    rec = _StopRecorder("eval")
    executor._maybe_stop_on_missing_baseline_accuracy(_stop_ctx(framework, rec), result)
    return rec.stop_reason


def test_eval_enablement_missing_accuracy_routes_not_stop(monkeypatch, tmp_path):
    cfg_path = str(_write_minimal_route_yaml(tmp_path))
    result = {"status": "succeeded", "run_eval_disabled": False, "materialized_config": cfg_path}
    reason = _route(monkeypatch, "sglang", result, tmp_path=tmp_path)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_UNAVAILABLE
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] is None
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == DEFAULT_ENABLEMENT_ACCURACY_FLOOR
    assert result[BASELINE_EVAL_EVIDENCE_KEY]
    assert result[BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY]
    assert result["eval_origin"] == "eval"


def test_eval_enablement_zero_accuracy_below_floor(monkeypatch):
    result = {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_BELOW_FLOOR
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == 0.0


def test_eval_enablement_probe_reports_generation_pathology(monkeypatch):
    """A tripped probe changes what a ~0 score means: the eval was cut short
    because the model never stopped generating, not because it answered and got
    them wrong. Without this the specialist is handed a bare ``accuracy=0.0``
    and goes looking for a quality regression that never happened.
    """
    result = {
        "status": "succeeded",
        "accuracy": 0.0,
        "run_eval_disabled": False,
        "eval_probe": {
            "kind": EVAL_KIND_GENERATION_PATHOLOGY,
            "observed_samples": 128,
            "cap_hits": 128,
            "max_completion_tokens_seen": 16384,
        },
    }
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_GENERATION_PATHOLOGY
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == 0.0
    evidence = result[BASELINE_EVAL_EVIDENCE_KEY]
    assert "128/128" in evidence
    assert "16384" in evidence


def test_eval_enablement_positive_below_floor(monkeypatch):
    observed = DEFAULT_ENABLEMENT_ACCURACY_FLOOR / 2
    result = {"status": "succeeded", "accuracy": observed, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_BELOW_FLOOR
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == observed
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == DEFAULT_ENABLEMENT_ACCURACY_FLOOR


def test_eval_enablement_accuracy_at_floor_passes(monkeypatch):
    result = {
        "status": "succeeded",
        "accuracy": DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
        "run_eval_disabled": False,
    }
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_eval_enablement_multi_node_falls_back_to_stop(monkeypatch):
    result = {"status": "succeeded", "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result, nodes=2)
    assert reason == "baseline_accuracy_failed"
    assert BASELINE_EVAL_FAILED_KEY not in result


def test_eval_enablement_operator_optout_is_routed(monkeypatch):
    """A disabled eval is not an opt-out: with enablement on it routes there.

    Rather than stopping, the missing reference is stamped as an eval-failure
    contract so enablement can repair the eval path. ``_is_promotable_result``
    then keeps this baseline from anchoring tput / accuracy / config.
    """
    result = {"status": "succeeded", "run_eval_disabled": True}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILED_KEY] is True


def test_eval_enablement_quality_ref_exempt_not_routed(monkeypatch):
    """Synthetic kernel-lane re-baselines are neither routed nor stopped."""
    result = {"status": "succeeded", "run_eval_disabled": True}
    executor = BaselineExecutor()
    rec = _StopRecorder("eval")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    executor._maybe_stop_on_missing_baseline_accuracy(
        _stop_ctx("sglang", rec, {"quality_ref_exempt": True}), result
    )
    assert rec.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result


# --- regression: the --concurrent-requests flag gate (2026-07-27 outage) ----
# Magpie re-copies its own generic *.sh scripts into <inferencex>/benchmarks/ on
# every run, and the InferenceX checkout is re-mirrored from scratch on every
# run, so an install-time-only patch does not survive. A Magpie installed by
# preflight (which never ran the patcher) re-introduced
#     run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
# into the executed copy; InferenceX's run_lm_eval rejected it and aborted the
# benchmark before any results*.json existed.
def _materialized_cfg(tmp_path: Path, *, run_eval: str, inferencex_path: str = "") -> Path:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "envs": {"TP": 1, "CONC": 8, "RUN_EVAL": run_eval},
            "inferencex_path": inferencex_path,
        }
    }
    path = tmp_path / "materialized.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_after_materialize_applies_eval_concurrency_compat(tmp_path):
    """The compat patch is re-asserted against the exact checkout that runs."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    cfg = _materialized_cfg(tmp_path, run_eval="true", inferencex_path=str(ix))
    seen: list[str | None] = []

    def fake_compat(*, inferencex_dir=None):
        seen.append(inferencex_dir)
        return True

    executor = BaselineExecutor()
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.ensure_eval_concurrency_compat",
        side_effect=fake_compat,
    ):
        assert executor._after_materialize_config(cfg, tmp_path) is None

    assert seen == [str(ix)]


def test_after_materialize_fails_loudly_when_flag_unpatchable(tmp_path):
    """Fail LOUDLY, never warn-and-continue: an unstrippable flag guarantees the
    benchmark aborts in run_lm_eval, so short-circuit before the server boots."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    cfg = _materialized_cfg(tmp_path, run_eval="true", inferencex_path=str(ix))

    executor = BaselineExecutor()
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.ensure_eval_concurrency_compat",
        return_value=False,
    ):
        out = executor._after_materialize_config(cfg, tmp_path)

    assert out is not None
    assert out["status"] == "failed"
    assert out["error_class"] == "eval_concurrency_flag_unpatchable"
    assert "--concurrent-requests" in out["error"]
    assert "EVAL_CONCURRENT_REQUESTS" in out["error"]
    # The failure is recognisably eval-rooted, so the accuracy stop gate fires.
    assert BaselineExecutor._is_eval_rooted_failure(out) is True


def test_after_materialize_skips_compat_gate_when_eval_disabled(tmp_path):
    """RUN_EVAL=false runs never reach run_lm_eval, so the flag cannot bite:
    an unpatchable script must not block a deliberately eval-less run."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    cfg = _materialized_cfg(tmp_path, run_eval="false", inferencex_path=str(ix))
    calls: list[int] = []

    def fake_compat(**_kwargs):
        calls.append(1)
        return False

    executor = BaselineExecutor()
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.ensure_eval_concurrency_compat",
        side_effect=fake_compat,
    ):
        assert executor._after_materialize_config(cfg, tmp_path) is None

    assert calls == []


def test_after_materialize_compat_exception_is_not_swallowed(tmp_path):
    """An exception from the patcher must surface as the same loud failure, not
    as a silent 'best-effort skip'."""
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    cfg = _materialized_cfg(tmp_path, run_eval="true", inferencex_path=str(ix))

    executor = BaselineExecutor()
    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.ensure_eval_concurrency_compat",
        side_effect=OSError("read-only fs"),
    ):
        out = executor._after_materialize_config(cfg, tmp_path)

    assert out is not None
    assert out["error_class"] == "eval_concurrency_flag_unpatchable"


def test_end_to_end_flagged_script_is_scrubbed_before_launch(tmp_path):
    """No mocks on the patcher: a real flagged sglang_mi355x.sh under
    $MAGPIE_PATH is scrubbed, and the real benchmark_lib.sh parser is taught to
    tolerate the flag, when the baseline materializes its config."""
    magpie = tmp_path / "site-packages"
    mbench = magpie / "Magpie" / "scripts" / "benchmark"
    mbench.mkdir(parents=True)
    (mbench / "sglang_mi355x.sh").write_text(
        '#!/bin/bash\n'
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?\n',
        encoding="utf-8",
    )
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text(
        "run_lm_eval() {\n"
        '    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC:-64}}"\n'
        "    while [[ $# -gt 0 ]]; do\n"
        "        case $1 in\n"
        '            --top-p)          top_p="$2"; shift 2 ;;\n'
        '            *)                echo "Unknown parameter: $1"; return 1 ;;\n'
        "        esac\n"
        "    done\n"
        "}\n"
        # This test runs the patcher unmocked, so the file has to carry the
        # anchors a real checkout carries: the hook now refuses to launch an
        # eval whose patches could not be applied.
        "run_eval() {\n"
        '    export EVAL_RESULT_DIR="$results_dir"\n'
        "}\n"
        "append_lm_eval_summary() {\n"
        '    mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2\n'
        "}\n",
        encoding="utf-8",
    )
    cfg = _materialized_cfg(tmp_path, run_eval="true", inferencex_path=str(ix))

    executor = BaselineExecutor()
    with patch.dict("os.environ", {"MAGPIE_PATH": str(magpie)}):
        assert executor._after_materialize_config(cfg, tmp_path) is None

    script = (mbench / "sglang_mi355x.sh").read_text(encoding="utf-8")
    assert "--concurrent-requests" not in script
    assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in script
    lib = (ix / "benchmarks" / "benchmark_lib.sh").read_text(encoding="utf-8")
    assert '--concurrent-requests|--concurrent_requests) concurrent_requests="$2"' in lib


def test_end_to_end_live_flag_blocks_launch_without_mocks(tmp_path):
    """Unmocked: a genuinely unremovable ``run_eval --concurrent-requests``
    (unrecognised value shape, and a benchmark_lib.sh whose parser cannot be
    taught to absorb it) short-circuits the baseline before the server boots."""
    magpie = tmp_path / "site-packages"
    mbench = magpie / "Magpie" / "scripts" / "benchmark"
    mbench.mkdir(parents=True)
    (mbench / "sglang_mi355x.sh").write_text(
        '        run_eval --framework lm-eval --port "$PORT" --concurrent-requests 64 || exit $?\n',
        encoding="utf-8",
    )
    ix = tmp_path / "ix"
    (ix / "benchmarks").mkdir(parents=True)
    (ix / "benchmarks" / "benchmark_lib.sh").write_text("run_lm_eval() { : ; }\n", encoding="utf-8")
    cfg = _materialized_cfg(tmp_path, run_eval="true", inferencex_path=str(ix))

    executor = BaselineExecutor()
    with patch.dict("os.environ", {"MAGPIE_PATH": str(magpie)}):
        out = executor._after_materialize_config(cfg, tmp_path)

    assert out is not None
    assert out["error_class"] == "eval_concurrency_flag_unpatchable"
