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


@pytest.fixture(autouse=True)
def _legacy_salvage_default(monkeypatch):
    """Exercise the legacy salvage/stop path by default; eval-origin enablement
    routing is opted into per-test."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", "0")


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
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", "1")
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
        }
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
    assert state.stop_reason == "baseline_accuracy_failed"


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
            return subprocess.CompletedProcess(cmd, 1, "", "ERROR: run_eval failed with exit code 1\n")
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


class _StopRecorder:
    """Minimal SharedState stub capturing ``set_stop_reason`` calls."""

    def __init__(self) -> None:
        self.stop_reason = ""
        self.baseline_accuracy = 0.0

    def set_stop_reason(self, value, **_kwargs):
        self.stop_reason = value
        return value


def _stop_ctx(framework: str, recorder, params: dict | None = None) -> SimpleNamespace:
    task = SimpleNamespace(
        task_id="t-bl",
        kind="baseline",
        params={"framework": framework, **(params or {})},
    )
    return SimpleNamespace(task=task, extra={"shared_state": recorder})


def _stopped(framework: str, result: dict, *, params: dict | None = None) -> str:
    """Run ``_maybe_stop_on_missing_baseline_accuracy`` and return the reason."""
    executor = BaselineExecutor()
    rec = _StopRecorder()
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


def test_salvage_ignores_discarded_warmup_and_stops(tmp_path):
    # Only a discarded warmup round carries a score; parse_eval_results excludes
    # warmup output, so there is nothing to salvage and the run stops.
    runs_baseline = tmp_path / "runs" / "baseline"
    _write_gsm8k_results(runs_baseline / "786a793e" / "warmup_round", 0.9)
    deciding = runs_baseline / "retry2_bootsafe"
    deciding.mkdir(parents=True, exist_ok=True)

    reason = _stopped(
        "vllm",
        {"status": "succeeded", "run_eval_disabled": False, "output_dir": str(deciding)},
    )
    assert reason == "baseline_accuracy_failed"


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
    BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY,
    BASELINE_EVAL_EVIDENCE_KEY,
    BASELINE_EVAL_FAILED_KEY,
    BASELINE_EVAL_FAILURE_KIND_KEY,
    BASELINE_EVAL_OBSERVED_ACCURACY_KEY,
    EVAL_KIND_ACCURACY_BELOW_FLOOR,
    EVAL_KIND_ACCURACY_UNAVAILABLE,
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


def _route(monkeypatch, framework, result, *, floor=None, nodes=None, tmp_path=None):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", "1")
    if floor is not None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR", str(floor))
    if nodes is not None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", str(nodes))
    if tmp_path is not None and "materialized_config" not in result:
        result["materialized_config"] = str(_write_minimal_route_yaml(tmp_path, framework))
    executor = BaselineExecutor()
    rec = _StopRecorder()
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
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == 0.0
    assert result[BASELINE_EVAL_EVIDENCE_KEY]
    assert result[BASELINE_EVAL_CONTRACT_FINGERPRINT_KEY]
    assert result["eval_origin"] == "eval"


def test_eval_enablement_zero_accuracy_below_floor(monkeypatch):
    result = {"status": "succeeded", "accuracy": 0.0, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_BELOW_FLOOR
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == 0.0


def test_eval_enablement_positive_below_configured_floor(monkeypatch):
    result = {"status": "succeeded", "accuracy": 0.2, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result, floor=0.5)
    assert reason == ""
    assert result[BASELINE_EVAL_FAILURE_KIND_KEY] == EVAL_KIND_ACCURACY_BELOW_FLOOR
    assert result[BASELINE_EVAL_OBSERVED_ACCURACY_KEY] == 0.2
    assert result[BASELINE_EVAL_ACCURACY_FLOOR_KEY] == 0.5


def test_eval_enablement_accuracy_at_floor_passes(monkeypatch):
    result = {"status": "succeeded", "accuracy": 0.5, "run_eval_disabled": False}
    reason = _route(monkeypatch, "sglang", result, floor=0.5)
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
    rec = _StopRecorder()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL", "1")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    executor._maybe_stop_on_missing_baseline_accuracy(
        _stop_ctx("sglang", rec, {"quality_ref_exempt": True}), result
    )
    assert rec.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result
