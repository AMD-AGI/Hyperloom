# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A genuine native AgentX baseline runs its accuracy round once its throughput round succeeded, unless switched off,
and takes its accuracy reference from the score that round wrote.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    ACCURACY_DIR,
    BASELINE_ACCURACY_STOP_REASON,
    BASELINE_EVAL_FAILED_KEY,
)
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.benchmark_run import accuracy
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import ACCURACY_ROUND_KEY, AccuracyRoundOutcome
from hyperloom.orchestrator.state.shared_state import SharedState


def _point(path: Path, *, agentx: bool) -> Path:
    bench = {"framework": "sglang", "model": "org/model", "agentx": {"enabled": agentx}, "envs": {"TP": "1"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _write_score(slot: Path, score: float) -> None:
    """Leave the results file the accuracy round of ``slot`` writes for ``score``."""
    eval_dir = slot / ACCURACY_DIR / "eval" / "model"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_gsm8k.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}})
    )


def _executor(
    tmp_path, monkeypatch, *, agentx=True, status="succeeded", ended="", score: float | None = None, round_dir=""
):
    """A baseline executor whose pass returns ``status`` from ``round_dir`` of its slot, and the accuracy rounds it
    asks for, each ending ``ended`` with ``score`` written when one is given."""
    slot = tmp_path / "runs" / "baseline" / "t1"
    config = _point(slot / "point.yaml", agentx=agentx)
    result = {"status": status, "output_dir": str(slot / round_dir), "materialized_config": str(config)}
    calls = []

    async def run_accuracy_round(request, *, session):
        calls.append((request.slot, request.config_path))
        if score is not None:
            _write_score(request.slot, score)
        return AccuracyRoundOutcome(request.slot / ACCURACY_DIR, ended)

    monkeypatch.setattr(accuracy, "run_accuracy_round", run_accuracy_round)
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    executor = BaselineExecutor(session_dir=tmp_path, shared_state=None)

    async def run_pass(ctx, **kwargs):
        return dict(result)

    monkeypatch.setattr(executor, "_run_pass", run_pass)
    return executor, calls, slot


def _baseline(tmp_path, monkeypatch, *, agentx_accuracy=True, agentx=True, kind="baseline", **played):
    """Run the baseline ``played`` describes with the accuracy ruling left out, and the accuracy rounds it asked for."""
    executor, calls, slot = _executor(tmp_path, monkeypatch, agentx=agentx, round_dir="measure_round", **played)
    monkeypatch.setattr(executor, "_maybe_stop_on_missing_baseline_accuracy", lambda ctx, result: None)
    state = SharedState(agentx_accuracy=agentx_accuracy, benchmark_mode="agentx" if agentx else "")
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", kind=kind, params={}), extra={"shared_state": state})
    return asyncio.run(executor._run_with_accuracy(ctx)), calls, slot


def test_a_native_agentx_baseline_runs_its_accuracy_round_in_its_slot(tmp_path, monkeypatch) -> None:
    result, calls, slot = _baseline(tmp_path, monkeypatch)

    assert calls == [(slot, slot / "point.yaml")]
    assert result[ACCURACY_ROUND_KEY] == {"run_dir": str(slot / ACCURACY_DIR), "error_class": ""}


@pytest.mark.parametrize(
    "case",
    [
        {"agentx_accuracy": False},
        {"agentx": False},
        {"status": "failed"},
        {"kind": "rebaseline"},
    ],
    ids=["switched_off", "fixed_workload", "failed_baseline", "not_the_quality_reference"],
)
def test_no_accuracy_round_runs_off_the_switch_or_off_a_succeeded_native_agentx_baseline(
    tmp_path, monkeypatch, case
) -> None:
    result, calls, _ = _baseline(tmp_path, monkeypatch, **case)

    assert calls == []
    assert ACCURACY_ROUND_KEY not in result


def test_an_mlperf_baseline_runs_no_accuracy_round(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(accuracy, "is_mlperf_backend", lambda: True)

    _, calls, _ = _baseline(tmp_path, monkeypatch)

    assert calls == []


def test_a_failed_accuracy_round_leaves_the_baseline_standing_with_a_warning(tmp_path, monkeypatch) -> None:
    result, _, slot = _baseline(tmp_path, monkeypatch, ended="eval_failed")

    assert result["status"] == "succeeded"
    assert result[ACCURACY_ROUND_KEY] == {"run_dir": str(slot / ACCURACY_DIR), "error_class": "eval_failed"}
    assert "agentx_accuracy_round_failed" in result["nonfatal_warnings"]


def test_the_baseline_takes_its_accuracy_from_the_score_its_round_wrote(tmp_path, monkeypatch) -> None:
    result, _, slot = _baseline(tmp_path, monkeypatch, score=0.945)

    assert (result["accuracy"], result["accuracy_task"]) == (0.945, "gsm8k")
    assert result["accuracy_source"].startswith(str(slot / ACCURACY_DIR))


def test_a_round_that_left_no_score_leaves_the_baseline_without_accuracy(tmp_path, monkeypatch) -> None:
    result, _, _ = _baseline(tmp_path, monkeypatch, ended="eval_failed")

    assert "accuracy" not in result


@pytest.mark.parametrize(("switch", "stops"), [(True, True), (False, False)], ids=["round_due", "switched_off"])
def test_only_a_baseline_that_owed_an_accuracy_round_stops_without_a_score(tmp_path, monkeypatch, switch, stops):
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    state = SharedState(agentx_accuracy=switch, benchmark_mode="agentx", enablement_mode="off")
    executor = BaselineExecutor(session_dir=tmp_path, shared_state=None)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", kind="baseline", params={}), extra={"shared_state": state})

    executor._maybe_stop_on_missing_baseline_accuracy(ctx, {"status": "succeeded"})

    assert (state.stop_reason == BASELINE_ACCURACY_STOP_REASON) is stops


@pytest.mark.parametrize("enablement_mode", ["off", "eval", "all"])
def test_a_baseline_is_ruled_on_the_score_its_own_accuracy_round_wrote(tmp_path, monkeypatch, enablement_mode):
    executor, calls, slot = _executor(tmp_path, monkeypatch, score=0.945)
    state = SharedState(agentx_accuracy=True, benchmark_mode="agentx", enablement_mode=enablement_mode)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", kind="baseline", params={}), extra={"shared_state": state})

    result = asyncio.run(executor._run_with_accuracy(ctx))

    assert calls == [(slot, slot / "point.yaml")]
    assert state.stop_reason == ""
    assert BASELINE_EVAL_FAILED_KEY not in result
    assert result["accuracy"] == 0.945
    assert result["accuracy_source"].startswith(str(slot / ACCURACY_DIR))


def test_a_baseline_whose_accuracy_round_left_no_score_stops_whatever_an_earlier_attempt_scored(tmp_path, monkeypatch):
    _write_score(tmp_path / "runs" / "baseline" / "t0", 0.93)
    executor, calls, slot = _executor(tmp_path, monkeypatch, ended="eval_failed")
    state = SharedState(agentx_accuracy=True, benchmark_mode="agentx", enablement_mode="off", baseline_accuracy=0.81)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", kind="baseline", params={}), extra={"shared_state": state})

    result = asyncio.run(executor._run_with_accuracy(ctx))

    assert calls == [(slot, slot / "point.yaml")]
    assert state.stop_reason == BASELINE_ACCURACY_STOP_REASON
    assert "accuracy" not in result
    assert result["nonfatal_warnings"] == ["agentx_accuracy_round_failed"]
    assert state.baseline_accuracy == 0.81
