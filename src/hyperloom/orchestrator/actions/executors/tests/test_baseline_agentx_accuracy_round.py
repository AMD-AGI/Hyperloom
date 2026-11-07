# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A genuine native AgentX baseline runs its accuracy round once its throughput round succeeded, unless switched off.

The round's score is not read yet: the baseline's AgentX accuracy still comes from ``parse_eval_results``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._accuracy_gate import parse_eval_results
from hyperloom.orchestrator.actions.executors._suite_config import ACCURACY_DIR
from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.benchmark_run import accuracy
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import ACCURACY_ROUND_KEY, AccuracyRoundOutcome
from hyperloom.orchestrator.state.shared_state import SharedState


def _point(path: Path, *, agentx: bool) -> Path:
    bench = {"framework": "sglang", "model": "org/model", "agentx": {"enabled": agentx}, "envs": {"TP": "1"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


def _baseline(
    tmp_path, monkeypatch, *, agentx_accuracy=True, agentx=True, status="succeeded", kind="baseline", ended=""
):
    """Run a baseline whose pass returns ``status``, and the accuracy rounds it asked for, each ending ``ended``."""
    slot = tmp_path / "runs" / "baseline" / "t1"
    config = _point(slot / "point.yaml", agentx=agentx)
    result = {"status": status, "output_dir": str(slot / "measure_round"), "materialized_config": str(config)}
    calls = []

    async def run_accuracy_round(slot, launch, **kwargs):
        calls.append((slot, launch))
        return AccuracyRoundOutcome(slot / ACCURACY_DIR, ended)

    monkeypatch.setattr(accuracy, "run_accuracy_round", run_accuracy_round)
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    executor = BaselineExecutor(magpie_python="python3", session_dir=tmp_path, shared_state=None)

    async def run_pass(ctx, **kwargs):
        return dict(result)

    monkeypatch.setattr(executor, "_run_pass", run_pass)
    monkeypatch.setattr(executor, "_maybe_stop_on_missing_baseline_accuracy", lambda ctx, result: None)
    state = SharedState(agentx_accuracy=agentx_accuracy, benchmark_mode="agentx" if agentx else "")
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", kind=kind, params={}), extra={"shared_state": state})
    return asyncio.run(executor._run_retrying(ctx)), calls, slot


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


def test_parse_eval_results_still_grades_an_agentx_slot_on_its_error_rate(tmp_path, monkeypatch) -> None:
    slot = tmp_path / "slot"
    (slot / "agentx").mkdir(parents=True)
    (slot / "agentx" / "inferencex_result.json").write_text(json.dumps({"request_error_rate": 0.0}))
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    before = parse_eval_results(slot, framework="sglang", benchmark_mode="agentx")
    eval_dir = slot / ACCURACY_DIR / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_gsm8k.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 0.0}}})
    )

    assert parse_eval_results(slot, framework="sglang", benchmark_mode="agentx") == before
    assert before["task"] == "agentx_error_rate" and before["accuracy"] == 1.0
