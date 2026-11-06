# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reverted patch still reports the change it was measured at, on the axis it was graded on."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.event_rows import blocking_gate
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.phases.framework_records import _record_attempt_gates
from hyperloom.orchestrator.state.shared_state import SharedState


def _intvty_axes(p50: float, output: float) -> dict:
    return {
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": p50,
        "output_throughput": output,
        "total_throughput": 2 * output,
        "duration_seconds": 60.0,
        "request_error_rate": 0.0,
    }


@pytest.mark.asyncio
async def test_an_agentx_revert_reports_its_measured_p50_change(tmp_path: Path):
    state = SharedState(
        baseline_tput=1000.0,
        current_best=_intvty_axes(100.0, 1000.0),
        grading={"objective": "e2e_norm_intvty_p90", "noise_pct": 2.0},
    )
    attempt = IntegrateAttempt(
        task_id="t-int", specialist_task_id="t-spec", shared_state=state, output_root=tmp_path / "run"
    )
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")
    lessons = []

    async def _record(lesson, **_kwargs):
        lessons.append(lesson)

    executor._maybe_write_framework_kb_record = _record
    out = await executor._gate_perf(
        attempt=attempt,
        params={"base_tput": 1000.0, "keep_threshold_pct": 1.0},
        bench_result=_intvty_axes(97.0, 1000.0),
        gate_evidence={"accuracy_pass": True},
    )

    assert out["status"] == "reverted"
    assert out["delta_pct"] == pytest.approx(-3.0)
    assert [(lesson.outcome, lesson.tps_delta_pct) for lesson in lessons] == [
        ("reverted_smoke_fail", pytest.approx(-3.0))
    ]
    assert out["reason"] == "interactivity p50 100.0->97.0 did not clear the interactivity KEEP gate"


async def _recorded_gates(tmp_path: Path, state: SharedState, bench_result: dict) -> tuple[dict, list[dict]]:
    """Gate one bench as integrate does, then record the attempt's gates as the framework timeline does."""
    attempt = IntegrateAttempt(
        task_id="t-int", specialist_task_id="t-spec", shared_state=state, output_root=tmp_path / "run"
    )
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session")

    async def _no_lesson(*_args, **_kwargs):
        return None

    executor._maybe_write_framework_kb_record = _no_lesson
    out = await executor._gate_perf(
        attempt=attempt,
        params={"base_tput": 1000.0, "keep_threshold_pct": 1.0},
        bench_result={"status": "succeeded", **bench_result},
        gate_evidence={"accuracy_pass": True},
    )
    gates: list[dict] = []
    recorder = SimpleNamespace(record_attempt_gate=lambda _task_id, verdict: gates.append(asdict(verdict)))
    _record_attempt_gates(recorder, "t-int", result=out)
    return out, gates


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("p50", "output"),
    [pytest.param(101.98, 1000.0, id="median-below-the-bar"), pytest.param(110.0, 900.0, id="output-guard-breach")],
)
async def test_an_agentx_revert_records_the_graded_gate_it_failed(tmp_path: Path, p50: float, output: float):
    state = SharedState(
        baseline_tput=1000.0,
        current_best=_intvty_axes(100.0, 1000.0),
        grading={"objective": "e2e_norm_intvty_p90", "noise_pct": 2.0},
    )

    out, gates = await _recorded_gates(tmp_path, state, _intvty_axes(p50, output))

    assert (out["status"], out["delta_pct"] > 1.0) == ("reverted", True)
    assert blocking_gate(gates) == "graded_axes"
    assert [gate["gate"] for gate in gates] == ["graded_axes", "accuracy"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tput", "status", "passed"), [(1050.0, "kept", True), (1005.0, "reverted", False)])
async def test_an_output_graded_attempt_still_records_its_keep_threshold(tmp_path: Path, tput, status, passed):
    state = SharedState(baseline_tput=1000.0, current_best={"tput": 1000.0})

    out, gates = await _recorded_gates(tmp_path, state, {"output_throughput": tput})

    assert out["status"] == status
    assert gates[0] == {
        "gate": "keep_threshold",
        "passed": passed,
        "reason": "",
        "observed": pytest.approx(tput / 10.0 - 100.0),
        "threshold": 1.0,
    }
