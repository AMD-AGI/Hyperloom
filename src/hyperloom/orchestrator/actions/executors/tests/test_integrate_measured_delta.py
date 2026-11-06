# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reverted patch still reports the change it was measured at, on the axis it was graded on."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
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
