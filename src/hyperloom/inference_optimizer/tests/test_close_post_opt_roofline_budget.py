# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLOSE sizes its post-opt roofline from the rooflines this session already ran."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hyperloom.common.deadline import Deadline
from hyperloom.orchestrator.phases import close as close_phase


def _phase(*, roofline_runtime_sec: float, remaining_sec: float | None):
    steps: list[dict] = []
    waited: list[float] = []

    async def _enqueue(**_kw):
        return SimpleNamespace(task_id="t-roofline")

    async def _run(task):
        return SimpleNamespace(state="succeeded")

    async def _record(step, **fields):
        steps.append({"step": step, **fields})

    phase = close_phase.ClosePhase.__new__(close_phase.ClosePhase)
    object.__setattr__(
        phase,
        "_coord",
        SimpleNamespace(
            shared_state=SimpleNamespace(
                closing_phase=False,
                stop_reason="global_converged",
                optimization_stack=[{"action": "integrate_patch"}],
                roofline_runtime_sec=roofline_runtime_sec,
                session_deadline=lambda: None if remaining_sec is None else Deadline.after(remaining_sec),
            ),
            phase_prelude=SimpleNamespace(
                internal_analysis_kind=lambda: "roofline",
                enqueue_internal_analysis_task=_enqueue,
            ),
            dispatcher=SimpleNamespace(run_task_registered=_run),
        ),
    )
    object.__setattr__(phase, "_record_close_step", _record)
    return phase, steps, waited


@pytest.mark.parametrize(("measured", "timeout"), [(0.0, 600.0), (1432.0, 1790.0)])
def test_the_timeout_follows_the_slowest_roofline_of_the_session(measured, timeout):
    phase, _, _ = _phase(roofline_runtime_sec=measured, remaining_sec=None)

    assert asyncio.run(phase._post_opt_roofline_timeout_sec("post_opt_roofline")) == pytest.approx(timeout)


def test_a_roofline_that_cannot_fit_the_session_is_skipped_up_front():
    phase, steps, _ = _phase(roofline_runtime_sec=1432.0, remaining_sec=900.0)

    asyncio.run(phase._maybe_run_close_post_opt_roofline())

    assert [(s["status"], s["detail"].split()[0]) for s in steps] == [("skipped", "insufficient_session_time")]


def test_a_roofline_that_fits_runs_under_the_derived_timeout(monkeypatch):
    phase, steps, _ = _phase(roofline_runtime_sec=1432.0, remaining_sec=7200.0)
    granted: list[float] = []
    real_wait_for = asyncio.wait_for

    async def _wait_for(awaitable, timeout):
        granted.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(close_phase.asyncio, "wait_for", _wait_for)

    asyncio.run(phase._maybe_run_close_post_opt_roofline())

    assert granted == [pytest.approx(1790.0)]
    assert steps[-1]["status"] == "done"
