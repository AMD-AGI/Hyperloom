# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework allocation expiry starts drainage before advisory operator holds."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_KERNEL, ESCALATE_HINT_SKIP_TO_SWEEP
from hyperloom.orchestrator.state.task_registry import TaskSpec

from .conftest import make_coordinator


HINTS = ["", ESCALATE_HINT_SKIP_TO_KERNEL, ESCALATE_HINT_SKIP_TO_SWEEP]


@pytest.fixture
def coord(session_dir, monkeypatch):
    now = time.time()
    c = make_coordinator(
        session_dir,
        shared_state_overrides={
            "phase": "FRAMEWORK_AGENT",
            "phase_started_unix": now - 1800,
            "max_minutes": 120,
            "elapsed_charged_sec": 1800,
            "leg_anchor_unix": now,
            "phase_budget_pct": {
                "PRELUDE": 0.1,
                "FRAMEWORK_AGENT": 0.1,
                "KERNEL_AGENT": 0.7,
                "SWEEP": 0.08,
                "CLOSE": 0.02,
            },
            "baseline_tput": 100.0,
            "baseline_perf": {"output_throughput": 100.0},
            "kernel_enabled": True,
            "framework_agent_phase_enabled": True,
        },
    )
    monkeypatch.setattr(c.phase_machine, "_on_phase_entered", AsyncMock())
    monkeypatch.setattr(c.phase_internal, "maybe_enqueue_trajectory_reviewer", AsyncMock())
    return c


def _leverage(coord, hint, plateau):
    coord.shared_state.pending_escalate_hint = hint
    coord.shared_state.framework_agent_phase_done = plateau
    coord.shared_state.attempts = [
        {"cycle": 0, "lever_kind": "config", "round_id": str(i), "adopted": False}
        for i in range(machine_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK if plateau else 1)
    ]


async def _operator_task(coord):
    task, _ = await coord.tasks.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={"operator_priority": True},
            idempotency_key="operator-budget-work",
        )
    )
    return task


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("plateau", [False, True])
@pytest.mark.parametrize("kernel_enabled", [True, False])
async def test_expired_allocation_outranks_advisory_operator_hold(coord, hint, plateau, kernel_enabled):
    _leverage(coord, hint, plateau)
    coord.shared_state.kernel_enabled = kernel_enabled
    task = await _operator_task(coord)
    assert machine_state.phase_budget_remaining_seconds(coord.shared_state) == 0
    assert machine_state.session_remaining_seconds(coord.shared_state) > 0
    assert await coord.phase_framework.operator_work_pending()

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == ("KERNEL_AGENT" if kernel_enabled else "SWEEP")
    assert coord.shared_state.phase_history[-1]["reason"] == "optimize_phase_budget_exhausted"
    assert coord.shared_state.pending_escalate_hint == ""
    assert coord.shared_state.stop_reason == ""
    assert (await coord.tasks.get(task.task_id)).state == "cancelled"
    inputs = coord.shared_state.phase_history[-1]["evidence"]["predicate_inputs"]
    assert machine_state.replay_next_phase(inputs)[0] == coord.shared_state.phase


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("plateau", [False, True])
@pytest.mark.parametrize("budget", ["remaining", "session_unlimited", "phase_unbounded"])
async def test_available_or_unbounded_allocation_keeps_operator_hold(coord, hint, plateau, budget):
    _leverage(coord, hint, plateau)
    if budget == "remaining":
        coord.shared_state.phase_started_unix = time.time() - 600
        coord.shared_state.elapsed_charged_sec = 600
        assert machine_state.phase_budget_remaining_seconds(coord.shared_state) > 0
    elif budget == "session_unlimited":
        coord.shared_state.max_minutes = 0
        assert machine_state.session_remaining_seconds(coord.shared_state) is None
        assert machine_state.phase_budget_remaining_seconds(coord.shared_state) > 0
    else:
        coord.shared_state.phase_budget_pct.pop("FRAMEWORK_AGENT")
        assert machine_state.phase_budget_remaining_seconds(coord.shared_state) is None
    task = await _operator_task(coord)

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == "FRAMEWORK_AGENT"
    assert not coord.dispatcher.admission_frozen
    assert coord.shared_state.pending_escalate_hint == hint
    assert (await coord.tasks.get(task.task_id)).state == "queued"


@pytest.mark.parametrize("hint", HINTS)
async def test_phase_expiry_drains_running_measurement_without_cancelling_it(coord, hint, monkeypatch):
    _leverage(coord, hint, True)
    task = await _operator_task(coord)
    await coord.tasks.transition(task.task_id, "running")
    cancel = AsyncMock(return_value=[])
    monkeypatch.setattr(coord.dispatcher, "cancel_inflight_actions", cancel)

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == "FRAMEWORK_AGENT"
    assert coord.dispatcher.admission_frozen
    assert (await coord.tasks.get(task.task_id)).state == "running"
    cancel.assert_not_awaited()
    await coord.tasks.transition(task.task_id, "succeeded")

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == "KERNEL_AGENT"
    assert coord.shared_state.phase_history[-1]["reason"] == "optimize_phase_budget_exhausted"
    assert not coord.dispatcher.admission_frozen
    assert coord.shared_state.stop_reason == ""
    cancel.assert_not_awaited()


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize(
    "override,reason", [({"stop_reason": "signal"}, "signal"), ({"closing_phase": True}, "time_exhausted")]
)
async def test_terminal_stop_retains_priority_over_phase_drain(coord, hint, override, reason):
    _leverage(coord, hint, True)
    for key, value in override.items():
        setattr(coord.shared_state, key, value)
    task = await _operator_task(coord)

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == "CLOSE"
    assert coord.shared_state.stop_reason == reason
    assert (await coord.tasks.get(task.task_id)).state == "cancelled"


@pytest.mark.parametrize("hint", HINTS)
@pytest.mark.parametrize("kernel_enabled", [True, False])
async def test_advisory_handoff_with_available_budget_is_unchanged(coord, hint, kernel_enabled):
    _leverage(coord, hint, True)
    coord.shared_state.phase_started_unix = time.time() - 600
    coord.shared_state.elapsed_charged_sec = 600
    coord.shared_state.kernel_enabled = kernel_enabled

    await coord.phase_machine.advance_phase_if_needed()

    assert coord.shared_state.phase == ("KERNEL_AGENT" if kernel_enabled else "SWEEP")
    assert coord.shared_state.phase_history[-1]["reason"] == "optimize_no_more_leverage"
    assert coord.shared_state.stop_reason == ""
