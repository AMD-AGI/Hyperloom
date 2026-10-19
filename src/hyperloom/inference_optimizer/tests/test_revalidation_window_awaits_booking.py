# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A revalidation baseline that passed is credited, whichever tick books it.

The task row goes terminal inside the execution; its result is booked by the
pump that runs after the next tick's reconcile pass. Closing the window on the
terminal row alone left the writeback a promoted revalidation with no window to
credit it to, so the enablement lane ended ``succeeded = False`` although the
patched server served and measured.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult


@pytest.fixture
def coord(tmp_path: Path):
    from hyperloom.orchestrator.loop.coordinator import AdvisoryServices, Coordinator
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, MockTurn, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="revalidation-booking")
    state.max_minutes = 30
    state.save(tmp_path)
    idle = ScriptedPlan(turns=[MockTurn(intents=[])])
    coord = Coordinator(
        session_dir=tmp_path,
        backends={name: MockBackend(idle) for name in ("orchestration", "critic")},
        role_registry=default_role_registry(),
        advisory=AdvisoryServices(knowledge_plane=None),
    )
    yield coord
    coord.db.close()


def _measured(tmp_path: Path) -> dict:
    """A baseline result that measured: served, replayed, no request errors."""
    return {
        "status": "succeeded",
        "reported_success": True,
        "valid_measurement": True,
        "submission_valid": True,
        "submission_invalid_reasons": [],
        "output_throughput": 804.35,
        "completed_requests": 3047,
        "duration_seconds": 3627.0,
        "request_error_rate": 0.0,
        "accuracy": 1.0,
        "accuracy_metric": "request_error_rate",
        "returncode": 0,
        "workspace": str(tmp_path / "runs" / "baseline" / "reval"),
    }


async def _finished_revalidation(coord, tmp_path: Path):
    """A revalidation whose row is terminal and whose result waits for the pump."""
    state = coord.shared_state
    state.phase = "ENABLEMENT"
    state.enablement.validation_pending = True
    state.enablement.revalidation_generation = 1
    task = await coord.tasks.create(
        kind="baseline",
        params={"source": "coordinator_internal", "reason": ENABLEMENT_REVALIDATION_REASON},
        idempotency_key="enablement_revalidation:gen1",
    )
    state.enablement.revalidation_task_id = task.task_id
    await coord.tasks.transition(task.task_id, "running")
    await coord.tasks.transition(task.task_id, "succeeded")
    result = SubAgentResult(task_id=task.task_id, state="succeeded", result=_measured(tmp_path))
    coord.dispatcher._completion_queue.append((await coord.tasks.get(task.task_id), result))
    return task


async def test_the_tick_after_the_revalidation_ends_still_credits_it(coord, tmp_path):
    """Reconcile runs first in the tick, the pump books the result after it."""
    await _finished_revalidation(coord, tmp_path)

    await coord.reconciler.run(time.time())

    assert coord.shared_state.enablement.validation_pending is True, (
        "the window closed on a terminal row whose result nobody had booked yet"
    )

    await coord.dispatcher.pump_dispatcher_once()

    enablement = coord.shared_state.enablement
    assert enablement.succeeded is True
    assert enablement.validation_pending is False
    assert enablement.revalidation_task_id == ""


async def test_a_terminal_revalidation_nobody_will_book_is_still_closed(coord, tmp_path):
    """The repair this pass exists for: a terminal row with no result coming."""
    task = await _finished_revalidation(coord, tmp_path)
    coord.dispatcher._completion_queue.clear()

    report = await coord.reconciler.run(time.time())

    assert report.closed_windows == [task.task_id]
    assert coord.shared_state.enablement.validation_pending is False
    assert coord.shared_state.enablement.succeeded is False


async def test_the_dispatcher_owes_a_result_until_the_pump_books_it(coord, tmp_path):
    task = await _finished_revalidation(coord, tmp_path)

    assert coord.dispatcher.owes_result(task.task_id)
    assert not coord.dispatcher.owes_result("some-other-task")
    assert not coord.dispatcher.owes_result("")

    await coord.dispatcher.pump_dispatcher_once()

    assert not coord.dispatcher.owes_result(task.task_id)


async def test_an_execution_that_raised_owes_nothing(coord, tmp_path):
    """A run that raised queues no completion; its leftover handle must not hold the window."""
    task = await _finished_revalidation(coord, tmp_path)
    coord.dispatcher._completion_queue.clear()

    async def _raised():
        raise RuntimeError("executor raised after the row went terminal")

    crashed = asyncio.ensure_future(_raised())
    await asyncio.wait({crashed})
    with pytest.raises(RuntimeError, match="executor raised"):
        crashed.result()
    coord.dispatcher._execution_of[task.task_id] = crashed

    assert not coord.dispatcher.owes_result(task.task_id)
    report = await coord.reconciler.run(time.time())
    assert report.closed_windows == [task.task_id]
    coord.dispatcher._execution_of.clear()


async def test_a_running_execution_owes_its_result(coord, tmp_path):
    task = await _finished_revalidation(coord, tmp_path)
    coord.dispatcher._completion_queue.clear()
    gate = asyncio.Event()
    running = asyncio.ensure_future(gate.wait())
    coord.dispatcher._execution_of[task.task_id] = running
    try:
        assert coord.dispatcher.owes_result(task.task_id)
        report = await coord.reconciler.run(time.time())
        assert report.closed_windows == []
        assert coord.shared_state.enablement.validation_pending is True
    finally:
        gate.set()
        await asyncio.wait_for(running, timeout=5)
        coord.dispatcher._execution_of.clear()


async def test_an_execution_outliving_its_cancelled_caller_still_owes_its_result(coord):
    """The registered handle ends on cancellation; the shielded execution does not."""
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def slow(_ctx) -> dict:
        entered.set()
        await asyncio.shield(gate.wait())
        return {"runner_status": "succeeded"}

    coord.sub.register_executor("slow_action", slow)
    task, _ = await coord.tasks.create_or_return_existing(
        kind="slow_action", params={}, idempotency_key="slow", requires_lanes=[]
    )
    await asyncio.wait_for(coord.dispatcher.pump_dispatcher_once(), timeout=5)
    await asyncio.wait_for(entered.wait(), timeout=5)
    handle = coord.dispatcher._inflight_actions[task.task_id]

    handle.atask.cancel()
    await asyncio.wait_for(asyncio.gather(handle.atask, return_exceptions=True), timeout=60)
    assert handle.atask.done()
    assert coord.dispatcher.owes_result(task.task_id), "the execution still runs; its result is still owed"

    gate.set()
    executions = tuple(coord.dispatcher._executions)
    await asyncio.wait_for(asyncio.gather(*executions, return_exceptions=True), timeout=10)
    assert coord.dispatcher.has_unbooked_completions(), "the execution queued its completion"
    assert coord.dispatcher.owes_result(task.task_id), "a queued completion is still owed"

    await coord.dispatcher.pump_dispatcher_once()
    assert not coord.dispatcher.owes_result(task.task_id)
