# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session wall-clock budget defences that live in the orchestrator loop.

Two of the four layers are here, the ones outside the executors:

* Admission -- an action whose expected cost cannot fit the budget that is left
  never starts. Covers the pure fit decision, the SharedState accessor both it
  and the grid deadline read, the dispatcher gate, the three intent paths that
  share it, and the pre-dispatch backstop for a task that sat queued until its
  budget drained.
* In-flight cancellation -- the backstop for work already running when the
  budget goes or the process is asked to stop. Covers the handles the dispatcher
  keeps, the cancellation itself, the closing-action carve-out, the task row
  landing terminal instead of stranding at ``running``, and the pump and
  ``Coordinator.stop`` paths that trigger it.

The remaining two layers are enforced inside the executors and tested next to
them: the timeout clamp in ``test_explore_executor``, and the subprocess session
reaper in ``test_kill_spawned_server``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    action_fits_time_budget,
)
from hyperloom.orchestrator.policy.gate import PolicyDenied
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import CLOSING_RESERVE_SEC, SharedState
from hyperloom.orchestrator.state.task_registry import Task

# An action costing an hour at p50, so a short budget cannot fit it.
_EXPENSIVE_ACTION = "kernel_opt"
_EXPENSIVE_COST_MIN = 60.0
# Cheap enough to fit anything but a nearly-spent budget.
_CHEAP_ACTION = "profile"


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _backends() -> dict[str, Backend]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {name: MockBackend(silent, name=name) for name in ("orchestration", "critic", "robustness")}


@pytest.fixture
def coord(session_dir) -> Coordinator:
    c = Coordinator(session_dir, backends=_backends())
    # Past the baseline prerequisite so the sequence gate stays out of the way.
    c.shared_state.baseline_tput = 800.0
    return c


def _set_budget(coord: Coordinator, *, minutes: float, elapsed_min: float = 0.0) -> None:
    """Give the session a finite budget with ``elapsed_min`` already spent."""
    coord.shared_state.max_minutes = int(minutes)
    coord.shared_state.elapsed_minutes = lambda **_kw: elapsed_min  # type: ignore[method-assign]


class TestFitDecision:
    """The pure fit rule, independent of any Coordinator."""

    def test_an_unbounded_budget_fits_everything(self):
        assert action_fits_time_budget(usable_sec=None, expected_cost_minutes=600.0)

    def test_an_action_with_no_cost_on_record_is_admitted(self):
        assert action_fits_time_budget(usable_sec=60.0, expected_cost_minutes=0.0)
        assert action_fits_time_budget(usable_sec=60.0, expected_cost_minutes=-1.0)

    def test_an_action_that_fits_is_admitted(self):
        assert action_fits_time_budget(usable_sec=30 * 60.0, expected_cost_minutes=30.0)

    def test_an_action_that_does_not_fit_is_refused(self):
        assert not action_fits_time_budget(usable_sec=30 * 60.0 - 1, expected_cost_minutes=30.0)

    def test_the_expected_cost_is_the_anchor_not_the_p75_backstop(self):
        """A 90-minute budget admits a 60/120 action: the tail is not the bar.

        Judging fit on p75 would refuse work that finishes in the budget half the
        time, abandoning usable minutes. The session reaper handles the overruns.
        """
        assert action_fits_time_budget(usable_sec=90 * 60.0, expected_cost_minutes=60.0)
        assert not action_fits_time_budget(usable_sec=90 * 60.0, expected_cost_minutes=120.0)


class TestUsableBudgetAccessor:
    """``session_budget_usable_sec`` is the one number admission and the grid share."""

    def test_an_unset_budget_reads_as_unbounded(self):
        assert SharedState(session_id="s").session_budget_usable_sec() is None

    def test_the_closing_reserve_is_held_back(self):
        state = SharedState(session_id="s", max_minutes=60)
        state.elapsed_minutes = lambda **_kw: 0.0  # type: ignore[method-assign]
        assert state.session_budget_usable_sec() == pytest.approx(3600.0 - CLOSING_RESERVE_SEC)

    def test_a_budget_inside_the_reserve_reads_as_spent(self):
        state = SharedState(session_id="s", max_minutes=60)
        state.elapsed_minutes = lambda **_kw: 59.9  # type: ignore[method-assign]
        assert state.session_budget_usable_sec() == 0.0

    def test_the_grid_deadline_is_derived_from_the_same_number(self, monkeypatch):
        """Both wall-clock layers must agree on how much budget is left."""
        import time as _time

        state = SharedState(session_id="s", max_minutes=60)
        state.elapsed_minutes = lambda **_kw: 10.0  # type: ignore[method-assign]
        monkeypatch.setattr(_time, "monotonic", lambda: 1000.0)
        usable = state.session_budget_usable_sec()
        assert state.grid_session_deadline_sec() == pytest.approx(1000.0 + usable)


class TestTimeBudgetGate:
    """The dispatcher gate that turns a fit failure into a refusal."""

    def test_an_action_too_big_for_the_budget_is_denied(self, coord: Coordinator):
        _set_budget(coord, minutes=20)
        denied = coord._time_budget_denial_for_action(_EXPENSIVE_ACTION)
        assert isinstance(denied, PolicyDenied)
        assert denied.rule == "time_budget"
        assert "60 min" in str(denied)
        assert "report" in str(getattr(denied, "hint", ""))

    def test_an_action_that_fits_is_admitted(self, coord: Coordinator):
        _set_budget(coord, minutes=20)
        assert coord._time_budget_denial_for_action(_CHEAP_ACTION) is None

    def test_an_unbounded_budget_admits_the_most_expensive_action(self, coord: Coordinator):
        coord.shared_state.max_minutes = 0
        assert coord._time_budget_denial_for_action(_EXPENSIVE_ACTION) is None

    def test_an_action_with_no_registry_entry_is_admitted(self, coord: Coordinator):
        _set_budget(coord, minutes=1)
        assert coord._time_budget_denial_for_action("frobnicate") is None

    def test_the_closing_actions_stay_startable_on_an_empty_budget(self, coord: Coordinator):
        """Refusing these would strand the session with nothing to show."""
        _set_budget(coord, minutes=60, elapsed_min=60.0)
        assert coord.shared_state.session_budget_usable_sec() == 0.0
        for action in TIME_BUDGET_EXEMPT_ACTIONS:
            assert coord._time_budget_denial_for_action(action) is None, action

    def test_a_stopping_session_leaves_the_gate_to_the_stop_path(self, coord: Coordinator):
        _set_budget(coord, minutes=1)
        coord.shared_state.stop_reason = "time_exhausted"
        assert coord._time_budget_denial_for_action(_EXPENSIVE_ACTION) is None

    def test_the_budget_shrinks_the_gate_as_the_session_runs(self, coord: Coordinator):
        _set_budget(coord, minutes=120, elapsed_min=0.0)
        assert coord._time_budget_denial_for_action(_EXPENSIVE_ACTION) is None
        _set_budget(coord, minutes=120, elapsed_min=70.0)
        assert coord._time_budget_denial_for_action(_EXPENSIVE_ACTION) is not None


class TestAdmissionGateOrder:
    """``_admission_denial_for_action`` chains the gates; the first one wins."""

    def test_the_baseline_prerequisite_is_reported_before_the_budget(self, coord: Coordinator):
        coord.shared_state.baseline_tput = 0.0
        _set_budget(coord, minutes=1)
        denied = coord._admission_denial_for_action("explore")
        assert denied is not None and denied.rule == "execution_order"

    def test_the_budget_gate_runs_once_the_sequence_gate_passes(self, coord: Coordinator):
        _set_budget(coord, minutes=20)
        denied = coord._admission_denial_for_action(_EXPENSIVE_ACTION)
        assert denied is not None and denied.rule == "time_budget"

    def test_an_action_clearing_both_gates_is_admitted(self, coord: Coordinator):
        _set_budget(coord, minutes=600)
        assert coord._admission_denial_for_action(_EXPENSIVE_ACTION) is None


def _delegate(action_name: str, key: str) -> Intent:
    return Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": action_name, "params": {}, "idempotency_key": key},
    )


class TestIntentPathsAreGated:
    """A refusal must land before a task row exists, so no ledger sees it."""

    @pytest.mark.asyncio
    async def test_delegating_an_over_budget_action_queues_nothing(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        _set_budget(coord, minutes=20)
        recorded: list[PolicyDenied] = []

        async def _rec(source, intent, denied, action_name=None):
            recorded.append(denied)

        monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
        await coord._handle_delegate("orchestration", _delegate(_EXPENSIVE_ACTION, "d-budget"))
        assert [d.rule for d in recorded] == ["time_budget"]
        assert [t for t in await coord.tasks.queued() if t.kind == _EXPENSIVE_ACTION] == []

    @pytest.mark.asyncio
    async def test_delegating_an_affordable_action_still_queues(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        _set_budget(coord, minutes=600)
        monkeypatch.setattr(coord.shared_state, "is_pruned", lambda a: False)
        await coord._handle_delegate("orchestration", _delegate(_EXPENSIVE_ACTION, "d-ok"))
        assert [t for t in await coord.tasks.queued() if t.kind == _EXPENSIVE_ACTION]

    @pytest.mark.asyncio
    async def test_proposing_an_over_budget_action_never_reaches_the_critic(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        _set_budget(coord, minutes=20)
        recorded: list[PolicyDenied] = []

        async def _rec(source, intent, denied, action_name=None):
            recorded.append(denied)

        monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
        intent = Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": _EXPENSIVE_ACTION, "predicted_gain_pct": 5.0},
        )
        await coord._handle_propose_action("orchestration", intent)
        assert [d.rule for d in recorded] == ["time_budget"]
        assert not coord.state.pending_proposals

    @pytest.mark.asyncio
    async def test_the_inline_runner_reports_the_refusal(self, coord: Coordinator, monkeypatch):
        _set_budget(coord, minutes=20)

        async def _rec(source, intent, denied, action_name=None):
            return None

        monkeypatch.setattr(coord.writeback, "_record_policy_denied", _rec)
        monkeypatch.setattr(coord.policy, "validate_intent", lambda *a, **k: None)
        out = await coord._run_action_now(_EXPENSIVE_ACTION, {})
        assert "denied" in out
        assert [t for t in await coord.tasks.queued() if t.kind == _EXPENSIVE_ACTION] == []


class TestPreDispatchBackstop:
    """A task can wait for a lane long enough for its budget to drain."""

    @pytest.mark.asyncio
    async def test_a_queued_task_the_budget_outlived_is_dropped_before_dispatch(
        self,
        coord: Coordinator,
    ):
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind=_EXPENSIVE_ACTION,
            params={},
            idempotency_key="q-drained",
        )
        # The budget drains while the task waits in the queue.
        _set_budget(coord, minutes=600, elapsed_min=590.0)

        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert [t.task_id for t, _, _ in spawned] == []
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    @pytest.mark.asyncio
    async def test_the_drop_is_not_recorded_as_an_action_failure(self, coord: Coordinator):
        """A task that never ran is not evidence about the action."""
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind=_EXPENSIVE_ACTION,
            params={},
            idempotency_key="q-no-failure",
        )
        _set_budget(coord, minutes=600, elapsed_min=590.0)

        await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert (await coord.tasks.get(task.task_id)).state == "cancelled"
        failures = list(getattr(coord.shared_state, "last_action_failures", []) or [])
        assert [f for f in failures if str(f.get("action") or "") == _EXPENSIVE_ACTION] == []

    @pytest.mark.asyncio
    async def test_a_queued_task_that_still_fits_is_left_alone(self, coord: Coordinator):
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind=_EXPENSIVE_ACTION,
            params={},
            idempotency_key="q-fits",
        )
        assert await coord.dispatcher._cancel_queued_task_over_budget(task) is False
        assert (await coord.tasks.get(task.task_id)).state == "queued"


# One of the closing actions, exempt from the budget because the closing reserve
# is held back so it can run.
_CLOSING_ACTION = "report"
# The lane ``_CHEAP_ACTION`` holds while it runs, so a leak is observable.
_CHEAP_ACTION_LANE = "profile_lane"


def _never_finishes(started: asyncio.Event):
    """Build an executor that only ever ends by being cancelled."""

    async def _run(_ctx) -> dict:
        started.set()
        await asyncio.sleep(3600.0)
        return {}

    return _run


async def _queue_action(coord: Coordinator, *, kind: str, key: str) -> tuple[Task, asyncio.Event]:
    """Queue an action that only ends by being cancelled, with its real lanes."""
    started = asyncio.Event()
    coord.sub.register_executor(kind, _never_finishes(started))
    lanes, ttl_sec = coord.dispatcher._registry_lanes_ttl(kind)
    task, _ = await coord.tasks.create_or_return_existing(
        kind=kind,
        params={},
        idempotency_key=key,
        requires_lanes=lanes,
        lease_ttl_sec=ttl_sec,
    )
    return task, started


async def _start_action(coord: Coordinator, *, kind: str, key: str) -> tuple[Task, asyncio.Task]:
    """Dispatch the action with no pump running, for the pieces under it."""
    task, started = await _queue_action(coord, kind=kind, key=key)
    spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())
    assert [t.task_id for t, _, _ in spawned] == [task.task_id]
    await asyncio.wait_for(started.wait(), timeout=5.0)
    return task, spawned[0][1]


async def _start_action_under_pump(
    coord: Coordinator,
    *,
    kind: str,
    key: str,
) -> tuple[Task, asyncio.Task, asyncio.Task]:
    """Let a running pump dispatch the action, the way a tick does.

    Returns ``(task, action task, pump task)``. The pump owns what it spawned,
    so the triggers can only be tested against a pump that spawned the work.
    """
    task, started = await _queue_action(coord, kind=kind, key=key)
    pump = asyncio.create_task(coord._pump_dispatcher_once())
    await asyncio.wait_for(started.wait(), timeout=5.0)
    return task, coord.dispatcher._inflight_actions[task.task_id][1], pump


async def _settle(atask: asyncio.Task) -> None:
    """Wait for an action to finish unwinding, however it ended."""
    await asyncio.wait_for(asyncio.gather(atask, return_exceptions=True), timeout=5.0)


class TestInflightHandles:
    """Something other than the pump has to be able to reach a running action."""

    @pytest.mark.asyncio
    async def test_a_running_action_is_reachable_by_task_id(self, coord: Coordinator):
        task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="h-live")
        try:
            assert coord.dispatcher._inflight_actions[task.task_id] == (_CHEAP_ACTION, atask)
        finally:
            atask.cancel()
            await _settle(atask)

    @pytest.mark.asyncio
    async def test_the_handle_retires_itself_when_the_action_ends(self, coord: Coordinator):
        """Self-removal is what keeps the set from outliving the work."""
        task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="h-retire")
        atask.cancel()
        await _settle(atask)
        assert task.task_id not in coord.dispatcher._inflight_actions

    @pytest.mark.asyncio
    async def test_an_action_that_finishes_normally_leaves_no_handle(self, coord: Coordinator):
        coord.sub.register_executor(_CHEAP_ACTION, lambda _ctx: _done({"ok": True}))
        task, _ = await coord.tasks.create_or_return_existing(
            kind=_CHEAP_ACTION,
            params={},
            idempotency_key="h-quick",
        )
        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())
        await _settle(spawned[0][1])
        assert task.task_id not in coord.dispatcher._inflight_actions


async def _done(payload: dict) -> dict:
    return payload


class TestCancellingInflightActions:
    """The cancellation itself, and who it spares."""

    @pytest.mark.asyncio
    async def test_it_stops_the_action_and_names_what_it_stopped(self, coord: Coordinator):
        task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="c-stop")
        cancelled = await coord.dispatcher.cancel_inflight_actions(reason="test")
        assert cancelled == [task.task_id]
        assert atask.cancelled()

    @pytest.mark.asyncio
    async def test_the_closing_actions_can_be_spared(self, coord: Coordinator):
        """Cancelling the report to save time would leave nothing to show for the run."""
        _, atask = await _start_action(coord, kind=_CLOSING_ACTION, key="c-exempt")
        try:
            assert (
                await coord.dispatcher.cancel_inflight_actions(
                    reason="test",
                    exempt=TIME_BUDGET_EXEMPT_ACTIONS,
                )
                == []
            )
            assert not atask.done()
        finally:
            atask.cancel()
            await _settle(atask)

    @pytest.mark.asyncio
    async def test_cancelling_with_nothing_running_is_a_no_op(self, coord: Coordinator):
        assert await coord.dispatcher.cancel_inflight_actions(reason="test") == []

    @pytest.mark.asyncio
    async def test_the_lane_is_free_again_afterwards(self, coord: Coordinator):
        """A cancelled action that kept its lane would wedge every later one."""
        await _start_action(coord, kind=_CHEAP_ACTION, key="c-lane")
        assert (await coord.locks.lane_holders()).get(_CHEAP_ACTION_LANE, 0) == 1
        await coord.dispatcher.cancel_inflight_actions(reason="test")
        assert (await coord.locks.lane_holders()).get(_CHEAP_ACTION_LANE, 0) == 0


class TestTheRunnerRecordsACancellation:
    """``CancelledError`` is not an ``Exception``, so the runner must name it."""

    @pytest.mark.asyncio
    async def test_a_cancelled_action_does_not_stay_running(self, coord: Coordinator):
        """A row stuck at ``running`` reads as live work to every phase gate."""
        task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="r-terminal")
        atask.cancel()
        await _settle(atask)
        row = await coord.tasks.get(task.task_id)
        assert row.state == "cancelled"
        assert "cancelled_in_flight" in str(row.history)

    @pytest.mark.asyncio
    async def test_the_cancellation_still_reaches_the_caller(self, coord: Coordinator):
        """Recording it must not turn a cancellation into a normal return."""
        _task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="r-propagates")
        atask.cancel()
        await _settle(atask)
        assert atask.cancelled()


class TestThePumpStopsWorkItCannotWaitFor:
    """The trigger side: a spent budget, and a shutdown request."""

    @staticmethod
    def _quick_poll(coord: Coordinator) -> None:
        coord._dispatcher_poll_sec = 0.05

    @pytest.mark.asyncio
    async def test_a_budget_that_runs_out_stops_the_action(self, coord: Coordinator):
        self._quick_poll(coord)
        _set_budget(coord, minutes=600)
        task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-budget")
        _set_budget(coord, minutes=600, elapsed_min=600.0)

        await asyncio.wait_for(pump, timeout=10.0)

        assert atask.cancelled()
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    @pytest.mark.asyncio
    async def test_the_closing_actions_keep_their_reserve(self, coord: Coordinator):
        """The budget hits zero with the closing window still to spend."""
        self._quick_poll(coord)
        _set_budget(coord, minutes=600, elapsed_min=600.0)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CLOSING_ACTION, key="p-closing")
        await asyncio.sleep(0.3)

        assert not atask.done()

        pump.cancel()
        await _settle(pump)

    @pytest.mark.asyncio
    async def test_a_shutdown_request_stops_the_action(self, coord: Coordinator):
        """SIGTERM sets the stop event; before this it only stopped the tick."""
        self._quick_poll(coord)
        _set_budget(coord, minutes=600)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-signal")
        coord._stop.set()

        await asyncio.wait_for(pump, timeout=10.0)

        assert atask.cancelled()

    @pytest.mark.asyncio
    async def test_a_cancelled_pump_does_not_orphan_its_actions(self, coord: Coordinator):
        """The handles live in the pump's frame; leaving must not drop them."""
        self._quick_poll(coord)
        _set_budget(coord, minutes=600)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-orphan")

        pump.cancel()
        await _settle(pump)

        assert atask.cancelled()
        assert coord.dispatcher._inflight_actions == {}


class TestInlineActionsAreReachableToo:
    """The inline path abandons its future, so it needs the same handle."""

    @staticmethod
    def _allow_inline(coord: Coordinator, monkeypatch) -> asyncio.Event:
        """Register a never-finishing executor and clear the gates around it."""
        started = asyncio.Event()
        coord.sub.register_executor(_CHEAP_ACTION, _never_finishes(started))
        monkeypatch.setattr(coord.policy, "validate_intent", lambda *a, **k: None)
        _set_budget(coord, minutes=600)
        return started

    @pytest.mark.asyncio
    async def test_an_inline_action_that_outlived_its_caller_can_be_stopped(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        """Before this, the only thing that ended it was the action itself."""
        started = self._allow_inline(coord, monkeypatch)
        inline = asyncio.create_task(coord.dispatcher._run_action_now(_CHEAP_ACTION, {}))
        await asyncio.wait_for(started.wait(), timeout=5.0)
        task_id = next(iter(coord.dispatcher._inflight_actions))

        assert await coord.dispatcher.cancel_inflight_actions(reason="test") == [task_id]

        await _settle(inline)
        assert inline.cancelled()
        assert (await coord.tasks.get(task_id)).state == "cancelled"
        assert coord.dispatcher._inflight_actions == {}

    @pytest.mark.asyncio
    async def test_an_inline_action_that_finishes_leaves_no_handle(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        monkeypatch.setattr(coord.policy, "validate_intent", lambda *a, **k: None)
        _set_budget(coord, minutes=600)
        coord.sub.register_executor(_CHEAP_ACTION, lambda _ctx: _done({"ok": True}))

        await coord.dispatcher._run_action_now(_CHEAP_ACTION, {})

        assert coord.dispatcher._inflight_actions == {}

    @pytest.mark.asyncio
    async def test_the_sync_bridge_reports_the_cancellation_instead_of_raising(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        """It runs on an agent's turn thread, which a ``CancelledError`` would end."""
        started = self._allow_inline(coord, monkeypatch)
        monkeypatch.setattr(
            coord.dispatcher,
            "_inline_action_whitelist",
            lambda: frozenset({_CHEAP_ACTION}),
        )
        coord._inline_fast_actions_enabled = True
        coord._coordinator_loop = asyncio.get_running_loop()

        outcome: list[str] = []
        caller = threading.Thread(
            target=lambda: outcome.append(coord.dispatcher._run_action_now_sync(_CHEAP_ACTION, {})),
            daemon=True,
        )
        caller.start()
        try:
            await asyncio.wait_for(started.wait(), timeout=5.0)
            await coord.dispatcher.cancel_inflight_actions(reason="test")
            await asyncio.to_thread(caller.join, 5.0)
        finally:
            caller.join(5.0)

        assert outcome and "was cancelled" in outcome[0]


class TestCoordinatorStop:
    """Teardown closes the database, so it cannot leave actions using it."""

    @pytest.mark.asyncio
    async def test_stop_cancels_the_actions_still_running(self, coord: Coordinator):
        _task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="s-stop")

        await coord.stop()

        assert atask.cancelled()
        assert coord.dispatcher._inflight_actions == {}
