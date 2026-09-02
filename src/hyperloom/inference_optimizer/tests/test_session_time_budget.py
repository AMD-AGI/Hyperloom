# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session wall-clock budget defences that live in the orchestrator loop.

Three of the defences are here, the ones outside the executors:

* Admission -- an action whose expected cost cannot fit the budget that is left
  never starts. Covers the pure fit decision, the SharedState accessor both it
  and the grid deadline read, the dispatcher gate, the three intent paths that
  share it, and the pre-dispatch backstop for a task that sat queued until its
  budget drained.
* In-flight cancellation -- the backstop for work already running when the
  budget goes or the process is asked to stop. Covers the handles the dispatcher
  keeps, the cancellation itself, the closing-action carve-out, the task row
  landing terminal instead of stranding at ``running``, which of those handles
  the pump owns on its way out, and the pump and ``Coordinator.stop`` paths that
  trigger it.
* Tick bound -- a reactor turn or phase-enter await that never returns is
  cancelled when the session (or closing) bound elapses, so the tick can still
  reach the wall-clock stop.

The remaining two layers are enforced inside the executors and tested next to
them: the timeout clamp in ``test_explore_executor``, and the subprocess session
reaper in ``test_kill_spawned_server``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.actions.cancel_channel import (
    CancelScope,
    current_cancel_scope,
    stop_was_asked_for,
    use_cancel_scope,
)
from hyperloom.orchestrator.actions.executors._ray_serving import CANCEL_ROUND_GRACE_SEC
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    COOPERATIVE_REAP_BUDGET_SEC,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    STOP_GATE_POLL_SECONDS,
    run_with_session_kill,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.dispatcher import (
    _CANCEL_NOTICE_SEC,
    _COOPERATIVE_CANCEL_GRACE_SEC,
)
from hyperloom.orchestrator.loop.coordinator_helpers import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    action_fits_time_budget,
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
)
from hyperloom.orchestrator.policy.gate import PolicyDenied
from hyperloom.orchestrator.roles import Backend, MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState, effective_closing_grace_sec
from hyperloom.orchestrator.state.task_registry import Task

# An action costing an hour at p50, so a short budget cannot fit it.
_EXPENSIVE_ACTION = "kernel_opt"
_EXPENSIVE_COST_MIN = 60.0
# Cheap enough to fit anything but a nearly-spent budget.
_CHEAP_ACTION = "profile"
# An action the catalogue prices at five minutes, and what one of the two
# sessions that motivated the wall-clock work actually measured for it.
_BASELINE_ACTION = "baseline"
_MEASURED_BASELINE_SEC = 51 * 60.0


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


def _budgeted_state(
    *,
    minutes: float,
    elapsed_min: float = 0.0,
    closing_grace_sec: float | None = None,
) -> SharedState:
    """A standalone state with a finite budget and ``elapsed_min`` already spent."""
    state = SharedState(session_id="s", max_minutes=int(minutes), closing_grace_sec=closing_grace_sec)
    state.elapsed_minutes = lambda **_kw: elapsed_min  # type: ignore[method-assign]
    return state


def _set_budget(coord: Coordinator, *, minutes: float, elapsed_min: float = 0.0) -> None:
    """Give the session a finite budget with ``elapsed_min`` already spent."""
    coord.shared_state.max_minutes = int(minutes)
    coord.shared_state.elapsed_minutes = lambda **_kw: elapsed_min  # type: ignore[method-assign]


class TestTheCostTheGateJudgesOn:
    """Where the expected cost comes from: the action catalogue, or nowhere."""

    def test_a_catalogued_action_reads_its_expected_runtime(self):
        assert expected_action_cost_minutes(ACTION_CATALOGUE[_EXPENSIVE_ACTION]) == pytest.approx(_EXPENSIVE_COST_MIN)

    def test_an_action_the_catalogue_does_not_carry_has_no_estimate(self):
        assert expected_action_cost_minutes(None) == 0.0

    def test_no_catalogued_action_reads_as_free(self):
        """A zero cost admits an action on any budget, so a whole catalogue of
        them is an admission gate that is not there — which is what reading a
        renamed field through a ``getattr`` default silently produced."""
        free = sorted(name for name, meta in ACTION_CATALOGUE.items() if expected_action_cost_minutes(meta) <= 0.0)
        assert free == []


class TestTheCostIsAnchoredOnWhatThisSessionMeasured:
    """The catalogue prices a baseline at five minutes; the field runs it in 51.

    Those estimates are calibrated on small models, so a gate anchored on them
    admits arms a real model cannot pay for -- it would not have stopped either
    of the two sessions that motivated the wall-clock work. PRELUDE's
    affordability gate already anchors on the session's own baseline round;
    admission now reads the same number through the same helper.
    """

    def test_a_measured_round_outprices_the_catalogue_for_an_action_that_benches(self):
        cost = expected_action_cost_minutes(
            ACTION_CATALOGUE["baseline"],
            measured_baseline_sec=_MEASURED_BASELINE_SEC,
        )
        assert cost == pytest.approx(51.0)

    def test_the_catalogue_wins_where_it_prices_more_than_one_round(self):
        """The measurement is a floor, not a replacement: only the catalogue
        knows an action benches a whole grid rather than a single variant."""
        cost = expected_action_cost_minutes(
            ACTION_CATALOGUE[_EXPENSIVE_ACTION],
            measured_baseline_sec=_MEASURED_BASELINE_SEC,
        )
        assert cost == pytest.approx(_EXPENSIVE_COST_MIN)

    def test_an_action_that_never_benches_keeps_its_own_estimate(self):
        """Writing the report costs what it costs; the model's size is not in it."""
        cost = expected_action_cost_minutes(
            ACTION_CATALOGUE["report"],
            measured_baseline_sec=_MEASURED_BASELINE_SEC,
        )
        assert cost == pytest.approx(ACTION_CATALOGUE["report"].typical_runtime_min)

    def test_a_session_with_no_baseline_yet_falls_back_to_the_catalogue(self):
        assert expected_action_cost_minutes(ACTION_CATALOGUE["baseline"]) == pytest.approx(5.0)

    def test_a_warm_replay_is_priced_as_the_baseline_round_it_is(self):
        """Warm replay is not a cheap re-attach to a server that is already hot.

        ``replay_warm_recipe`` is dispatched to ``BaselineExecutor`` with the
        recipe's ``extra_server_args``/``extra_envs``/``patches``, so it boots
        its own server and runs the same benchmark the baseline ran; the recipe
        changes what is measured, not how long measuring takes. Being refused
        near the tail on a 51-minute price is therefore the gate working, not
        the gate being timid — and if warm replay ever does learn to re-attach,
        this is where the price stops being right.
        """
        cost = expected_action_cost_minutes(
            ACTION_CATALOGUE["replay_warm_recipe"],
            measured_baseline_sec=_MEASURED_BASELINE_SEC,
        )
        assert cost == pytest.approx(_MEASURED_BASELINE_SEC / 60.0)
        assert ACTION_CATALOGUE["replay_warm_recipe"].requires_lanes == ACTION_CATALOGUE["baseline"].requires_lanes

    def test_a_measurement_that_is_not_a_number_is_not_a_cost(self):
        assert measured_baseline_runtime_sec(None) == 0.0
        assert measured_baseline_runtime_sec(SimpleNamespace(baseline_runtime_sec="not-a-number")) == 0.0
        assert measured_baseline_runtime_sec(SimpleNamespace(baseline_runtime_sec=-1.0)) == 0.0
        assert measured_baseline_runtime_sec(SimpleNamespace(baseline_runtime_sec=_MEASURED_BASELINE_SEC)) == (
            pytest.approx(_MEASURED_BASELINE_SEC)
        )


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
        state = _budgeted_state(minutes=60)
        assert state.session_budget_usable_sec() == pytest.approx(3600.0 - 72.0)

    def test_a_budget_inside_the_reserve_reads_as_spent(self):
        state = _budgeted_state(minutes=60, elapsed_min=59.9)
        assert state.session_budget_usable_sec() == 0.0

    def test_the_grid_deadline_is_derived_from_the_same_number(self, monkeypatch):
        """Both wall-clock layers must agree on how much budget is left."""
        import time as _time

        state = _budgeted_state(minutes=60, elapsed_min=10.0)
        monkeypatch.setattr(_time, "monotonic", lambda: 1000.0)
        usable = state.session_budget_usable_sec()
        assert state.grid_session_deadline_sec() == pytest.approx(1000.0 + usable)


class TestTheReserveIsTheClosingGraceWindow:
    """The budget held back must be the budget the CLOSE phase actually gets.

    A fixed 120s reserve was only ever right for sessions of at least 100
    minutes: a shorter one was charged more than its closing phase can spend,
    and an operator who passed ``--closing-grace-sec 0`` to disable that phase
    paid 120 seconds for work that never runs.
    """

    @pytest.mark.parametrize(
        ("minutes", "closing_grace_sec", "expected"),
        [
            (120, None, 120.0),  # the default session: unchanged by this fix
            (60, None, 72.0),  # min(120, 2% of the budget)
            (60, 0.0, 0.0),  # closing phase disabled: reserve nothing
            (60, 600.0, 600.0),  # an explicit window wins verbatim
            (0, None, 0.0),  # unbounded budget: nothing to reserve from
        ],
    )
    def test_the_reserve_tracks_the_resolved_grace_window(self, minutes, closing_grace_sec, expected):
        state = SharedState(session_id="s", max_minutes=minutes, closing_grace_sec=closing_grace_sec)
        assert state.closing_reserve_sec() == pytest.approx(expected)
        assert state.closing_reserve_sec() == pytest.approx(effective_closing_grace_sec(minutes, closing_grace_sec))

    @pytest.mark.parametrize("closing_grace_sec", [None, 0.0, 600.0])
    def test_admission_and_the_grid_deadline_agree_on_every_reserve(self, closing_grace_sec, monkeypatch):
        import time as _time

        state = _budgeted_state(minutes=60, elapsed_min=20.0, closing_grace_sec=closing_grace_sec)
        monkeypatch.setattr(_time, "monotonic", lambda: 1000.0)
        usable = state.session_budget_usable_sec()
        assert usable == pytest.approx(max(0.0, 2400.0 - state.closing_reserve_sec()))
        assert state.grid_session_deadline_sec() == pytest.approx(1000.0 + usable)

    def test_a_disabled_closing_phase_leaves_the_last_minutes_spendable(self):
        """The 120s a disabled phase used to cost is the difference here."""
        spent = _budgeted_state(minutes=60, elapsed_min=59.0)
        kept = _budgeted_state(minutes=60, elapsed_min=59.0, closing_grace_sec=0.0)
        assert spent.session_budget_usable_sec() == 0.0
        assert kept.session_budget_usable_sec() == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_the_coordinator_hands_the_operators_window_to_the_state(self, coord: Coordinator):
        """The reserve lives on SharedState, but the flag arrives at the Coordinator."""
        try:
            await coord.run(max_ticks=1, max_minutes=60, closing_grace_sec=0.0)
        finally:
            await coord.stop()
        assert coord.shared_state.closing_grace_sec == 0.0
        assert coord.shared_state.closing_reserve_sec() == 0.0


class TestTimeBudgetGate:
    """The dispatcher gate that turns a fit failure into a refusal."""

    def test_an_action_too_big_for_the_budget_is_denied(self, coord: Coordinator):
        _set_budget(coord, minutes=20)
        denied = coord._time_budget_denial_for_action(_EXPENSIVE_ACTION)
        assert isinstance(denied, PolicyDenied)
        assert denied.rule == "time_budget"
        assert f"{_EXPENSIVE_COST_MIN:.0f} min" in str(denied)
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

    def test_only_the_closing_actions_are_exempt_from_the_budget(self):
        """Recover restarts the server; it is not how a session ends."""
        assert TIME_BUDGET_EXEMPT_ACTIONS == frozenset({"report", "session_breakdown"})

    def test_the_closing_actions_stay_startable_on_an_empty_budget(self, coord: Coordinator):
        """Refusing these would strand the session with nothing to show."""
        _set_budget(coord, minutes=60, elapsed_min=60.0)
        assert coord.shared_state.session_budget_usable_sec() == 0.0
        for action in TIME_BUDGET_EXEMPT_ACTIONS:
            assert coord._time_budget_denial_for_action(action) is None, action

    def test_recover_is_refused_on_an_empty_budget(self, coord: Coordinator):
        """A spent session that still starts recover cannot close."""
        _set_budget(coord, minutes=60, elapsed_min=60.0)
        assert coord.shared_state.session_budget_usable_sec() == 0.0
        denied = coord._time_budget_denial_for_action("recover")
        assert isinstance(denied, PolicyDenied)
        assert denied.rule == "time_budget"

    def test_a_stopping_session_leaves_the_gate_to_the_stop_path(self, coord: Coordinator):
        _set_budget(coord, minutes=1)
        coord.shared_state.stop_reason = "time_exhausted"
        assert coord._time_budget_denial_for_action(_EXPENSIVE_ACTION) is None

    def test_this_session_s_own_baseline_changes_the_answer(self, coord: Coordinator):
        """Half an hour left admits a baseline the catalogue prices at five
        minutes -- until this session has measured one and knows better."""
        _set_budget(coord, minutes=30)
        assert coord._time_budget_denial_for_action(_BASELINE_ACTION) is None

        coord.shared_state.baseline_runtime_sec = _MEASURED_BASELINE_SEC
        denied = coord._time_budget_denial_for_action(_BASELINE_ACTION)

        assert isinstance(denied, PolicyDenied)
        assert denied.rule == "time_budget"
        assert "51 min" in str(denied)

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

    @pytest.mark.asyncio
    async def test_a_queued_recover_is_dropped_when_the_budget_is_spent(
        self,
        coord: Coordinator,
    ):
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind="recover",
            params={},
            idempotency_key="q-recover",
        )
        _set_budget(coord, minutes=600, elapsed_min=600.0)

        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert [t.task_id for t, _, _ in spawned] == []
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    @pytest.mark.asyncio
    async def test_a_queued_targeted_build_is_dropped_when_the_budget_is_spent(
        self,
        coord: Coordinator,
    ):
        """The kind the pump does not join is still subject to the budget gate.

        It is exempt from being *joined*, not from admission: a compile started
        against a spent budget runs on past the session it was charged to.
        """
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind="targeted_build",
            params={},
            idempotency_key="q-build",
            requires_lanes=["build_lane"],
            lease_ttl_sec=900,
        )
        _set_budget(coord, minutes=600, elapsed_min=600.0)

        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert [t.task_id for t, _, _ in spawned] == []
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"
        assert task.task_id not in coord.dispatcher._inflight_actions

    @pytest.mark.asyncio
    async def test_a_targeted_build_that_fits_is_dispatched_but_not_joined(
        self,
        coord: Coordinator,
    ):
        """It is registered for cancellation and excluded, but never joined."""
        _set_budget(coord, minutes=600)
        task, _ = await coord.tasks.create_or_return_existing(
            kind="targeted_build",
            params={},
            idempotency_key="q-build-fits",
            requires_lanes=["build_lane"],
            lease_ttl_sec=900,
        )
        exclude: set[str] = set()

        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=exclude)

        assert [t.task_id for t, _, _ in spawned] == []
        assert task.task_id in coord.dispatcher._inflight_actions
        assert task.task_id in exclude
        await coord.dispatcher.cancel_inflight_actions(reason="test_teardown")

    @pytest.mark.asyncio
    async def test_a_queued_conc_sweep_the_budget_outlived_is_recorded_as_skipped(
        self,
        coord: Coordinator,
    ):
        """Cancelling conc_sweep at dispatch must stamp last_conc_sweep so SWEEP can close."""
        from hyperloom.orchestrator.phases.machine_state import exit_normal_sweep

        _set_budget(coord, minutes=180)
        task, _ = await coord.tasks.create_or_return_existing(
            kind="conc_sweep",
            params={},
            idempotency_key="q-conc-sweep",
        )
        _set_budget(coord, minutes=180, elapsed_min=166.0)

        spawned = await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert [t.task_id for t, _, _ in spawned] == []
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"
        assert coord.shared_state.last_conc_sweep["status"] == "skipped"
        assert coord.shared_state.last_conc_sweep["skip_reason"] == "session_time_budget"
        assert coord.shared_state.last_conc_sweep["was_skipped"] is True
        result = exit_normal_sweep(coord.shared_state)
        assert result is not None
        reason, evidence = result
        assert reason == "sweep_done"
        assert evidence["sweep_status"] == "skipped"

    @pytest.mark.asyncio
    async def test_dropping_an_over_budget_conc_sweep_does_not_erase_a_prior_result(
        self,
        coord: Coordinator,
    ):
        """A later cancel must not overwrite a conc_sweep the session already measured."""
        _set_budget(coord, minutes=180)
        coord.shared_state.record_conc_sweep(
            {"status": "succeeded", "was_skipped": False, "summary": {"successful_pairs": 3}}
        )
        task, _ = await coord.tasks.create_or_return_existing(
            kind="conc_sweep",
            params={},
            idempotency_key="q-conc-sweep-prior",
        )
        _set_budget(coord, minutes=180, elapsed_min=166.0)

        await coord.dispatcher._spawn_fitting_queued(exclude_ids=set())

        assert (await coord.tasks.get(task.task_id)).state == "cancelled"
        assert coord.shared_state.last_conc_sweep["status"] == "succeeded"


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


async def _queue_action(
    coord: Coordinator,
    *,
    kind: str,
    key: str,
    make_executor: Callable[[asyncio.Event], Any] = _never_finishes,
) -> tuple[Task, asyncio.Event]:
    """Queue an action with its real lanes; ``make_executor`` shapes what it does."""
    started = asyncio.Event()
    coord.sub.register_executor(kind, make_executor(started))
    lanes, ttl_sec = coord.dispatcher._registry_lanes_ttl(kind)
    task, _ = await coord.tasks.create_or_return_existing(
        kind=kind,
        params={},
        idempotency_key=key,
        requires_lanes=lanes,
        lease_ttl_sec=ttl_sec,
    )
    return task, started


async def _start_action(
    coord: Coordinator,
    *,
    kind: str,
    key: str,
    make_executor: Callable[[asyncio.Event], Any] = _never_finishes,
) -> tuple[Task, asyncio.Task]:
    """Dispatch the action with no pump running, for the pieces under it."""
    task, started = await _queue_action(coord, kind=kind, key=key, make_executor=make_executor)
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
            entry = coord.dispatcher._inflight_actions[task.task_id]
            assert (entry.kind, entry.atask) == (_CHEAP_ACTION, atask)
            assert not entry.scope.cancelled
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


# Long enough that a round which ran to completion is unmistakable in the
# elapsed time, short enough that an abandoned thread cannot outlive the suite.
_BLOCKING_SEC = 30


def _blocks_in_a_thread(started: asyncio.Event, *, outcome: dict[str, Any]):
    """Build an executor shaped like every benchmark one: a subprocess in a thread.

    ``asyncio.to_thread`` is where all of them spend their time, and a thread
    that has started cannot be cancelled, so this is the shape the last defence
    actually has to stop. ``outcome`` is written after the thread returns, which
    is what makes "the work is over" observable rather than inferred.
    """

    async def _run(_ctx) -> dict:
        started.set()
        proc = await asyncio.to_thread(
            run_with_session_kill,
            ["sleep", str(_BLOCKING_SEC)],
            timeout=_BLOCKING_SEC * 4,
        )
        outcome["returncode"] = proc.returncode
        return {"returncode": proc.returncode}

    return _run


def _sleeps_in_a_thread(started: asyncio.Event, *, seconds: float = 2.0):
    """Build an executor whose thread has no way to hear a cancel."""

    async def _run(_ctx) -> dict:
        started.set()
        await asyncio.to_thread(time.sleep, seconds)
        return {}

    return _run


class TestTheCooperativeStopWindowsCompose:
    """Three waits on the same stop, which only mean anything together.

    Each was picked to look reasonable beside the others -- ten seconds at the
    dispatcher, eight for a round in a Ray actor, five for the SIGTERM grace --
    and composed they said the dispatcher gives up before the work it is waiting
    for can finish. A window a hair short of what stopping costs does not expire
    occasionally: it expires every time, and what it discards is the attributed
    sentinel the round was about to return.

    The components are spelled out here rather than re-derived from the constants
    under test, so a change to one of them has to be argued for -- and the sum is
    spelled out too, so a serial step the unwind takes and no term covers has to
    be argued for as well, rather than quietly making the window short again.
    """

    def test_the_reap_budget_is_what_stopping_a_round_costs(self):
        # Notice at the 0.5s poll, SIGTERM and wait out the 5s grace, collect the
        # SIGKILL'd child for 1s, drain its pipes for 2s.
        assert COOPERATIVE_REAP_BUDGET_SEC == 0.5 + 5.0 + 1.0 + 2.0

    def test_the_ray_grace_outlasts_a_round_stopping_itself(self):
        """A round in an actor stops the same way; the grace has to cover it."""
        assert CANCEL_ROUND_GRACE_SEC >= COOPERATIVE_REAP_BUDGET_SEC

    def test_the_dispatcher_outlasts_the_slowest_honest_stop(self):
        # The Ray path is the long one: 8.5s for the round to stop itself, 0.25s
        # for the answer to be seen, then up to 10s to release the lease it held.
        assert _COOPERATIVE_CANCEL_GRACE_SEC >= 8.5 + 0.25 + 10.0

    def test_reaping_a_server_and_dropping_its_lease_are_both_paid(self):
        """The two release waits are a sequence, so the window has to cover both.

        A Ray round's unwind reaps the server it left behind and only then closes
        the lease it ran in -- that order is a requirement, not an accident, so no
        GPU process outlives the lease. Taking the longer of the two leaves the
        window five seconds short of what that unwind costs, which is the same
        shortfall these windows were derived to remove.
        """
        assert _COOPERATIVE_CANCEL_GRACE_SEC >= 8.5 + 0.25 + 5.0 + 10.0

    def test_the_window_is_exactly_the_terms_it_names(self):
        """An upper bound, so a term the unwind pays and the sum omits is a bug.

        Spelled as a total and not only as a floor: a fifth serial step was found
        in the unwind that no term covered, and a floor would have gone on passing
        while the sum stayed short of what stopping costs.
        """
        assert _COOPERATIVE_CANCEL_GRACE_SEC == 8.5 + 0.25 + 5.0 + 10.0

    def test_a_cancelled_scope_is_visible_to_the_work_inside_it(self):
        """The unwind's steps read the scope, not a returncode.

        A cooperative stop returns its sentinel rather than raising, so a step
        can be reached on the ordinary path with the cancel already outstanding
        and has to ask.
        """
        scope = CancelScope()
        with use_cancel_scope(scope):
            assert not stop_was_asked_for()
            scope.cancel(reason="session_time_exhausted")
            assert stop_was_asked_for()

    def test_work_outside_an_action_is_never_told_to_skip(self):
        """No scope means no cancel, so a bare call keeps every step it had."""
        assert not stop_was_asked_for()

    def test_the_notice_window_is_the_poll_the_scope_is_checked_at(self):
        """Nothing is listening yet is a claim about the poll, not about the work."""
        assert _CANCEL_NOTICE_SEC >= STOP_GATE_POLL_SECONDS
        assert _CANCEL_NOTICE_SEC < COOPERATIVE_REAP_BUDGET_SEC


class TestTheCancelChannel:
    """The channel itself: what it carries, and how far it reaches."""

    @pytest.mark.asyncio
    async def test_a_worker_thread_sees_the_scope_of_the_task_that_started_it(self):
        """The whole idiom rests on ``to_thread`` copying the context."""
        scope = CancelScope()
        with use_cancel_scope(scope):
            seen = await asyncio.to_thread(current_cancel_scope)
        assert seen is scope

    @pytest.mark.asyncio
    async def test_code_outside_an_action_finds_no_scope(self):
        """A Ray worker and a bare call are the same case: nothing to check."""
        assert await asyncio.to_thread(current_cancel_scope) is None

    def test_the_first_reason_is_the_one_kept(self):
        """A blanket cancel arriving second must not overwrite the specific cause."""
        scope = CancelScope()
        scope.cancel(reason="session_time_exhausted")
        scope.cancel(reason="dispatcher_pump_exit")
        assert scope.cancelled
        assert scope.reason == "session_time_exhausted"

    def test_a_scope_reports_whether_anything_is_watching_it(self):
        scope = CancelScope()
        assert not scope.has_listeners
        with scope.listening():
            assert scope.has_listeners
        assert not scope.has_listeners


class TestCancellingWorkThatBlocksInAThread:
    """Cancelling the coroutine does not stop the thread it is waiting on.

    The canceller gets a clean ``CancelledError`` off the ``await`` while the
    subprocess runs on to its own hard timeout, so the lanes and the GPU lease
    are released, and the database closed, with the benchmark still holding the
    card. Stopping it takes a channel the thread itself checks.
    """

    @pytest.mark.asyncio
    async def test_the_work_is_over_before_the_cancel_returns(self, coord: Coordinator):
        outcome: dict[str, Any] = {}
        task, atask = await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-thread",
            make_executor=lambda started: _blocks_in_a_thread(started, outcome=outcome),
        )
        began = time.monotonic()

        assert await coord.dispatcher.cancel_inflight_actions(reason="test") == [task.task_id]

        assert outcome, "the cancel returned while the thread was still running"
        assert time.monotonic() - began < _BLOCKING_SEC
        await _settle(atask)

    @pytest.mark.asyncio
    async def test_the_stop_is_attributed_to_the_orchestrator(self, coord: Coordinator):
        """A cancel is not a timeout and not a slow variant; the ledger reads returncodes."""
        outcome: dict[str, Any] = {}
        await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-thread-rc",
            make_executor=lambda started: _blocks_in_a_thread(started, outcome=outcome),
        )

        await coord.dispatcher.cancel_inflight_actions(reason="test_reason")

        assert outcome["returncode"] == ORCHESTRATOR_CANCELLED_RETURNCODE

    @pytest.mark.asyncio
    async def test_a_thread_with_nothing_listening_is_still_not_waited_for(self, coord: Coordinator):
        """The channel is cooperative, so work that cannot hear it is left behind.

        Waiting on it anyway would trade a leaked thread for a shutdown that
        hangs on one, which is the worse of the two.
        """
        _task, atask = await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-deaf",
            make_executor=_sleeps_in_a_thread,
        )
        began = time.monotonic()

        await coord.dispatcher.cancel_inflight_actions(reason="test")

        assert time.monotonic() - began < _COOPERATIVE_CANCEL_GRACE_SEC
        assert atask.cancelled()


def _runs_a_round_in_a_lease(started: asyncio.Event, *, outcome: dict[str, Any], lease: Any):
    """An executor shaped like the production default: a round inside a Ray lease.

    ``_should_use_ray_backend`` is off under pytest and on by default on a single
    node, so this is the branch every real run takes and no test did.
    """

    async def _run(_ctx) -> dict:
        started.set()
        rc, _out, _err = await asyncio.to_thread(
            lease.run_session_kill,
            [sys.executable, "-c", f"import time; time.sleep({_BLOCKING_SEC})"],
            timeout=_BLOCKING_SEC * 4,
        )
        outcome["returncode"] = rc
        return {"returncode": rc}

    return _run


class TestCancellingARoundInsideARayLease:
    """The production default routes rounds through a Ray actor, not a local child.

    The scope is a ContextVar, so it does not exist in the actor's process: the
    lease has to notice the cancel on this side and forward it, or the four-layer
    defence has no reach at all on the path every real single-node run takes.
    """

    @pytest.fixture
    def lease(self, serving_lease_on_a_ray_double: Any) -> Any:
        return serving_lease_on_a_ray_double

    @pytest.mark.asyncio
    async def test_the_round_in_the_actor_stops_before_the_cancel_returns(
        self,
        coord: Coordinator,
        lease: Any,
    ):
        outcome: dict[str, Any] = {}
        task, atask = await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-ray",
            make_executor=lambda started: _runs_a_round_in_a_lease(started, outcome=outcome, lease=lease),
        )
        began = time.monotonic()

        assert await coord.dispatcher.cancel_inflight_actions(reason="test") == [task.task_id]

        assert outcome, "the cancel returned while the round was still running in the actor"
        assert time.monotonic() - began < _BLOCKING_SEC
        await _settle(atask)

    @pytest.mark.asyncio
    async def test_the_stop_is_attributed_to_the_orchestrator(self, coord: Coordinator, lease: Any):
        """The actor reaps its own tree, so the sentinel is the same one the local path returns."""
        outcome: dict[str, Any] = {}
        await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-ray-rc",
            make_executor=lambda started: _runs_a_round_in_a_lease(started, outcome=outcome, lease=lease),
        )

        await coord.dispatcher.cancel_inflight_actions(reason="test_reason")

        assert outcome["returncode"] == ORCHESTRATOR_CANCELLED_RETURNCODE

    @pytest.mark.asyncio
    async def test_an_actor_that_will_not_answer_is_killed_and_the_stop_still_named(
        self,
        coord: Coordinator,
        lease: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A wedged actor must not hold the lease open, and must not go unattributed."""
        from hyperloom.orchestrator.actions.executors import _ray_serving as rs

        monkeypatch.setattr(rs, "CANCEL_ROUND_GRACE_SEC", 0.5)
        monkeypatch.setattr(rs.ServingLease, "_ask_actor_to_cancel", lambda _self, _reason: False)
        outcome: dict[str, Any] = {}
        await _start_action(
            coord,
            kind=_CHEAP_ACTION,
            key="c-ray-wedged",
            make_executor=lambda started: _runs_a_round_in_a_lease(started, outcome=outcome, lease=lease),
        )

        await coord.dispatcher.cancel_inflight_actions(reason="test_reason")

        assert outcome["returncode"] == ORCHESTRATOR_CANCELLED_RETURNCODE
        assert lease._actor is None, "the lease must be released when its actor is killed"


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


def _quick_poll(coord: Coordinator) -> None:
    """Shorten the pump's re-scan interval so a pump test is not a wall-clock test."""
    coord._dispatcher_poll_sec = 0.05


class TestThePumpStopsWorkItCannotWaitFor:
    """The trigger side: a spent budget, and a shutdown request."""

    @pytest.mark.asyncio
    async def test_a_budget_that_runs_out_stops_the_action(self, coord: Coordinator):
        _quick_poll(coord)
        _set_budget(coord, minutes=600)
        task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-budget")
        _set_budget(coord, minutes=600, elapsed_min=600.0)

        await asyncio.wait_for(pump, timeout=10.0)

        assert atask.cancelled()
        assert (await coord.tasks.get(task.task_id)).state == "cancelled"

    @pytest.mark.asyncio
    async def test_the_closing_actions_keep_their_reserve(self, coord: Coordinator):
        """The budget hits zero with the closing window still to spend."""
        _quick_poll(coord)
        _set_budget(coord, minutes=600, elapsed_min=600.0)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CLOSING_ACTION, key="p-closing")
        await asyncio.sleep(0.3)

        assert not atask.done()

        pump.cancel()
        await _settle(pump)

    @pytest.mark.asyncio
    async def test_a_shutdown_request_stops_the_action(self, coord: Coordinator):
        """SIGTERM sets the stop event; before this it only stopped the tick."""
        _quick_poll(coord)
        _set_budget(coord, minutes=600)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-signal")
        coord._stop.set()

        await asyncio.wait_for(pump, timeout=10.0)

        assert atask.cancelled()

    @pytest.mark.asyncio
    async def test_a_cancelled_pump_does_not_orphan_its_actions(self, coord: Coordinator):
        """The handles live in the pump's frame; leaving must not drop them."""
        _quick_poll(coord)
        _set_budget(coord, minutes=600)
        _task, atask, pump = await _start_action_under_pump(coord, kind=_CHEAP_ACTION, key="p-orphan")

        pump.cancel()
        await _settle(pump)

        assert atask.cancelled()
        assert coord.dispatcher._inflight_actions == {}


def _allow_inline(coord: Coordinator, monkeypatch) -> asyncio.Event:
    """Register a never-finishing executor and clear the gates around it."""
    started = asyncio.Event()
    coord.sub.register_executor(_CHEAP_ACTION, _never_finishes(started))
    monkeypatch.setattr(coord.policy, "validate_intent", lambda *a, **k: None)
    _set_budget(coord, minutes=600)
    return started


async def _start_inline_action(coord: Coordinator, monkeypatch) -> asyncio.Task:
    """Run an inline action and wait until it is registered and under way."""
    started = _allow_inline(coord, monkeypatch)
    inline = asyncio.create_task(coord.dispatcher._run_action_now(_CHEAP_ACTION, {}))
    await asyncio.wait_for(started.wait(), timeout=5.0)
    return inline


class TestInlineActionsAreReachableToo:
    """The inline path abandons its future, so it needs the same handle."""

    @pytest.mark.asyncio
    async def test_an_inline_action_that_outlived_its_caller_can_be_stopped(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        """Before this, the only thing that ended it was the action itself."""
        inline = await _start_inline_action(coord, monkeypatch)
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
        started = _allow_inline(coord, monkeypatch)
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


class TestThePumpOnlyCancelsWhatItSpawned:
    """The registry is dispatcher-wide; the pump's exit sweep is not.

    An inline action is registered by whoever ran it, not by the pump, and is
    designed to keep going after that caller stops waiting. A tick with nothing
    queued returns immediately, so an exit sweep over the whole registry would
    make the emptiest possible pump the thing that kills it.
    """

    @pytest.mark.asyncio
    async def test_a_tick_with_nothing_queued_leaves_an_inline_action_running(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        inline = await _start_inline_action(coord, monkeypatch)
        try:
            await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=10.0)

            assert not inline.done()
            assert coord.dispatcher._inflight_actions
        finally:
            inline.cancel()
            await _settle(inline)

    @pytest.mark.asyncio
    async def test_a_cancelled_pump_takes_its_own_and_only_its_own(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        """Narrowing the sweep must not cost the pump the actions it does own."""
        _quick_poll(coord)
        inline = await _start_inline_action(coord, monkeypatch)
        _task, spawned, pump = await _start_action_under_pump(coord, kind=_CLOSING_ACTION, key="own-spawn")
        try:
            pump.cancel()
            await _settle(pump)

            assert spawned.cancelled()
            assert not inline.done()
        finally:
            inline.cancel()
            await _settle(inline)

    @pytest.mark.asyncio
    async def test_a_shutdown_still_reaches_an_inline_action(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        """The narrower sweep must not blunt the trigger that has to reach everything."""
        inline = await _start_inline_action(coord, monkeypatch)
        coord._stop.set()

        await asyncio.wait_for(coord._pump_dispatcher_once(), timeout=10.0)

        await _settle(inline)
        assert inline.cancelled()


class TestCoordinatorStop:
    """Teardown closes the database, so it cannot leave actions using it."""

    @pytest.mark.asyncio
    async def test_stop_cancels_the_actions_still_running(self, coord: Coordinator):
        _task, atask = await _start_action(coord, kind=_CHEAP_ACTION, key="s-stop")

        await coord.stop()

        assert atask.cancelled()
        assert coord.dispatcher._inflight_actions == {}


async def _idle(*_args, **_kwargs) -> None:
    return None


async def _hang_forever(*_args, **_kwargs) -> None:
    await asyncio.sleep(3600)


class TestATickCannotOutliveTheSessionBound:
    """A step that never returns used to skip the wall-clock stop at tick end."""

    @pytest.mark.asyncio
    async def test_a_hanging_reactor_still_stops_when_the_budget_ends(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        monkeypatch.setattr(coord, "_advance_phase_if_needed", _idle)
        monkeypatch.setattr(coord, "_reactor_pass", _hang_forever)
        monkeypatch.setattr(coord, "_pump_dispatcher_once", _idle)
        started = time.monotonic()
        try:
            reason = await asyncio.wait_for(
                coord.run(max_minutes=0.05, closing_grace_sec=0.0, tick_interval_sec=0.0),
                timeout=15.0,
            )
        finally:
            await coord.stop()
        assert reason == "time_exhausted"
        assert time.monotonic() - started < 10.0

    @pytest.mark.asyncio
    async def test_a_hanging_phase_enter_still_stops_when_the_budget_ends(
        self,
        coord: Coordinator,
        monkeypatch,
    ):
        monkeypatch.setattr(coord, "_advance_phase_if_needed", _hang_forever)
        monkeypatch.setattr(coord, "_reactor_pass", _idle)
        monkeypatch.setattr(coord, "_pump_dispatcher_once", _idle)
        started = time.monotonic()
        try:
            reason = await asyncio.wait_for(
                coord.run(max_minutes=0.05, closing_grace_sec=0.0, tick_interval_sec=0.0),
                timeout=15.0,
            )
        finally:
            await coord.stop()
        assert reason == "time_exhausted"
        assert time.monotonic() - started < 10.0

    @pytest.mark.asyncio
    async def test_a_spent_bound_does_not_start_the_next_step(self, coord: Coordinator):
        coord._run_deadline = time.monotonic() - 1.0
        started: list[bool] = []

        async def _must_not_run() -> None:
            started.append(True)

        await coord._await_within_session_bound(_must_not_run, stage="test")
        assert started == []

    @pytest.mark.asyncio
    async def test_no_deadline_still_runs_the_step(self, coord: Coordinator):
        started: list[bool] = []

        async def _ok() -> None:
            started.append(True)

        await coord._await_within_session_bound(_ok, stage="test")
        assert started == [True]

    @pytest.mark.asyncio
    async def test_closing_uses_the_grace_bound_not_the_session_deadline(self, coord: Coordinator):
        coord._run_deadline = time.monotonic() - 10.0
        coord._closing_deadline = time.monotonic() + 60.0
        coord.shared_state.closing_phase = True
        started: list[bool] = []

        async def _ok() -> None:
            started.append(True)

        await coord._await_within_session_bound(_ok, stage="close")
        assert started == [True]


class TestThePersistedDeadlineIsTheLoopDeadline:
    """Coordinator.run must not reissue a full max_minutes on a spent session.

    The GPU CI e2e run is a smoke check, not a wall-clock assertion. This is
    the in-process stand-in for #1146 item 5: a session whose deadline is
    already in the past must stop as ``time_exhausted`` instead of running out
    the remaining ticks against a freshly computed budget.
    """

    @pytest.mark.asyncio
    async def test_run_stamps_deadline_unix_from_start_ts(self, coord: Coordinator):
        from hyperloom.common.coerce import to_unix

        try:
            await coord.run(max_ticks=1, max_minutes=60, closing_grace_sec=0.0)
        finally:
            await coord.stop()
        stamped = coord.shared_state.deadline_unix
        start = to_unix(coord.shared_state.start_ts)
        assert stamped == pytest.approx(start + 3600.0, abs=2.0)

    @pytest.mark.asyncio
    async def test_run_stamps_a_fractional_budget_before_int_truncation(self, coord: Coordinator):
        from hyperloom.common.coerce import to_unix

        try:
            await coord.run(max_ticks=1, max_minutes=0.0001, closing_grace_sec=0.0)
        finally:
            await coord.stop()
        start = to_unix(coord.shared_state.start_ts)
        assert coord.shared_state.deadline_unix == pytest.approx(start + 0.006, abs=0.05)

    @pytest.mark.asyncio
    async def test_run_keeps_a_deadline_stamped_before_this_process(self, coord: Coordinator):
        from datetime import datetime, timedelta, timezone

        start = datetime.now(timezone.utc) - timedelta(hours=3)
        original = start.timestamp() + 180 * 60.0
        coord.shared_state.start_ts = start.isoformat()
        coord.shared_state.max_minutes = 180
        coord.shared_state.deadline_unix = original
        try:
            await coord.run(max_ticks=1, max_minutes=180, closing_grace_sec=0.0)
        finally:
            await coord.stop()
        assert coord.shared_state.deadline_unix == pytest.approx(original)

    @pytest.mark.asyncio
    async def test_a_spent_session_stops_instead_of_reissuing_the_budget(self, coord: Coordinator):
        from datetime import datetime, timedelta, timezone

        start = datetime.now(timezone.utc) - timedelta(hours=3)
        coord.shared_state.start_ts = start.isoformat()
        coord.shared_state.max_minutes = 180
        coord.shared_state.deadline_unix = start.timestamp() + 180 * 60.0
        started = time.monotonic()
        try:
            reason = await coord.run(
                max_minutes=180,
                closing_grace_sec=0.0,
                max_ticks=8,
            )
        finally:
            await coord.stop()
        assert reason == "time_exhausted"
        assert time.monotonic() - started < 15.0
        assert "close_backends" in coord.shared_state.teardown_timings_sec
        assert coord.shared_state.teardown_timings_sec["total"] >= 0.0
