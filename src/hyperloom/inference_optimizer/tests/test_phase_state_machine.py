# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""phase state machine tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.phases import machine_state as phase_state
from hyperloom.inference_optimizer.breakdown.stop_reasons import is_valid_stop_reason
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import (
    PolicyGate,
)
from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir

from ._geak_helpers import stop_geak_before_launch


def _no_controller_run(**kwargs: Any) -> dict[str, Any]:
    return {"status": "no_opportunity", "patch_count": 0, "task_count": 0, "output_dir": str(kwargs["output_dir"])}


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    from hyperloom.orchestrator.kernel import controller_submit

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    stop_geak_before_launch(monkeypatch)
    monkeypatch.setattr(controller_submit, "run_controller_subprocess", _no_controller_run)
    return make_session_dir()


def test_phase_names_are_monotonic():
    assert phase_state.PHASE_NAMES == (
        "PRELUDE",
        "ENABLEMENT",
        "FRAMEWORK_AGENT",
        "KERNEL_AGENT",
        "SWEEP",
        "CLOSE",
    )
    for i, name in enumerate(phase_state.PHASE_NAMES):
        assert phase_state.phase_index(name) == i
    assert phase_state.phase_index("unknown") == -1


def test_allowed_actions_disjoint_phases():
    # Kernel-agent-owned actions only run in KERNEL (Inv-2.1).
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        assert "recover" not in allowed
    assert "baseline" in phase_state.PHASE_ALLOWED_ACTIONS["PRELUDE"]
    assert "baseline" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    # ENABLEMENT carries baseline so the Coordinator's revalidation survives the
    # phase sweep, but reserves it so no agent can propose one.
    assert "baseline" in phase_state.PHASE_ALLOWED_ACTIONS["ENABLEMENT"]
    assert "baseline" not in phase_state.allowed_actions_for("ENABLEMENT")
    assert "baseline" in phase_state.allowed_actions_for("PRELUDE")
    # kernel_opt and gemm_tuning are Coordinator-owned: dispatched once at KERNEL entry from a lane budget, so they
    # are proposable in no phase at all.
    assert "integrate" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL_AGENT"]
    for phase in phase_state.PHASE_NAMES:
        assert "kernel_opt" not in phase_state.PHASE_ALLOWED_ACTIONS[phase]
        assert "gemm_tuning" not in phase_state.PHASE_ALLOWED_ACTIONS[phase]
    assert "conc_sweep" in phase_state.PHASE_ALLOWED_ACTIONS["SWEEP"]
    assert "conc_sweep" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    assert "report" in phase_state.PHASE_ALLOWED_ACTIONS["CLOSE"]


def test_stop_reason_vocab_includes_v06_and_v08():
    for reason in (
        "target_reached",
        "time_exhausted",
        "max_ticks",
        "baseline_failed",
        "emergency",
        "coordinator_exception",
        "sweep_failed",
        "baseline_arg_error",
    ):
        assert is_valid_stop_reason(reason), reason
    assert not is_valid_stop_reason("totally_invented")


def test_set_stop_reason_keeps_baseline_arg_error(tmp_path):
    """baseline_arg_error must survive set_stop_reason and not map to unknown."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState(session_id="t", model_name="m", model_path="m")
    written = state.set_stop_reason("baseline_arg_error")
    assert written == "baseline_arg_error"
    assert state.stop_reason == "baseline_arg_error"


def test_normalize_budget_pct_falls_back_to_defaults():
    out = phase_state.normalize_budget_pct(None)
    assert out == phase_state.DEFAULT_PHASE_BUDGET_PCT
    out = phase_state.normalize_budget_pct({"FRAMEWORK_AGENT": 0.5, "BOGUS": 0.9})
    assert out["FRAMEWORK_AGENT"] == 0.5
    assert out["PRELUDE"] == phase_state.DEFAULT_PHASE_BUDGET_PCT["PRELUDE"]
    assert "BOGUS" not in out


def test_prelude_exits_on_baseline_tput():
    state = SharedState(baseline_tput=0.0, phase="PRELUDE", phase_budget_pct={}, phase_started_unix=0.0, max_minutes=0)
    assert phase_state.compute_next_phase(state) is None
    state.baseline_tput = 1234.5
    out = phase_state.compute_next_phase(state)
    assert out is not None
    _next_phase, reason, evidence = out
    assert reason == "prelude_done"
    assert evidence["predicate_inputs"]["baseline"]["tput"] == 1234.5


def test_prelude_blocked_while_warm_replay_in_flight():
    """PRELUDE must not advance to FRAMEWORK until warm-replay settles."""
    state = SharedState(
        phase="PRELUDE",
        phase_budget_pct={},
        phase_started_unix=0.0,
        max_minutes=0,
        baseline_tput=1234.5,
        warm_replay_outcome={"status": "in_flight", "replay_task_id": "abc"},
    )
    assert phase_state.compute_next_phase(state) is None
    state.warm_replay_outcome = {"status": "failed"}
    out = phase_state.compute_next_phase(state)
    assert out is not None and out[1] == "prelude_done"


def test_prelude_waits_for_the_initial_analysis():
    """Leaving before the initial roofline is booked hands that analysis to the next phase instead of seeding this one."""
    state = SharedState(
        phase="PRELUDE",
        phase_budget_pct={},
        phase_started_unix=0.0,
        max_minutes=0,
        baseline_tput=1234.5,
        auto_roofline_pending_task_id="rl-1",
    )
    assert phase_state.compute_next_phase(state) is None
    state.auto_roofline_pending_task_id = ""
    out = phase_state.compute_next_phase(state)
    assert out is not None and out[1] == "prelude_done"


def _prelude_state(
    *,
    max_minutes: int = 180,
    spent_sec: float = 0.0,
    usable_sec: float | None = None,
    baseline_tput: float = 0.0,
    baseline_runtime_sec: float = 0.0,
    baseline_post_ready_runtime_sec: float = 0.0,
    baseline_warm_runtime_sec: float = 0.0,
    baseline_measure_round_dropped: bool = False,
    baseline_failure_streak: int = 0,
    auto_roofline_pending_task_id: str = "",
) -> SharedState:
    """A PRELUDE-phase state with an explicit clock, as the budget policy reads it."""
    state = SharedState(
        auto_roofline_pending_task_id=auto_roofline_pending_task_id,
        phase="PRELUDE",
        max_minutes=max_minutes,
        baseline_failure_streak=baseline_failure_streak,
        phase_elapsed_totals={"PRELUDE": spent_sec},
        phase_started_unix=0.0,
        phase_budget_pct={},
        baseline_tput=baseline_tput,
        baseline_runtime_sec=baseline_runtime_sec,
        baseline_post_ready_runtime_sec=baseline_post_ready_runtime_sec,
        baseline_warm_runtime_sec=baseline_warm_runtime_sec,
        baseline_measure_round_dropped=baseline_measure_round_dropped,
        resumed_ts="",
        pending_escalate_hint="",
    )
    state.session_budget_usable_sec = lambda: usable_sec
    return state


def test_prelude_can_afford_an_arm_the_budget_still_covers():
    """The normal case must be untouched: a cheap arm early in a session runs."""
    state = _prelude_state(spent_sec=600.0, usable_sec=10_000.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=300.0)
    assert affordable is True
    # Half of 180 minutes is held for the optimization phases; the rest is PRELUDE's.
    assert evidence["affordable_sec"] == pytest.approx(4600.0)


def test_prelude_refuses_an_arm_that_would_eat_the_optimization_reserve():
    """The Qwen3.5-397B shape: 51 minutes of baseline, then a roofline that costs another 45+."""
    state = _prelude_state(spent_sec=3090.0, usable_sec=7700.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=2706.0)
    assert affordable is False
    assert evidence["bound"] == "optimization_reserve"
    # 7700s left, 5400s of it spoken for, so the arm may cost at most 2300s.
    assert evidence["affordable_sec"] == pytest.approx(2300.0)


def test_a_resumed_prelude_is_not_charged_for_what_the_earlier_leg_spent():
    """Banked phase spend and the session clock answer to different origins."""
    state = _prelude_state(spent_sec=10_000.0, usable_sec=10_000.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=2706.0)
    assert affordable is True
    assert evidence["affordable_sec"] == pytest.approx(4600.0)


def test_prelude_budget_policy_is_inert_without_a_clock():
    """An unbounded run has no budget to protect, so nothing is refused."""
    state = _prelude_state(max_minutes=0, usable_sec=None)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=99_999.0)
    assert affordable is True
    assert evidence["reason"] == "unbounded_budget"


def test_time_exhausted_during_prelude_finally_has_a_producer():
    """The reason was in the vocabulary and in the report glossary with no code path to it."""
    state = _prelude_state(spent_sec=10_800.0, usable_sec=0.0)
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None
    next_phase, reason, evidence = out
    assert (next_phase, reason) == ("CLOSE", "time_exhausted_during_prelude")
    assert evidence["terminal"] is True
    assert is_valid_stop_reason(reason)
    assert phase_state.replay_next_phase(evidence["predicate_inputs"]) == out


def test_the_prelude_allowance_never_reads_as_a_debt():
    """Past the reserve the phase can buy nothing; the figure says zero seconds, not minus an hour."""
    # A 12h session reserves 21600 s for optimization, and 14718.5 s of clock is left.
    state = _prelude_state(max_minutes=720, usable_sec=14_718.5)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=300.0)
    assert affordable is False
    assert evidence["affordable_sec"] == pytest.approx(0.0)


def test_a_prelude_that_can_buy_nothing_closes_instead_of_idling():
    """The phase that ticked for 4.1h: nothing it may spend, no arm of its own, and a clock that is not zero."""
    state = _prelude_state(max_minutes=720, spent_sec=25_000.0, usable_sec=14_718.5, baseline_failure_streak=2)
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None
    next_phase, reason, evidence = out
    assert (next_phase, reason) == ("CLOSE", "prelude_budget_exhausted")
    assert evidence["terminal"] is True
    assert evidence["prelude_affordable_sec"] == pytest.approx(0.0)
    assert evidence["session_usable_sec"] == pytest.approx(14_718.5)
    assert is_valid_stop_reason(reason)
    assert phase_state.replay_next_phase(evidence["predicate_inputs"]) == out


def test_a_prelude_with_budget_of_its_own_keeps_retrying():
    """The same failed baseline, before the reserve line: the phase still has rounds it can pay for."""
    state = _prelude_state(max_minutes=720, spent_sec=6_000.0, usable_sec=30_000.0, baseline_failure_streak=2)
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None


def test_a_round_still_in_flight_is_not_an_idle_prelude():
    """A warm replay that has not landed is work in progress, not a phase waiting on nothing."""
    state = _prelude_state(max_minutes=720, spent_sec=25_000.0, usable_sec=14_718.5, baseline_failure_streak=2)
    state.warm_replay_outcome = {"status": "in_flight", "replay_task_id": "abc"}
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None


def test_a_plain_baseline_retry_in_flight_is_not_an_idle_prelude():
    """A retried ``baseline`` is PRELUDE's own work too: closing on it would cancel the task."""
    state = _prelude_state(max_minutes=720, spent_sec=25_000.0, usable_sec=14_718.5, baseline_failure_streak=2)
    flags = phase_state.PhaseFlags(kernel_enabled=True)

    in_flight = phase_state.WorkInFlight(prelude=True)

    assert phase_state.compute_next_phase(state, flags=flags, in_flight=in_flight) is None

    # The fact travels with the decision, so a replay of it refuses the terminal as well.
    inputs = phase_state.workflow_predicate_inputs(state, flags=flags, in_flight=in_flight)
    assert inputs["pending_work"] == {"prelude_work_in_flight": True}
    assert phase_state.replay_next_phase(inputs) is None

    out = phase_state.compute_next_phase(state, flags=flags)
    assert out is not None and out[1] == "prelude_budget_exhausted"


def test_an_unbounded_prelude_never_reaches_the_budget_line():
    """Without a clock there is no reserve to cross, so a failed baseline keeps its retries."""
    state = _prelude_state(max_minutes=0, usable_sec=None, baseline_failure_streak=2)
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None


def test_a_landed_baseline_outranks_the_exhausted_clock():
    """With a baseline in hand the run has something to optimize; the later phases judge for themselves."""
    state = _prelude_state(spent_sec=10_800.0, usable_sec=0.0, baseline_tput=1074.7)
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None
    assert out[1] == "prelude_done"


def test_prelude_holds_while_the_initial_roofline_is_pending():
    state = _prelude_state(usable_sec=10_000.0, baseline_tput=1074.7, auto_roofline_pending_task_id="rl-1")
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None
    state.auto_roofline_pending_task_id = ""
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None and out[1] == "prelude_done"
    assert phase_state.replay_next_phase(out[2]["predicate_inputs"]) == out


def test_a_spent_clock_does_not_wait_for_the_initial_roofline():
    state = _prelude_state(
        spent_sec=10_800.0, usable_sec=0.0, baseline_tput=1074.7, auto_roofline_pending_task_id="rl-1"
    )
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None and out[1] == "prelude_done"


def test_the_initial_roofline_wait_stops_at_the_optimization_reserve():
    # 180 min session reserves 5400 s for optimization; 5000 s usable means PRELUDE's own allowance is spent.
    state = _prelude_state(usable_sec=5_000.0, baseline_tput=1074.7, auto_roofline_pending_task_id="rl-1")
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None and out[1] == "prelude_done"
    assert phase_state.replay_next_phase(out[2]["predicate_inputs"]) == out
    state.session_budget_usable_sec = lambda: 6_000.0
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None


def test_the_phase_block_shows_the_initial_roofline_hold_exactly_while_it_holds():
    """A spent phase share under the hold must not read as a spent session."""

    def hold_lines(state: SharedState) -> list[str]:
        return [line for line in phase_state.phase_status_summary(state).splitlines() if line.startswith("hold")]

    state = _prelude_state(usable_sec=10_000.0, baseline_tput=1074.7, auto_roofline_pending_task_id="rl-1")
    [hold] = hold_lines(state)
    assert "rl-1" in hold and "4600" in hold
    state.session_budget_usable_sec = lambda: 5_000.0
    assert hold_lines(state) == []
    state.session_budget_usable_sec = lambda: 10_000.0
    state.auto_roofline_pending_task_id = ""
    assert hold_lines(state) == []


def test_prelude_exit_states_whether_one_optimization_round_still_fits():
    """The plain statement neither field session ever got: preparation spent the run."""
    state = _prelude_state(baseline_tput=1074.7, baseline_runtime_sec=2705.7, usable_sec=2796.0)
    out = phase_state.compute_next_phase(state)
    assert out is not None
    _next, reason, evidence = out
    assert reason == "prelude_done"
    assert evidence["fits_one_optimization_round"] is True
    assert evidence["affordable_rounds"] == pytest.approx(1.03, abs=0.01)

    state.session_budget_usable_sec = lambda: 1200.0
    out2 = phase_state.compute_next_phase(state)
    assert out2 is not None
    assert out2[2]["fits_one_optimization_round"] is False


def test_an_infinite_baseline_runtime_prices_no_optimization_round():
    state = _prelude_state(baseline_tput=1074.7, baseline_runtime_sec=float("inf"), usable_sec=2796.0)
    out = phase_state.compute_next_phase(state)
    assert out is not None
    _next, reason, evidence = out
    assert reason == "prelude_done"
    assert "fits_one_optimization_round" not in evidence
    assert evidence["predicate_inputs"]["baseline"]["runtime_sec"] is None


# The workload the cold-anchor cases below are priced against: a 900s cold round whose last 550s was the benchmark, so
# the boot took 350s, and a 400s hot pass.
_COLD_ANCHOR_WORKLOAD = {
    "baseline_tput": 1074.7,
    "baseline_runtime_sec": 900.0,
    "baseline_post_ready_runtime_sec": 550.0,
    "baseline_warm_runtime_sec": 400.0,
}
_RETRY_COST_SEC = 1300.0 + 750.0


class TestAColdAnchorIsNotAFinishedPrelude:
    """What happens to a session whose baseline could only keep its cold figure."""

    def test_a_dropped_hot_pass_does_not_finish_the_phase(self):
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=_RETRY_COST_SEC + 60.0,
        )

        assert phase_state.compute_next_phase(state) is None

        state.baseline_measure_round_dropped = False
        assert phase_state.compute_next_phase(state)[1] == "prelude_done"

    def test_a_session_that_cannot_afford_another_baseline_closes(self):
        """2050s buys a round and a variant to read against it; 1200s buys neither."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=1200.0,
        )

        out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))

        assert out is not None
        next_phase, reason, evidence = out
        assert (next_phase, reason) == ("CLOSE", "prelude_cold_anchor_low_budget")
        assert evidence["terminal"] is True
        assert evidence["baseline_anchor"] == "cold"
        assert evidence["retry_round_sec"] == pytest.approx(1300.0)
        assert is_valid_stop_reason(reason)

    def test_a_retry_already_in_flight_is_waited_for(self):
        """The marker stays up until a round lands, so the shortfall must not cancel the round itself."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=1200.0,
        )
        flags = phase_state.PhaseFlags(kernel_enabled=True)

        in_flight = phase_state.WorkInFlight(prelude=True)

        assert phase_state.compute_next_phase(state, flags=flags, in_flight=in_flight) is None

        out = phase_state.compute_next_phase(state, flags=flags)
        assert out is not None and out[1] == "prelude_cold_anchor_low_budget"

    def test_a_session_resumed_with_a_fresh_clock_measures_another_baseline(self):
        """The marker outlives the shortfall, so it must not decide on its own."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=_RETRY_COST_SEC + 60.0,
        )

        assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True)) is None

    def test_a_single_round_baseline_is_not_mistaken_for_a_dropped_one(self):
        """A cold figure by configuration is consistent with what follows it."""
        state = _prelude_state(
            baseline_tput=1074.7,
            baseline_runtime_sec=900.0,
            baseline_post_ready_runtime_sec=550.0,
            usable_sec=1200.0,
        )

        out = phase_state.compute_next_phase(state)
        assert out is not None and out[1] == "prelude_done"

    def test_a_session_with_no_clock_is_not_closed_for_a_budget_it_does_not_have(self):
        """An unbounded run cannot fail an affordability test, so it retries."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=None,
        )

        assert phase_state.compute_next_phase(state) is None


def test_prelude_baseline_failed_after_three_failures():
    state = SharedState(
        phase="PRELUDE", phase_budget_pct={}, phase_started_unix=0.0, max_minutes=0, baseline_failure_streak=2
    )
    assert phase_state.compute_next_phase(state) is None
    state.baseline_failure_streak = 3
    out = phase_state.compute_next_phase(state)
    assert out is not None and out[1] == "prelude_baseline_failed"


def test_optimize_phase_budget_exhaustion_advances():
    # Elapsed exceeds the phase budget.
    from hyperloom.orchestrator.phases.machine_state import normalize_budget_pct

    state = SharedState(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1.0,
        max_minutes=10,  # 600s total; 60% explore budget = 360s
        phase_budget_pct=normalize_budget_pct(None),
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[{"action": "explore"}],
        resumed_ts="",
        pending_escalate_hint="",
    )
    state._now_unix = lambda: 1_000_000.0
    out = phase_state.compute_next_phase(state)
    assert out is not None and out[1] == "optimize_phase_budget_exhausted"


def test_compute_next_phase_no_kernel_skips_kernel_phase():
    state = SharedState(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        stop_reason="",
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        # At least one specialist round this cycle, required for skip_to_kernel to fire at all (see
        # test_exit_normal_optimize_skip_to_kernel_*).
        specialist_rounds=[{"proposals_total": 1, "proposals_kept": 0}],
        optimization_stack=[{"action": "explore"}],
    )
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=False))
    assert out is not None
    next_phase, reason, evidence = out
    assert next_phase == "SWEEP"
    assert reason == "optimize_no_more_leverage"
    assert evidence["kernel_skipped"] is True


def test_skip_to_kernel_requires_a_tested_round():
    """A skip_to_kernel hint must not end EXPLORE with zero validated work."""
    state = SharedState(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1_000_000.0,
        max_minutes=0,
        phase_budget_pct={},
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        specialist_rounds=[],
        macro_cycle=0,
        optimization_stack=[{"action": "explore"}],
        resumed_ts="",
    )
    state._now_unix = lambda: 1_000_000.0
    out = phase_state.compute_next_phase(state)
    assert out is None


_ENTERED_UNIX = 1_000_000.0


def _framework_state_with_rounds(rounds: list[dict[str, Any]]) -> SharedState:
    """FRAMEWORK_AGENT entered at ``_ENTERED_UNIX`` in cycle 0, with a skip_to_kernel hint pending."""
    state = SharedState(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=_ENTERED_UNIX,
        max_minutes=0,
        phase_budget_pct={},
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        specialist_rounds=rounds,
        macro_cycle=0,
        optimization_stack=[{"action": "explore"}],
        resumed_ts="",
    )
    state._now_unix = lambda: _ENTERED_UNIX + 600.0
    return state


def _completed(offset_sec: float) -> str:
    return datetime.fromtimestamp(_ENTERED_UNIX + offset_sec, timezone.utc).isoformat()


@pytest.mark.parametrize("source_phase", ["FRAMEWORK_AGENT", "EXPLORE"])
def test_skip_to_kernel_fires_once_a_round_ran(source_phase):
    state = _framework_state_with_rounds(
        [{"proposals_total": 1, "source_phase": source_phase, "cycle": 0, "completed_at": _completed(60.0)}]
    )
    out = phase_state.compute_next_phase(state)
    assert out is not None
    _next, reason, evidence = out
    assert reason == "optimize_no_more_leverage"
    assert evidence.get("hint") == "skip_to_kernel"


@pytest.mark.parametrize(
    "round_row",
    [
        # PRELUDE's own scout, stamped with the phase it ran in.
        {"domain": "research_scout_specialist", "source_phase": "PRELUDE"},
        # A domain specialist delegated during PRELUDE: the router stamps the explore lane, not the live phase.
        {"domain": "serving_specialist", "source_phase": "EXPLORE"},
    ],
)
def test_skip_to_kernel_ignores_rounds_completed_before_phase_entry(round_row):
    """PRELUDE shares macro-cycle 0 with FRAMEWORK_AGENT, so only the entry time separates its rounds."""
    state = _framework_state_with_rounds([{**round_row, "cycle": 0, "completed_at": _completed(-60.0)}])
    assert phase_state.compute_next_phase(state) is None


def test_skip_to_kernel_ignores_a_round_with_no_completion_time():
    state = _framework_state_with_rounds([{"domain": "serving_specialist", "source_phase": "EXPLORE", "cycle": 0}])
    assert phase_state.compute_next_phase(state) is None


def test_compute_next_phase_terminal_overrides_phase():
    state = SharedState(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        stop_reason="target_reached",
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[],
    )
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True))
    assert out is not None
    assert out[0] == "CLOSE" and out[1] == "target_reached"
    assert out[2].get("terminal") is True


class TestAMetTargetDoesNotOutrankTheGuards:
    """The forward jump to SWEEP runs after the terminal checks, never before."""

    def _state(self, phase: str, **kw) -> SharedState:
        base = dict(
            phase=phase,
            phase_started_unix=0.0,
            max_minutes=0,
            phase_budget_pct={},
            stop_reason="",
            closing_phase=False,
            target_reached_at="2026-09-04T00:00:00+00:00",
            params_no_promote_streak=0,
            explore_search={},
            optimization_stack=[],
            resumed_ts="",
            pending_escalate_hint="",
            kernel_optimizer="geak",
            geak_result={},
            kernel_rewrite_controller_result={},
        )
        base.update(kw)
        return SharedState(**base)

    def test_a_coordinator_stop_reason_still_wins(self):
        out = phase_state.compute_next_phase(
            self._state(phase_state.PHASE_KERNEL_AGENT, stop_reason="baseline_failed"),
            flags=phase_state.PhaseFlags(kernel_enabled=True),
        )
        assert out is not None
        assert out[0] == phase_state.PHASE_CLOSE and out[1] == "baseline_failed"

    def test_the_closing_phase_still_wins(self):
        out = phase_state.compute_next_phase(
            self._state(phase_state.PHASE_KERNEL_AGENT, closing_phase=True),
            flags=phase_state.PhaseFlags(kernel_enabled=True),
        )
        assert out is not None
        assert out[0] == phase_state.PHASE_CLOSE and out[1] == "time_exhausted"

    def test_prelude_keeps_its_own_guards(self):
        """A baseline PRELUDE has not accepted is not one SWEEP can measure."""
        out = phase_state.compute_next_phase(
            self._state(phase_state.PHASE_PRELUDE), flags=phase_state.PhaseFlags(kernel_enabled=True)
        )
        assert out is None or out[0] != phase_state.PHASE_SWEEP

    def test_a_later_phase_does_jump(self):
        out = phase_state.compute_next_phase(
            self._state(phase_state.PHASE_KERNEL_AGENT), flags=phase_state.PhaseFlags(kernel_enabled=True)
        )
        assert out is not None
        assert out[0] == phase_state.PHASE_SWEEP and out[1] == "target_reached"
        assert out[2].get("terminal") is not True


def test_a_met_target_renames_a_budget_limited_sweep_exit():
    """``sweep_budget_exhausted`` is outside STOP_REASON_VOCAB, so CLOSE would recover it as ``time_exhausted`` -- a met target reported as a timeout."""
    now = 1_800_000_000.0
    started = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(hours=2)
    state = SharedState(
        phase=phase_state.PHASE_SWEEP,
        phase_started_unix=started.timestamp(),
        start_ts=started.isoformat(),
        max_minutes=60,
        phase_budget_pct={phase_state.PHASE_SWEEP: 0.5},
        stop_reason="",
        closing_phase=False,
        target_reached_at="2026-09-04T00:00:00+00:00",
        last_conc_sweep={},
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[],
        macro_cycle=0,
        saturated_directions={},
        cumulative_gain_validated=0.0,
        gain_at_cycle_start=0.0,
        no_gain_cycle_streak=0,
        resumed_ts="",
    )
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(kernel_enabled=True), now_unix=now)
    assert out is not None
    assert out[0] == phase_state.PHASE_CLOSE and out[1] == "target_reached"
    assert out[2].get("terminal") is True


def test_shared_state_phase_fields_default_to_empty():
    s = SharedState()
    assert s.phase == ""
    assert s.phase_history == []
    assert s.phase_started_unix == 0.0
    assert s.phase_budget_pct == {}


def test_record_phase_transition_writes_row_and_updates_phase():
    s = SharedState()
    row = phase_state.record_phase_transition(
        s,
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={"trigger": "fresh_session"},
        at=phase_state.PhaseClock(ts="2026-05-19T00:00:00+00:00", ts_unix=1747600000.0),
    )
    assert s.phase == "PRELUDE"
    assert s.phase_started_unix == 1747600000.0
    assert s.phase_history == [row]
    assert row["from_phase"] == "" and row["to_phase"] == "PRELUDE"
    # History is append-only.
    row2 = phase_state.record_phase_transition(
        s,
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="prelude_done",
        evidence={"baseline_tput": 100.0},
        at=phase_state.PhaseClock(ts="2026-05-19T00:01:00+00:00", ts_unix=1747600060.0),
    )
    assert s.phase == phase_state.PHASE_FRAMEWORK_AGENT
    assert len(s.phase_history) == 2
    assert s.phase_history[-1] == row2
    assert row2["from_phase"] == "PRELUDE"


def test_explore_elapsed_accumulates_completed_and_live_segments():
    s = SharedState()
    phase_state.record_phase_transition(
        s,
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="phase_entered",
        evidence={},
        at=phase_state.PhaseClock(ts="2026-05-19T00:00:00+00:00", ts_unix=100.0),
    )
    phase_state.record_phase_transition(
        s,
        to_phase="KERNEL_AGENT",
        reason="optimize_no_more_leverage",
        evidence={},
        at=phase_state.PhaseClock(ts="2026-05-19T00:02:00+00:00", ts_unix=220.0),
    )
    assert s.phase_elapsed_totals[phase_state.PHASE_FRAMEWORK_AGENT] == 120.0
    assert phase_state.phase_cumulative_seconds(s, phase=phase_state.PHASE_FRAMEWORK_AGENT, now_unix=300.0) == 120.0

    phase_state.record_phase_transition(
        s,
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="sweep_reloop",
        evidence={},
        at=phase_state.PhaseClock(ts="2026-05-19T00:03:00+00:00", ts_unix=280.0),
    )
    assert phase_state.phase_cumulative_seconds(s, phase=phase_state.PHASE_FRAMEWORK_AGENT, now_unix=310.0) == 150.0


def test_langfuse_status_includes_explore_runtime_and_kb_hit():
    s = SharedState()
    s.start_ts = "2026-05-19T00:00:00+00:00"
    s.phase = phase_state.PHASE_FRAMEWORK_AGENT
    s.phase_started_unix = 100.0
    s.phase_elapsed_totals = {phase_state.PHASE_FRAMEWORK_AGENT: 120.0}
    s.warm_start_context = {"status": "hit"}

    summary = s._langfuse_status_summary()

    assert summary["kb_hit"] == "hit"
    assert summary["explore_elapsed_s"] >= 120
    assert "explore_ratio" in summary
    assert "session_elapsed_s" in summary


def _make_role_registry():
    from hyperloom.orchestrator.roles.agent_role import default_role_registry

    return default_role_registry()


def test_policy_gate_phase_strict_allows_in_phase_action():
    state = SharedState()
    phase_state.record_phase_transition(
        state,
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={},
        at=phase_state.PhaseClock(ts="2026-05-19T00:00:00+00:00", ts_unix=1.0),
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
    )
    gate.validate_intent("orchestration", intent)  # no exception


@pytest.fixture
def coordinator_with_mocks(session_dir):
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
    }
    return Coordinator(session_dir, backends=backends)


def test_coordinator_init_writes_phase_prelude_for_fresh_session(coordinator_with_mocks):
    c = coordinator_with_mocks
    assert c.shared_state.phase == "PRELUDE"
    assert len(c.shared_state.phase_history) == 1
    row = c.shared_state.phase_history[0]
    assert row["to_phase"] == "PRELUDE"
    assert row["reason"] == "phase_entered"
    # A fresh session starts on the defaults rather than an empty split.
    assert c.shared_state.phase_budget_pct == dict(phase_state.DEFAULT_PHASE_BUDGET_PCT)


def test_a_session_recorded_at_an_unknown_phase_refuses_to_resume(coordinator_with_mocks):
    """A phase this build does not have was written by a build whose machine differed."""
    c = coordinator_with_mocks
    c.shared_state.phase = "EXPLORE"

    with pytest.raises(RuntimeError) as excinfo:
        c.phase_machine.ensure_phase_initialised(None)

    assert "EXPLORE" in str(excinfo.value)
    assert c.shared_state.phase == "EXPLORE"


def _fail_phase_init_save(monkeypatch, c, exc: BaseException) -> None:
    c.shared_state.phase = ""
    c.shared_state.phase_history = []

    def _save(session_dir=None) -> None:
        raise exc

    monkeypatch.setattr(c.shared_state, "save", _save)


def test_a_failed_save_after_phase_init_is_logged_and_the_phase_stands(coordinator_with_mocks, monkeypatch, caplog):
    c = coordinator_with_mocks
    err = OSError("disk full")
    _fail_phase_init_save(monkeypatch, c, err)

    with caplog.at_level("ERROR", logger="hyperloom.orchestrator.phases.machine"):
        c.phase_machine.ensure_phase_initialised(None)

    assert c.shared_state.phase == "PRELUDE"
    assert [r.exc_info[1] for r in caplog.records if r.exc_info] == [err]


def test_a_serialization_defect_on_the_phase_init_save_is_not_swallowed(coordinator_with_mocks, monkeypatch):
    c = coordinator_with_mocks
    _fail_phase_init_save(monkeypatch, c, TypeError("not JSON serializable"))

    with pytest.raises(TypeError, match="not JSON serializable"):
        c.phase_machine.ensure_phase_initialised(None)


_NO_KERNEL_CYCLE_SEC = 600 * 60.0
_NO_KERNEL_CLI_BUDGET_PCT = {phase_state.PHASE_FRAMEWORK_AGENT: 0.2}


@pytest.fixture
def no_kernel_coordinator(session_dir):
    """A real Coordinator on an unbounded ``--no-kernel`` session, so FRAMEWORK owns a flat share of the cycle."""
    from hyperloom.orchestrator.roles import MockBackend, MockCriticBackend, ScriptedPlan
    from hyperloom.orchestrator.loop.coordinator import Coordinator, CoordinatorSettings

    SharedState(
        kernel_enabled=False,
        max_minutes=0,
        cycle_minutes=_NO_KERNEL_CYCLE_SEC / 60.0,
        baseline_tput=100.0,
    ).save(session_dir)
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"}),
    )
    return Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="orch"), "critic": MockCriticBackend()},
        settings=CoordinatorSettings(phase_budget_pct=dict(_NO_KERNEL_CLI_BUDGET_PCT)),
    )


def _enter_framework_for(c, elapsed_sec: float) -> None:
    import time

    phase_state.record_phase_transition(
        c.shared_state,
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="prelude_done",
        evidence={},
        at=phase_state.PhaseClock(ts_unix=time.time() - elapsed_sec),
    )


async def _phase_block(c) -> str:
    prompt = (await c.conversation.compose_prompt("orchestration")).text
    return prompt.split("=== Phase ===", 1)[1]


@pytest.mark.asyncio
async def test_no_kernel_framework_share_is_the_same_for_machine_dispatch_and_prompt(no_kernel_coordinator):
    c = no_kernel_coordinator
    unredistributed = _NO_KERNEL_CLI_BUDGET_PCT[phase_state.PHASE_FRAMEWORK_AGENT]
    share = phase_state.redistribute_budget_pct(
        phase_state.normalize_budget_pct(_NO_KERNEL_CLI_BUDGET_PCT), kernel_enabled=False
    )[phase_state.PHASE_FRAMEWORK_AGENT]
    assert share > unredistributed
    assert c.shared_state.phase_budget_pct[phase_state.PHASE_FRAMEWORK_AGENT] == pytest.approx(share)
    try:
        # Past the unredistributed share's allotment, well inside the redistributed one.
        _enter_framework_for(c, _NO_KERNEL_CYCLE_SEC * (unredistributed + share) / 2)
        assert c.dispatcher.dispatch_paused_for_phase_budget() is False
        block = await _phase_block(c)
        assert f"pct={share:.2f}" in block
        assert "remaining_sec=0 " not in block
        await c.phase_machine.advance_phase_if_needed()
        assert c.shared_state.phase == phase_state.PHASE_FRAMEWORK_AGENT

        _enter_framework_for(c, _NO_KERNEL_CYCLE_SEC * share + 60.0)
        assert c.dispatcher.dispatch_paused_for_phase_budget() is True
        assert "remaining_sec=0" in await _phase_block(c)
        await c.phase_machine.advance_phase_if_needed()
        framework_exit = c.shared_state.phase_history[-1]
        assert framework_exit["from_phase"] == phase_state.PHASE_FRAMEWORK_AGENT
        assert framework_exit["reason"] == "optimize_phase_budget_exhausted"
        assert framework_exit["evidence"]["kernel_skipped"] is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_advances_to_the_optimize_phase_when_baseline_present(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        # Simulate baseline KEEP to trigger prelude_done.
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert c.shared_state.phase == phase_state.PHASE_FRAMEWORK_AGENT
        # 2 rows: PRELUDE entry + PRELUDE -> the optimisation phase.
        assert len(c.shared_state.phase_history) == 2
        last = c.shared_state.phase_history[-1]
        assert last["from_phase"] == "PRELUDE"
        assert last["to_phase"] == phase_state.PHASE_FRAMEWORK_AGENT
        assert last["reason"] == "prelude_done"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_phase_idempotent_within_same_tick(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert c.shared_state.phase == phase_state.PHASE_FRAMEWORK_AGENT
        first_history = list(c.shared_state.phase_history)
        # No state change → no new transition.
        await c.tick(1)
        assert c.shared_state.phase_history == first_history
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_queued_or_running_baseline_reads_as_prelude_work(coordinator_with_mocks):
    """The fact the terminal reads comes off the task store the transition drain cancels from."""
    from hyperloom.orchestrator.state.task_registry import TaskSpec

    c = coordinator_with_mocks
    try:
        assert await c.phase_machine._prelude_work_in_flight() is False

        task, _ = await c.tasks.create_or_return_existing(
            TaskSpec(kind="baseline", params={}, idempotency_key="prelude-baseline-retry")
        )
        assert await c.phase_machine._prelude_work_in_flight() is True
        await c.tasks.transition(task.task_id, "running")
        assert await c.phase_machine._prelude_work_in_flight() is True

        await c.tasks.transition(task.task_id, "succeeded", evidence={"result_keys": []})
        assert await c.phase_machine._prelude_work_in_flight() is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_first_failure_with_no_allowance_left_closes_the_session(coordinator_with_mocks):
    """One owner decides whether PRELUDE may still retry: admission refuses it, the terminal records that.

    The tick that books the first baseline failure finds the allowance already at zero, so there is no
    retry the Orchestration agent could get launched and nothing is lost by closing here. The enablement
    lane is off, so no other phase takes the failed baseline over and the budget line is the live one.
    """
    from hyperloom.orchestrator.actions.executors import baseline as baseline_executor

    c = coordinator_with_mocks
    try:
        c.shared_state.enablement_mode = "off"
        c.shared_state.max_minutes = 720
        c.shared_state.baseline_failure_streak = 1
        c.shared_state.session_budget_usable_sec = lambda: 14_718.5

        refused, shortfall = baseline_executor._a_retry_past_the_reserve_line(c.shared_state)
        assert refused is True, "a retry past the line must be refused before the machine reads the terminal"
        assert shortfall["affordable_sec"] == 0.0

        await c.phase_machine.advance_phase_if_needed()

        assert c.shared_state.phase == phase_state.PHASE_CLOSE
        row = c.shared_state.phase_history[-1]
        assert row["from_phase"] == "PRELUDE"
        assert row["reason"] == "prelude_budget_exhausted"
        assert row["evidence"]["baseline_failure_streak"] == 1
        assert row["evidence"]["prelude_affordable_sec"] == 0.0
        assert phase_state.is_valid_stop_reason(row["reason"])
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_work_no_phase_owns_is_not_prelude_work(coordinator_with_mocks):
    """Only PRELUDE's own actions hold the phase open; a queued kernel task must not."""
    from hyperloom.orchestrator.state.task_registry import TaskSpec

    c = coordinator_with_mocks
    try:
        await c.tasks.create_or_return_existing(
            TaskSpec(kind="kernel_agent", params={}, idempotency_key="not-a-prelude-action")
        )
        assert await c.phase_machine._prelude_work_in_flight() is False
    finally:
        await c.stop()


# ENABLEMENT phase predicate tests


def _enablement_state(phase, *, tput=0.0, streak=1, validation_pending=False, auto_roofline_pending_task_id=""):
    """A state the enablement entry and exit branches read."""
    return SharedState(
        auto_roofline_pending_task_id=auto_roofline_pending_task_id,
        phase=phase,
        stop_reason="",
        closing_phase=False,
        baseline_tput=tput,
        baseline_failure_streak=streak,
        enablement=EnablementRound(validation_pending=validation_pending),
        phase_budget_pct={},
        phase_started_unix=0.0,
        max_minutes=0,
        resumed_ts="",
        pending_escalate_hint="",
    )


def test_prelude_enters_enablement_on_a_baseline_failure_streak():
    """A failed baseline routes PRELUDE to ENABLEMENT when the lane is admitted."""
    state = _enablement_state("PRELUDE")
    phase, reason, _ = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True))
    assert phase == phase_state.PHASE_ENABLEMENT
    assert reason == "enablement_entered"


def test_prelude_skips_enablement_when_the_lane_is_not_admitted():
    """An unadmitted lane leaves PRELUDE waiting for a baseline rather than entering the phase."""
    state = _enablement_state("PRELUDE")
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=False)) is None


def test_enablement_exits_once_the_baseline_lands_and_work_drains():
    """All three conjuncts satisfied is the only way out through the normal exit."""
    state = _enablement_state("ENABLEMENT", tput=1000.0)
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True))
    assert out is not None
    phase, reason, evidence = out
    assert phase != phase_state.PHASE_ENABLEMENT
    assert reason == "enablement_done"
    assert phase_state.replay_next_phase(evidence["predicate_inputs"]) == out


def test_enablement_holds_while_work_is_in_flight():
    """A build outliving its round must not let a later phase reopen validation."""
    state = _enablement_state("ENABLEMENT", tput=1000.0)
    assert (
        phase_state.compute_next_phase(
            state,
            in_flight=phase_state.WorkInFlight(enablement=True),
            flags=phase_state.PhaseFlags(enablement_enabled=True),
        )
        is None
    )
    inputs = phase_state.workflow_predicate_inputs(
        state,
        in_flight=phase_state.WorkInFlight(enablement=True),
        flags=phase_state.PhaseFlags(enablement_enabled=True),
    )
    assert phase_state.replay_next_phase(inputs) is None


def test_enablement_waits_for_the_initial_analysis_like_prelude_does():
    """The first anchor of a session whose baseline failed is promoted here, and that enqueues the roofline."""
    state = _enablement_state("ENABLEMENT", tput=1000.0, auto_roofline_pending_task_id="rl-1")
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True)) is None
    state.auto_roofline_pending_task_id = ""
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True))
    assert out is not None and out[1] == "enablement_done"
    assert phase_state.replay_next_phase(out[2]["predicate_inputs"]) == out


def test_the_enablement_wait_stops_at_the_optimization_reserve():
    """Unbounded in an unbounded session; bounded by the reserve once the session has a budget."""
    state = _enablement_state("ENABLEMENT", tput=1000.0, auto_roofline_pending_task_id="rl-1")
    # 180 min session reserves 5400 s for optimization; 5000 s usable leaves nothing to hold with.
    state.max_minutes = 180
    state.session_budget_usable_sec = lambda: 5_000.0
    out = phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True))
    assert out is not None and out[1] == "enablement_done"
    assert phase_state.replay_next_phase(out[2]["predicate_inputs"]) == out
    state.session_budget_usable_sec = lambda: 6_000.0
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True)) is None


def test_enablement_holds_while_revalidation_is_pending():
    """An eval-origin KEEP owes a genuine baseline before the run counts as enabled."""
    state = _enablement_state("ENABLEMENT", tput=1000.0, validation_pending=True)
    assert phase_state.compute_next_phase(state, flags=phase_state.PhaseFlags(enablement_enabled=True)) is None


def test_replay_recomputes_from_primitive_baseline_facts():
    state = _prelude_state(baseline_tput=100.0, usable_sec=1000.0)
    out = phase_state.compute_next_phase(state)
    assert out is not None
    inputs = deepcopy(out[2]["predicate_inputs"])

    assert "matched" not in inputs["baseline"]
    inputs["baseline"]["tput"] = 0.0

    assert phase_state.replay_next_phase(inputs) is None
