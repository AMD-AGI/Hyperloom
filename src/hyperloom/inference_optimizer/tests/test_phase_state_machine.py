# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""phase state machine tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases import machine_state as phase_state
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import (
    CORE_STATE_FIELDS,
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def test_phase_names_are_monotonic():
    assert phase_state.PHASE_NAMES == (
        "PRELUDE",
        "FRAMEWORK_AGENT",
        "KERNEL_AGENT",
        "SWEEP",
        "CLOSE",
    )
    for i, name in enumerate(phase_state.PHASE_NAMES):
        assert phase_state.phase_index(name) == i
    assert phase_state.phase_index("unknown") == -1


def test_allowed_actions_disjoint_phases():
    # recover is in every phase; kernel_agent-owned actions only in KERNEL (Inv-2.1).
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        assert "recover" in allowed
    assert "baseline" in phase_state.PHASE_ALLOWED_ACTIONS["PRELUDE"]
    assert "baseline" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    assert "kernel_opt" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL_AGENT"]
    assert "kernel_opt" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    assert "gemm_tuning" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL_AGENT"]
    assert "gemm_tuning" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    assert "conc_sweep" in phase_state.PHASE_ALLOWED_ACTIONS["SWEEP"]
    assert "conc_sweep" not in phase_state.PHASE_ALLOWED_ACTIONS["FRAMEWORK_AGENT"]
    assert "report" in phase_state.PHASE_ALLOWED_ACTIONS["CLOSE"]


def test_is_action_allowed_in_phase_handles_unknowns():
    assert phase_state.is_action_allowed_in_phase("baseline", "PRELUDE")
    assert not phase_state.is_action_allowed_in_phase("baseline", "FRAMEWORK_AGENT")
    # Unknown phase → deny by default.
    assert not phase_state.is_action_allowed_in_phase("baseline", "UNKNOWN")
    assert not phase_state.is_action_allowed_in_phase("baseline", "")
    # Empty action name → deny.
    assert not phase_state.is_action_allowed_in_phase("", "PRELUDE")






def test_every_reason_an_exit_rule_can_return_is_in_the_vocabulary():
    """The closed vocabulary must actually close over what the rules emit.

    Restating the frozenset here would only assert that a copy matches its
    original. What matters is that no exit rule can hand PolicyGate a reason it
    will then reject, stranding the transition.
    """
    import itertools

    rules = (
        phase_state.exit_normal_prelude,
        phase_state.exit_normal_optimize,
        phase_state.exit_normal_kernel,
        phase_state.exit_normal_sweep,
    )
    # A spread of states wide enough to reach each rule's branches.
    states = [
        SharedState(),
        SharedState(baseline_tput=1234.5),
        SharedState(baseline_tput=1234.5, framework_agent_phase_done=True),
        SharedState(baseline_tput=1234.5, phase_budget_pct={p: 0.01 for p in phase_state.PHASE_NAMES}),
    ]
    seen = set()
    for rule, state in itertools.product(rules, states):
        try:
            out = rule(state)
        except TypeError:
            continue  # rule needs kwargs this sweep does not supply
        if out is None:
            continue
        reason = out[0]
        seen.add(reason)
        assert phase_state.is_valid_phase_exit_reason(reason), reason
    assert seen, "no exit rule fired; this guard would pass vacuously"


def test_phase_exit_reason_vocabulary_is_closed():
    assert not phase_state.is_valid_phase_exit_reason("totally_invented")
    assert not phase_state.is_valid_phase_exit_reason("")
    # Stripped before comparison, so a stray newline in a history row still matches.
    assert phase_state.is_valid_phase_exit_reason("  prelude_done \n")


def test_stop_reason_vocab_includes_v06_and_v08():
    for reason in (
        "target_reached",
        "time_exhausted",
        "max_ticks",
        "policy_loop",
        "baseline_failed",
        "emergency",
        "coordinator_exception",
        "crash_threshold_exceeded",
        "user_stop_requested",
        "recipe_kb_drain_failed",
        "plateau_explore",
        "sweep_failed",
        "baseline_arg_error",
    ):
        assert phase_state.is_valid_stop_reason(reason), reason
    assert not phase_state.is_valid_stop_reason("totally_invented")


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


def test_exit_normal_prelude_triggers_on_baseline_tput():
    state = SimpleNamespace(baseline_tput=0.0)
    assert phase_state.exit_normal_prelude(state) is None
    state.baseline_tput = 1234.5
    out = phase_state.exit_normal_prelude(state)
    assert out is not None
    reason, evidence = out
    assert reason == "prelude_done"
    assert evidence["baseline_tput"] == 1234.5


def test_exit_normal_prelude_blocked_while_warm_replay_in_flight():
    """PRELUDE must not advance to FRAMEWORK until warm-replay settles."""
    state = SimpleNamespace(
        baseline_tput=1234.5,
        warm_replay_outcome={"status": "in_flight", "replay_task_id": "abc"},
    )
    assert phase_state.exit_normal_prelude(state) is None
    state.warm_replay_outcome = {"status": "failed"}
    out = phase_state.exit_normal_prelude(state)
    assert out is not None and out[0] == "prelude_done"


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
    baseline_double_run: bool = False,
) -> SimpleNamespace:
    """A PRELUDE-phase state with an explicit clock, as the budget policy reads it."""
    return SimpleNamespace(
        phase="PRELUDE",
        max_minutes=max_minutes,
        phase_elapsed_totals={"PRELUDE": spent_sec},
        phase_started_unix=0.0,
        baseline_tput=baseline_tput,
        baseline_runtime_sec=baseline_runtime_sec,
        baseline_post_ready_runtime_sec=baseline_post_ready_runtime_sec,
        baseline_warm_runtime_sec=baseline_warm_runtime_sec,
        baseline_measure_round_dropped=baseline_measure_round_dropped,
        baseline_double_run=baseline_double_run,
        session_budget_usable_sec=lambda: usable_sec,
    )


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
    """Banked phase spend and the session clock answer to different origins.

    A resume that reanchors the budget restarts the session clock while the
    phase ledger keeps every second the earlier leg banked. A bound read off
    the ledger therefore declared preparation overspent on a session that had
    its whole budget ahead of it, and the measured half of the baseline was
    refused on every resumed run. Only the clock decides.
    """
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
    out = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    next_phase, reason, evidence = out
    assert (next_phase, reason) == ("CLOSE", "time_exhausted_during_prelude")
    assert evidence["terminal"] is True
    assert phase_state.is_valid_stop_reason(reason)


def test_a_landed_baseline_outranks_the_exhausted_clock():
    """With a baseline in hand the run has something to optimize; the later phases judge for themselves."""
    state = _prelude_state(spent_sec=10_800.0, usable_sec=0.0, baseline_tput=1074.7)
    out = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    assert out[1] == "prelude_done"


def test_prelude_exit_states_whether_one_optimization_round_still_fits():
    """The plain statement neither field session ever got: preparation spent the run."""
    state = _prelude_state(baseline_tput=1074.7, baseline_runtime_sec=2705.7, usable_sec=2796.0)
    out = phase_state.exit_normal_prelude(state)
    assert out is not None
    evidence = out[1]
    assert evidence["fits_one_optimization_round"] is True
    assert evidence["affordable_rounds"] == pytest.approx(1.03, abs=0.01)

    state.session_budget_usable_sec = lambda: 1200.0
    evidence = phase_state.exit_normal_prelude(state)[1]
    assert evidence["fits_one_optimization_round"] is False


# The workload the cold-anchor cases below are priced against: a 900s cold round
# whose last 550s was the benchmark, so the boot took 350s, and a 400s hot pass.
# One further measured variant therefore costs 750s -- its own boot and a
# benchmark on a populated JIT cache -- while a double-run round costs the whole
# measured cold pass plus a second benchmark, 1300s. Together they are what a
# session must afford before measuring another baseline is worth doing.
_COLD_ANCHOR_WORKLOAD = {
    "baseline_tput": 1074.7,
    "baseline_runtime_sec": 900.0,
    "baseline_post_ready_runtime_sec": 550.0,
    "baseline_warm_runtime_sec": 400.0,
    "baseline_double_run": True,
}
_RETRY_COST_SEC = 1300.0 + 750.0


class TestAColdAnchorIsNotAFinishedPrelude:
    """What happens to a session whose baseline could only keep its cold figure.

    The figure exists, so every rule that asks only whether a baseline landed
    reads preparation as done. It is not: the number carries the boot, the first
    request's compile and the graph capture, so every variant measured against it
    reads as an improvement over a baseline that was never the baseline.

    Two outcomes are correct and the budget picks between them -- measure another
    baseline, or stop and say why -- and neither is "optimize against it".
    """

    def test_a_dropped_hot_pass_does_not_finish_the_phase(self):
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=_RETRY_COST_SEC + 60.0,
        )

        assert phase_state.exit_normal_prelude(state) is None

        state.baseline_measure_round_dropped = False
        assert phase_state.exit_normal_prelude(state)[0] == "prelude_done"

    def test_a_session_that_cannot_afford_another_baseline_closes(self):
        """2050s buys a round and a variant to read against it; 1200s buys neither."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=1200.0,
        )

        out = phase_state.compute_next_phase(state, kernel_enabled=True)

        assert out is not None
        next_phase, reason, evidence = out
        assert (next_phase, reason) == ("CLOSE", "prelude_cold_anchor_low_budget")
        assert evidence["terminal"] is True
        assert evidence["baseline_anchor"] == "cold"
        assert evidence["retry_round_sec"] == pytest.approx(1300.0)
        assert phase_state.is_valid_stop_reason(reason)
        assert phase_state.is_valid_phase_exit_reason(reason)

    def test_a_session_resumed_with_a_fresh_clock_measures_another_baseline(self):
        """The marker outlives the shortfall, so it must not decide on its own.

        A resume reanchors the session clock while the marker from the earlier leg
        persists. Closing on the marker would end every resumed run before it
        began, and no later baseline could clear the marker because none would
        run. So the phase stays open with nothing to advance it but a new
        baseline.
        """
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=_RETRY_COST_SEC + 60.0,
        )

        assert phase_state.exit_cold_anchor_prelude(state) is None
        assert phase_state.compute_next_phase(state, kernel_enabled=True) is None

    def test_a_single_round_baseline_is_not_mistaken_for_a_dropped_one(self):
        """A cold figure by configuration is consistent with what follows it.

        A session that never asked for a hot pass measures everything the same
        way, so its comparisons hold. Only a pass that was *dropped* leaves a
        denominator out of step with the numerators.
        """
        state = _prelude_state(
            baseline_tput=1074.7,
            baseline_runtime_sec=900.0,
            baseline_post_ready_runtime_sec=550.0,
            usable_sec=1200.0,
        )

        assert phase_state.exit_cold_anchor_prelude(state) is None
        assert phase_state.exit_normal_prelude(state)[0] == "prelude_done"

    def test_a_session_with_no_clock_is_not_closed_for_a_budget_it_does_not_have(self):
        """An unbounded run cannot fail an affordability test, so it retries."""
        state = _prelude_state(
            **_COLD_ANCHOR_WORKLOAD,
            baseline_measure_round_dropped=True,
            usable_sec=None,
        )

        assert phase_state.exit_cold_anchor_prelude(state) is None


def test_exit_terminal_prelude_after_three_baseline_failures():
    state = SimpleNamespace(baseline_failure_streak=2)
    assert phase_state.exit_terminal_prelude(state) is None
    state.baseline_failure_streak = 3
    out = phase_state.exit_terminal_prelude(state)
    assert out is not None and out[0] == "prelude_baseline_failed"


def test_exit_normal_optimize_uses_budget_exhaustion():
    # Elapsed exceeds the phase budget.
    state = SimpleNamespace(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1.0,
        max_minutes=10,  # 600s total; 60% explore budget = 360s
        phase_budget_pct={},
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[{"action": "explore"}],
        _now_unix=lambda: 1_000_000.0,
    )
    out = phase_state.exit_normal_optimize(state)
    assert out is not None and out[0] == "optimize_phase_budget_exhausted"


def test_compute_next_phase_no_kernel_skips_kernel_phase():
    state = SimpleNamespace(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        stop_reason="",
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        # At least one specialist round this cycle, required for skip_to_kernel
        # to fire at all (see test_exit_normal_optimize_skip_to_kernel_*).
        specialist_rounds=[{"proposals_total": 1, "proposals_kept": 0}],
        optimization_stack=[{"action": "explore"}],
    )
    out = phase_state.compute_next_phase(state, kernel_enabled=False)
    assert out is not None
    next_phase, reason, evidence = out
    assert next_phase == "SWEEP"
    assert reason == "no_kernel_skipped"
    assert evidence.get("passed_through_reason") == "optimize_no_more_leverage"


def test_exit_normal_optimize_skip_to_kernel_requires_a_tested_round():
    """A skip_to_kernel hint must not end EXPLORE with zero validated work.

    Reproduces the cumulative_gain_validated=0.00% session: the hint arrived
    before EXPLORE ever dispatched a specialist round this cycle, and must not
    be honored until one actually has.
    """
    state = SimpleNamespace(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1_000_000.0,
        max_minutes=0,
        phase_budget_pct={},
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        specialist_rounds=[],
        macro_cycle=0,
        optimization_stack=[{"action": "explore"}],
        _now_unix=lambda: 1_000_000.0,
    )
    out = phase_state.exit_normal_optimize(state)
    assert out is None


def test_exit_normal_optimize_skip_to_kernel_fires_once_a_round_ran():
    state = SimpleNamespace(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=1_000_000.0,
        max_minutes=0,
        phase_budget_pct={},
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        specialist_rounds=[{"proposals_total": 1, "proposals_kept": 0}],
        macro_cycle=0,
        optimization_stack=[{"action": "explore"}],
        _now_unix=lambda: 1_000_000.0,
    )
    out = phase_state.exit_normal_optimize(state)
    assert out is not None
    reason, evidence = out
    assert reason == "optimize_no_more_leverage"
    assert evidence.get("hint") == "skip_to_kernel"


def test_compute_next_phase_terminal_overrides_phase():
    state = SimpleNamespace(
        phase=phase_state.PHASE_FRAMEWORK_AGENT,
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        stop_reason="target_reached",
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[],
    )
    out = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert out is not None
    assert out[0] == "CLOSE" and out[1] == "target_reached"
    assert out[2].get("terminal") is True


def test_shared_state_phase_fields_default_to_empty():
    s = SharedState()
    assert s.phase == ""
    assert s.phase_history == []
    assert s.phase_started_ts == ""
    assert s.phase_started_unix == 0.0
    assert s.phase_budget_pct == {}


def test_record_phase_transition_writes_row_and_updates_phase():
    s = SharedState()
    row = s.record_phase_transition(
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={"trigger": "fresh_session"},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1747600000.0,
    )
    assert s.phase == "PRELUDE"
    assert s.phase_started_ts == "2026-05-19T00:00:00+00:00"
    assert s.phase_started_unix == 1747600000.0
    assert s.phase_history == [row]
    assert row["from_phase"] == "" and row["to_phase"] == "PRELUDE"
    # History is append-only.
    row2 = s.record_phase_transition(
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="prelude_done",
        evidence={"baseline_tput": 100.0},
        ts="2026-05-19T00:01:00+00:00",
        ts_unix=1747600060.0,
    )
    assert s.phase == phase_state.PHASE_FRAMEWORK_AGENT
    assert len(s.phase_history) == 2
    assert s.phase_history[-1] == row2
    assert row2["from_phase"] == "PRELUDE"


def test_explore_elapsed_accumulates_completed_and_live_segments():
    s = SharedState()
    s.record_phase_transition(
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=100.0,
    )
    s.record_phase_transition(
        to_phase="KERNEL_AGENT",
        reason="optimize_no_more_leverage",
        evidence={},
        ts="2026-05-19T00:02:00+00:00",
        ts_unix=220.0,
    )
    assert s.explore_elapsed_accum_s == 120.0
    assert phase_state.explore_elapsed_seconds(s, now_unix=300.0) == 120.0

    s.record_phase_transition(
        to_phase=phase_state.PHASE_FRAMEWORK_AGENT,
        reason="sweep_reloop",
        evidence={},
        ts="2026-05-19T00:03:00+00:00",
        ts_unix=280.0,
    )
    assert phase_state.explore_elapsed_seconds(s, now_unix=310.0) == 150.0


def test_langfuse_status_includes_explore_runtime_and_kb_hit():
    s = SharedState()
    s.start_ts = "2026-05-19T00:00:00+00:00"
    s.phase = phase_state.PHASE_FRAMEWORK_AGENT
    s.phase_started_unix = 100.0
    s.explore_elapsed_accum_s = 120.0
    s.warm_start_context = {"status": "hit"}

    summary = s._langfuse_status_summary()

    assert summary["kb_hit"] == "hit"
    assert summary["explore_elapsed_s"] >= 120
    assert "explore_ratio" in summary
    assert "session_elapsed_s" in summary


def test_legacy_resume_keeps_explore_runtime_unknown():
    raw = SharedState().to_dict()
    raw.pop("explore_elapsed_accum_s")
    raw.update(
        {
            "start_ts": "2026-05-19T00:00:00+00:00",
            "phase": "EXPLORE",
            "phase_started_unix": 100.0,
        }
    )

    s = SharedState.from_dict(raw)
    assert s.explore_elapsed_accum_s is None
    assert phase_state.explore_elapsed_seconds(s, now_unix=220.0) is None

    summary = s._langfuse_status_summary()
    assert "session_elapsed_s" in summary
    assert "explore_elapsed_s" not in summary
    assert "explore_ratio" not in summary

    s.record_phase_transition(
        to_phase="KERNEL_AGENT",
        reason="optimize_no_more_leverage",
        evidence={},
        ts="2026-05-19T00:02:00+00:00",
        ts_unix=220.0,
    )
    assert s.explore_elapsed_accum_s is None


def test_core_state_fields_includes_phase_fields():
    for f in (
        "phase",
        "phase_started_ts",
        "phase_started_unix",
        "phase_history",
        "phase_budget_pct",
        "explore_elapsed_accum_s",
    ):
        assert f in CORE_STATE_FIELDS, f


def _make_role_registry():
    from hyperloom.orchestrator.roles.agent_role import default_role_registry

    return default_role_registry()






def test_policy_gate_phase_strict_allows_in_phase_action():
    state = SharedState()
    state.record_phase_transition(
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
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










def test_every_kernel_request_kind_is_gated_or_explicitly_exempt():
    """A new handler kind cannot arrive silently ungated."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_OWNED_KERNEL_REQUEST_KINDS,
        REQUEST_KIND_TO_OWNED_ACTION,
    )
    from hyperloom.orchestrator.kernel.request_handlers import KERNEL_REQUEST_HANDLERS

    # trace_analyze has no owning action and no phase membership, so there is
    # nothing to gate it against; it is refreshed on demand from any phase.
    exempt = {"trace_analyze"}
    ungated = (
        set(KERNEL_REQUEST_HANDLERS)
        - set(REQUEST_KIND_TO_OWNED_ACTION)
        - COORDINATOR_OWNED_KERNEL_REQUEST_KINDS
        - exempt
    )
    assert ungated == set(), f"request kinds reach no phase gate: {sorted(ungated)}"






@pytest.fixture
def coordinator_with_mocks(session_dir):
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
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
        "robustness": MockRobustnessBackend(),
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
    """A phase this build does not have was written by a build whose machine differed.

    Treating it as a fresh start re-runs PRELUDE on top of a session that
    already has a baseline, a KEPT stack and hours of measurement, and the
    numbers of the two builds end up in one report.
    """
    c = coordinator_with_mocks
    c.shared_state.phase = "EXPLORE"

    with pytest.raises(RuntimeError) as excinfo:
        c._ensure_phase_initialised()

    assert "EXPLORE" in str(excinfo.value)
    assert c.shared_state.phase == "EXPLORE"


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
        c.shared_state.framework_agent_phase_enabled = False
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        first_history = list(c.shared_state.phase_history)
        # No state change → no new transition.
        await c.tick(1)
        assert c.shared_state.phase_history == first_history
    finally:
        await c.stop()


def test_collect_phase_segments_groups_actions_by_window():
    from hyperloom.inference_optimizer.breakdown.collectors import collect_phase_segments

    state = {
        "phase_history": [
            {
                "from_phase": "",
                "to_phase": "PRELUDE",
                "reason": "phase_entered",
                "evidence": {"trigger": "fresh_session"},
                "ts": "2026-05-19T00:00:00+00:00",
                "ts_unix": 1.0,
            },
            {
                "from_phase": "PRELUDE",
                "to_phase": "EXPLORE",
                "reason": "prelude_done",
                "evidence": {"baseline_tput": 100.0},
                "ts": "2026-05-19T00:05:00+00:00",
                "ts_unix": 301.0,
            },
        ],
    }
    timeline = [
        {"ts": "2026-05-19T00:01:00+00:00", "action": "baseline"},
        {"ts": "2026-05-19T00:10:00+00:00", "action": "params"},
    ]
    segments = collect_phase_segments(state, timeline, warnings=[])
    assert len(segments) == 2
    prelude, explore = segments
    assert prelude["phase"] == "PRELUDE"
    assert prelude["exit_reason"] == "prelude_done"
    assert prelude["elapsed_seconds"] == 300.0
    assert len(prelude["actions"]) == 1
    assert prelude["actions"][0]["action"] == "baseline"
    assert explore["phase"] == "EXPLORE"
    assert explore["exit_reason"] == ""  # currently active segment
    assert explore["elapsed_seconds"] is None  # no exit_unix yet
    assert explore["actions"][0]["action"] == "params"


def test_collect_phase_segments_empty_when_history_missing():
    from hyperloom.inference_optimizer.breakdown.collectors import collect_phase_segments

    assert collect_phase_segments({}, [], warnings=[]) == []
    assert collect_phase_segments({"phase_history": []}, [], warnings=[]) == []
