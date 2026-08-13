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
        "EXPLORE",
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
    assert "baseline" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "kernel_opt" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL_AGENT"]
    assert "kernel_opt" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "gemm_tuning" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL_AGENT"]
    assert "gemm_tuning" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "sweep" in phase_state.PHASE_ALLOWED_ACTIONS["SWEEP"]
    assert "sweep" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "report" in phase_state.PHASE_ALLOWED_ACTIONS["CLOSE"]


def test_is_action_allowed_in_phase_handles_unknowns():
    assert phase_state.is_action_allowed_in_phase("baseline", "PRELUDE")
    assert not phase_state.is_action_allowed_in_phase("baseline", "EXPLORE")
    # Unknown phase → deny by default.
    assert not phase_state.is_action_allowed_in_phase("baseline", "UNKNOWN")
    assert not phase_state.is_action_allowed_in_phase("baseline", "")
    # Empty action name → deny.
    assert not phase_state.is_action_allowed_in_phase("", "PRELUDE")


def test_llm_proposable_set_drops_coordinator_internal_actions():
    from hyperloom.inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_INTERNAL_ACTIONS,
        ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
    )

    # Proposable = allowlist minus Coordinator-internal and robustness-delegate-only actions.
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        proposable = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS[phase]
        assert proposable == (allowed - COORDINATOR_INTERNAL_ACTIONS - ROBUSTNESS_DELEGATE_ONLY_ACTIONS)
        assert proposable.isdisjoint(COORDINATOR_INTERNAL_ACTIONS)
        # recover stays phase-allowed but is never LLM-proposable.
        assert "recover" in allowed
        assert "recover" not in proposable
    # The advertised analysis / framework names are never proposable.
    explore = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["EXPLORE"]
    assert "roofline" not in explore
    assert "profile" not in explore
    assert "explore" in explore and "specialist" in explore
    framework = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["FRAMEWORK_AGENT"]
    assert "framework" not in framework
    assert "integrate_patch" in framework
    # Investigation reaches the two mid-chain phases but not the wind-down ones.
    assert "specialist" in framework
    assert "specialist" in phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["KERNEL_AGENT"]
    assert "specialist" not in phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["SWEEP"]
    assert "specialist" not in phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["CLOSE"]


def test_is_action_llm_proposable_in_phase_handles_unknowns():
    assert phase_state.is_action_llm_proposable_in_phase("baseline", "PRELUDE")
    assert phase_state.is_action_llm_proposable_in_phase("explore", "EXPLORE")
    # roofline lives in the allowlist but is never LLM-proposable.
    assert phase_state.is_action_allowed_in_phase("roofline", "EXPLORE")
    assert not phase_state.is_action_llm_proposable_in_phase("roofline", "EXPLORE")
    assert not phase_state.is_action_llm_proposable_in_phase("framework_agent", "FRAMEWORK_AGENT")
    # Unknown phase / empty action → deny by default.
    assert not phase_state.is_action_llm_proposable_in_phase("baseline", "UNKNOWN")
    assert not phase_state.is_action_llm_proposable_in_phase("", "PRELUDE")
    # llm_proposable_actions_for is sorted and excludes internal names.
    explore = phase_state.llm_proposable_actions_for("EXPLORE")
    assert explore == tuple(sorted(explore))
    assert "roofline" not in explore and "profile" not in explore


def test_phase_exit_reasons_includes_required_vocab():
    for reason in (
        "prelude_done",
        "plateau_explore",
        "plateau_kernel",
        "explore_phase_budget_exhausted",
        "kernel_phase_budget_exhausted",
        "explore_budget_cap",
        "kernel_budget_cap",
        "sweep_budget_cap",
        "sweep_done",
        "conc_sweep_done",
        "conc_sweep_failed",
        "sweep_budget_exhausted",
        "kernel_phase_aborted_no_trace",
        "explore_force_exit_low_budget",
        "explore_no_more_leverage",
        "kernel_no_more_leverage",
        "framework_agent_phase_done",
        "framework_agent_plateau",
        "framework_agent_force_exit_low_budget",
        "cycle_reloop",
        "global_converged",
        "robustness_escalated",
        "target_reached",
        "time_exhausted",
        "time_exhausted_during_prelude",
        "user_stop_requested",
        "recipe_kb_t0_failed",
        "recipe_kb_drain_failed",
        "recipe_kb_commit_failed",
        "prelude_baseline_failed",
        "prelude_policy_loop",
        "policy_loop",
        "crash_threshold_exceeded",
        "baseline_failed",
        "emergency",
        "max_ticks",
        "signal",
        "no_kernel_skipped",
        "phase_entered",
    ):
        assert phase_state.is_valid_phase_exit_reason(reason), reason


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
        "conc_sweep_failed",
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
    out = phase_state.normalize_budget_pct({"EXPLORE": 0.5, "BOGUS": 0.9})
    assert out["EXPLORE"] == 0.5
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
) -> SimpleNamespace:
    """A PRELUDE-phase state with an explicit clock, as the budget policy reads it."""
    return SimpleNamespace(
        phase="PRELUDE",
        max_minutes=max_minutes,
        phase_elapsed_totals={"PRELUDE": spent_sec},
        phase_started_unix=0.0,
        baseline_tput=baseline_tput,
        baseline_runtime_sec=baseline_runtime_sec,
        session_budget_usable_sec=lambda: usable_sec,
    )


def test_prelude_can_afford_an_arm_the_budget_still_covers():
    """The normal case must be untouched: a cheap arm early in a session runs."""
    state = _prelude_state(spent_sec=600.0, usable_sec=10_000.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=300.0)
    assert affordable is True
    assert evidence["prelude_spent_sec"] == 600.0


def test_prelude_refuses_an_arm_past_its_own_ceiling():
    """The Qwen3.5-397B shape: 51 minutes of baseline, then a roofline that costs another 45+."""
    state = _prelude_state(spent_sec=3090.0, usable_sec=7700.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=2706.0)
    assert affordable is False
    assert evidence["bound"] == "prelude_ceiling"
    # 40% of 180 minutes is 4320s; 3090 spent leaves 1230s, well under the arm.
    assert evidence["affordable_sec"] == pytest.approx(1230.0)


def test_prelude_refuses_an_arm_that_would_eat_the_optimization_reserve():
    """Under the phase ceiling but over the session reserve: the tighter bound wins."""
    state = _prelude_state(spent_sec=60.0, usable_sec=6000.0)
    affordable, evidence = phase_state.prelude_can_afford(state, expected_cost_sec=3000.0)
    assert affordable is False
    assert evidence["bound"] == "optimization_reserve"


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


def test_exit_terminal_prelude_after_three_baseline_failures():
    state = SimpleNamespace(baseline_failure_streak=2)
    assert phase_state.exit_terminal_prelude(state) is None
    state.baseline_failure_streak = 3
    out = phase_state.exit_terminal_prelude(state)
    assert out is not None and out[0] == "prelude_baseline_failed"


def test_exit_normal_explore_uses_budget_exhaustion():
    # Elapsed exceeds budget; IR-6 force-exit disabled to isolate the budget path.
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=1.0,
        max_minutes=10,  # 600s total; 60% explore budget = 360s
        phase_budget_pct={},
        params_no_promote_streak=0,
        explore_search={},
        optimization_stack=[{"action": "explore"}],
        _now_unix=lambda: 1_000_000.0,
    )
    out = phase_state.exit_normal_explore(
        state,
        force_exit_hours_remaining=0.0,
        force_exit_budget_pct=0.0,
    )
    assert out is not None and out[0] == "explore_phase_budget_exhausted"


def test_compute_next_phase_no_kernel_skips_kernel_phase():
    state = SimpleNamespace(
        phase="EXPLORE",
        phase_started_unix=0.0,
        max_minutes=0,
        phase_budget_pct={},
        stop_reason="",
        pending_escalate_hint="skip_to_kernel",
        explore_search={},
        optimization_stack=[{"action": "explore"}],
    )
    out = phase_state.compute_next_phase(state, kernel_enabled=False)
    assert out is not None
    next_phase, reason, evidence = out
    assert next_phase == "SWEEP"
    assert reason == "no_kernel_skipped"
    assert evidence.get("passed_through_reason") == "plateau_explore"


def test_compute_next_phase_terminal_overrides_phase():
    state = SimpleNamespace(
        phase="EXPLORE",
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
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={"baseline_tput": 100.0},
        ts="2026-05-19T00:01:00+00:00",
        ts_unix=1747600060.0,
    )
    assert s.phase == "EXPLORE"
    assert len(s.phase_history) == 2
    assert s.phase_history[-1] == row2
    assert row2["from_phase"] == "PRELUDE"


def test_explore_elapsed_accumulates_completed_and_live_segments():
    s = SharedState()
    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=100.0,
    )
    s.record_phase_transition(
        to_phase="KERNEL_AGENT",
        reason="explore_done",
        evidence={},
        ts="2026-05-19T00:02:00+00:00",
        ts_unix=220.0,
    )
    assert s.explore_elapsed_accum_s == 120.0
    assert phase_state.explore_elapsed_seconds(s, now_unix=300.0) == 120.0

    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="sweep_reloop",
        evidence={},
        ts="2026-05-19T00:03:00+00:00",
        ts_unix=280.0,
    )
    assert phase_state.explore_elapsed_seconds(s, now_unix=310.0) == 150.0


def test_langfuse_status_includes_explore_runtime_and_kb_hit():
    s = SharedState()
    s.start_ts = "2026-05-19T00:00:00+00:00"
    s.phase = "EXPLORE"
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
        reason="explore_done",
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


def test_policy_gate_phase_strict_denies_kernel_in_prelude():
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
        strict_phase=True,
    )
    # ``explore`` is an EXPLORE-phase action; proposing it in PRELUDE is
    # phase-incompatible.
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "explore", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"
    assert "PRELUDE" in (excinfo.value.hint or "")


def test_policy_gate_phase_warn_mode_does_not_raise():
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
        strict_phase=False,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "explore", "predicted_gain_pct": 1.0},
    )
    # warn-mode just bumps the audit counter, no raise.
    gate.validate_intent("orchestration", intent)
    assert state.policy_denial_streak.get("explore:phase_incompatible", 0) >= 1


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
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
    )
    gate.validate_intent("orchestration", intent)  # no exception


def test_policy_gate_phase_strict_blocks_explore_action_in_prelude():
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
        strict_phase=True,
    )
    # ``sweep`` is proposable only in SWEEP, so proposing it in PRELUDE
    # lands on R1 phase_incompatible.
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "sweep", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_denies_kernel_request_in_explore():
    """A kernel_agent-owned REQUEST in EXPLORE is denied by R1."""
    state = SharedState()
    state.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "kernel_opt",
            "params": {},
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_gates_apply_patch_alias_like_integrate():
    # apply_patch is a REQUEST-kind alias of integrate and must be phase-gated
    # identically. In EXPLORE, a kernel-owned integrate REQUEST is denied by
    # R1; the alias must be denied with the same rule.
    state = SharedState()
    state.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )

    def _rule_for(kind):
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel_agent", "kind": kind, "params": {}},
        )
        with pytest.raises(PolicyDenied) as excinfo:
            gate.validate_intent("orchestration", intent)
        return excinfo.value.rule

    assert _rule_for("integrate") == "phase_incompatible"
    assert _rule_for("apply_patch") == "phase_incompatible"


def test_policy_gate_does_not_widen_explore_for_kernel_request():
    """EXPLORE no longer widens to kernel_agent-owned kinds."""
    state = SharedState()
    state.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "kernel_opt",
            "params": {},
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_does_not_widen_kernel_for_explore_propose():
    """KERNEL does not accept explore proposals."""
    state = SharedState()
    state.record_phase_transition(
        to_phase="KERNEL_AGENT",
        reason="plateau_explore",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "explore", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_sweep_rejects_explore_lever():
    """SWEEP rejects the explore lever under R1."""
    state = SharedState()
    state.record_phase_transition(
        to_phase="SWEEP",
        reason="plateau_kernel",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    # ``explore`` is not in the SWEEP proposable set, so R1 rejects it.
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "explore", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


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
    # FRAMEWORK 0.20, EXPLORE 0.35, PRELUDE 0.03; budgets sum to 1.0.
    assert c.shared_state.phase_budget_pct["FRAMEWORK_AGENT"] == 0.20
    assert c.shared_state.phase_budget_pct["EXPLORE"] == 0.35
    assert c.shared_state.phase_budget_pct["PRELUDE"] == 0.03


@pytest.mark.asyncio
async def test_coordinator_advances_to_explore_when_baseline_present(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        # Skip FRAMEWORK to exercise the PRELUDE→EXPLORE contract.
        c.shared_state.framework_agent_phase_enabled = False
        # Simulate baseline KEEP to trigger prelude_done.
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert c.shared_state.phase == "EXPLORE"
        # 2 rows: PRELUDE entry + PRELUDE→EXPLORE.
        assert len(c.shared_state.phase_history) == 2
        last = c.shared_state.phase_history[-1]
        assert last["from_phase"] == "PRELUDE"
        assert last["to_phase"] == "EXPLORE"
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
