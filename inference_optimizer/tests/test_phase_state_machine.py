"""v0.8 M2 — phase state machine tests (KB_design §3.2 + §3.8 + §3.11 R1).

Covers the additive subset of M2 implemented in this PR:

* ``phase_state`` pure functions (allowed actions, exit reasons,
  stop_reason vocab, plateau/budget judges, v0.6 phase inference).
* SharedState new fields + ``record_phase_transition`` helper.
* PolicyGate R1 ``phase_incompatible`` (warn-vs-enforce modes).
* Coordinator initialises ``phase`` to PRELUDE on fresh sessions and
  records a baseline ``phase_entered`` row in ``phase_history``.
* Coordinator advances PRELUDE → EXPLORE once ``baseline_tput > 0``
  (with FRAMEWORK_PR phase opt-out via ``framework_phase_enabled=False``).
* breakdown.collect_phase_segments groups action events by phase
  window with proper ``elapsed_seconds`` math.
* PolicyGate adds the new phase fields to ``CORE_STATE_FIELDS``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import phase_state
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    CORE_STATE_FIELDS,
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# ===========================================================================
# phase_state pure function tests
# ===========================================================================
def test_phase_names_are_monotonic():
    assert phase_state.PHASE_NAMES == (
        "PRELUDE", "FRAMEWORK_PR", "EXPLORE", "KERNEL", "SWEEP", "CLOSE",
    )
    # phase_index strictly increases.
    for i, name in enumerate(phase_state.PHASE_NAMES):
        assert phase_state.phase_index(name) == i
    assert phase_state.phase_index("unknown") == -1


def test_allowed_actions_disjoint_phases():
    # Every kernel-owned action belongs only to KERNEL phase (Inv-2.1
    # protects the strictly-monotonic flow). recover is in every phase.
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        assert "recover" in allowed
    # Cross-phase exclusion sanity.
    assert "baseline" in phase_state.PHASE_ALLOWED_ACTIONS["PRELUDE"]
    assert "baseline" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "kernel_opt" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL"]
    assert "kernel_opt" not in phase_state.PHASE_ALLOWED_ACTIONS["EXPLORE"]
    assert "gemm_tuning" in phase_state.PHASE_ALLOWED_ACTIONS["KERNEL"]
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


def test_phase_exit_reasons_includes_required_vocab():
    for reason in (
        "prelude_done", "plateau_explore", "plateau_kernel",
        "sweep_done", "robustness_escalated", "target_reached",
        "time_exhausted", "user_stop_requested", "cortex_drain_failed",
        "no_kernel_skipped",
    ):
        assert phase_state.is_valid_phase_exit_reason(reason), reason


def test_stop_reason_vocab_includes_v06_and_v08():
    for reason in (
        # v0.6 sentinels
        "target_reached", "time_exhausted", "max_ticks", "policy_loop",
        "baseline_failed", "emergency", "coordinator_exception",
        # v0.8 additions
        "crash_threshold_exceeded", "user_stop_requested",
        "cortex_drain_failed", "plateau_explore",
    ):
        assert phase_state.is_valid_stop_reason(reason), reason
    assert not phase_state.is_valid_stop_reason("totally_invented")


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
    """PRELUDE must not advance to FRAMEWORK_PR until warm-replay settles."""
    state = SimpleNamespace(
        baseline_tput=1234.5,
        warm_replay_outcome={"status": "in_flight", "replay_task_id": "abc"},
    )
    assert phase_state.exit_normal_prelude(state) is None
    state.warm_replay_outcome = {"status": "failed"}
    out = phase_state.exit_normal_prelude(state)
    assert out is not None and out[0] == "prelude_done"


def test_exit_terminal_prelude_after_three_baseline_failures():
    state = SimpleNamespace(baseline_failure_streak=2)
    assert phase_state.exit_terminal_prelude(state) is None
    state.baseline_failure_streak = 3
    out = phase_state.exit_terminal_prelude(state)
    assert out is not None and out[0] == "prelude_baseline_failed"


def test_exit_normal_explore_uses_budget_exhaustion():
    # Provide an in-the-past phase_started_unix so elapsed exceeds budget.
    # IR-6 force-exit is disabled (thresholds=0) for this test so we
    # isolate the budget_exhausted path; a dedicated suite in
    # test_phase_force_exit.py exercises the force-exit gate.
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


# ===========================================================================
# SharedState writer
# ===========================================================================
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
        to_phase="PRELUDE", reason="phase_entered",
        evidence={"trigger": "fresh_session"},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1747600000.0,
    )
    assert s.phase == "PRELUDE"
    assert s.phase_started_ts == "2026-05-19T00:00:00+00:00"
    assert s.phase_started_unix == 1747600000.0
    assert s.phase_history == [row]
    assert row["from_phase"] == "" and row["to_phase"] == "PRELUDE"
    # Second transition keeps history append-only.
    row2 = s.record_phase_transition(
        to_phase="EXPLORE", reason="prelude_done",
        evidence={"baseline_tput": 100.0},
        ts="2026-05-19T00:01:00+00:00", ts_unix=1747600060.0,
    )
    assert s.phase == "EXPLORE"
    assert len(s.phase_history) == 2
    assert s.phase_history[-1] == row2
    assert row2["from_phase"] == "PRELUDE"


# ===========================================================================
# CORE_STATE_FIELDS includes phase fields (Inv-1 single writer)
# ===========================================================================
def test_core_state_fields_includes_phase_fields():
    for f in (
        "phase", "phase_started_ts", "phase_started_unix",
        "phase_history", "phase_budget_pct",
    ):
        assert f in CORE_STATE_FIELDS, f


# ===========================================================================
# PolicyGate R1 phase_incompatible
# ===========================================================================
def _make_role_registry():
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    return default_role_registry()


def test_policy_gate_phase_strict_denies_kernel_in_prelude():
    state = SharedState()
    state.record_phase_transition(
        to_phase="PRELUDE", reason="phase_entered", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "kernel_opt", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"
    assert "PRELUDE" in (excinfo.value.hint or "")


def test_policy_gate_phase_warn_mode_does_not_raise():
    state = SharedState()
    state.record_phase_transition(
        to_phase="PRELUDE", reason="phase_entered", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=False,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "kernel_opt", "predicted_gain_pct": 1.0},
    )
    # Should NOT raise — warn-mode just bumps the audit counter.
    gate.validate_intent("orchestration", intent)
    assert state.policy_denial_streak.get("kernel_opt:phase_incompatible", 0) >= 1


def test_policy_gate_phase_strict_allows_in_phase_action():
    state = SharedState()
    state.record_phase_transition(
        to_phase="PRELUDE", reason="phase_entered", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
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
        to_phase="PRELUDE", reason="phase_entered", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    # ``profile`` / ``roofline`` are now LLM-denied earlier via the
    # ``analysis_action_not_llm_proposable`` rule (Coordinator-internal
    # analysis actions never reach R1), and ``params`` is denied with
    # ``action_deprecated``. We pick ``sweep`` instead — a non-deprecated,
    # non-internal action that is allowed only in the SWEEP phase, so the
    # propose lands cleanly on R1 phase_incompatible while in PRELUDE.
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "sweep", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


# ===========================================================================
# Coordinator initialises phase on fresh sessions
# ===========================================================================
@pytest.fixture
def coordinator_with_mocks(session_dir):
    from inference_optimizer.orchestrator.backends import (
        MockBackend, MockCriticBackend, MockKernelBackend, MockRobustnessBackend,
        ScriptedPlan,
    )
    from inference_optimizer.orchestrator.coordinator import Coordinator
    silent = ScriptedPlan(turns=[], default_intent=Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    ))
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }
    return Coordinator(session_dir, backends=backends)


def test_coordinator_init_writes_phase_prelude_for_fresh_session(coordinator_with_mocks):
    c = coordinator_with_mocks
    assert c.shared_state.phase == "PRELUDE"
    assert len(c.shared_state.phase_history) == 1
    row = c.shared_state.phase_history[0]
    assert row["to_phase"] == "PRELUDE"
    assert row["reason"] == "phase_entered"
    # Budget dict populated. 2026-06 rebalance reshaped the slice:
    # PRELUDE 0.08 → 0.05 (only ever used ~5min of 27min in practice),
    # EXPLORE 0.47 → 0.40 (force-exit at phase_remaining_pct=0.176
    # confirmed it was over-provisioned by ~7pp), KERNEL held at 0.35
    # (GEAK quick-mode needs full cycles), and SWEEP 0.08 → 0.18 to
    # fit the sweep + conc_sweep pair that field telemetry showed
    # running ~2.5x over the old 8% slice. Sum across phases stays
    # at 1.0.
    assert c.shared_state.phase_budget_pct["EXPLORE"] == 0.40
    assert c.shared_state.phase_budget_pct["PRELUDE"] == 0.05


@pytest.mark.asyncio
async def test_coordinator_advances_to_explore_when_baseline_present(
    coordinator_with_mocks, session_dir,
):
    c = coordinator_with_mocks
    try:
        # Skip the FRAMEWORK_PR phase so the legacy PRELUDE→EXPLORE
        # contract is what this test exercises (the FRAMEWORK_PR
        # transition is covered separately in
        # ``test_phase_state_framework_pr.py``).
        c.shared_state.framework_phase_enabled = False
        # Simulate baseline KEEP without running the executor: write the
        # SharedState event that triggers the prelude_done condition.
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert c.shared_state.phase == "EXPLORE"
        # phase_history now has 2 rows: PRELUDE entry + PRELUDE→EXPLORE
        # (FRAMEWORK_PR is skipped here because the fixture sets
        # ``framework_phase_enabled=False``).
        assert len(c.shared_state.phase_history) == 2
        last = c.shared_state.phase_history[-1]
        assert last["from_phase"] == "PRELUDE"
        assert last["to_phase"] == "EXPLORE"
        assert last["reason"] == "prelude_done"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_phase_idempotent_within_same_tick(
    coordinator_with_mocks, session_dir,
):
    c = coordinator_with_mocks
    try:
        c.shared_state.framework_phase_enabled = False
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        first_history = list(c.shared_state.phase_history)
        # Another tick without further state change → no new transition.
        await c.tick(1)
        assert c.shared_state.phase_history == first_history
    finally:
        await c.stop()


# ===========================================================================
# breakdown collect_phase_segments
# ===========================================================================
def test_collect_phase_segments_groups_actions_by_window():
    from inference_optimizer.breakdown.collectors import collect_phase_segments
    state = {
        "phase_history": [
            {
                "from_phase": "",
                "to_phase":   "PRELUDE",
                "reason":     "phase_entered",
                "evidence":   {"trigger": "fresh_session"},
                "ts":         "2026-05-19T00:00:00+00:00",
                "ts_unix":    1.0,
            },
            {
                "from_phase": "PRELUDE",
                "to_phase":   "EXPLORE",
                "reason":     "prelude_done",
                "evidence":   {"baseline_tput": 100.0},
                "ts":         "2026-05-19T00:05:00+00:00",
                "ts_unix":    301.0,
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
    assert explore["exit_reason"] == ""        # currently active segment
    assert explore["elapsed_seconds"] is None  # no exit_unix yet
    assert explore["actions"][0]["action"] == "params"


def test_collect_phase_segments_empty_when_history_missing():
    from inference_optimizer.breakdown.collectors import collect_phase_segments
    assert collect_phase_segments({}, [], warnings=[]) == []
    assert collect_phase_segments({"phase_history": []}, [], warnings=[]) == []
