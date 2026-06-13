# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""v0.8 M2 — phase state machine tests (KB_design §3.2 + §3.8 + §3.11 R1)."""

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


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# phase_state pure function tests
def test_phase_names_are_monotonic():
    assert phase_state.PHASE_NAMES == (
        "PRELUDE", "FRAMEWORK_PR", "EXPLORE", "KERNEL", "SWEEP", "CLOSE",
    )
    for i, name in enumerate(phase_state.PHASE_NAMES):
        assert phase_state.phase_index(name) == i
    assert phase_state.phase_index("unknown") == -1


def test_allowed_actions_disjoint_phases():
    # recover is in every phase; kernel-owned actions only in KERNEL (Inv-2.1).
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        assert "recover" in allowed
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


def test_llm_proposable_set_drops_coordinator_internal_actions():
    from inference_optimizer.protocol.action_surfaces import (
        COORDINATOR_INTERNAL_ACTIONS,
        ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
    )

    # Proposable = allowlist minus Coordinator-internal and robustness-delegate-only actions.
    for phase in phase_state.PHASE_NAMES:
        allowed = phase_state.PHASE_ALLOWED_ACTIONS[phase]
        proposable = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS[phase]
        assert proposable == (
            allowed
            - COORDINATOR_INTERNAL_ACTIONS
            - ROBUSTNESS_DELEGATE_ONLY_ACTIONS
        )
        assert proposable.isdisjoint(COORDINATOR_INTERNAL_ACTIONS)
        # recover stays phase-allowed (robustness delegate) but is
        # never LLM-proposable by Orchestration.
        assert "recover" in allowed
        assert "recover" not in proposable
    # The advertised analysis / framework_pr names are never proposable.
    explore = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["EXPLORE"]
    assert "roofline" not in explore
    assert "profile" not in explore
    assert "explore" in explore and "specialist" in explore
    framework_pr = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS["FRAMEWORK_PR"]
    assert "framework_pr" not in framework_pr
    assert "integrate_patch" in framework_pr


def test_is_action_llm_proposable_in_phase_handles_unknowns():
    assert phase_state.is_action_llm_proposable_in_phase("baseline", "PRELUDE")
    assert phase_state.is_action_llm_proposable_in_phase("explore", "EXPLORE")
    # roofline lives in the allowlist but is never LLM-proposable.
    assert phase_state.is_action_allowed_in_phase("roofline", "EXPLORE")
    assert not phase_state.is_action_llm_proposable_in_phase("roofline", "EXPLORE")
    assert not phase_state.is_action_llm_proposable_in_phase("framework_pr", "FRAMEWORK_PR")
    # Unknown phase / empty action → deny by default.
    assert not phase_state.is_action_llm_proposable_in_phase("baseline", "UNKNOWN")
    assert not phase_state.is_action_llm_proposable_in_phase("", "PRELUDE")
    # llm_proposable_actions_for is sorted and excludes internal names.
    explore = phase_state.llm_proposable_actions_for("EXPLORE")
    assert explore == tuple(sorted(explore))
    assert "roofline" not in explore and "profile" not in explore


def test_phase_interleave_off_matches_default_proposable_set():
    """With the env flag off, the interleave-aware helpers match the default set."""
    for phase in phase_state.PHASE_NAMES:
        base = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS.get(
            phase, frozenset(),
        )
        assert (
            phase_state.llm_proposable_actions_for_with_interleave(
                phase, interleave=False,
            )
            == base
        )
    # And kernel-owned actions are denied in EXPLORE under default mode.
    assert not phase_state.is_action_llm_proposable_in_phase_with_interleave(
        "kernel_opt", "EXPLORE", interleave=False,
    )
    assert not phase_state.is_action_llm_proposable_in_phase_with_interleave(
        "explore", "KERNEL", interleave=False,
    )


def test_phase_interleave_on_widens_explore_and_kernel():
    """With interleave=True, EXPLORE gains kernel-owned actions and KERNEL gains explore/specialist/integrate_patch."""
    explore = phase_state.llm_proposable_actions_for_with_interleave(
        "EXPLORE", interleave=True,
    )
    assert "kernel_opt" in explore
    assert "integrate" in explore
    assert "gemm_tuning" in explore
    assert "explore" in explore  # native EXPLORE actions still present
    assert "specialist" in explore
    kernel = phase_state.llm_proposable_actions_for_with_interleave(
        "KERNEL", interleave=True,
    )
    assert "explore" in kernel
    assert "specialist" in kernel
    assert "integrate_patch" in kernel
    assert "kernel_opt" in kernel  # native KERNEL actions still present
    # SWEEP / CLOSE / PRELUDE / FRAMEWORK_PR are unchanged.
    for phase in ("PRELUDE", "FRAMEWORK_PR", "SWEEP", "CLOSE"):
        base = phase_state.PHASE_LLM_PROPOSABLE_ACTIONS.get(
            phase, frozenset(),
        )
        assert (
            phase_state.llm_proposable_actions_for_with_interleave(
                phase, interleave=True,
            )
            == base
        )


def test_phase_interleave_env_flag_is_picked_up(monkeypatch):
    """The helpers honour the env flag; interleave is OFF by default, only an explicit on value enables it."""
    # Unset / empty => OFF by default.
    monkeypatch.delenv(phase_state.PHASE_INTERLEAVE_ENV, raising=False)
    assert phase_state.is_phase_interleave_enabled() is False
    assert not phase_state.is_action_llm_proposable_in_phase_with_interleave(
        "kernel_opt", "EXPLORE",
    )
    # Explicit on values enable interleave.
    monkeypatch.setenv(phase_state.PHASE_INTERLEAVE_ENV, "1")
    assert phase_state.is_phase_interleave_enabled() is True
    assert phase_state.is_action_llm_proposable_in_phase_with_interleave(
        "kernel_opt", "EXPLORE",
    )
    # Explicit off values stay off.
    monkeypatch.setenv(phase_state.PHASE_INTERLEAVE_ENV, "0")
    assert phase_state.is_phase_interleave_enabled() is False
    assert not phase_state.is_action_llm_proposable_in_phase_with_interleave(
        "kernel_opt", "EXPLORE",
    )
    monkeypatch.setenv(phase_state.PHASE_INTERLEAVE_ENV, "false")
    assert phase_state.is_phase_interleave_enabled() is False
    monkeypatch.setenv(phase_state.PHASE_INTERLEAVE_ENV, "off")
    assert phase_state.is_phase_interleave_enabled() is False


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
        # #522: fast baseline arg-error stop reason
        "baseline_arg_error",
    ):
        assert phase_state.is_valid_stop_reason(reason), reason
    assert not phase_state.is_valid_stop_reason("totally_invented")


def test_set_stop_reason_keeps_baseline_arg_error(tmp_path):
    """#522: baseline_arg_error must survive set_stop_reason (not map to unknown)."""
    from inference_optimizer.orchestrator.shared_state import SharedState
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
    # Past phase_started_unix so elapsed exceeds budget; IR-6 force-exit disabled to isolate budget path.
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


# SharedState writer
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


# CORE_STATE_FIELDS includes phase fields (Inv-1 single writer)
def test_core_state_fields_includes_phase_fields():
    for f in (
        "phase", "phase_started_ts", "phase_started_unix",
        "phase_history", "phase_budget_pct",
    ):
        assert f in CORE_STATE_FIELDS, f


# PolicyGate R1 phase_incompatible
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
    # ``profile`` / ``roofline`` are Coordinator-managed and denied by
    # R1 ``phase_incompatible`` regardless of phase, and ``params`` is
    # denied with ``action_deprecated``. We pick ``sweep`` instead — a
    # non-deprecated, non-internal action that is proposable only in the
    # SWEEP phase, so the propose lands on R1 phase_incompatible via the
    # per-phase set while in PRELUDE.
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action_name": "sweep", "predicted_gain_pct": 1.0},
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_phase_interleave_off_denies_kernel_request_in_explore(
    monkeypatch,
):
    """With interleave off, a kernel-owned REQUEST in EXPLORE is denied by R1."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "0")
    state = SharedState()
    state.record_phase_transition(
        to_phase="EXPLORE", reason="prelude_done", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "kernel_opt",
            "params": {},
        },
    )
    with pytest.raises(PolicyDenied) as excinfo:
        gate.validate_intent("orchestration", intent)
    assert excinfo.value.rule == "phase_incompatible"


def test_policy_gate_phase_interleave_on_allows_kernel_request_in_explore(
    monkeypatch,
):
    """With interleave on, EXPLORE widens to kernel-owned kinds, so R1 lets the REQUEST through."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    state = SharedState()
    state.record_phase_transition(
        to_phase="EXPLORE", reason="prelude_done", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
    )
    gate = PolicyGate(
        role_registry=_make_role_registry(),
        shared_state=state,
        strict_phase=True,
    )
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "kernel_opt",
            "params": {},
        },
    )
    gate.validate_intent("orchestration", intent)  # no exception


def test_policy_gate_phase_interleave_on_allows_explore_propose_in_kernel(
    monkeypatch,
):
    """With interleave on, KERNEL also accepts explore/specialist/integrate_patch proposals."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    state = SharedState()
    state.record_phase_transition(
        to_phase="KERNEL", reason="plateau_explore", evidence={},
        ts="2026-05-19T00:00:00+00:00", ts_unix=1.0,
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
    gate.validate_intent("orchestration", intent)  # no exception


def test_policy_gate_phase_interleave_does_not_widen_other_phases(monkeypatch):
    """Interleave widening is scoped to EXPLORE/KERNEL; SWEEP still rejects explore/kernel_opt under R1."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    state = SharedState()
    state.record_phase_transition(
        to_phase="SWEEP", reason="plateau_kernel", evidence={},
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


# Coordinator initialises phase on fresh sessions
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
    # P3_22 rebalance: EXPLORE/KERNEL carry a larger slice; PRELUDE 3%, SWEEP 12%; sum stays 1.0.
    assert c.shared_state.phase_budget_pct["EXPLORE"] == 0.45
    assert c.shared_state.phase_budget_pct["PRELUDE"] == 0.03


@pytest.mark.asyncio
async def test_coordinator_advances_to_explore_when_baseline_present(
    coordinator_with_mocks, session_dir,
):
    c = coordinator_with_mocks
    try:
        # Skip FRAMEWORK_PR so this exercises the legacy PRELUDE→EXPLORE contract.
        c.shared_state.framework_phase_enabled = False
        # Simulate baseline KEEP: write the event that triggers prelude_done.
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert c.shared_state.phase == "EXPLORE"
        # 2 rows: PRELUDE entry + PRELUDE→EXPLORE (FRAMEWORK_PR skipped).
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
        # Another tick without state change → no new transition.
        await c.tick(1)
        assert c.shared_state.phase_history == first_history
    finally:
        await c.stop()


# breakdown collect_phase_segments
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
