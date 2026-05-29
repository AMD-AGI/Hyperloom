"""IR-7 — session_steward_specialist + assess_remaining_gaps tests.

Covers:

* :func:`phase_state.wants_steward_assessment` predicate.
* :func:`phase_state.exit_normal_explore` steward gate routing.
* :meth:`SharedState.record_steward_assessment` audit + history cap.
* Coordinator's ``_route_steward_verdict`` routing the three
  recommendations (continue_explore, advance_to_kernel, stop_session)
  + the antiloop coercion of a second ``continue_explore``.
* PolicyGate's ``assess_remaining_gaps_throttle`` on LLM-side proposes.
* ``session_steward_specialist`` is registered in the domain catalogue
  and gated as M5-active.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import phase_state
from inference_optimizer.orchestrator.specialist_domains import (
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_DOMAINS_M5,
    get_domain,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ===========================================================================
# fixtures
# ===========================================================================
def _make_plateau_state(*, kernel_enabled: bool = True) -> SharedState:
    """Build a SharedState that triggers plateau but NOT force-exit."""
    now = datetime.now(timezone.utc)
    state = SharedState(
        session_id="test-steward",
        phase=phase_state.PHASE_EXPLORE,
        phase_started_ts=(now - timedelta(minutes=30)).isoformat(),
        phase_started_unix=time.time() - 30 * 60,
        start_ts=(now - timedelta(hours=1)).isoformat(),
        max_minutes=600,  # 10h budget — well above force-exit thresholds
        kernel_enabled=kernel_enabled,
    )
    # Plateau signals: 3 consecutive empty specialist rounds and a
    # winners_history below the keep_gain threshold.
    state.explore_search = {
        "schema_version": 1,
        "tested": {},
        "accepted": [],
        "rejected": [],
        "winners_history": [
            {"round_id": 1, "gain_pct": 0.1, "variant_name": "v1"},
            {"round_id": 2, "gain_pct": 0.0, "variant_name": "v2"},
            {"round_id": 3, "gain_pct": 0.05, "variant_name": "v3"},
        ],
        "cursor": 3,
    }
    state.specialist_rounds = [
        {"round_id": str(i), "empty": True, "proposals_total": 0}
        for i in range(1, 4)
    ]
    state.optimization_stack = [
        {"action": "explore", "name": f"v{i}"} for i in range(1, 3)
    ]
    return state


# ===========================================================================
# domain catalogue
# ===========================================================================
def test_session_steward_in_catalogue():
    assert "session_steward_specialist" in SPECIALIST_DOMAIN_KEYS
    assert "session_steward_specialist" in SPECIALIST_DOMAINS_M5
    d = get_domain("session_steward_specialist")
    assert d is not None
    assert d.kb_anchor == "pr_intelligence"
    assert d.available_in == "M5"


# ===========================================================================
# wants_steward_assessment predicate
# ===========================================================================
def test_wants_steward_assessment_fires_on_plateau():
    state = _make_plateau_state()
    assert phase_state.wants_steward_assessment(state) is True


def test_wants_steward_assessment_no_plateau_signals():
    state = SharedState(
        session_id="t1",
        phase=phase_state.PHASE_EXPLORE,
        max_minutes=600,
    )
    # No v0.8 signals at all — should not request steward.
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_already_have_verdict():
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.foo",
    }
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_disabled_via_override():
    state = _make_plateau_state()
    state.plateau_overrides = {"steward_disabled": True}
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_skipped_during_force_exit():
    """When HARD force-exit fires, the steward path is no-op."""
    now = datetime.now(timezone.utc)
    state = _make_plateau_state()
    # Override start_ts so session_remaining_seconds is well below 3h.
    state.start_ts = (now - timedelta(hours=7.5)).isoformat()
    assert phase_state.wants_steward_assessment(state) is False


# ===========================================================================
# exit_normal_explore steward gate routing
# ===========================================================================
def test_exit_normal_explore_holds_when_steward_pending():
    """Plateau triggered, no assessment yet -> stay in EXPLORE (None)."""
    state = _make_plateau_state()
    out = phase_state.exit_normal_explore(state)
    assert out is None


def test_exit_normal_explore_holds_on_continue_explore():
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.X",
    }
    state.steward_continuation_used = False
    out = phase_state.exit_normal_explore(state)
    assert out is None


def test_exit_normal_explore_routes_on_advance_to_kernel():
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "advance_to_kernel",
    }
    out = phase_state.exit_normal_explore(state)
    assert out is not None
    reason, evidence = out
    assert reason == "plateau_explore"
    assert evidence["steward_recommendation"] == "advance_to_kernel"


def test_exit_normal_explore_routes_on_stop_session():
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "stop_session",
    }
    out = phase_state.exit_normal_explore(state)
    assert out is not None
    reason, evidence = out
    assert reason == "plateau_explore"
    assert evidence["steward_recommendation"] == "stop_session"


def test_exit_normal_explore_continuation_exhausted():
    """Second continue_explore (after first was used) -> exit plateau."""
    state = _make_plateau_state()
    state.steward_continuation_used = True
    state.last_remaining_gaps_assessment = {
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.Y",
    }
    out = phase_state.exit_normal_explore(state)
    assert out is not None
    reason, _evidence = out
    assert reason == "plateau_explore"


def test_exit_normal_explore_steward_disabled():
    state = _make_plateau_state()
    state.plateau_overrides = {"steward_disabled": True}
    out = phase_state.exit_normal_explore(state)
    assert out is not None
    reason, evidence = out
    assert reason == "plateau_explore"
    assert evidence.get("steward_disabled") is True


# ===========================================================================
# SharedState.record_steward_assessment
# ===========================================================================
def test_record_steward_assessment_writes_history():
    state = SharedState(session_id="t-rec")
    row = state.record_steward_assessment(
        recommendation="stop_session",
        next_gap_canonical_id="",
        remaining_potential_pct_estimate=0.5,
        rationale="exhausted single-flag space",
        task_id="task-abc",
        round_at_assessment=5,
    )
    assert row["recommendation"] == "stop_session"
    assert state.last_remaining_gaps_assessment == row
    assert len(state.remaining_gaps_assessments) == 1


def test_record_steward_assessment_history_cap():
    state = SharedState(session_id="t-cap")
    cap = SharedState._STEWARD_ASSESSMENT_HISTORY_CAP
    for i in range(cap + 5):
        state.record_steward_assessment(
            recommendation="continue_explore",
            next_gap_canonical_id=f"gap.{i}",
            remaining_potential_pct_estimate=1.0,
            rationale=f"iter {i}",
            task_id=f"task-{i}",
            round_at_assessment=i,
        )
    assert len(state.remaining_gaps_assessments) == cap


def test_record_steward_assessment_truncates_rationale():
    state = SharedState(session_id="t-trunc")
    long_text = "x" * 5000
    state.record_steward_assessment(
        recommendation="stop_session",
        next_gap_canonical_id="",
        remaining_potential_pct_estimate=0.0,
        rationale=long_text,
        task_id="t",
        round_at_assessment=0,
    )
    assert len(state.last_remaining_gaps_assessment["rationale"]) <= 2000


# ===========================================================================
# Coordinator routing
# ===========================================================================
@pytest.mark.asyncio
async def test_route_stop_session():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-stop", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    fake = SimpleNamespace(
        shared_state=state,
        session_dir="/tmp",
        append_gap_attempt=lambda *a, **kw: None,
    )
    task = SimpleNamespace(task_id="task-stop", params={})
    payload = {
        "domain": "session_steward_specialist",
        "recommendation": "stop_session",
        "remaining_potential_pct_estimate": 0.0,
        "rationale": "no more single-flag levers",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    assert state.stop_reason == "no_more_leverage"
    assert state.last_remaining_gaps_assessment["recommendation"] == "stop_session"


@pytest.mark.asyncio
async def test_route_advance_to_kernel():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-adv", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    fake = SimpleNamespace(
        shared_state=state,
        session_dir="/tmp",
        append_gap_attempt=lambda *a, **kw: None,
    )
    task = SimpleNamespace(task_id="task-adv", params={})
    payload = {
        "domain": "session_steward_specialist",
        "recommendation": "advance_to_kernel",
        "remaining_potential_pct_estimate": 5.0,
        "rationale": "kernel phase has leverage",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    assert state.pending_escalate_hint == "skip_to_kernel"
    assert state.last_remaining_gaps_assessment["recommendation"] == "advance_to_kernel"


@pytest.mark.asyncio
async def test_route_continue_explore_grants_first_then_coerces_second():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-cont", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    state.params_no_promote_streak = 5
    state.specialist_domain_empty_streak = {"serving_specialist": 3}
    # First call: continue_explore -> granted.
    fake = SimpleNamespace(
        shared_state=state,
        session_dir="/tmp",
    )
    task = SimpleNamespace(task_id="task-cont-1", params={})
    payload_1 = {
        "domain": "session_steward_specialist",
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.scheduler.overlap",
        "remaining_potential_pct_estimate": 8.0,
        "rationale": "haven't tried SpecV2 yet",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload_1,
    )
    assert state.steward_continuation_used is True
    assert state.params_no_promote_streak == 0
    assert state.specialist_domain_empty_streak == {}
    assert state.last_remaining_gaps_assessment["recommendation"] == "continue_explore"
    # Stop reason should NOT be set yet.
    assert state.stop_reason in ("", None)

    # Second call: continue_explore again -> coerced to advance_to_kernel.
    task2 = SimpleNamespace(task_id="task-cont-2", params={})
    payload_2 = {
        "domain": "session_steward_specialist",
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.another",
        "remaining_potential_pct_estimate": 3.0,
        "rationale": "still hopeful",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task2, done_payload=payload_2,
    )
    assert state.pending_escalate_hint == "skip_to_kernel"
    assert state.last_remaining_gaps_assessment["recommendation"] == "advance_to_kernel"


@pytest.mark.asyncio
async def test_route_coerces_out_of_vocab():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-oov", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    fake = SimpleNamespace(
        shared_state=state,
        session_dir="/tmp",
    )
    task = SimpleNamespace(task_id="task-oov", params={})
    payload = {
        "domain": "session_steward_specialist",
        "recommendation": "make_coffee",  # not a valid recommendation
        "rationale": "hallucinated verdict",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    # OOV -> coerced to stop_session.
    assert state.stop_reason == "no_more_leverage"
    assert state.last_remaining_gaps_assessment["recommendation"] == "stop_session"


@pytest.mark.asyncio
async def test_route_continue_explore_missing_gap_id_coerced():
    """``continue_explore`` without next_gap_canonical_id -> advance_to_kernel."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-missgap", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    fake = SimpleNamespace(
        shared_state=state,
        session_dir="/tmp",
    )
    task = SimpleNamespace(task_id="task-missgap", params={})
    payload = {
        "domain": "session_steward_specialist",
        "recommendation": "continue_explore",
        # next_gap_canonical_id missing
        "rationale": "vague continuation request",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    assert state.pending_escalate_hint == "skip_to_kernel"


# ===========================================================================
# PolicyGate throttle (LLM-side propose path)
# ===========================================================================
def _ready_to_throttle_state(session_id: str) -> SharedState:
    """Return a SharedState that satisfies the Issue-A preconditions
    (phase=EXPLORE + len(optimization_stack)>=3) so the throttle path
    is the only remaining gate. The throttle tests below all start
    from this fixture; the new precondition tests use a bare state."""
    state = SharedState(session_id=session_id)
    state.phase = "EXPLORE"
    state.optimization_stack = [
        {"variant_name": f"stub-{i}", "tput": 100.0 + i}
        for i in range(3)
    ]
    return state


def test_assess_remaining_gaps_throttle_blocks_back_to_back(monkeypatch):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = _ready_to_throttle_state("t-throttle")
    state.last_remaining_gaps_assessment = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "recommendation": "advance_to_kernel",
    }
    fake = SimpleNamespace(shared_state=state)
    denial = Coordinator._assess_remaining_gaps_throttle_denial(fake)
    assert denial is not None
    assert denial.rule == "assess_remaining_gaps_throttle"


def test_assess_remaining_gaps_throttle_permits_after_window(monkeypatch):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = _ready_to_throttle_state("t-throttle-ok")
    state.last_remaining_gaps_assessment = {
        "ts": (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat(),
        "recommendation": "advance_to_kernel",
    }
    fake = SimpleNamespace(shared_state=state)
    denial = Coordinator._assess_remaining_gaps_throttle_denial(fake)
    assert denial is None


def test_assess_remaining_gaps_throttle_first_call_passes():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = _ready_to_throttle_state("t-throttle-first")
    # No prior assessment -> no throttle (and preconditions satisfied).
    fake = SimpleNamespace(shared_state=state)
    denial = Coordinator._assess_remaining_gaps_throttle_denial(fake)
    assert denial is None


# ---------------------------------------------------------------------------
# Issue-A (Saturday May 2026): preconditions enforced before the throttle
# ---------------------------------------------------------------------------
def test_assess_remaining_gaps_denied_outside_explore():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-precondition-phase")
    state.phase = "PRELUDE"
    state.optimization_stack = [
        {"variant_name": f"stub-{i}"} for i in range(5)
    ]
    fake = SimpleNamespace(shared_state=state)
    denial = Coordinator._assess_remaining_gaps_throttle_denial(fake)
    assert denial is not None
    assert denial.rule == "assess_remaining_gaps_phase"


def test_assess_remaining_gaps_denied_when_stack_too_short():
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-precondition-stack")
    state.phase = "EXPLORE"
    state.optimization_stack = [{"variant_name": "stub-0"}]
    fake = SimpleNamespace(shared_state=state)
    denial = Coordinator._assess_remaining_gaps_throttle_denial(fake)
    assert denial is not None
    assert denial.rule == "assess_remaining_gaps_min_stack"


# ===========================================================================
# Phase allowlist
# ===========================================================================
def test_assess_remaining_gaps_in_explore_allowlist():
    assert "assess_remaining_gaps" in phase_state.PHASE_ALLOWED_ACTIONS[
        phase_state.PHASE_EXPLORE
    ]
