"""session_steward_specialist + assess_remaining_gaps tests.

After loosen plan P2_13 the steward is **advisory only**: it no longer
drives phase transitions and there is no LLM-side throttle. The depth
gate that used to rewrite stop/advance into continue_explore is gone.

Surviving coverage:

* :func:`phase_state.wants_steward_assessment` predicate (Coordinator
  still auto-enqueues an advisory verdict on plateau).
* :meth:`SharedState.record_steward_assessment` audit + history cap.
* Coordinator's ``_route_steward_verdict`` records the recommendation
  verbatim, treats infrastructure failures as retries, coerces OOV
  strings to ``stop_session``, and **does not** set any
  ``pending_escalate_hint``.
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
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_already_have_verdict():
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "continue_explore",
        "next_gap_canonical_id": "gap.foo",
    }
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_after_infra_failure_empty_rec():
    """Empty recommendation after a transport failure should allow retry."""
    state = _make_plateau_state()
    state.last_remaining_gaps_assessment = {
        "recommendation": "",
        "rationale": "steward infrastructure failure (subprocess_stale_heartbeat)",
    }
    assert phase_state.wants_steward_assessment(state) is True


def test_wants_steward_assessment_disabled_via_override():
    state = _make_plateau_state()
    state.plateau_overrides = {"steward_disabled": True}
    assert phase_state.wants_steward_assessment(state) is False


def test_wants_steward_assessment_skipped_during_force_exit():
    """When HARD force-exit fires, the steward path is no-op."""
    now = datetime.now(timezone.utc)
    state = _make_plateau_state()
    state.start_ts = (now - timedelta(hours=7.5)).isoformat()
    assert phase_state.wants_steward_assessment(state) is False


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
# Coordinator routing — advisory, never drives phase
# ===========================================================================
@pytest.mark.asyncio
async def test_route_stop_session_is_advisory_only():
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
    assert not state.stop_reason
    assert not state.pending_escalate_hint
    assert state.last_remaining_gaps_assessment["recommendation"] == "stop_session"


@pytest.mark.asyncio
async def test_route_advance_to_kernel_is_advisory_only():
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
    assert not state.pending_escalate_hint
    assert state.last_remaining_gaps_assessment["recommendation"] == "advance_to_kernel"


@pytest.mark.asyncio
async def test_route_continue_explore_resets_plateau_counters():
    """``continue_explore`` still resets the plateau proxy counters as a
    neutral aid for the next round, even though it no longer feeds back
    into a steward gate."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    state = SharedState(session_id="t-route-cont", phase="EXPLORE")
    state.explore_search = {"cursor": 1}
    state.params_no_promote_streak = 5
    state.specialist_domain_empty_streak = {"serving_specialist": 3}
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
    assert state.last_remaining_gaps_assessment["recommendation"] == "continue_explore"
    assert state.stop_reason in ("", None)
    assert not state.pending_escalate_hint


@pytest.mark.asyncio
async def test_route_infra_failure_retries_without_stop(tmp_path):
    from inference_optimizer.orchestrator.coordinator import Coordinator

    class _StewardTaskRegistry:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def create_or_return_existing(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(
                task_id="retry-steward",
                kind=kwargs.get("kind"),
                state="queued",
            ), False

    state = SharedState(session_id="t-route-infra", phase="EXPLORE")
    state.explore_search = {"cursor": 2}
    coord = Coordinator.__new__(Coordinator)
    coord.shared_state = state
    coord.session_dir = tmp_path
    coord.tasks = _StewardTaskRegistry()
    coord.knowledge_plane = None

    task = SimpleNamespace(task_id="task-infra-1", params={})
    payload = {
        "domain": "session_steward_specialist",
        "empty": True,
        "reason": "subprocess_stale_heartbeat",
        "summary": "subprocess_stale_heartbeat",
    }
    await coord._route_steward_verdict(task=task, done_payload=payload)
    assert state.stop_reason in ("", None)
    assert state.last_remaining_gaps_assessment["recommendation"] == ""
    assert state.steward_infra_failures_by_round == {"2": 1}
    assert len(coord.tasks.calls) == 1
    assert "retry1" in coord.tasks.calls[0]["idempotency_key"]


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
        "recommendation": "make_coffee",
        "rationale": "hallucinated verdict",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    assert not state.stop_reason
    assert not state.pending_escalate_hint
    assert state.last_remaining_gaps_assessment["recommendation"] == "stop_session"


@pytest.mark.asyncio
async def test_route_continue_explore_missing_gap_id_coerced():
    """``continue_explore`` without next_gap_canonical_id -> advance_to_kernel
    (still advisory; no phase hint set)."""
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
        "rationale": "vague continuation request",
    }
    await Coordinator._route_steward_verdict(
        fake, task=task, done_payload=payload,
    )
    assert state.last_remaining_gaps_assessment["recommendation"] == "advance_to_kernel"
    assert not state.pending_escalate_hint


# ===========================================================================
# Phase allowlist
# ===========================================================================
def test_assess_remaining_gaps_in_explore_allowlist():
    assert "assess_remaining_gaps" in phase_state.PHASE_ALLOWED_ACTIONS[
        phase_state.PHASE_EXPLORE
    ]


# ===========================================================================
# Coordinator no longer exposes the throttle / depth-gate helpers
# ===========================================================================
def test_throttle_and_depth_gate_helpers_removed():
    from inference_optimizer.orchestrator.coordinator import Coordinator
    for attr in (
        "_assess_remaining_gaps_throttle_denial",
        "_apply_depth_gate_to_verdict",
        "_depth_gate_thresholds",
    ):
        assert not hasattr(Coordinator, attr), (
            f"Coordinator.{attr} unexpectedly resurrected — loosen P2_13 "
            f"removed the steward throttle / depth gate."
        )
