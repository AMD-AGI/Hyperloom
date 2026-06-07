# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""IR-6 — EXPLORE HARD force-exit gate tests.

Covers ``phase_state.should_force_exit_explore`` and its integration via
``exit_normal_explore`` / ``compute_next_phase``. The gate fires when:

* total session wall-clock remaining drops below
  ``force_exit_hours_remaining`` hours, OR
* EXPLORE phase remaining budget pct drops below
  ``force_exit_budget_pct``.

Either gate alone is sufficient. Both gates feed evidence into the
phase_history audit row regardless of which (or both) fired.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone


from inference_optimizer.orchestrator import phase_state
from inference_optimizer.orchestrator.shared_state import SharedState


def _make_explore_state(
    *,
    max_minutes: int,
    started_hours_ago: float,
    phase_started_hours_ago: float,
    phase_budget_pct: dict | None = None,
) -> SharedState:
    now = datetime.now(timezone.utc)
    start_ts = (now - timedelta(hours=started_hours_ago)).isoformat()
    phase_started_unix = time.time() - phase_started_hours_ago * 3600.0
    state = SharedState(
        session_id="test-force-exit",
        model_name="test",
        gpu_type="MI300X",
        tp=8,
        precision="fp8",
        conc=64,
        isl=1024,
        osl=1024,
        baseline_tput=1500.0,
        phase=phase_state.PHASE_EXPLORE,
        phase_started_ts=(now - timedelta(hours=phase_started_hours_ago)).isoformat(),
        phase_started_unix=phase_started_unix,
        start_ts=start_ts,
        max_minutes=max_minutes,
    )
    if phase_budget_pct is None:
        phase_budget_pct = dict(phase_state.DEFAULT_PHASE_BUDGET_PCT)
    state.phase_budget_pct = phase_budget_pct
    # Seed plateau signals so any later plateau judgment paths don't
    # interfere; we focus the test on the force-exit gate.
    state.explore_search = {
        "schema_version": 1,
        "tested": {},
        "accepted": [],
        "rejected": [],
        "winners_history": [{"round_id": 1, "gain_pct": 5.0}],
    }
    return state


def test_force_exit_total_remaining_below_threshold():
    """7.5h elapsed of a 10h budget -> 2.5h remaining < 3h threshold."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.5,
        phase_started_hours_ago=4.0,
    )
    fired, evidence = phase_state.should_force_exit_explore(
        state,
        hours_remaining_threshold=3.0,
        budget_pct_threshold=0.20,
    )
    assert fired is True
    assert "session_remaining" in evidence["fired_reasons"]
    assert evidence["session_remaining_seconds"] < 3 * 3600 + 60
    assert evidence["hours_remaining_threshold"] == 3.0


def test_force_exit_phase_pct_below_threshold():
    """Phase elapsed close to its slice; session_remaining still OK."""
    # 10h budget, EXPLORE pct=0.60 -> 6h slice. Elapsed 5.7h in EXPLORE ->
    # remaining=0.3h=5% of slice <= 20%. start_ts is 6h ago so session
    # remaining is 4h > 3h threshold.
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=6.0,
        phase_started_hours_ago=5.7,
    )
    fired, evidence = phase_state.should_force_exit_explore(
        state,
        hours_remaining_threshold=3.0,
        budget_pct_threshold=0.20,
    )
    assert fired is True
    assert "phase_remaining_pct" in evidence["fired_reasons"]
    assert evidence["phase_remaining_pct"] <= 0.20
    assert evidence["session_remaining_seconds"] > 3 * 3600


def test_force_exit_neither_trigger_fires():
    """Plenty of budget remaining: gate must NOT fire."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=1.0,
        phase_started_hours_ago=0.5,
    )
    fired, evidence = phase_state.should_force_exit_explore(
        state,
        hours_remaining_threshold=3.0,
        budget_pct_threshold=0.20,
    )
    assert fired is False
    assert evidence["fired_reasons"] == []
    # Evidence still populated for diagnostics so the prompt can show
    # the buffer.
    assert evidence["session_remaining_seconds"] > 3 * 3600
    assert evidence["phase_remaining_pct"] > 0.20


def test_force_exit_both_triggers_fire():
    """Both gates trigger; evidence lists both reasons."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.6,
        phase_started_hours_ago=5.7,
    )
    fired, evidence = phase_state.should_force_exit_explore(
        state,
        hours_remaining_threshold=3.0,
        budget_pct_threshold=0.20,
    )
    assert fired is True
    assert set(evidence["fired_reasons"]) >= {
        "session_remaining", "phase_remaining_pct",
    }


def test_force_exit_unlimited_run_never_fires():
    """max_minutes=0 -> unlimited; gate cannot fire on session_remaining."""
    state = _make_explore_state(
        max_minutes=0,
        started_hours_ago=100.0,
        phase_started_hours_ago=100.0,
    )
    fired, evidence = phase_state.should_force_exit_explore(state)
    assert fired is False
    # Without max_minutes neither phase_remaining nor session_remaining
    # are computable.
    assert "session_remaining_seconds" not in evidence


def test_exit_normal_explore_force_exit_takes_priority_over_plateau():
    """Force-exit must win even if plateau also has a verdict."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.6,
        phase_started_hours_ago=4.0,
    )
    # Seed enough plateau-shaped data that a plateau judgment would
    # otherwise fire (3 empty rounds + low cumulative gain).
    state.specialist_rounds = [
        {"round_id": i, "empty_streak": i + 1, "proposal_count": 0}
        for i in range(3)
    ]
    result = phase_state.exit_normal_explore(state)
    assert result is not None
    reason, evidence = result
    assert reason == "explore_force_exit_low_budget"
    assert evidence["evidence"] == "force_exit"


def test_compute_next_phase_routes_to_kernel_with_kernel_enabled():
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.6,
        phase_started_hours_ago=4.0,
    )
    state.kernel_enabled = True
    nxt = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == phase_state.PHASE_KERNEL
    assert reason == "explore_force_exit_low_budget"


def test_compute_next_phase_routes_to_sweep_when_kernel_disabled():
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.6,
        phase_started_hours_ago=4.0,
    )
    state.kernel_enabled = False
    nxt = phase_state.compute_next_phase(state, kernel_enabled=False)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == phase_state.PHASE_SWEEP
    # When kernel is disabled we route through ``no_kernel_skipped`` and
    # pass the original reason through evidence.
    assert reason == "no_kernel_skipped"
    assert evidence.get("passed_through_reason") == "explore_force_exit_low_budget"


def test_explore_force_exit_low_budget_is_in_vocabularies():
    assert "explore_force_exit_low_budget" in phase_state.PHASE_EXIT_REASONS
    assert "explore_force_exit_low_budget" in phase_state.STOP_REASON_VOCAB


def test_force_exit_thresholds_routed_through_overrides():
    """CLI overrides at ``state.plateau_overrides`` get picked up."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=2.0,
        phase_started_hours_ago=1.0,
    )
    state.plateau_overrides = {
        "force_exit_hours_remaining": 9.0,
        "force_exit_budget_pct": 0.95,
    }
    # With absurd thresholds (9h remaining required, 95% pct floor),
    # ANY in-progress session triggers.
    nxt = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert nxt is not None
    target, reason, _ = nxt
    assert target == phase_state.PHASE_KERNEL
    assert reason == "explore_force_exit_low_budget"
