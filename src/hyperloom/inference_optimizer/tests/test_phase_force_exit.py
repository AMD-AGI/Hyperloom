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

import pytest

from hyperloom.orchestrator.phases import machine_state as phase_state
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def _interleave_off():
    """These tests assert the strict non-interleave IR-6 thresholds."""
    assert phase_state.is_phase_interleave_enabled() is False


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
    # Seed plateau signals so plateau judgment paths don't interfere.
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
    # EXPLORE slice nearly exhausted (~5% left <= 20%) while session
    # remaining (4h) stays above the 3h threshold.
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
    # Evidence still populated for diagnostics.
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
        "session_remaining",
        "phase_remaining_pct",
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
    # Without max_minutes nothing is computable.
    assert "session_remaining_seconds" not in evidence


def test_exit_normal_explore_force_exit_takes_priority_over_plateau():
    """Force-exit must win even if plateau also has a verdict."""
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.6,
        phase_started_hours_ago=4.0,
    )
    # Seed plateau-shaped data that would otherwise fire.
    state.specialist_rounds = [{"round_id": i, "empty_streak": i + 1, "proposal_count": 0} for i in range(3)]
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
    assert target == phase_state.PHASE_KERNEL_AGENT
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
    # Kernel disabled: route through ``no_kernel_skipped``, original reason
    # passed through evidence.
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
    # With absurd thresholds, any in-progress session triggers.
    nxt = phase_state.compute_next_phase(state, kernel_enabled=True)
    assert nxt is not None
    target, reason, _ = nxt
    assert target == phase_state.PHASE_KERNEL_AGENT
    assert reason == "explore_force_exit_low_budget"


def test_interleave_env_no_longer_narrows_force_exit_hours(monkeypatch):
    """The retired interleave env no longer narrows the strict 3h gate."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    # 2.5h remaining; strict mode fires.
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.5,
        phase_started_hours_ago=4.0,
    )
    fired, evidence = phase_state.should_force_exit_explore(state)
    assert fired is True
    assert evidence["hours_remaining_threshold"] == phase_state.DEFAULT_EXPLORE_FORCE_EXIT_HOURS_REMAINING
    assert evidence["budget_pct_threshold"] == phase_state.DEFAULT_EXPLORE_FORCE_EXIT_BUDGET_PCT
    assert evidence["interleave_aware_ir6"] is False


def test_interleave_still_fires_inside_close_buffer(monkeypatch):
    """The retired interleave env does not disable the strict close buffer."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    # 0.5h remaining, so strict mode fires.
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=9.5,
        phase_started_hours_ago=6.0,
    )
    fired, evidence = phase_state.should_force_exit_explore(state)
    assert fired is True
    assert "session_remaining" in evidence["fired_reasons"]
    assert evidence["interleave_aware_ir6"] is False


def test_interleave_respects_explicit_override(monkeypatch):
    """An explicit non-default threshold from the caller still wins."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PHASE_INTERLEAVE", "1")
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=7.5,
        phase_started_hours_ago=4.0,
    )
    fired, evidence = phase_state.should_force_exit_explore(
        state,
        hours_remaining_threshold=3.0001,
    )
    assert fired is True  # 2.5h remaining < 3.0001h explicit threshold
    assert evidence["hours_remaining_threshold"] == 3.0001
    assert evidence["interleave_aware_ir6"] is False


def test_force_exit_phase_remaining_pct_uses_chargeback_denominator():
    """phase_remaining_pct numerator/denominator come from the same helper.

    On a short bounded run the EXPLORE budget is charge-back based (share of the
    session time still remaining), so the fraction must divide the charge-back
    remaining by the charge-back total — not by the legacy ``max_minutes*pct``.
    A freshly entered phase has remaining == total, i.e. a full 1.0 fraction.
    """
    now = time.time()
    state = _make_explore_state(
        max_minutes=600,
        started_hours_ago=2.0,        # 8h remain of a 10h session
        phase_started_hours_ago=0.0,  # EXPLORE just entered → 0 elapsed
    )
    total = phase_state._phase_budget_total_seconds(state, now_unix=now)
    remaining = phase_state.phase_budget_remaining_seconds(state, now_unix=now)
    assert total is not None and total > 0
    # Charge-back engaged (not the legacy whole-session*pct allotment).
    legacy = 600 * 60.0 * phase_state.DEFAULT_PHASE_BUDGET_PCT["EXPLORE"]
    assert total != pytest.approx(legacy)
    assert remaining == pytest.approx(total)

    _, evidence = phase_state.should_force_exit_explore(state, now_unix=now)
    assert evidence["phase_remaining_pct"] == pytest.approx(1.0)
