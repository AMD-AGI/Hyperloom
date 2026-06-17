# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Acceptance tests for the long-run optimization refinements.

Covers the decaying acceptance curve, decaying-gain convergence, the absolute
per-phase wall-clock cap (incl. the unbounded 14-day ceiling), the FRAMEWORK_PR
reloop target, and the trailing-window crash-rate emergency stop.

All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from inference_optimizer.orchestrator import phase_state as ps
from inference_optimizer.orchestrator.shared_state import SharedState


CYCLIC_ENV = "INFERENCE_OPTIMIZER_CYCLIC_PHASES"


# ==========================================================================
# Decaying acceptance curve: threshold(N) = 0.1 + 0.9 / N (N = macro_cycle + 1)
# ==========================================================================
@pytest.mark.parametrize(
    "macro_cycle, expected",
    [(0, 1.00), (1, 0.55), (2, 0.40), (4, 0.28), (9, 0.19)],
)
def test_decaying_keep_threshold_curve(macro_cycle, expected):
    assert ps.decaying_keep_threshold_pct(macro_cycle) == pytest.approx(expected, abs=1e-9)


def test_decaying_keep_threshold_floor():
    # As N → ∞ the curve approaches the 0.1% floor from above.
    assert ps.decaying_keep_threshold_pct(10_000) == pytest.approx(0.1, abs=1e-3)
    assert ps.decaying_keep_threshold_pct(10_000) > 0.1


def test_decaying_keep_threshold_multi_node_scales_by_two():
    for n in (0, 1, 4):
        single = ps.decaying_keep_threshold_pct(n)
        multi = ps.decaying_keep_threshold_pct(n, multi_node=True)
        assert multi == pytest.approx(2.0 * single)
    # N=1 multi-node reproduces the legacy 2.0% baseline.
    assert ps.decaying_keep_threshold_pct(0, multi_node=True) == pytest.approx(2.0)


# ==========================================================================
# Decaying-gain convergence: a cycle only "gains" when it clears its own bar
# ==========================================================================
def _sweep_state(*, macro_cycle, cycle_delta, no_gain_streak):
    now = datetime.now(timezone.utc)
    st = SharedState(
        session_id="t",
        phase=ps.PHASE_SWEEP,
        start_ts=(now - timedelta(hours=1)).isoformat(),
        max_minutes=96 * 60,
        macro_cycle=macro_cycle,
        gain_at_cycle_start=5.0,
        cumulative_gain_validated=5.0 + cycle_delta,
        no_gain_cycle_streak=no_gain_streak,
    )
    return st


def test_subthreshold_gain_does_not_reset_streak(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    # cycle 3 bar = 0.40%; a 0.2% delta is below it → counts as no-gain.
    st = _sweep_state(macro_cycle=2, cycle_delta=0.2, no_gain_streak=1)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert ev["min_gain_pct"] == pytest.approx(0.40)
    assert ev["cycle_gained"] is False
    assert ev["no_gain_cycle_streak_effective"] == 2
    assert reloop is True  # streak 2 < 3, still loops


def test_three_subthreshold_cycles_converge(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    st = _sweep_state(macro_cycle=2, cycle_delta=0.1, no_gain_streak=2)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "global_converged"


def test_suprathreshold_gain_resets_streak(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    # cycle 3 bar = 0.40%; a 0.5% delta clears it → streak resets to 0.
    st = _sweep_state(macro_cycle=2, cycle_delta=0.5, no_gain_streak=2)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert ev["cycle_gained"] is True
    assert ev["no_gain_cycle_streak_effective"] == 0
    assert reloop is True


# ==========================================================================
# Absolute per-phase cap + 14-day ceiling for unbounded runs
# ==========================================================================
def test_phase_cap_binds_on_session_term_for_short_runs():
    # 2h bounded run: proportional term (2h*0.45) < 24h*0.45 cap → proportional.
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=120)
    cap = ps.phase_cap_seconds(st)
    assert cap == pytest.approx(120 * 60 * 0.45)


def test_phase_cap_binds_on_24h_reference_for_unbounded_runs():
    import math
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=0)
    cap = ps.phase_cap_seconds(st)
    # 24h * 0.45 (ceil to minutes) is far below the 14-day proportional term.
    assert cap == pytest.approx(math.ceil(24 * 60 * 0.45) * 60)


def test_effective_max_minutes_unbounded_is_14_days():
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=0)
    assert ps.effective_max_minutes(st) == ps.DEFAULT_LONGRUN_MAX_MINUTES
    assert ps.DEFAULT_LONGRUN_MAX_MINUTES == 14 * 24 * 60


def test_unbounded_explore_exits_when_cap_exceeded(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "0")
    now = 1_000_000.0
    # Started just over the 24h*0.45 cap ago, unbounded run.
    cap = ps.phase_cap_seconds(
        SharedState(phase=ps.PHASE_EXPLORE, max_minutes=0)
    )
    st = SharedState(
        phase=ps.PHASE_EXPLORE,
        max_minutes=0,
        phase_started_unix=now - (cap + 10),
    )
    out = ps.exit_normal_explore(st, now_unix=now)
    assert out is not None
    assert out[0] == "explore_budget_cap"


def test_bounded_explore_does_not_hit_absolute_cap(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "0")
    now = 1_000_000.0
    # 10h bounded run, 1 min into EXPLORE → well under the phase budget and the
    # absolute cap, and above the 3.0h force-exit wall-clock buffer (the session
    # auto-stamps start_ts at construction, so session_remaining ≈ max_minutes).
    st = SharedState(
        phase=ps.PHASE_EXPLORE,
        max_minutes=600,
        phase_started_unix=now - 60,
        phase_budget_pct=dict(ps.DEFAULT_PHASE_BUDGET_PCT),
    )
    assert ps.exit_normal_explore(st, now_unix=now) is None


# ==========================================================================
# Vocab: renamed reasons are phase-exit only, never terminal stop reasons
# ==========================================================================
def test_renamed_leverage_reasons_are_phase_exit_not_stop_reason():
    assert ps.is_valid_phase_exit_reason("explore_no_more_leverage")
    assert ps.is_valid_phase_exit_reason("kernel_no_more_leverage")
    assert not ps.is_valid_stop_reason("no_more_leverage")
    assert not ps.is_valid_stop_reason("explore_no_more_leverage")


# ==========================================================================
# Trailing-window crash rate
# ==========================================================================
def test_recent_crash_count_ages_out_old_crashes():
    st = SharedState(session_id="t")
    now = 1_000_000.0
    # 24 crashes inside the window, 24 well outside it.
    st.crash_timestamps = (
        [now - 25 * 3600 for _ in range(24)]
        + [now - 60 for _ in range(24)]
    )
    st.crash_count = 48
    assert st.recent_crash_count(window_sec=24 * 3600, now=now) == 24


def test_increment_crash_count_records_timestamp_and_caps():
    st = SharedState(session_id="t")
    for _ in range(5):
        st.increment_crash_count()
    assert st.crash_count == 5
    assert len(st.crash_timestamps) == 5
    assert st.recent_crash_count(window_sec=24 * 3600) == 5
