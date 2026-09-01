# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance tests for the long-run optimization refinements.

Covers the decaying acceptance curve, decaying-gain convergence, the absolute
per-phase wall-clock cap (incl. the unbounded 14-day ceiling), the FRAMEWORK
reloop target, and the trailing-window crash-rate emergency stop.

All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import SharedState


# Decaying acceptance curve: threshold(N) = 0.1 + 0.9 / N (N = macro_cycle + 1)
@pytest.mark.parametrize(
    "macro_cycle, expected",
    [(0, 1.00), (1, 0.55), (2, 0.40), (4, 0.28), (9, 0.19)],
)
def test_decaying_keep_threshold_curve(macro_cycle, expected):
    assert ps.decaying_keep_threshold_pct(macro_cycle) == pytest.approx(expected, abs=1e-9)


def test_decaying_keep_threshold_floor():
    assert ps.decaying_keep_threshold_pct(10_000) == pytest.approx(0.1, abs=1e-3)
    assert ps.decaying_keep_threshold_pct(10_000) > 0.1


def test_decaying_keep_threshold_multi_node_scales_by_two():
    for n in (0, 1, 4):
        single = ps.decaying_keep_threshold_pct(n)
        multi = ps.decaying_keep_threshold_pct(n, multi_node=True)
        assert multi == pytest.approx(2.0 * single)
    assert ps.decaying_keep_threshold_pct(0, multi_node=True) == pytest.approx(2.0)


# Decaying-gain convergence: a cycle only "gains" when it clears its own bar
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


def test_subthreshold_gain_does_not_reset_streak():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.2, no_gain_streak=1)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert ev["min_gain_pct"] == pytest.approx(0.40)
    assert ev["cycle_gained"] is False
    assert ev["no_gain_cycle_streak_effective"] == 2
    assert reloop is True


def test_three_subthreshold_cycles_converge():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.1, no_gain_streak=2)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "global_converged"


def test_suprathreshold_gain_resets_streak():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.5, no_gain_streak=2)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert ev["cycle_gained"] is True
    assert ev["no_gain_cycle_streak_effective"] == 0
    assert reloop is True


def test_all_saturated_directions_stop_reloop():
    st = _sweep_state(macro_cycle=2, cycle_delta=1.0, no_gain_streak=0)
    st.saturated_directions = {
        "kernel_switch_specialist": {"saturated": True},
        "comm_specialist": {"saturated": True},
    }
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "all_directions_saturated"


def test_saturation_convergence_is_always_enabled():
    st = _sweep_state(macro_cycle=2, cycle_delta=1.0, no_gain_streak=0)
    st.saturated_directions = {"kernel_switch_specialist": {"saturated": True}}
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "all_directions_saturated"


# Absolute per-phase cap + 14-day ceiling for unbounded runs
def test_phase_cap_binds_on_session_term_for_short_runs():
    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]
    st = SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=120)
    cap = ps.phase_cap_seconds(st)
    assert cap == pytest.approx(120 * 60 * pct)


def test_phase_cap_binds_on_24h_reference_for_unbounded_runs():
    import math

    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]
    st = SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=0)
    cap = ps.phase_cap_seconds(st)
    assert cap == pytest.approx(math.ceil(24 * 60 * pct) * 60)


def test_effective_max_minutes_unbounded_is_14_days():
    st = SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=0)
    assert ps.effective_max_minutes(st) == ps.DEFAULT_LONGRUN_MAX_MINUTES
    assert ps.DEFAULT_LONGRUN_MAX_MINUTES == 14 * 24 * 60


def test_unbounded_explore_exits_when_cap_exceeded():
    now = 1_000_000.0
    cap = ps.phase_cap_seconds(SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=0))
    st = SharedState(
        phase=ps.PHASE_FRAMEWORK_AGENT,
        max_minutes=0,
        phase_started_unix=now - (cap + 10),
        phase_budget_pct=dict(ps.DEFAULT_PHASE_BUDGET_PCT),
    )
    out = ps.exit_normal_optimize(st, now_unix=now)
    assert out is not None
    assert out[0] == "optimize_budget_cap"


def test_bounded_explore_does_not_hit_absolute_cap():
    now = 1_000_000.0
    st = SharedState(
        phase=ps.PHASE_FRAMEWORK_AGENT,
        max_minutes=600,
        phase_started_unix=now - 60,
        phase_budget_pct=dict(ps.DEFAULT_PHASE_BUDGET_PCT),
    )
    assert ps.exit_normal_optimize(st, now_unix=now) is None


# Vocab: renamed reasons are phase-exit only, never terminal stop reasons
def test_renamed_leverage_reasons_are_phase_exit_not_stop_reason():
    assert ps.is_valid_phase_exit_reason("optimize_no_more_leverage")
    assert ps.is_valid_phase_exit_reason("kernel_no_more_leverage")
    assert not ps.is_valid_stop_reason("no_more_leverage")
    assert not ps.is_valid_stop_reason("optimize_no_more_leverage")


# Trailing-window crash rate
def test_recent_crash_count_ages_out_old_crashes():
    st = SharedState(session_id="t")
    now = 1_000_000.0
    st.crash_timestamps = [now - 25 * 3600 for _ in range(24)] + [now - 60 for _ in range(24)]
    st.crash_count = 48
    assert st.recent_crash_count(window_sec=24 * 3600, now=now) == 24


def test_increment_crash_count_records_timestamp_and_caps():
    st = SharedState(session_id="t")
    for _ in range(5):
        st.increment_crash_count()
    assert st.crash_count == 5
    assert len(st.crash_timestamps) == 5
    assert st.recent_crash_count(window_sec=24 * 3600) == 5
