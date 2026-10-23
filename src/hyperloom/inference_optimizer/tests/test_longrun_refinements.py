# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Acceptance tests for the long-run optimization refinements."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.inference_optimizer.breakdown.stop_reasons import is_valid_stop_reason
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
    return SharedState(
        session_id="t",
        phase=ps.PHASE_SWEEP,
        start_ts=(now - timedelta(hours=1)).isoformat(),
        max_minutes=96 * 60,
        macro_cycle=macro_cycle,
        gain_at_cycle_start=5.0,
        cumulative_gain_validated=5.0 + cycle_delta,
        no_gain_cycle_streak=no_gain_streak,
    )


def test_subthreshold_gain_does_not_reset_streak():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.2, no_gain_streak=1)
    reloop, ev = ps._reloop_decision(ps.workflow_predicate_inputs(st))
    assert ev["min_gain_pct"] == pytest.approx(0.40)
    assert ev["cycle_gained"] is False
    assert ev["no_gain_cycle_streak_effective"] == 2
    assert reloop is True


def test_three_subthreshold_cycles_converge():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.1, no_gain_streak=2)
    reloop, ev = ps._reloop_decision(ps.workflow_predicate_inputs(st))
    assert reloop is False
    assert ev["reloop_blocked"] == "global_converged"


def test_suprathreshold_gain_resets_streak():
    st = _sweep_state(macro_cycle=2, cycle_delta=0.5, no_gain_streak=2)
    reloop, ev = ps._reloop_decision(ps.workflow_predicate_inputs(st))
    assert ev["cycle_gained"] is True
    assert ev["no_gain_cycle_streak_effective"] == 0
    assert reloop is True


def test_all_saturated_directions_stop_reloop():
    st = _sweep_state(macro_cycle=2, cycle_delta=1.0, no_gain_streak=0)
    st.saturated_directions = {
        "kernel_switch_specialist": {"saturated": True},
        "comm_specialist": {"saturated": True},
    }
    reloop, ev = ps._reloop_decision(ps.workflow_predicate_inputs(st))
    assert reloop is False
    assert ev["reloop_blocked"] == "all_directions_saturated"

    st.last_conc_sweep = {"status": "succeeded", "summary": {"successful_pairs": 1}}
    target, reason, evidence = ps.compute_next_phase(st)
    assert (target, reason) == (ps.PHASE_CLOSE, "global_converged")
    assert evidence["terminal"] is True
    assert evidence["saturated_directions"] == ["comm_specialist", "kernel_switch_specialist"]


def test_saturation_convergence_is_always_enabled():
    st = _sweep_state(macro_cycle=2, cycle_delta=1.0, no_gain_streak=0)
    st.saturated_directions = {"kernel_switch_specialist": {"saturated": True}}
    reloop, ev = ps._reloop_decision(ps.workflow_predicate_inputs(st))
    assert reloop is False
    assert ev["reloop_blocked"] == "all_directions_saturated"


# Vocab: the leverage reasons close a phase, they never stop the run
def test_leverage_reasons_are_not_stop_reasons():
    assert not is_valid_stop_reason("no_more_leverage")
    assert not is_valid_stop_reason("optimize_no_more_leverage")
    assert not is_valid_stop_reason("kernel_no_more_leverage")


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
