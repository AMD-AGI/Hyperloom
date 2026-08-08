# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-phase budgets are a share of the RUN, not a share of every entry.

``phase_started_unix`` is reset on every phase entry, so a budget guard reading
it alone measures only the current entry. Because the pipeline is a macro-cycle
(FRAMEWORK_AGENT -> EXPLORE -> KERNEL_AGENT -> SWEEP -> reloop), each phase is
re-entered once per cycle and used to be handed its whole allotment again on
every entry. A real 24h session entered KERNEL_AGENT three times, burned a fresh
3.6h each time, and finished at 288% of a cap that never fired.

These tests pin the fixed contract: ``phase_cumulative_seconds`` totals every
entry, the cap/budget guards read that total, and ``phase_elapsed_seconds``
keeps its per-entry meaning for renderers and evidence dicts.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import SharedState


# A 24h run with KERNEL_AGENT capped at 15%: cap = 1440 * 60 * 0.15 = 12960s.
SESSION_MINUTES = 24 * 60
KERNEL_PCT = 0.15
KERNEL_CAP_SEC = 12_960.0
# Three entries of 3h each: every single entry stays under the cap, the sum does
# not (32400s = 250% of the cap).
ENTRY_SEC = 3 * 3600.0
# Interleaved SWEEP segment, so KERNEL's total can never be confused with plain
# wall-clock elapsed since its first entry.
GAP_SEC = 600.0
# A real epoch: zero reads as "phase not entered yet" throughout the machine.
T0 = 1_700_000_000.0
T0_ISO = "2023-11-14T22:13:20+00:00"


def _kernel_state() -> SharedState:
    state = SharedState()
    state.max_minutes = SESSION_MINUTES
    state.phase_budget_pct = {ps.PHASE_KERNEL_AGENT: KERNEL_PCT}
    return state


def _enter(state: SharedState, phase: str, at_unix: float) -> None:
    state.record_phase_transition(
        to_phase=phase,
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=at_unix,
    )


def _three_kernel_entries(state: SharedState) -> float:
    """Run KERNEL -> SWEEP three times; return the unix time after the third exit."""
    now = T0
    for _ in range(3):
        _enter(state, ps.PHASE_KERNEL_AGENT, now)
        now += ENTRY_SEC
        _enter(state, ps.PHASE_SWEEP, now)
        now += GAP_SEC
    return now


def test_three_entries_under_per_entry_cap_exceed_the_cumulative_cap():
    state = _kernel_state()
    now = T0
    for entry in range(3):
        _enter(state, ps.PHASE_KERNEL_AGENT, now)
        exit_at = now + ENTRY_SEC
        # Each individual entry stays comfortably under the absolute cap, which
        # is exactly why per-entry accounting never fired.
        assert ps.phase_elapsed_seconds(state, now_unix=exit_at) == ENTRY_SEC
        assert ENTRY_SEC < KERNEL_CAP_SEC
        assert ps.phase_cumulative_seconds(state, now_unix=exit_at) == pytest.approx(
            ENTRY_SEC * (entry + 1)
        )
        _enter(state, ps.PHASE_SWEEP, exit_at)
        now = exit_at + GAP_SEC

    assert ps.phase_cumulative_seconds(
        state,
        phase=ps.PHASE_KERNEL_AGENT,
        now_unix=now,
    ) == pytest.approx(3 * ENTRY_SEC)

    # Re-enter KERNEL: the cap must fire immediately, because the phase has
    # already spent 250% of its share of the run.
    _enter(state, ps.PHASE_KERNEL_AGENT, now)
    assert ps.phase_cap_seconds(state) == KERNEL_CAP_SEC
    assert ps.phase_elapsed_seconds(state, now_unix=now) == 0.0
    assert ps.phase_cap_exceeded(state, now_unix=now) is True


def test_per_entry_elapsed_keeps_its_meaning():
    # phase_elapsed_seconds feeds renderers and "time in the current phase"
    # evidence; the cumulative fix must not change what it reports.
    state = _kernel_state()
    now = _three_kernel_entries(state)
    _enter(state, ps.PHASE_KERNEL_AGENT, now)
    live = now + 1800.0
    assert ps.phase_elapsed_seconds(state, now_unix=live) == 1800.0
    assert ps.phase_cumulative_seconds(state, now_unix=live) == pytest.approx(
        3 * ENTRY_SEC + 1800.0
    )


def test_phase_budget_remaining_charges_every_entry():
    state = _kernel_state()
    state.start_ts = T0_ISO
    now = _three_kernel_entries(state)
    _enter(state, ps.PHASE_KERNEL_AGENT, now)

    total = ps._phase_budget_total_seconds(state, now_unix=now)
    remaining = ps.phase_budget_remaining_seconds(state, now_unix=now)
    assert total is not None and total > 0.0
    # The per-entry numerator this used to subtract is zero at a fresh entry, so
    # the old contract handed the phase its whole allotment back on re-entry.
    assert ps.phase_elapsed_seconds(state, now_unix=now) == 0.0
    assert remaining == pytest.approx(total - 3 * ENTRY_SEC)
    assert remaining < total


def test_phase_budget_remaining_hits_zero_once_the_share_is_spent():
    state = _kernel_state()
    state.start_ts = T0_ISO
    long_entry = 4 * 3600.0
    now = T0
    for _ in range(3):
        _enter(state, ps.PHASE_KERNEL_AGENT, now)
        now += long_entry
        _enter(state, ps.PHASE_SWEEP, now)
        now += GAP_SEC
    _enter(state, ps.PHASE_KERNEL_AGENT, now)

    assert ps.phase_budget_remaining_seconds(state, now_unix=now) == 0.0


def test_totals_accumulate_for_every_phase_not_just_explore():
    state = SharedState()
    _enter(state, ps.PHASE_EXPLORE, T0)
    _enter(state, ps.PHASE_KERNEL_AGENT, T0 + 100.0)
    _enter(state, ps.PHASE_SWEEP, T0 + 250.0)
    _enter(state, ps.PHASE_EXPLORE, T0 + 300.0)

    assert state.phase_elapsed_totals == {
        ps.PHASE_EXPLORE: 100.0,
        ps.PHASE_KERNEL_AGENT: 150.0,
        ps.PHASE_SWEEP: 50.0,
    }
    # EXPLORE's dedicated accumulator still agrees; it is kept because it
    # carries a tri-state "unknown" for legacy resumes that the budget totals
    # deliberately do not have.
    assert state.explore_elapsed_accum_s == 100.0


def test_fresh_state_has_no_banked_totals():
    state = SharedState()
    assert state.phase_elapsed_totals == {}
    assert ps.phase_cumulative_seconds(state) == 0.0


def test_resume_without_totals_rebuilds_them_from_phase_history():
    # A state written before phase_elapsed_totals existed must not silently
    # re-arm the per-entry bug for the rest of the resumed run.
    source = _kernel_state()
    now = _three_kernel_entries(source)
    _enter(source, ps.PHASE_KERNEL_AGENT, now)

    raw = source.to_dict()
    raw.pop("phase_elapsed_totals")
    resumed = SharedState.from_dict(raw)

    # The trailing (still-active) segment is excluded: phase_cumulative_seconds
    # adds the live segment itself.
    assert resumed.phase_elapsed_totals[ps.PHASE_KERNEL_AGENT] == pytest.approx(3 * ENTRY_SEC)
    assert resumed.phase_elapsed_totals[ps.PHASE_SWEEP] == pytest.approx(3 * GAP_SEC)
    assert ps.phase_cap_exceeded(resumed, now_unix=now) is True


def test_history_rebuild_excludes_the_active_segment():
    # Rows the cap has evicted are simply absent, and the trailing row is the
    # live segment; the rebuild must under-count rather than extrapolate,
    # because over-charging would end a phase early.
    history = [
        {"to_phase": "KERNEL_AGENT", "ts_unix": T0},
        {"to_phase": "SWEEP", "ts_unix": T0 + 300.0},
        {"to_phase": "KERNEL_AGENT", "ts_unix": T0 + 400.0},
    ]
    assert ps.phase_elapsed_totals_from_history(history) == {
        "KERNEL_AGENT": 300.0,
        "SWEEP": 100.0,
    }


def test_history_rebuild_skips_unusable_rows():
    history = [
        # No phase name.
        {"to_phase": "", "ts_unix": T0},
        # Unset entry timestamp reads as "never entered".
        {"to_phase": "KERNEL_AGENT", "ts_unix": 0.0},
        {"to_phase": "EXPLORE", "ts_unix": T0 + 100.0},
        # Non-advancing timestamp (clock skew) contributes nothing.
        {"to_phase": "SWEEP", "ts_unix": T0 + 100.0},
        {"to_phase": "CLOSE", "ts_unix": T0 + 900.0},
    ]
    assert ps.phase_elapsed_totals_from_history(history) == {"SWEEP": 800.0}
    assert ps.phase_elapsed_totals_from_history(None) == {}
    assert ps.phase_elapsed_totals_from_history("nope") == {}
