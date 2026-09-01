# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-phase time accounting across macro-cycle re-entries.

``phase_started_unix`` is reset on every phase entry, so a guard reading it alone
measures only the current entry. Because the pipeline reloops (FRAMEWORK_AGENT ->
EXPLORE -> KERNEL_AGENT -> SWEEP -> reloop), each phase is re-entered once per
cycle. A real 24h session entered KERNEL_AGENT three times, burned a fresh 3.6h
each time, and finished at 288% of a cap that never fired.

These tests pin which guard reads which clock: the absolute cap
(``phase_cap_exceeded``) totals every entry, while the per-cycle budget
(``phase_budget_remaining_seconds``) charges only the current one — its allotment
is already charge-back-reduced by earlier entries, so billing the total twice
starved every re-entry.

They also pin the resume half of it: a phase entry is not closed by the process
exiting, so the current entry spans the idle gap between two run legs unless the
leg boundary floors it.
"""

from __future__ import annotations

from datetime import datetime, timezone

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

# A 3h session that stopped half an hour into PRELUDE and was resumed 3 days
# later: long enough that charging the gap to PRELUDE dwarfs the whole budget.
RESUMED_SESSION_MINUTES = 180
PRELUDE_LEG_SEC = 1800.0
IDLE_GAP_SEC = 3 * 24 * 3600.0
RESUME_UNIX = T0 + PRELUDE_LEG_SEC + IDLE_GAP_SEC
# What the resumed leg has run by the time the guards are asked.
NEW_LEG_SEC = 120.0


def _iso(at_unix: float) -> str:
    return datetime.fromtimestamp(at_unix, tz=timezone.utc).isoformat()


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
        assert ps.phase_cumulative_seconds(state, now_unix=exit_at) == pytest.approx(ENTRY_SEC * (entry + 1))
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
    assert ps.phase_cumulative_seconds(state, now_unix=live) == pytest.approx(3 * ENTRY_SEC + 1800.0)


def test_phase_budget_remaining_charges_only_the_current_entry():
    state = _kernel_state()
    state.start_ts = T0_ISO
    now = _three_kernel_entries(state)
    _enter(state, ps.PHASE_KERNEL_AGENT, now)

    total = ps._phase_budget_total_seconds(state, now_unix=now)
    assert total is not None and total > 0.0
    # Charge-back already pays for the three banked entries via a smaller
    # `total`, not via a second subtraction on top of it.
    assert ps.phase_elapsed_seconds(state, now_unix=now) == 0.0
    assert ps.phase_budget_remaining_seconds(state, now_unix=now) == pytest.approx(total)

    mid = now + 900.0
    assert ps.phase_budget_remaining_seconds(state, now_unix=mid) == pytest.approx(
        ps._phase_budget_total_seconds(state, now_unix=mid) - 900.0
    )


def test_the_absolute_cap_stops_a_phase_that_outspent_its_share():
    # With charge-back active the per-cycle budget no longer ends such a phase,
    # so the cap has to.
    state = _kernel_state()
    state.start_ts = T0_ISO
    now = T0
    for _ in range(3):
        _enter(state, ps.PHASE_KERNEL_AGENT, now)
        now += 4 * 3600.0
        _enter(state, ps.PHASE_SWEEP, now)
        now += GAP_SEC
    _enter(state, ps.PHASE_KERNEL_AGENT, now)

    assert ps.phase_cumulative_seconds(state, now_unix=now) > KERNEL_CAP_SEC
    assert ps.phase_cap_exceeded(state, now_unix=now) is True
    assert ps.exit_normal_kernel(state, now_unix=now) is not None


def test_a_macro_cycle_reentry_gets_budget_while_the_session_has_time_left():
    # An 18h --no-kernel session: FRAMEWORK_AGENT spent 27567s in cycle 0, then
    # cycle_reloop re-entered it with 27584s of session left and the 51840s cap
    # far away. Charging the cumulative total returned 0, so the dispatcher
    # paused and the phase exited in ~60s on each of the next two cycles.
    state = SharedState()
    state.max_minutes = 18 * 60.0
    state.start_ts = T0_ISO
    state.phase_budget_pct = {
        ps.PHASE_PRELUDE: 0.03,
        ps.PHASE_FRAMEWORK_AGENT: 0.80,
        ps.PHASE_KERNEL_AGENT: 0.0,
        ps.PHASE_SWEEP: 0.10555555555555557,
        ps.PHASE_CLOSE: 0.02,
    }

    now = T0
    _enter(state, ps.PHASE_PRELUDE, now)
    now += 9579.0
    _enter(state, ps.PHASE_FRAMEWORK_AGENT, now)
    now += 27567.0
    _enter(state, ps.PHASE_SWEEP, now)
    now += 67.0
    _enter(state, ps.PHASE_FRAMEWORK_AGENT, now)

    assert ps.session_remaining_seconds(state, now_unix=now) > 7 * 3600.0
    assert ps.phase_cap_exceeded(state, now_unix=now) is False
    assert ps.phase_budget_remaining_seconds(state, now_unix=now) > 6 * 3600.0
    assert ps.exit_normal_optimize(state, now_unix=now) is None


def test_totals_accumulate_for_every_phase_not_just_explore():
    state = SharedState()
    _enter(state, ps.PHASE_FRAMEWORK_AGENT, T0)
    _enter(state, ps.PHASE_KERNEL_AGENT, T0 + 100.0)
    _enter(state, ps.PHASE_SWEEP, T0 + 250.0)
    _enter(state, ps.PHASE_FRAMEWORK_AGENT, T0 + 300.0)

    assert state.phase_elapsed_totals == {
        ps.PHASE_FRAMEWORK_AGENT: 100.0,
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


def _resumed_prelude_state() -> SharedState:
    """A session still in PRELUDE whose previous leg stopped 3 days ago."""
    state = SharedState()
    state.max_minutes = RESUMED_SESSION_MINUTES
    state.phase = ps.PHASE_PRELUDE
    state.phase_started_unix = T0
    state.phase_started_ts = T0_ISO
    state.resumed_ts = _iso(RESUME_UNIX)
    return state


def test_a_resume_does_not_charge_the_phase_for_the_gap_it_was_not_running():
    # phase_started_unix is only rewritten on a phase transition, and exiting
    # the process is not one, so the current entry spans both legs.
    state = _resumed_prelude_state()
    now = RESUME_UNIX + NEW_LEG_SEC

    assert ps.phase_elapsed_seconds(state, now_unix=now) == pytest.approx(NEW_LEG_SEC)
    assert ps.phase_cumulative_seconds(state, now_unix=now) == pytest.approx(NEW_LEG_SEC)


def test_a_resumed_phase_is_not_capped_by_time_the_process_was_down():
    state = _resumed_prelude_state()
    state.phase_budget_pct = {ps.PHASE_PRELUDE: 0.4}
    now = RESUME_UNIX + NEW_LEG_SEC

    assert ps.phase_cap_seconds(state) == pytest.approx(RESUMED_SESSION_MINUTES * 60.0 * 0.4)
    assert ps.phase_cap_exceeded(state, now_unix=now) is False


def test_the_charge_back_base_of_a_resumed_phase_stays_inside_the_session():
    # The crash / recorded-stop branch re-anchors start_ts, so the session clock
    # restarts. A phase clock that still spans the gap reconstructs a base of
    # "the whole budget plus three days" and hands the phase an allotment the
    # session cannot pay for.
    state = _resumed_prelude_state()
    state.start_ts = state.resumed_ts
    now = RESUME_UNIX + NEW_LEG_SEC

    budget = ps.normalize_budget_pct(None)
    denom = sum(pct for pct in budget.values() if pct > 0.0)
    session_sec = RESUMED_SESSION_MINUTES * 60.0
    total = ps._phase_budget_total_seconds(state, now_unix=now)

    assert total == pytest.approx(session_sec * budget[ps.PHASE_PRELUDE] / denom)


def test_a_kept_budget_anchor_charges_the_resumed_phase_the_smaller_base():
    # The clean-stop branch keeps start_ts, so the session stays charged for the
    # idle gap while the phase clock no longer is. The base is then what the
    # session had left when THIS leg began, not when the phase was entered.
    idle_gap = 3600.0
    state = _resumed_prelude_state()
    state.start_ts = T0_ISO
    state.resumed_ts = _iso(T0 + PRELUDE_LEG_SEC + idle_gap)
    now = T0 + PRELUDE_LEG_SEC + idle_gap + NEW_LEG_SEC

    budget = ps.normalize_budget_pct(None)
    denom = sum(pct for pct in budget.values() if pct > 0.0)
    base = RESUMED_SESSION_MINUTES * 60.0 - (PRELUDE_LEG_SEC + idle_gap)
    total = ps._phase_budget_total_seconds(state, now_unix=now)

    assert total == pytest.approx(base * budget[ps.PHASE_PRELUDE] / denom)


def test_a_later_phase_entry_supersedes_the_resume_boundary():
    # The leg boundary only floors the entry it interrupted; once the phase is
    # re-entered its own stamp is the later of the two.
    state = _resumed_prelude_state()
    _enter(state, ps.PHASE_FRAMEWORK_AGENT, RESUME_UNIX + NEW_LEG_SEC)

    assert state.phase_elapsed_totals[ps.PHASE_PRELUDE] == pytest.approx(NEW_LEG_SEC)
    assert ps.phase_elapsed_seconds(
        state,
        now_unix=RESUME_UNIX + NEW_LEG_SEC + 180.0,
    ) == pytest.approx(180.0)


def test_budget_exit_evidence_reports_the_time_it_judged_on():
    """A cap decided on cumulative time must not be evidenced by one entry's clock.

    The guards moved to cumulative accounting; the phase_history evidence kept
    writing ``phase_elapsed_seconds``. On a re-entered phase that reads as a
    contradiction — a row claiming the budget is spent while showing a few
    minutes elapsed — and phase_history is the record a stalled run is
    reconstructed from.
    """
    state = _kernel_state()
    # Two entries already banked, a third under way: no single entry is over the
    # cap, the total is.
    state.phase_elapsed_totals = {ps.PHASE_KERNEL_AGENT: 2 * ENTRY_SEC}
    state.phase = ps.PHASE_KERNEL_AGENT
    state.phase_started_unix = T0
    state.phase_started_ts = T0_ISO
    now = T0 + ENTRY_SEC

    result = ps.exit_normal_kernel(state, now_unix=now)

    assert result is not None
    reason, evidence = result
    assert reason in {"kernel_budget_cap", "kernel_phase_budget_exhausted"}
    assert evidence["entry_elapsed_seconds"] == pytest.approx(ENTRY_SEC)
    assert evidence["cumulative_elapsed_seconds"] == pytest.approx(3 * ENTRY_SEC)
    # The number that justifies the exit is the one over the cap.
    assert evidence["cumulative_elapsed_seconds"] > KERNEL_CAP_SEC
    assert evidence["entry_elapsed_seconds"] < KERNEL_CAP_SEC
