# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cyclic phase machine acceptance tests.

Covers ``compute_next_phase`` SWEEP back-edge branches, the per-cycle budget
window, Coordinator loopback application, PolicyGate re-entry after a loopback,
and short-run macro-loop behaviour. All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import SharedState


def _sweep_state(
    *,
    macro_cycle: int = 0,
    validated_gain: float = 5.0,
    gain_at_cycle_start: float = 0.0,
    no_gain_streak: int = 0,
    max_minutes: int = 96 * 60,
    started_hours_ago: float = 1.0,
) -> SharedState:
    now = datetime.now(timezone.utc)
    st = SharedState(
        session_id="t",
        phase=ps.PHASE_SWEEP,
        start_ts=(now - timedelta(hours=started_hours_ago)).isoformat(),
        max_minutes=max_minutes,
        macro_cycle=macro_cycle,
        cumulative_gain_validated=validated_gain,
        gain_at_cycle_start=gain_at_cycle_start,
        no_gain_cycle_streak=no_gain_streak,
    )
    # sweep_done trigger: the concurrency ladder is the phase's sweep.
    st.last_conc_sweep = {"status": "succeeded"}
    return st


# compute_next_phase SWEEP back-edge
def test_sweep_reloops_to_explore_when_budget_and_leverage():
    st = _sweep_state(macro_cycle=0, validated_gain=5.0, gain_at_cycle_start=0.0)
    nxt = ps.compute_next_phase(st)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == ps.PHASE_FRAMEWORK_AGENT
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


def test_sweep_closes_on_failed_conc_sweep_even_when_reloop_available():
    st = _sweep_state(macro_cycle=0, validated_gain=5.0, gain_at_cycle_start=0.0)
    st.last_conc_sweep = {}
    st.last_conc_sweep = {"status": "failed"}
    target, reason, evidence = ps.compute_next_phase(st)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_failed"
    assert evidence.get("sweep_status") == "failed"
    assert "loopback" not in evidence


def test_sweep_closes_when_globally_converged():
    # No gain this cycle + streak at 2 → effective 3 ≥ threshold.
    st = _sweep_state(
        macro_cycle=2,
        validated_gain=5.0,
        gain_at_cycle_start=5.0,
        no_gain_streak=2,
    )
    target, reason, evidence = ps.compute_next_phase(st)
    assert target == ps.PHASE_CLOSE
    assert reason == "global_converged"
    assert evidence["terminal"] is True
    assert evidence["reloop_blocked"] == "global_converged"


def test_sweep_closes_when_insufficient_remaining():
    # Long run (48h) but only ~10min remain → below the reloop floor.
    st = _sweep_state(max_minutes=48 * 60, started_hours_ago=48 - 10 / 60.0)
    target, reason, evidence = ps.compute_next_phase(st)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert evidence["reloop_blocked"] == "insufficient_remaining"


def test_sweep_skip_to_close_does_not_override_a_settled_conc_sweep():
    """LLM skip_to_close after a refused conc_sweep must not become robustness_escalated."""
    st = _sweep_state(max_minutes=180, started_hours_ago=166 / 60.0)
    st.last_conc_sweep = {}
    st.last_conc_sweep = {
        "status": "skipped",
        "was_skipped": True,
        "skip_reason": "session_time_budget",
    }
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_CLOSE)
    nxt = ps.compute_next_phase(st)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert evidence.get("sweep_status") == "skipped"


def test_sweep_skip_to_close_still_escalates_when_conc_sweep_never_settled():
    """skip_to_close remains a robustness abort when SWEEP has nothing to close on."""
    st = _sweep_state(max_minutes=180, started_hours_ago=1.0)
    st.last_conc_sweep = {}
    st.last_conc_sweep = {}
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_CLOSE)
    nxt = ps.compute_next_phase(st)
    assert nxt is not None
    target, reason, _evidence = nxt
    assert target == ps.PHASE_CLOSE
    assert reason == "robustness_escalated"


def test_sweep_skip_to_close_yields_to_reloop_when_conc_sweep_was_skipped():
    """A skipped conc_sweep with budget left must not be aborted by skip_to_close."""
    st = _sweep_state(macro_cycle=0, validated_gain=5.0, gain_at_cycle_start=0.0)
    st.last_conc_sweep = {}
    st.last_conc_sweep = {
        "status": "skipped",
        "was_skipped": True,
        "skip_reason": "session_time_budget",
    }
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_CLOSE)
    nxt = ps.compute_next_phase(st)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == ps.PHASE_FRAMEWORK_AGENT
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True


def test_short_bounded_run_reloops_when_budget_and_leverage_remain():
    # 12h bounded run: macro-loop is available even though budget accounting
    # stays in short-run charge-back mode.
    st = _sweep_state(
        max_minutes=12 * 60,
        started_hours_ago=1.0,
        validated_gain=5.0,
        gain_at_cycle_start=0.0,
    )
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is True
    assert ev["reloop"] is True
    assert ev["next_cycle"] == 1

    target, reason, evidence = ps.compute_next_phase(st)
    assert target == ps.PHASE_FRAMEWORK_AGENT
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


def test_short_bounded_run_closes_when_insufficient_remaining():
    # 12h bounded run with ~10min left: below the 3h reloop floor.
    st = _sweep_state(max_minutes=12 * 60, started_hours_ago=12 - 10 / 60.0)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "insufficient_remaining"

    target, reason, evidence = ps.compute_next_phase(st)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert "loopback" not in evidence
    assert evidence["reloop_blocked"] == "insufficient_remaining"


def test_reloop_blocked_when_insufficient_budget_remains():
    # 12h session: effective floor = min(10800, 12*3600*0.15) = min(10800, 6480) = 6480s.
    # Reloop is blocked when remaining < 6480s, i.e. elapsed > 12h - 1.8h = 10.2h.
    st = _sweep_state(max_minutes=12 * 60, started_hours_ago=0.0)
    start_unix = datetime.fromisoformat(st.start_ts).timestamp()

    # Well inside budget (3h remaining for a 12h session).
    reloop, ev = ps.should_reloop_to_explore(
        st,
        now_unix=start_unix + 9 * 3600,
    )
    assert reloop is True
    assert ev["reloop"] is True
    assert "min_remaining_sec_effective" in ev

    # Just past the proportional floor (remaining drops below 6480s).
    reloop, ev = ps.should_reloop_to_explore(
        st,
        now_unix=start_unix + 12 * 3600 - 6479,
    )
    assert reloop is False
    assert ev["reloop_blocked"] == "insufficient_remaining"
    assert ev["min_remaining_sec_effective"] == pytest.approx(6480.0, abs=1.0)


def test_exactly_24h_is_long_run():
    st = _sweep_state(max_minutes=24 * 60, started_hours_ago=1.0)
    assert ps.is_long_run(st) is True
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is True
    assert ev["reloop"] is True
    assert ev["next_cycle"] == 1


def test_long_and_unbounded_runs_are_long():
    # >= 24h bounded → long.
    st_long = _sweep_state(max_minutes=24 * 60, started_hours_ago=1.0)
    assert ps.is_long_run(st_long) is True
    # Unbounded (max_minutes == 0) → long.
    st_unbounded = _sweep_state(max_minutes=0, started_hours_ago=1.0)
    assert ps.is_long_run(st_unbounded) is True


def test_should_reloop_respects_max_cycles():
    st = _sweep_state(macro_cycle=5)
    reloop, ev = ps.should_reloop_to_explore(st, max_cycles=6)
    assert reloop is False
    assert ev["reloop_blocked"] == "max_cycles"


# per-cycle budget window
def test_per_cycle_budget_shrinks_phase_window():
    now = 1_000_000.0
    common = dict(
        session_id="t",
        phase=ps.PHASE_FRAMEWORK_AGENT,
        max_minutes=96 * 60,
        phase_started_unix=now,
    )
    whole_run = SharedState(**common, cycle_minutes=0.0)
    per_cycle = SharedState(**common, cycle_minutes=360.0)  # 6h cycle
    budget = dict(ps.DEFAULT_PHASE_BUDGET_PCT)
    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]

    # Long bounded runs charge back (base * pct / denom); the per-cycle window
    # caps the base, so a 6h cycle plans a smaller EXPLORE than the 96h run.
    denom = sum(budget[p] for p in ps.PHASE_NAMES[ps.phase_index(ps.PHASE_FRAMEWORK_AGENT) :] if budget[p] > 0)
    rem_run = ps.phase_budget_remaining_seconds(
        whole_run,
        budget_pct=budget,
        now_unix=now,
    )
    rem_cycle = ps.phase_budget_remaining_seconds(
        per_cycle,
        budget_pct=budget,
        now_unix=now,
    )
    # whole-run base = full 96h session; per-cycle base = capped to the 6h window.
    assert rem_run == pytest.approx(96 * 3600 * pct / denom)
    assert rem_cycle == pytest.approx(6 * 3600 * pct / denom)
    assert rem_cycle < rem_run


def test_long_run_chargeback_cap_and_tail():
    now = 1_000_000.0
    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]
    denom = sum(
        ps.DEFAULT_PHASE_BUDGET_PCT[p]
        for p in ps.PHASE_NAMES[ps.phase_index(ps.PHASE_FRAMEWORK_AGENT) :]
        if ps.DEFAULT_PHASE_BUDGET_PCT[p] > 0
    )
    # A 48h long bounded run with a 24h cycle window: early on, remaining session
    # (>24h) exceeds the window, so the window caps the charge-back base.
    early_start = datetime.fromtimestamp(now - 1 * 3600.0, tz=timezone.utc).isoformat()
    early = SharedState(
        session_id="t",
        phase=ps.PHASE_FRAMEWORK_AGENT,
        start_ts=early_start,
        max_minutes=48 * 60,
        cycle_minutes=24 * 60.0,
        phase_started_unix=now,
    )
    total_early = ps._phase_budget_total_seconds(early, now_unix=now)
    assert total_early == pytest.approx(24 * 3600 * pct / denom)  # capped at the window

    # Near the tail (only 3h of session left, < the 24h window), the remaining
    # session time — not the window — is the charge-back base.
    tail_start = datetime.fromtimestamp(now - 45 * 3600.0, tz=timezone.utc).isoformat()
    tail = SharedState(
        session_id="t",
        phase=ps.PHASE_FRAMEWORK_AGENT,
        start_ts=tail_start,
        max_minutes=48 * 60,
        cycle_minutes=24 * 60.0,
        phase_started_unix=now,
    )
    total_tail = ps._phase_budget_total_seconds(tail, now_unix=now)
    assert total_tail == pytest.approx(3 * 3600 * pct / denom)  # session tail < window


def test_budget_minutes_falls_back_to_max_minutes_when_disabled():
    # Long run (48h): cycle_minutes when set defines the per-cycle window.
    st = SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=48 * 60, cycle_minutes=0.0)
    assert ps._budget_minutes(st) == 48 * 60.0
    st.cycle_minutes = 120.0
    assert ps._budget_minutes(st) == 120.0


def test_budget_minutes_ignores_cycle_window_for_short_run():
    # Short bounded run (10h < 24h): the per-cycle window must NOT apply; phase
    # budgets stay anchored on the whole session even if cycle_minutes was pinned.
    st = SharedState(phase=ps.PHASE_FRAMEWORK_AGENT, max_minutes=600, cycle_minutes=360.0)
    assert ps._budget_minutes(st) == 600.0


def test_short_run_keeps_chargeback_budgeting_across_cycles():
    now = 1_000_000.0
    start_ts = datetime.fromtimestamp(now - 2 * 3600.0, tz=timezone.utc).isoformat()
    common = dict(
        session_id="t",
        phase=ps.PHASE_FRAMEWORK_AGENT,
        start_ts=start_ts,
        max_minutes=600,
        cycle_minutes=360.0,
        phase_started_unix=now,
    )
    cycle0 = SharedState(**common, macro_cycle=0)
    cycle1 = SharedState(**common, macro_cycle=1)

    total0 = ps._phase_budget_total_seconds(cycle0, now_unix=now)
    total1 = ps._phase_budget_total_seconds(cycle1, now_unix=now)
    legacy_whole_run = 600 * 60.0 * ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]
    cycle_window = 360.0 * 60.0 * ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_FRAMEWORK_AGENT]

    assert total0 is not None and total0 > 0
    assert total0 == pytest.approx(total1)
    assert total0 != pytest.approx(legacy_whole_run)
    assert total0 != pytest.approx(cycle_window)


# Coordinator loopback application
@pytest.fixture
def cyclic_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(sd, backends=backends)
    yield c


@pytest.mark.asyncio
async def test_coordinator_applies_loopback(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_SWEEP
    st.start_ts = (now - timedelta(hours=1)).isoformat()
    st.max_minutes = 96 * 60
    st.macro_cycle = 0
    st.cumulative_gain_validated = 7.0
    st.gain_at_cycle_start = 0.0
    st.last_conc_sweep = {"status": "succeeded"}
    st.last_conc_sweep = {"status": "succeeded"}

    await c._advance_phase_if_needed()

    # Reloop targets the highest-leverage layer (FRAMEWORK enabled by default).
    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1
    # Per-cycle sweep markers cleared so the new cycle's SWEEP runs fresh.
    assert st.last_conc_sweep == {}
    assert st.last_conc_sweep == {}
    # Gain anchored for the new cycle; gained this cycle → streak reset.
    assert st.gain_at_cycle_start == pytest.approx(7.0)
    assert st.no_gain_cycle_streak == 0
    # The loopback transition row is stamped with the new cycle number.
    loopback_row = next(r for r in reversed(st.phase_history) if r.get("to_phase"))
    assert loopback_row["to_phase"] == "FRAMEWORK_AGENT"
    assert loopback_row["cycle"] == 1


@pytest.mark.asyncio
async def test_skip_to_close_is_consumed_when_sweep_already_settled(
    cyclic_coordinator,
    monkeypatch,
):
    """A suppressed skip_to_close must not leak into the next phase."""
    c = cyclic_coordinator
    st = c.shared_state
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_SWEEP
    st.start_ts = (now - timedelta(minutes=166)).isoformat()
    st.max_minutes = 180
    st.macro_cycle = 0
    st.last_conc_sweep = {}
    st.last_conc_sweep = {
        "status": "skipped",
        "was_skipped": True,
        "skip_reason": "session_time_budget",
    }
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_CLOSE)

    async def _entered(*, from_phase, to_phase):
        return None

    monkeypatch.setattr(c.phase_machine, "_on_phase_entered", _entered)
    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_CLOSE
    assert st.pending_escalate_hint == ""
    assert st.last_consumed_escalate_hint == ps.ESCALATE_HINT_SKIP_TO_CLOSE


@pytest.mark.asyncio
async def test_coordinator_converged_close_sets_stop_reason(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_SWEEP
    st.start_ts = (now - timedelta(hours=1)).isoformat()
    st.max_minutes = 96 * 60
    st.macro_cycle = 3
    st.cumulative_gain_validated = 5.0
    st.gain_at_cycle_start = 5.0  # no gain this cycle
    st.no_gain_cycle_streak = 2  # effective 3 ≥ threshold
    st.last_conc_sweep = {"status": "succeeded"}

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_CLOSE
    assert st.stop_reason == "global_converged"
    assert st.no_gain_cycle_streak == 3


# PolicyGate re-entry after loopback is not falsely denied


# Regression — short-run path now uses macro-loop while budget remains.
def test_regression_short_run_sweep_evidence_carries_loopback():
    st = _sweep_state(max_minutes=12 * 60)
    target, reason, evidence = ps.compute_next_phase(st)
    assert (target, reason) == (ps.PHASE_FRAMEWORK_AGENT, "cycle_reloop")
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


# Stage 5 additions — adaptive reloop floor


def test_unbounded_run_uses_absolute_floor():
    st = _sweep_state(max_minutes=0, started_hours_ago=0.0)
    _, ev = ps.should_reloop_to_explore(st)
    # Unbounded run (max_minutes=0): effective floor == absolute floor (10800).
    assert ev["min_remaining_sec_effective"] == pytest.approx(10800.0, abs=1.0)


def test_short_bounded_run_scales_floor():
    # 2h session: effective = min(10800, 2*3600*0.15) = min(10800, 1080) = 1080s.
    st = _sweep_state(max_minutes=2 * 60, started_hours_ago=0.0)
    _, ev = ps.should_reloop_to_explore(st)
    assert ev["min_remaining_sec_effective"] == pytest.approx(1080.0, abs=1.0)


def test_long_bounded_run_caps_at_absolute_floor():
    # 48h session: effective = min(10800, 48*3600*0.15) = min(10800, 25920) = 10800s.
    st = _sweep_state(max_minutes=48 * 60, started_hours_ago=0.0)
    _, ev = ps.should_reloop_to_explore(st)
    assert ev["min_remaining_sec_effective"] == pytest.approx(10800.0, abs=1.0)


def test_env_override_changes_absolute_floor(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC", "3600")
    assert ps._default_cycle_reloop_min_remaining_sec() == pytest.approx(3600.0)


def test_malformed_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC", "not-a-number")
    assert ps._default_cycle_reloop_min_remaining_sec() == pytest.approx(10800.0)


def test_evidence_keys_present():
    st = _sweep_state(max_minutes=12 * 60, started_hours_ago=0.0)
    _, ev = ps.should_reloop_to_explore(st)
    for key in (
        "macro_cycle",
        "min_gain_pct",
        "cycle_gain_delta",
        "cycle_gained",
        "no_gain_cycle_streak_effective",
        "min_remaining_sec_effective",
    ):
        assert key in ev, f"missing evidence key: {key}"
