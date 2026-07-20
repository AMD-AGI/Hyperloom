# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
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
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _sweep_state(
    *,
    macro_cycle: int = 0,
    cumulative_gain: float = 5.0,
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
        cumulative_gain_validated=cumulative_gain,
        gain_at_cycle_start=gain_at_cycle_start,
        no_gain_cycle_streak=no_gain_streak,
    )
    # sweep_done trigger.
    st.last_sweep = {"status": "succeeded"}
    return st


# compute_next_phase SWEEP back-edge
def test_sweep_reloops_to_explore_when_budget_and_leverage():
    st = _sweep_state(macro_cycle=0, cumulative_gain=5.0, gain_at_cycle_start=0.0)
    nxt = ps.compute_next_phase(st, max_hours=96.0)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == ps.PHASE_EXPLORE
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


def test_sweep_closes_on_failed_conc_sweep_even_when_reloop_available():
    st = _sweep_state(macro_cycle=0, cumulative_gain=5.0, gain_at_cycle_start=0.0)
    st.last_sweep = {}
    st.last_conc_sweep = {"status": "failed"}
    target, reason, evidence = ps.compute_next_phase(st, max_hours=96.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "conc_sweep_failed"
    assert evidence.get("conc_sweep_status") == "failed"
    assert "loopback" not in evidence


def test_sweep_closes_when_globally_converged():
    # No gain this cycle + streak at 2 → effective 3 ≥ threshold.
    st = _sweep_state(
        macro_cycle=2,
        cumulative_gain=5.0,
        gain_at_cycle_start=5.0,
        no_gain_streak=2,
    )
    target, reason, evidence = ps.compute_next_phase(st, max_hours=96.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "global_converged"
    assert evidence["terminal"] is True
    assert evidence["reloop_blocked"] == "global_converged"


def test_sweep_closes_when_insufficient_remaining():
    # Long run (48h) but only ~10min remain → below the reloop floor.
    st = _sweep_state(max_minutes=48 * 60, started_hours_ago=48 - 10 / 60.0)
    target, reason, evidence = ps.compute_next_phase(st, max_hours=48.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert evidence["reloop_blocked"] == "insufficient_remaining"


def test_short_bounded_run_reloops_when_budget_and_leverage_remain():
    # 12h bounded run: macro-loop is available even though budget accounting
    # stays in short-run charge-back mode.
    st = _sweep_state(
        max_minutes=12 * 60,
        started_hours_ago=1.0,
        cumulative_gain=5.0,
        gain_at_cycle_start=0.0,
    )
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is True
    assert ev["reloop"] is True
    assert ev["next_cycle"] == 1

    target, reason, evidence = ps.compute_next_phase(st, max_hours=12.0)
    assert target == ps.PHASE_EXPLORE
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


def test_short_bounded_run_closes_when_insufficient_remaining():
    # 12h bounded run with ~10min left: still below the 30min reloop floor.
    st = _sweep_state(max_minutes=12 * 60, started_hours_ago=12 - 10 / 60.0)
    reloop, ev = ps.should_reloop_to_explore(st)
    assert reloop is False
    assert ev["reloop_blocked"] == "insufficient_remaining"

    target, reason, evidence = ps.compute_next_phase(st, max_hours=12.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert "loopback" not in evidence
    assert evidence["reloop_blocked"] == "insufficient_remaining"


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
        phase=ps.PHASE_EXPLORE,
        max_minutes=96 * 60,
        phase_started_unix=now,
    )
    whole_run = SharedState(**common, cycle_minutes=0.0)
    per_cycle = SharedState(**common, cycle_minutes=360.0)  # 6h cycle
    budget = dict(ps.DEFAULT_PHASE_BUDGET_PCT)
    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_EXPLORE]

    # Long bounded runs charge back (base * pct / denom); the per-cycle window
    # caps the base, so a 6h cycle plans a smaller EXPLORE than the 96h run.
    denom = sum(budget[p] for p in ps.PHASE_NAMES[ps.phase_index(ps.PHASE_EXPLORE) :] if budget[p] > 0)
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
    pct = ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_EXPLORE]
    denom = sum(
        ps.DEFAULT_PHASE_BUDGET_PCT[p]
        for p in ps.PHASE_NAMES[ps.phase_index(ps.PHASE_EXPLORE) :]
        if ps.DEFAULT_PHASE_BUDGET_PCT[p] > 0
    )
    # A 48h long bounded run with a 24h cycle window: early on, remaining session
    # (>24h) exceeds the window, so the window caps the charge-back base.
    early_start = datetime.fromtimestamp(now - 1 * 3600.0, tz=timezone.utc).isoformat()
    early = SharedState(
        session_id="t",
        phase=ps.PHASE_EXPLORE,
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
        phase=ps.PHASE_EXPLORE,
        start_ts=tail_start,
        max_minutes=48 * 60,
        cycle_minutes=24 * 60.0,
        phase_started_unix=now,
    )
    total_tail = ps._phase_budget_total_seconds(tail, now_unix=now)
    assert total_tail == pytest.approx(3 * 3600 * pct / denom)  # session tail < window


def test_budget_minutes_falls_back_to_max_minutes_when_disabled():
    # Long run (48h): cycle_minutes when set defines the per-cycle window.
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=48 * 60, cycle_minutes=0.0)
    assert ps._budget_minutes(st) == 48 * 60.0
    st.cycle_minutes = 120.0
    assert ps._budget_minutes(st) == 120.0


def test_budget_minutes_ignores_cycle_window_for_short_run():
    # Short bounded run (10h < 24h): the per-cycle window must NOT apply; phase
    # budgets stay anchored on the whole session even if cycle_minutes was pinned.
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=600, cycle_minutes=360.0)
    assert ps._budget_minutes(st) == 600.0


def test_short_run_keeps_chargeback_budgeting_across_cycles():
    now = 1_000_000.0
    start_ts = datetime.fromtimestamp(now - 2 * 3600.0, tz=timezone.utc).isoformat()
    common = dict(
        session_id="t",
        phase=ps.PHASE_EXPLORE,
        start_ts=start_ts,
        max_minutes=600,
        cycle_minutes=360.0,
        phase_started_unix=now,
    )
    cycle0 = SharedState(**common, macro_cycle=0)
    cycle1 = SharedState(**common, macro_cycle=1)

    total0 = ps._phase_budget_total_seconds(cycle0, now_unix=now)
    total1 = ps._phase_budget_total_seconds(cycle1, now_unix=now)
    legacy_whole_run = 600 * 60.0 * ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_EXPLORE]
    cycle_window = 360.0 * 60.0 * ps.DEFAULT_PHASE_BUDGET_PCT[ps.PHASE_EXPLORE]

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
        "kernel_agent": MockBackend(ScriptedPlan(turns=[]), name="kernel_agent"),
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
    st.last_sweep = {"status": "succeeded"}
    st.last_conc_sweep = {"status": "succeeded"}

    await c._advance_phase_if_needed()

    # Reloop targets the highest-leverage layer (FRAMEWORK enabled by default).
    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1
    # Per-cycle sweep markers cleared so the new cycle's SWEEP runs fresh.
    assert st.last_sweep == {}
    assert st.last_conc_sweep == {}
    # Gain anchored for the new cycle; gained this cycle → streak reset.
    assert st.gain_at_cycle_start == pytest.approx(7.0)
    assert st.no_gain_cycle_streak == 0
    # The loopback transition row is stamped with the new cycle number.
    loopback_row = next(r for r in reversed(st.phase_history) if r.get("to_phase"))
    assert loopback_row["to_phase"] == "FRAMEWORK_AGENT"
    assert loopback_row["cycle"] == 1


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
    st.last_sweep = {"status": "succeeded"}

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_CLOSE
    assert st.stop_reason == "global_converged"
    assert st.no_gain_cycle_streak == 3


# PolicyGate re-entry after loopback is not falsely denied
def test_policygate_allows_explore_action_after_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.orchestrator.policy.gate import PolicyGate
    from hyperloom.orchestrator.roles.agent_role import default_role_registry

    sd = make_session_dir()
    st = SharedState(session_id="t", phase=ps.PHASE_EXPLORE, macro_cycle=2)
    # Simulate a history that already passed through SWEEP in a prior cycle.
    st.phase_history = [
        {
            "from_phase": "SWEEP",
            "to_phase": "EXPLORE",
            "reason": "cycle_reloop",
            "evidence": {},
            "ts": "",
            "ts_unix": 0.0,
            "cycle": 2,
        },
    ]
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=st,
    )
    # Must not raise phase_incompatible: current phase is EXPLORE.
    gate._validate_phase_action(
        gate.role_registry.get("orchestration"),
        "specialist",
        intent_kind="propose_action",
    )


# Regression — short-run path now uses macro-loop while budget remains.
def test_regression_short_run_sweep_evidence_carries_loopback():
    st = _sweep_state(max_minutes=12 * 60)
    target, reason, evidence = ps.compute_next_phase(st, max_hours=12.0)
    assert (target, reason) == (ps.PHASE_EXPLORE, "cycle_reloop")
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1
