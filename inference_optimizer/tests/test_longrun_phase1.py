# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase 1 (cyclic phase machine) acceptance tests — R1 / R2 / R7.

Covers:
* ``compute_next_phase`` SWEEP back-edge branches (reloop / converged / no
  budget / cyclic-off).
* per-cycle budget window (R2).
* Coordinator loopback application (macro_cycle bump, marker reset, R7 streak).
* PolicyGate re-entry after a loopback is not falsely denied.
* 12h single-cycle regression: cyclic-off behaviour byte-for-byte unchanged.

All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from inference_optimizer.orchestrator import phase_state as ps
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


CYCLIC_ENV = "INFERENCE_OPTIMIZER_CYCLIC_PHASES"


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


# ==========================================================================
# R1/R7 — compute_next_phase SWEEP back-edge
# ==========================================================================
def test_sweep_reloops_to_explore_when_budget_and_leverage(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    st = _sweep_state(macro_cycle=0, cumulative_gain=5.0, gain_at_cycle_start=0.0)
    nxt = ps.compute_next_phase(st, max_hours=96.0)
    assert nxt is not None
    target, reason, evidence = nxt
    assert target == ps.PHASE_EXPLORE
    assert reason == "cycle_reloop"
    assert evidence["loopback"] is True
    assert evidence["next_cycle"] == 1


def test_sweep_closes_when_globally_converged(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    # No gain this cycle + streak already at 2 → effective 3 ≥ threshold.
    st = _sweep_state(
        macro_cycle=2, cumulative_gain=5.0, gain_at_cycle_start=5.0,
        no_gain_streak=2,
    )
    target, reason, evidence = ps.compute_next_phase(st, max_hours=96.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "global_converged"
    assert evidence["terminal"] is True
    assert evidence["reloop_blocked"] == "global_converged"


def test_sweep_closes_when_insufficient_remaining(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    # 10-min total budget, started now → < 30-min reloop floor.
    st = _sweep_state(max_minutes=10, started_hours_ago=0.0)
    target, reason, evidence = ps.compute_next_phase(st, max_hours=10 / 60.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"
    assert evidence["reloop_blocked"] == "insufficient_remaining"


def test_sweep_closes_when_cyclic_disabled(monkeypatch):
    monkeypatch.delenv(CYCLIC_ENV, raising=False)
    st = _sweep_state()
    target, reason, _ = ps.compute_next_phase(st, max_hours=96.0)
    assert target == ps.PHASE_CLOSE
    assert reason == "sweep_done"


def test_should_reloop_respects_max_cycles(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    st = _sweep_state(macro_cycle=5)
    reloop, ev = ps.should_reloop_to_explore(st, max_cycles=6)
    assert reloop is False
    assert ev["reloop_blocked"] == "max_cycles"


# ==========================================================================
# R2 — per-cycle budget window
# ==========================================================================
def test_per_cycle_budget_shrinks_phase_window():
    now = 1_000_000.0
    common = dict(
        session_id="t", phase=ps.PHASE_EXPLORE, max_minutes=96 * 60,
        phase_started_unix=now,
    )
    whole_run = SharedState(**common, cycle_minutes=0.0)
    per_cycle = SharedState(**common, cycle_minutes=360.0)  # 6h cycle
    budget = dict(ps.DEFAULT_PHASE_BUDGET_PCT)  # EXPLORE = 0.45

    rem_run = ps.phase_budget_remaining_seconds(
        whole_run, budget_pct=budget, now_unix=now,
    )
    rem_cycle = ps.phase_budget_remaining_seconds(
        per_cycle, budget_pct=budget, now_unix=now,
    )
    # whole-run: 96h*0.45; per-cycle: 6h*0.45 → much smaller.
    assert rem_run == pytest.approx(96 * 3600 * 0.45)
    assert rem_cycle == pytest.approx(6 * 3600 * 0.45)
    assert rem_cycle < rem_run


def test_budget_minutes_falls_back_to_max_minutes_when_disabled():
    st = SharedState(phase=ps.PHASE_EXPLORE, max_minutes=600, cycle_minutes=0.0)
    assert ps._budget_minutes(st) == 600.0
    st.cycle_minutes = 120.0
    assert ps._budget_minutes(st) == 120.0


# ==========================================================================
# R1 — Coordinator loopback application
# ==========================================================================
@pytest.fixture
def cyclic_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(CYCLIC_ENV, "1")
    from inference_optimizer.paths import make_session_dir as _msd
    from inference_optimizer.orchestrator.coordinator import Coordinator
    from inference_optimizer.orchestrator.backends import (
        MockBackend, MockCriticBackend, MockKernelBackend,
        MockRobustnessBackend, ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "kernel": MockKernelBackend(),
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

    assert st.phase == ps.PHASE_EXPLORE
    assert st.macro_cycle == 1
    # Per-cycle sweep markers cleared so the new cycle's SWEEP runs fresh.
    assert st.last_sweep == {}
    assert st.last_conc_sweep == {}
    # Gain anchored for the new cycle; gained this cycle → streak reset.
    assert st.gain_at_cycle_start == pytest.approx(7.0)
    assert st.no_gain_cycle_streak == 0
    # The loopback row is stamped with the new cycle number.
    last_row = st.phase_history[-1]
    assert last_row["to_phase"] == "EXPLORE"
    assert last_row["cycle"] == 1


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
    st.no_gain_cycle_streak = 2   # → effective 3 ≥ threshold
    st.last_sweep = {"status": "succeeded"}

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_CLOSE
    assert st.stop_reason == "global_converged"
    assert st.no_gain_cycle_streak == 3


# ==========================================================================
# R1 — PolicyGate re-entry after loopback is not falsely denied
# ==========================================================================
def test_policygate_allows_explore_action_after_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from inference_optimizer.orchestrator.policy import PolicyGate
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    from inference_optimizer.protocol.intent import Intent, IntentType

    sd = make_session_dir()
    st = SharedState(session_id="t", phase=ps.PHASE_EXPLORE, macro_cycle=2)
    # Simulate a history that already passed through SWEEP in a prior cycle.
    st.phase_history = [
        {"from_phase": "SWEEP", "to_phase": "EXPLORE", "reason": "cycle_reloop",
         "evidence": {}, "ts": "", "ts_unix": 0.0, "cycle": 2},
    ]
    gate = PolicyGate(
        role_registry=default_role_registry(),
        session_dir=sd,
        shared_state=st,
    )
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={"action": {"name": "specialist", "params": {}}},
    )
    # Must not raise phase_incompatible: current phase is EXPLORE.
    gate._validate_phase_action(
        gate.role_registry.get("orchestration"),
        "specialist",
        intent_kind="propose_action",
    )


# ==========================================================================
# Regression — cyclic-off path unchanged (12h single-cycle behaviour)
# ==========================================================================
def test_regression_sweep_to_close_evidence_carries_no_loopback(monkeypatch):
    monkeypatch.delenv(CYCLIC_ENV, raising=False)
    st = _sweep_state()
    target, reason, evidence = ps.compute_next_phase(st, max_hours=12.0)
    assert (target, reason) == (ps.PHASE_CLOSE, "sweep_done")
    assert "loopback" not in evidence
    assert evidence.get("cyclic") is False
