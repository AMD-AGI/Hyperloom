# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase 2 (exploration depth — bottleneck re-direction) acceptance tests — R3.

Covers:
* cyclic EXPLORE plateau is *actionable* (winds the cycle down via
  ``no_more_leverage`` + ``switch_bottleneck``); advisory-only when cyclic off.
* ``compute_next_phase`` routes a plateaued EXPLORE → SWEEP (skips KERNEL).
* Coordinator stamps the bottleneck-switch handoff onto SharedState.
* The redirect advisory renders in the next cycle's EXPLORE and names a
  suggested specialist domain; it clears once the live bottleneck drifts.
* Rejected/tested fingerprints are bucketed per macro-cycle and stay de-duped.

All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from inference_optimizer.orchestrator import phase_state as ps
from inference_optimizer.orchestrator.shared_state import SharedState


CYCLIC_ENV = "INFERENCE_OPTIMIZER_CYCLIC_PHASES"


def _plateaued_explore_state(
    *,
    macro_cycle: int = 1,
    max_minutes: int = 96 * 60,
    started_hours_ago: float = 0.5,
    top_bottleneck: str = "MoE_fused",
) -> SharedState:
    """EXPLORE state that satisfies compute_plateau_explore (no winners + 3
    trailing empty specialist rounds) with budget remaining."""
    now = datetime.now(timezone.utc)
    st = SharedState(
        session_id="t",
        phase=ps.PHASE_EXPLORE,
        start_ts=(now - timedelta(hours=started_hours_ago)).isoformat(),
        phase_started_unix=(now - timedelta(hours=started_hours_ago)).timestamp(),
        max_minutes=max_minutes,
        macro_cycle=macro_cycle,
    )
    # No winners → recent_keep_gain 0 < threshold.
    st.explore_search = {"schema_version": 1, "winners_history": []}
    # Three trailing empty specialist rounds → empty_streak >= 3.
    st.specialist_rounds = [
        {"proposals_total": 0, "proposals_kept": 0} for _ in range(3)
    ]
    if top_bottleneck:
        st.roofline_snapshots = [{"snapshot_id": 1, "top_bottleneck": top_bottleneck}]
    return st


# ==========================================================================
# R3 — plateau → actionable
# ==========================================================================
def test_explore_plateau_is_actionable_in_cyclic(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    st = _plateaued_explore_state()
    out = ps.exit_normal_explore(st)
    assert out is not None
    reason, evidence = out
    assert reason == "explore_no_more_leverage"
    assert evidence.get("switch_bottleneck") is True
    assert evidence.get("plateau") is True


def test_explore_plateau_is_advisory_only_when_cyclic_off(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "0")
    st = _plateaued_explore_state()
    # Budget remains + no escalate hint + cyclic off → plateau does NOT force exit.
    assert ps.exit_normal_explore(st) is None


def test_compute_next_phase_plateau_routes_explore_to_kernel(monkeypatch):
    monkeypatch.setenv(CYCLIC_ENV, "1")
    st = _plateaued_explore_state()
    target, reason, evidence = ps.compute_next_phase(st, max_hours=96.0)
    # Exhausted explore leverage switches lever to KERNEL (non-terminal).
    assert target == ps.PHASE_KERNEL
    assert reason == "explore_no_more_leverage"
    assert evidence.get("switch_bottleneck") is True


# ==========================================================================
# R3 — Coordinator stamps the bottleneck-switch handoff
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
async def test_coordinator_marks_bottleneck_switch_on_plateau(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    src = _plateaued_explore_state(macro_cycle=1, top_bottleneck="MoE_fused")
    # Copy the plateau-relevant fields onto the coordinator's state.
    st.phase = src.phase
    st.start_ts = src.start_ts
    st.phase_started_unix = src.phase_started_unix
    st.max_minutes = src.max_minutes
    st.macro_cycle = src.macro_cycle
    st.explore_search = src.explore_search
    st.specialist_rounds = src.specialist_rounds
    st.roofline_snapshots = src.roofline_snapshots

    await c._advance_phase_if_needed()

    # Exhausted explore leverage switches lever to KERNEL (non-terminal).
    assert st.phase == ps.PHASE_KERNEL
    assert st.pending_bottleneck_switch is True
    assert st.last_cycle_bottleneck == "MoE_fused"


# ==========================================================================
# R3 — redirect advisory block
# ==========================================================================
def test_redirect_advisory_renders_with_suggested_domain(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.phase = ps.PHASE_EXPLORE
    st.macro_cycle = 2
    st.mark_bottleneck_switch(prev_bottleneck="MoE_fused")
    # A fresh roofline whose dominant direction is comm.
    st.roofline_snapshots = [{
        "snapshot_id": 2,
        "top_bottleneck": "MoE_fused",
        "compute_pct": 20.0,
        "idle_pct": 5.0,
        "comm_pct": 70.0,
        "roofline_bound_kind": "compute",
    }]
    block = c._bottleneck_redirect_advisory_block()
    assert "plateaued_bottleneck=MoE_fused" in block
    assert "comm_specialist" in block
    assert "macro_cycle=2" in block


def test_redirect_advisory_empty_when_cyclic_off(cyclic_coordinator, monkeypatch):
    c = cyclic_coordinator
    monkeypatch.setenv(CYCLIC_ENV, "0")
    st = c.shared_state
    st.phase = ps.PHASE_EXPLORE
    st.mark_bottleneck_switch(prev_bottleneck="MoE_fused")
    assert c._bottleneck_redirect_advisory_block() == ""


def test_redirect_advisory_empty_outside_explore(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.phase = ps.PHASE_SWEEP
    st.mark_bottleneck_switch(prev_bottleneck="MoE_fused")
    assert c._bottleneck_redirect_advisory_block() == ""


def test_acceptance_threshold_advisory_lists_unblocked(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.macro_cycle = 2  # cycle 3 → KEEP bar 0.40%
    st.explore_search = {
        "tested": {
            "fp1": {"name": "v_hi", "outcome": "REVERT", "gain_pct": 0.6},
            "fp2": {"name": "v_lo", "outcome": "REVERT", "gain_pct": 0.2},
            "fp3": {"name": "v_keep", "outcome": "KEEP", "gain_pct": 3.0},
        },
        "rejected": [],
    }
    block = c._acceptance_threshold_advisory_block()
    assert "KEEP>=0.40%" in block
    assert "v_hi" in block          # 0.6% >= 0.40% → re-testable
    assert "v_lo" in block          # 0.2% < 0.40% → reference only
    assert "v_keep" not in block    # KEEP'd → never surfaced for re-test


def test_acceptance_threshold_advisory_empty_first_cycle(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.macro_cycle = 0  # first cycle: bar == legacy default, nothing decayed
    assert c._acceptance_threshold_advisory_block() == ""


def test_acceptance_threshold_advisory_empty_when_cyclic_off(
    cyclic_coordinator, monkeypatch,
):
    c = cyclic_coordinator
    monkeypatch.setenv(CYCLIC_ENV, "0")
    c.shared_state.macro_cycle = 3
    assert c._acceptance_threshold_advisory_block() == ""


# ==========================================================================
# R3 — drift clears the pending switch
# ==========================================================================
def test_switch_clears_on_bottleneck_drift():
    st = SharedState(session_id="t")
    st.mark_bottleneck_switch(prev_bottleneck="MoE_fused")
    assert st.pending_bottleneck_switch is True
    # Same bottleneck → still pending.
    assert st.maybe_clear_bottleneck_switch_on_drift("MoE_fused") is False
    assert st.pending_bottleneck_switch is True
    # Drifted → cleared.
    assert st.maybe_clear_bottleneck_switch_on_drift("gemm_fp8") is True
    assert st.pending_bottleneck_switch is False
    assert st.last_cycle_bottleneck == ""


def test_mark_switch_falls_back_to_live_top_bottleneck():
    st = SharedState(session_id="t")
    st.roofline_snapshots = [{"snapshot_id": 1, "top_bottleneck": "attn_decode"}]
    st.mark_bottleneck_switch()  # no explicit prev → uses live top
    assert st.last_cycle_bottleneck == "attn_decode"


# ==========================================================================
# R3 — rejected/tested fingerprints bucketed per cycle (no re-explore)
# ==========================================================================
def test_tested_and_rejected_stamped_with_macro_cycle():
    st = SharedState(session_id="t", macro_cycle=0)
    st.apply_explore_search_update({
        "schema_version": 1,
        "tested": {"fp_a": {"name": "a", "fingerprint": "fp_a"}},
        "rejected": [{"name": "a", "fingerprint": "fp_a"}],
    })
    assert st.explore_search["tested"]["fp_a"]["cycle"] == 0
    assert st.explore_search["rejected"][0]["cycle"] == 0

    # Next cycle: a new rejection is bucketed under cycle 1; the old one keeps
    # its cycle-0 attribution (never re-explored, just re-attributed-safe).
    st.macro_cycle = 1
    st.apply_explore_search_update({
        "schema_version": 1,
        "tested": {
            "fp_a": {"name": "a", "fingerprint": "fp_a", "cycle": 0},
            "fp_b": {"name": "b", "fingerprint": "fp_b"},
        },
        "rejected": [
            {"name": "a", "fingerprint": "fp_a", "cycle": 0},
            {"name": "b", "fingerprint": "fp_b"},
        ],
    })
    tested = st.explore_search["tested"]
    assert tested["fp_a"]["cycle"] == 0
    assert tested["fp_b"]["cycle"] == 1
    cycles = {r["fingerprint"]: r["cycle"] for r in st.explore_search["rejected"]}
    assert cycles == {"fp_a": 0, "fp_b": 1}


def test_veto_fingerprints_bucketed_by_bottleneck():
    st = SharedState(session_id="t", macro_cycle=0)
    # cycle 0 worked the MoE bottleneck.
    st.roofline_snapshots = [{"snapshot_id": 1, "top_bottleneck": "MoE_fused"}]
    st.apply_explore_search_update({
        "schema_version": 1,
        "tested": {"fp_a": {"name": "a", "fingerprint": "fp_a"}},
        "rejected": [{"name": "a", "fingerprint": "fp_a"}],
    })
    assert st.explore_search["tested"]["fp_a"]["bottleneck"] == "MoE_fused"
    assert st.explore_search["rejected"][0]["bottleneck"] == "MoE_fused"

    # cycle 1 drifted to a comm bottleneck; new rejection carries the new bucket
    # while the old one keeps its MoE attribution.
    st.macro_cycle = 1
    st.roofline_snapshots = [{"snapshot_id": 2, "top_bottleneck": "all_reduce"}]
    st.apply_explore_search_update({
        "schema_version": 1,
        "tested": {
            "fp_a": {"name": "a", "fingerprint": "fp_a", "cycle": 0,
                     "bottleneck": "MoE_fused"},
            "fp_b": {"name": "b", "fingerprint": "fp_b"},
        },
        "rejected": [],
    })
    tested = st.explore_search["tested"]
    assert tested["fp_a"]["bottleneck"] == "MoE_fused"
    assert tested["fp_b"]["bottleneck"] == "all_reduce"
    assert tested["fp_b"]["cycle"] == 1
