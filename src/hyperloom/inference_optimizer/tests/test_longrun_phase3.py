# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-run resilience and periodic soft restart acceptance tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_SWEEP
from hyperloom.orchestrator.state.task_registry import TaskRegistry
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import ensure_schema


SOFT_RESTART_DISABLE_ENV = "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART"


@pytest.fixture
def conn(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield db
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, 60, 600])
async def test_old_running_rows_without_death_evidence_are_retained(conn, ttl):
    reg = TaskRegistry(conn)
    task = await reg.create(kind="bench", params={}, idempotency_key="old", lease_ttl_sec=ttl)
    await reg.transition(task.task_id, "running")
    await conn.execute("UPDATE tasks SET updated_at='2020-01-01T00:00:00+00:00'")
    assert await reg.reclaim_dead_running() == []
    assert (await reg.get(task.task_id)).state == "running"


# cycle-boundary soft restart
@pytest.fixture
def cyclic_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv(SOFT_RESTART_DISABLE_ENV, raising=False)
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
    }
    c = Coordinator(sd, backends=backends)
    yield c


def _arm_sweep_loopback(st):
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_SWEEP
    st.start_ts = (now - timedelta(hours=1)).isoformat()
    st.max_minutes = 96 * 60
    st.macro_cycle = 0
    st.cumulative_gain_validated = 7.0
    st.gain_at_cycle_start = 0.0
    st.last_conc_sweep = {"status": "succeeded"}
    st.last_conc_sweep = {"status": "succeeded"}


@pytest.mark.asyncio
async def test_soft_restart_runs_at_loopback(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    _arm_sweep_loopback(st)

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1


@pytest.mark.asyncio
async def test_soft_restart_preserves_best_and_ledger(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    _arm_sweep_loopback(st)
    st.current_best = {"tput": 123.0, "extra_server_args": "--foo"}
    st.optimization_stack = [{"name": "v1", "gain_pct": 5.0}]
    st.apply_explore_search_update(
        {
            "schema_version": 1,
            "tested": {"fp_a": {"name": "a", "fingerprint": "fp_a"}},
            "rejected": [{"name": "a", "fingerprint": "fp_a"}],
        }
    )

    await c._advance_phase_if_needed()

    assert st.current_best == {"tput": 123.0, "extra_server_args": "--foo"}
    assert st.optimization_stack == [{"name": "v1", "gain_pct": 5.0}]
    assert "fp_a" in st.explore_search["tested"]


@pytest.mark.asyncio
async def test_soft_restart_can_be_disabled(cyclic_coordinator, monkeypatch):
    c = cyclic_coordinator
    monkeypatch.setenv(SOFT_RESTART_DISABLE_ENV, "1")
    # Flip the in-memory toggle to emulate a disabled run.
    c._cycle_soft_restart = False
    st = c.shared_state
    _arm_sweep_loopback(st)

    await c._advance_phase_if_needed()

    assert st.macro_cycle == 1


@pytest.mark.asyncio
async def test_soft_restart_summary_idempotent(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.macro_cycle = 1
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert summary is not None
    assert summary["new_cycle"] == 1
    assert summary["memory_captured"] is True
    again = await c._run_cycle_soft_restart(prior_cycle=1, new_cycle=2)
    assert again["running_tasks_reclaimed"] == 0


async def _noop_phase_side_effects(c):
    async def _noop(*_args, **_kwargs):
        return None

    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.specialist_dispatch._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]


def _arm_explore_to_sweep(st):
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_FRAMEWORK_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_enabled = False
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)


@pytest.mark.asyncio
async def test_phase_transition_cancels_queued_specialist(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    _arm_explore_to_sweep(c.shared_state)

    queued = await c.tasks.create(
        kind="specialist",
        params={"needs_gpu": True},
        idempotency_key="queued-specialist",
    )

    await c._advance_phase_if_needed()

    updated = await c.tasks.get(queued.task_id)
    assert c.shared_state.phase == ps.PHASE_SWEEP
    assert updated.state == "cancelled"
    assert updated.history[-1]["evidence"]["reason"] == "phase_transition:FRAMEWORK_AGENT->SWEEP"


@pytest.mark.asyncio
async def test_phase_transition_waits_for_a_running_specialist(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    _arm_explore_to_sweep(c.shared_state)

    running = await c.tasks.create(
        kind="specialist",
        params={"needs_gpu": True},
        idempotency_key="running-specialist",
    )
    await c.tasks.transition(running.task_id, "running")

    await c._advance_phase_if_needed()
    assert c.shared_state.phase == ps.PHASE_FRAMEWORK_AGENT

    await c.tasks.transition(running.task_id, "cancelled", evidence={"reason": "stopped"})
    await c._advance_phase_if_needed()
    assert c.shared_state.phase == ps.PHASE_SWEEP


@pytest.mark.asyncio
async def test_phase_transition_preserves_target_phase_queued_task(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    _arm_explore_to_sweep(c.shared_state)

    queued = await c.tasks.create(
        kind="conc_sweep",
        params={},
        idempotency_key="queued-conc-sweep",
    )

    await c._advance_phase_if_needed()

    assert c.shared_state.phase == ps.PHASE_SWEEP
    assert (await c.tasks.get(queued.task_id)).state == "queued"


@pytest.mark.asyncio
async def test_phase_transition_preserves_close_report_task(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    c.shared_state.phase = ps.PHASE_FRAMEWORK_AGENT
    c.shared_state.set_stop_reason("target_reached")

    queued = await c.tasks.create(
        kind="report",
        params={},
        idempotency_key="queued-report",
    )

    await c._advance_phase_if_needed()

    assert c.shared_state.phase == ps.PHASE_CLOSE
    assert (await c.tasks.get(queued.task_id)).state == "queued"


# Dispatcher pump: every-tick expired-running reclaim (pump_watchdog path)


async def _build_minimal_coord(tmp_path: Path, monkeypatch):
    """Build a minimal Coordinator with no real GPU pool (capacity=0)."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles.mock_backend import MockBackend, ScriptedPlan
    from hyperloom.orchestrator.state.shared_state import SharedState

    for _var in (
        "TP",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY",
        "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES",
    ):
        monkeypatch.delenv(_var, raising=False)

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd

    sd = _msd()
    state = SharedState(session_id="pump-test")
    state.gpu_specialist_capacity = 0
    state.save(sd)

    idle_plan = ScriptedPlan(turns=[])
    backends = {
        "orchestration": MockBackend(idle_plan),
        "critic": MockBackend(idle_plan),
    }
    from hyperloom.orchestrator.roles.agent_role import default_role_registry
    from .conftest import seed_target_analysis_marker

    seed_target_analysis_marker(sd)
    return Coordinator(
        session_dir=sd,
        backends=backends,
        role_registry=default_role_registry(),
        recipe_kb=None,
        knowledge_plane=None,
    )


@pytest.mark.asyncio
async def test_pump_reclaims_expired_running_task(tmp_path: Path, monkeypatch):
    """_pump_dispatcher_once flips an orphaned expired-running task to failed."""
    coord = await _build_minimal_coord(tmp_path, monkeypatch)

    # Orphaned task: TTL expired via backdated updated_at.
    orphan = await coord.tasks.create(
        kind="integrate_patch",
        params={},
        idempotency_key="zombie-orphan",
        lease_ttl_sec=60,
    )
    await coord.tasks.transition(orphan.task_id, "running")
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await coord.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        (stale_ts, orphan.task_id),
    )

    # In-window task: age < ttl.
    live = await coord.tasks.create(
        kind="integrate_patch",
        params={},
        idempotency_key="live-running",
        lease_ttl_sec=3600,
    )
    await coord.tasks.transition(live.task_id, "running")

    # No-TTL task: never reclaimed.
    no_ttl = await coord.tasks.create(
        kind="sweep",
        params={},
        idempotency_key="no-ttl-running",
        lease_ttl_sec=0,
    )
    await coord.tasks.transition(no_ttl.task_id, "running")

    await coord._pump_dispatcher_once()

    assert (await coord.tasks.get(orphan.task_id)).state == "running", "age alone cannot establish worker death"
    assert (await coord.tasks.get(live.task_id)).state == "running", "in-window running task must not be reclaimed"
    assert (await coord.tasks.get(no_ttl.task_id)).state == "running", "no-TTL running task must never be reclaimed"


@pytest.mark.asyncio
async def test_pump_reclaim_idempotent(tmp_path: Path, monkeypatch):
    """Running the pump twice on an already-failed task is a no-op (idempotent)."""
    coord = await _build_minimal_coord(tmp_path, monkeypatch)

    orphan = await coord.tasks.create(
        kind="integrate_patch",
        params={},
        idempotency_key="zombie-idem",
        lease_ttl_sec=60,
    )
    await coord.tasks.transition(orphan.task_id, "running")
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    await coord.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        (stale_ts, orphan.task_id),
    )

    await coord._pump_dispatcher_once()
    assert (await coord.tasks.get(orphan.task_id)).state == "running"

    await coord._pump_dispatcher_once()
    assert (await coord.tasks.get(orphan.task_id)).state == "running"
