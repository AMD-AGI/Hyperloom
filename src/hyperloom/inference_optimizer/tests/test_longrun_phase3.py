# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Long-run resilience and periodic soft restart acceptance tests.

Covers:
* ``TaskRegistry.reclaim_expired_running`` (lease-expiry watchdog): orphaned
  running tasks → failed, idempotent, fresh / no-ttl tasks untouched.
* the cycle-boundary soft restart runs at the SWEEP reloop (to FRAMEWORK_AGENT,
  else EXPLORE): resets the orchestration conversation, reclaims orphaned tasks,
  and PRESERVES the global best + negative ledger (no data loss, no duplicate
  tasks).
* the soft restart honours its opt-out env flag.

All deterministic + offline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom.orchestrator.phases import machine_state as ps
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


# TaskRegistry.reclaim_expired_running (watchdog)
@pytest.mark.asyncio
async def test_reclaim_expired_running_orphan(conn):
    reg = TaskRegistry(conn)
    t = await reg.create(
        kind="bench",
        params={},
        idempotency_key="k1",
        lease_ttl_sec=60,
    )
    await reg.transition(t.task_id, "running")
    future = datetime.now(timezone.utc).timestamp() + 10_000
    reclaimed = await reg.reclaim_expired_running(now_unix=future)
    assert reclaimed == [t.task_id]
    assert (await reg.get(t.task_id)).state == "failed"
    assert await reg.reclaim_expired_running(now_unix=future) == []


@pytest.mark.asyncio
async def test_reclaim_leaves_fresh_and_no_ttl_running(conn):
    reg = TaskRegistry(conn)
    fresh = await reg.create(
        kind="bench",
        params={},
        idempotency_key="fresh",
        lease_ttl_sec=600,
    )
    await reg.transition(fresh.task_id, "running")
    no_ttl = await reg.create(
        kind="bench",
        params={},
        idempotency_key="nottl",
        lease_ttl_sec=0,
    )
    await reg.transition(no_ttl.task_id, "running")
    future = datetime.now(timezone.utc).timestamp() + 10_000
    reclaimed = await reg.reclaim_expired_running(now_unix=future)
    assert fresh.task_id in reclaimed
    assert no_ttl.task_id not in reclaimed
    assert (await reg.get(no_ttl.task_id)).state == "running"


@pytest.mark.asyncio
async def test_reclaim_respects_lease_window(conn):
    reg = TaskRegistry(conn)
    t = await reg.create(
        kind="bench",
        params={},
        idempotency_key="k",
        lease_ttl_sec=600,
    )
    await reg.transition(t.task_id, "running")
    soon = datetime.now(timezone.utc).timestamp() + 10
    assert await reg.reclaim_expired_running(now_unix=soon) == []
    assert (await reg.get(t.task_id)).state == "running"


# cycle-boundary soft restart
@pytest.fixture
def cyclic_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv(SOFT_RESTART_DISABLE_ENV, raising=False)
    # Don't let the soft restart's /proc server sweep run against the real host.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SERVER_RESTART", "1")
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
    t = await c.tasks.create(
        kind="bench",
        params={},
        idempotency_key="orphan",
        lease_ttl_sec=1,
    )
    await c.tasks.transition(t.task_id, "running")
    await c.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        ((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), t.task_id),
    )

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1
    assert c._orchestration_seeded is False
    assert (await c.tasks.get(t.task_id)).state == "failed"


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
    t = await c.tasks.create(
        kind="bench",
        params={},
        idempotency_key="orphan2",
        lease_ttl_sec=1,
    )
    await c.tasks.transition(t.task_id, "running")
    await c.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        ((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), t.task_id),
    )

    await c._advance_phase_if_needed()

    assert st.macro_cycle == 1
    assert (await c.tasks.get(t.task_id)).state == "running"


@pytest.mark.asyncio
async def test_soft_restart_summary_idempotent(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    st.macro_cycle = 1
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert summary is not None
    assert summary["new_cycle"] == 1
    assert summary["conversation_reset"] is True
    again = await c._run_cycle_soft_restart(prior_cycle=1, new_cycle=2)
    assert again["running_tasks_reclaimed"] == 0


@pytest.mark.asyncio
async def test_soft_restart_invokes_server_deep_clean(cyclic_coordinator):
    c = cyclic_coordinator
    # Enable the server-restart step but stub the real /proc kill.
    c._cycle_restart_servers = True
    calls: list[int] = []
    c.phase_explore._restart_inference_servers = lambda: calls.append(1)  # type: ignore[method-assign]
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert calls == [1]
    assert summary["servers_restarted"] is True


@pytest.mark.asyncio
async def test_soft_restart_skips_server_clean_when_disabled(cyclic_coordinator):
    c = cyclic_coordinator
    assert c._cycle_restart_servers is False
    calls: list[int] = []
    c.phase_explore._restart_inference_servers = lambda: calls.append(1)  # type: ignore[method-assign]
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert calls == []
    assert "servers_restarted" not in summary


async def _noop_phase_side_effects(c):
    async def _noop(*_args, **_kwargs):
        return None

    c.phase_internal._maybe_enqueue_explore_research_scout = _noop  # type: ignore[method-assign]
    c.phase_explore._maybe_force_stalled_domain_specialist = _noop  # type: ignore[method-assign]
    c.phase_internal._maybe_enqueue_trajectory_reviewer = _noop  # type: ignore[method-assign]


def _arm_explore_to_sweep(st):
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_FRAMEWORK_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_enabled = False
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)


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
async def test_geak_revalidation_blocks_sweep_transition_while_queued(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    now = datetime.now(timezone.utc)
    st = c.shared_state
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    st.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": "geak-revalidate-1",
    }
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)

    queued = await c.tasks.create(
        kind="explore",
        params={
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
        },
        idempotency_key="geak-revalidate",
        task_id="geak-revalidate-1",
    )

    await c._advance_phase_if_needed()

    assert st.phase == ps.PHASE_KERNEL_AGENT
    assert (await c.tasks.get(queued.task_id)).state == "queued"


@pytest.mark.asyncio
async def test_failed_geak_revalidation_releases_sweep_transition(cyclic_coordinator):
    c = cyclic_coordinator
    await _noop_phase_side_effects(c)
    now = datetime.now(timezone.utc)
    st = c.shared_state
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok"}
    st.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": "geak-revalidate-failed",
    }
    st.set_pending_escalate_hint(ps.ESCALATE_HINT_SKIP_TO_SWEEP)

    task = await c.tasks.create(
        kind="explore",
        params={
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
        },
        idempotency_key="geak-revalidate-failed",
        task_id="geak-revalidate-failed",
    )

    await c._handle_unpromotable_result(
        task,
        {
            "status": "failed",
            "error_class": "subprocess_nonzero",
            "error": "revalidation failed",
        },
    )
    await c._advance_phase_if_needed()

    assert not st.geak_pending
    assert st.phase == ps.PHASE_SWEEP


@pytest.mark.asyncio
async def test_phase_transition_does_not_cancel_running_specialist(cyclic_coordinator):
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

    assert c.shared_state.phase == ps.PHASE_SWEEP
    assert (await c.tasks.get(running.task_id)).state == "running"


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
        "robustness": MockBackend(idle_plan),
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

    assert (await coord.tasks.get(orphan.task_id)).state == "failed", (
        "orphaned expired-running task must be failed by the pump"
    )
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
    assert (await coord.tasks.get(orphan.task_id)).state == "failed"

    await coord._pump_dispatcher_once()
    assert (await coord.tasks.get(orphan.task_id)).state == "failed"
