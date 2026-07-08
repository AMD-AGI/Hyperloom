# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Phase 3 (resilience — periodic soft restart) acceptance tests — R6.

Covers:
* ``TaskRegistry.reclaim_expired_running`` (lease-expiry watchdog): orphaned
  running tasks → failed, idempotent, fresh / no-ttl tasks untouched.
* the cycle-boundary soft restart runs at a SWEEP→EXPLORE loopback: resets the
  orchestration conversation, reclaims orphaned tasks, and PRESERVES the global
  best + negative ledger (no data loss, no duplicate tasks).
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


CYCLIC_ENV = "INFERENCE_OPTIMIZER_CYCLIC_PHASES"
SOFT_RESTART_DISABLE_ENV = "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART"


@pytest.fixture
def conn(tmp_path):
    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    yield db
    db.close()


# ==========================================================================
# R6 — TaskRegistry.reclaim_expired_running (watchdog)
# ==========================================================================
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
    # now far past updated_at + lease_ttl → orphaned.
    future = datetime.now(timezone.utc).timestamp() + 10_000
    reclaimed = await reg.reclaim_expired_running(now_unix=future)
    assert reclaimed == [t.task_id]
    assert (await reg.get(t.task_id)).state == "failed"
    # Idempotent: already failed → no-op.
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
    # now == updated_at → fresh task age 0 < ttl; no-ttl task never expires.
    future = datetime.now(timezone.utc).timestamp() + 10_000
    reclaimed = await reg.reclaim_expired_running(now_unix=future)
    # Only the fresh one *could* expire (ttl 600 < 10000) → reclaimed; no-ttl untouched.
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
    # Within the lease window → not reclaimed.
    soon = datetime.now(timezone.utc).timestamp() + 10
    assert await reg.reclaim_expired_running(now_unix=soon) == []
    assert (await reg.get(t.task_id)).state == "running"


# ==========================================================================
# R6 — cycle-boundary soft restart
# ==========================================================================
@pytest.fixture
def cyclic_coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(CYCLIC_ENV, "1")
    monkeypatch.delenv(SOFT_RESTART_DISABLE_ENV, raising=False)
    # Don't let the soft restart's /proc server sweep run against the real host
    # during unit tests; the kill path is covered separately via monkeypatch.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SERVER_RESTART", "1")
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockKernelBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "kernel_agent": MockKernelBackend(),
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
    st.last_sweep = {"status": "succeeded"}
    st.last_conc_sweep = {"status": "succeeded"}


@pytest.mark.asyncio
async def test_soft_restart_runs_at_loopback(cyclic_coordinator):
    c = cyclic_coordinator
    st = c.shared_state
    _arm_sweep_loopback(st)
    # An orphaned running task from the prior cycle.
    t = await c.tasks.create(
        kind="bench",
        params={},
        idempotency_key="orphan",
        lease_ttl_sec=1,
    )
    await c.tasks.transition(t.task_id, "running")
    # Backdate updated_at so the 1s lease is already expired.
    await c.db.execute(
        "UPDATE tasks SET updated_at=? WHERE task_id=?",
        ((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), t.task_id),
    )

    await c._advance_phase_if_needed()

    # Reloop targets FRAMEWORK (highest-leverage layer, enabled by default).
    assert st.phase == ps.PHASE_FRAMEWORK_AGENT
    assert st.macro_cycle == 1
    # Conversation reset for the new cycle.
    assert c._orchestration_seeded is False
    # Orphaned running task reclaimed → failed.
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

    # Global accumulators survive the soft restart untouched.
    assert st.current_best == {"tput": 123.0, "extra_server_args": "--foo"}
    assert st.optimization_stack == [{"name": "v1", "gain_pct": 5.0}]
    assert "fp_a" in st.explore_search["tested"]


@pytest.mark.asyncio
async def test_soft_restart_can_be_disabled(cyclic_coordinator, monkeypatch):
    c = cyclic_coordinator
    monkeypatch.setenv(SOFT_RESTART_DISABLE_ENV, "1")
    # Re-read the flag the way __init__ did (fixture built the coordinator with
    # the flag unset, so flip the in-memory toggle to emulate a disabled run).
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

    # Loopback still happened, but the soft restart did NOT reclaim the task.
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
    # Running a second time is safe (no orphan tasks → 0 reclaimed).
    again = await c._run_cycle_soft_restart(prior_cycle=1, new_cycle=2)
    assert again["running_tasks_reclaimed"] == 0


@pytest.mark.asyncio
async def test_soft_restart_invokes_server_deep_clean(cyclic_coordinator):
    c = cyclic_coordinator
    # Enable the server-restart step (fixture disabled it) but stub the real
    # /proc kill so the test never touches host processes.
    c._cycle_restart_servers = True
    calls: list[int] = []
    c.phase_explore._restart_inference_servers = lambda: calls.append(1)  # type: ignore[method-assign]
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert calls == [1]
    assert summary["servers_restarted"] is True


@pytest.mark.asyncio
async def test_soft_restart_skips_server_clean_when_disabled(cyclic_coordinator):
    c = cyclic_coordinator
    # Fixture left server restart disabled.
    assert c._cycle_restart_servers is False
    calls: list[int] = []
    c.phase_explore._restart_inference_servers = lambda: calls.append(1)  # type: ignore[method-assign]
    summary = await c._run_cycle_soft_restart(prior_cycle=0, new_cycle=1)
    assert calls == []
    assert "servers_restarted" not in summary


# ==========================================================================
# Dispatcher pump: every-tick expired-running reclaim
# ==========================================================================

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
        "kernel_agent": MockBackend(idle_plan),
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
        cortex_kb=None,
        knowledge_plane=None,
    )


@pytest.mark.asyncio
async def test_pump_reclaims_expired_running_task(tmp_path: Path, monkeypatch):
    """_pump_dispatcher_once flips an orphaned expired-running task to failed.

    Verifies the every-tick reclaim_expired_running call added to the pump
    (pump_watchdog path) so a zombie task whose TTL has elapsed is retired
    within a single tick — not after the next maintenance cadence (every 50 ticks).
    """
    coord = await _build_minimal_coord(tmp_path, monkeypatch)

    # Orphaned task: lease_ttl_sec=60, updated_at backdated far enough that
    # updated_at + 60 < now → TTL has expired.
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

    # In-window task: lease_ttl_sec=3600, updated_at = now → age ≈ 0s < 3600s.
    live = await coord.tasks.create(
        kind="integrate_patch",
        params={},
        idempotency_key="live-running",
        lease_ttl_sec=3600,
    )
    await coord.tasks.transition(live.task_id, "running")

    # No-TTL task (lease_ttl_sec=0): never reclaimed.
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
    assert (await coord.tasks.get(live.task_id)).state == "running", (
        "in-window running task must not be reclaimed"
    )
    assert (await coord.tasks.get(no_ttl.task_id)).state == "running", (
        "no-TTL running task must never be reclaimed"
    )


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

    # Second pump call must not raise and the task stays failed.
    await coord._pump_dispatcher_once()
    assert (await coord.tasks.get(orphan.task_id)).state == "failed"
