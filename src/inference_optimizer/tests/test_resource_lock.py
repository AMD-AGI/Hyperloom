"""Tests for orchestrator/resource_lock.py — atomic acquire_many."""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.orchestrator.resource_lock import (
    LaneBusy,
    ResourceLockManager,
    SqliteLeaseBackend,
    StaleLeaseError,
)


@pytest.mark.asyncio
async def test_acquire_release_single_lane(db):
    backend = SqliteLeaseBackend(db)
    lease = await backend.acquire_many(
        ["benchmark_lane"],
        holder_id="h1",
        task_id="t1",
        action="bench",
        ttl_sec=30,
    )
    # benchmark_lane conflicts with profile_lane and server_lifecycle, so the
    # expanded set is 3 lanes:
    assert set(lease.lanes) == {"benchmark_lane", "profile_lane",
                                 "server_lifecycle"}
    rows = await db.fetchall("SELECT lane FROM leases ORDER BY lane")
    assert {r["lane"] for r in rows} == set(lease.lanes)

    deleted = await backend.release(lease)
    assert deleted == len(lease.lanes)


@pytest.mark.asyncio
async def test_acquire_many_is_atomic(db):
    """If any required lane is busy, *no* row is inserted."""
    backend = SqliteLeaseBackend(db)

    held = await backend.acquire_many(
        ["benchmark_lane"],
        holder_id="h1",
        task_id="t1",
        action="bench",
        ttl_sec=60,
    )

    # Profile conflicts via cross-lane rule; whole txn must rollback.
    with pytest.raises(LaneBusy) as exc:
        await backend.acquire_many(
            ["profile_lane"],
            holder_id="h2",
            task_id="t2",
            action="profile",
            ttl_sec=30,
        )
    busy = set(exc.value.busy_lanes)
    # at least one of the conflicting lanes was reported
    assert busy & set(held.lanes)

    # leases table still only has h1's rows
    rows = await db.fetchall("SELECT holder_id FROM leases")
    holders = {r["holder_id"] for r in rows}
    assert holders == {"h1"}


@pytest.mark.asyncio
async def test_workspace_mutation_lane_independent(db):
    """workspace_mutation has no transitive conflicts so it can co-exist."""
    backend = SqliteLeaseBackend(db)
    a = await backend.acquire_many(
        ["workspace_mutation"], holder_id="h1", task_id="t1",
        action="patch", ttl_sec=10,
    )
    # Bench can still acquire (does not touch workspace_mutation)
    b = await backend.acquire_many(
        ["benchmark_lane"], holder_id="h2", task_id="t2",
        action="bench", ttl_sec=10,
    )
    assert "workspace_mutation" in a.lanes
    assert "workspace_mutation" not in b.lanes


@pytest.mark.asyncio
async def test_heartbeat_extends_lease(db):
    backend = SqliteLeaseBackend(db)
    lease = await backend.acquire_many(
        ["workspace_mutation"], holder_id="h1", task_id="t1",
        action="patch", ttl_sec=5,
    )
    row_before = await db.fetchone(
        "SELECT expires_at FROM leases WHERE lane='workspace_mutation'"
    )
    await asyncio.sleep(0.01)
    await backend.heartbeat(lease, ttl_sec=120)
    row_after = await db.fetchone(
        "SELECT expires_at FROM leases WHERE lane='workspace_mutation'"
    )
    assert row_after["expires_at"] > row_before["expires_at"]


@pytest.mark.asyncio
async def test_heartbeat_rejects_wrong_holder(db):
    backend = SqliteLeaseBackend(db)
    lease = await backend.acquire_many(
        ["workspace_mutation"], holder_id="h1", task_id="t1",
        action="patch", ttl_sec=5,
    )
    fake = type(lease)(
        holder_id="someone-else",
        task_id=lease.task_id,
        action=lease.action,
        lanes=lease.lanes,
        acquired_at=lease.acquired_at,
        expires_at=lease.expires_at,
    )
    with pytest.raises(StaleLeaseError):
        await backend.heartbeat(fake, ttl_sec=10)


@pytest.mark.asyncio
async def test_release_is_idempotent(db):
    backend = SqliteLeaseBackend(db)
    lease = await backend.acquire_many(
        ["workspace_mutation"], holder_id="h1", task_id="t1",
        action="patch", ttl_sec=5,
    )
    n1 = await backend.release(lease)
    n2 = await backend.release(lease)
    assert n1 == 1
    assert n2 == 0


@pytest.mark.asyncio
async def test_reap_expired_emits_event(db):
    """Stale leases produce a lease_expired event in the same txn."""
    backend = SqliteLeaseBackend(db)
    # Insert a lease with an expires_at in the past directly
    await db.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("workspace_mutation", "ghost", "t-old", "patch", 9999,
         "2020-01-01T00:00:00+00:00",
         "2020-01-01T00:00:01+00:00",
         "2020-01-01T00:00:01+00:00"),
    )
    reaped = await backend.reap_expired()
    assert len(reaped) == 1
    assert reaped[0]["lane"] == "workspace_mutation"
    rows = await db.fetchall("SELECT * FROM events WHERE topic='lease_expired'")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_manager_retry_succeeds_after_release(db):
    """ResourceLockManager retries with back-off."""
    backend = SqliteLeaseBackend(db)
    mgr = ResourceLockManager(backend)
    held = await mgr.acquire(
        ["benchmark_lane"], task_id="t1", action="bench", ttl_sec=10,
    )

    async def release_after_delay():
        await asyncio.sleep(0.2)
        await mgr.release(held)

    asyncio.create_task(release_after_delay())
    new_lease = await mgr.acquire(
        ["profile_lane"], task_id="t2", action="profile", ttl_sec=10,
    )
    assert "profile_lane" in new_lease.lanes
    await mgr.release(new_lease)


@pytest.mark.asyncio
async def test_unknown_lane_rejected(db):
    backend = SqliteLeaseBackend(db)
    with pytest.raises(ValueError):
        await backend.acquire_many(
            ["nonexistent_lane"], holder_id="h", task_id="t",
            action="x", ttl_sec=10,
        )
