# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""research_lane capacity + concurrent dispatcher tests."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
from hyperloom.orchestrator.bus.resource_lock import (
    BRINGUP_ROUND_LANE,
    KNOWN_LANES,
    LANE_CONFLICTS,
    LaneBusy,
    LaneFull,
    ResourceLockManager,
    SqliteLeaseBackend,
    StaleLeaseError,
    hold_round_lane,
)
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.bus.storage.schema import (
    DEFAULT_LANE_CAPACITIES,
    SCHEMA_VERSION,
    ensure_schema,
    get_lane_capacity,
    set_lane_capacity,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "coordinator.db"


@pytest.fixture
def conn(db_path):
    db = SqliteConnection(db_path)
    ensure_schema(db.raw)
    yield db
    db.close()


@pytest.fixture
def locks(conn):
    return ResourceLockManager(SqliteLeaseBackend(conn))


def test_schema_version_records_owner_provenance():
    assert SCHEMA_VERSION == 6


def test_fresh_db_has_composite_pk(conn):
    cur = conn.raw.execute("PRAGMA table_info(leases)")
    pk_cols = sorted(row["name"] for row in cur.fetchall() if int(row["pk"] or 0) > 0)
    assert pk_cols == ["holder_id", "lane"]


def test_fresh_db_seeds_default_lane_capacity(conn):
    cur = conn.raw.execute(
        "SELECT lane, capacity FROM lane_capacity ORDER BY lane",
    )
    rows = {r["lane"]: int(r["capacity"]) for r in cur.fetchall()}
    for lane, cap in DEFAULT_LANE_CAPACITIES.items():
        assert rows[lane] == cap


def test_fresh_db_has_gpu_leases_table(conn):
    cur = conn.raw.execute("PRAGMA table_info(gpu_leases)")
    cols = {row["name"] for row in cur.fetchall()}
    assert {"gpu_id", "holder_id", "task_id", "expires_at"} <= cols


def test_set_lane_capacity_upserts(conn):
    set_lane_capacity(conn.raw, "research_lane", 6)
    assert get_lane_capacity(conn.raw, "research_lane") == 6
    set_lane_capacity(conn.raw, "research_lane", 1)
    assert get_lane_capacity(conn.raw, "research_lane") == 1


def test_v2_ensure_schema_is_idempotent(conn):
    """Calling ensure_schema twice on the same DB doesn't lose data."""
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane",
            "h1",
            "t1",
            "specialist",
            1,
            "2026-05-19T18:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-05-19T18:00:00+00:00",
        ),
    )
    conn.raw.commit()
    ensure_schema(conn.raw)
    cur = conn.raw.execute("SELECT COUNT(*) AS n FROM leases")
    assert int(cur.fetchone()["n"]) == 1


@pytest.mark.asyncio
async def test_serving_lane_capacity_1_raises_LaneBusy(locks):
    a = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="ha",
        task_id="ta",
        action="bench",
        ttl_sec=60,
    )
    with pytest.raises(LaneBusy) as exc:
        await locks.acquire_many(
            ["benchmark_lane"],
            holder_id="hb",
            task_id="tb",
            action="bench",
            ttl_sec=60,
        )
    assert "benchmark_lane" in exc.value.busy_lanes
    await locks.release(a)


@pytest.mark.asyncio
async def test_research_lane_capacity_admits_multiple_holders(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 3)
    leases = []
    for i in range(3):
        l = await locks.acquire_many(
            ["research_lane"],
            holder_id=f"s{i}",
            task_id=f"t{i}",
            action="specialist",
            ttl_sec=60,
        )
        leases.append(l)
    holders = await locks.lane_holders()
    assert holders["research_lane"] == 3
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_research_lane_overflow_raises_LaneFull(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    leases = [
        await locks.acquire_many(
            ["research_lane"],
            holder_id=f"s{i}",
            task_id=f"t{i}",
            action="specialist",
            ttl_sec=60,
        )
        for i in range(2)
    ]
    with pytest.raises(LaneFull) as exc:
        await locks.acquire_many(
            ["research_lane"],
            holder_id="s2",
            task_id="t2",
            action="specialist",
            ttl_sec=60,
        )
    assert "research_lane" in exc.value.full_lanes
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_specialist_gpu_pool_allocates_and_releases(conn):
    pool = SpecialistGpuPool(conn, gpu_ids=[0, 1])
    lease = await pool.try_acquire(
        count=1,
        holder_id="gpu-a",
        task_id="task-a",
        ttl_sec=60,
    )
    assert lease is not None
    assert list(lease.gpu_ids) == [0]
    second = await pool.try_acquire(
        count=1,
        holder_id="gpu-b",
        task_id="task-b",
        ttl_sec=60,
    )
    assert second is not None
    assert list(second.gpu_ids) == [1]
    full = await pool.try_acquire(
        count=1,
        holder_id="gpu-c",
        task_id="task-c",
        ttl_sec=60,
    )
    assert full is None
    await pool.release(lease)
    reacquired = await pool.try_acquire(
        count=1,
        holder_id="gpu-c",
        task_id="task-c",
        ttl_sec=60,
    )
    assert reacquired is not None
    assert list(reacquired.gpu_ids) == [0]
    await pool.release(second)
    await pool.release(reacquired)


@pytest.mark.asyncio
async def test_ray_observation_admits_up_to_pending_limit(conn):
    """§3.2: under single-node Ray, admission is COUNT-based (pending limit), not physical-capacity-based — so multiple
    specialists queue on ONE GPU.
    """
    # A single physical GPU: the legacy try_acquire caps at 1 concurrent...
    pool = SpecialistGpuPool(conn, gpu_ids=[0])
    a = await pool.try_acquire_ray_observation(holder_id="h-a", task_id="t-a", pending_limit=3, ttl_sec=60)
    b = await pool.try_acquire_ray_observation(holder_id="h-b", task_id="t-b", pending_limit=3, ttl_sec=60)
    c = await pool.try_acquire_ray_observation(holder_id="h-c", task_id="t-c", pending_limit=3, ttl_sec=60)
    # ...but the Ray observation ledger admits up to pending_limit (3) at once, each on a distinct synthetic slot id
    # above the real device id space.
    assert a is not None and b is not None and c is not None
    slots = sorted(list(a.gpu_ids) + list(b.gpu_ids) + list(c.gpu_ids))
    assert slots == [100000, 100001, 100002]
    # Over the limit -> backpressure (None; caller keeps the task queued).
    d = await pool.try_acquire_ray_observation(holder_id="h-d", task_id="t-d", pending_limit=3, ttl_sec=60)
    assert d is None
    # Releasing frees a slot for the next admission (reuses the low slot).
    await pool.release(a)
    e = await pool.try_acquire_ray_observation(holder_id="h-e", task_id="t-e", pending_limit=3, ttl_sec=60)
    assert e is not None and list(e.gpu_ids) == [100000]
    await pool.release(b)
    await pool.release(c)
    await pool.release(e)


@pytest.mark.asyncio
async def test_specialist_gpu_pool_rejects_oversized_request(conn):
    pool = SpecialistGpuPool(conn, gpu_ids=[0])
    assert (
        await pool.try_acquire(
            count=2,
            holder_id="gpu-a",
            task_id="task-a",
            ttl_sec=60,
        )
        is None
    )


@pytest.mark.asyncio
async def test_capacity_zero_means_lane_disabled(conn, locks):
    """``--research-lane-capacity 0`` disables the research lane."""
    set_lane_capacity(conn.raw, "research_lane", 0)
    with pytest.raises(LaneFull):
        await locks.acquire_many(
            ["research_lane"],
            holder_id="s0",
            task_id="t0",
            action="specialist",
            ttl_sec=60,
        )


@pytest.mark.asyncio
async def test_same_holder_retry_is_idempotent(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 1)
    a = await locks.acquire_many(
        ["research_lane"],
        holder_id="s0",
        task_id="t0",
        action="specialist",
        ttl_sec=30,
    )
    b = await locks.acquire_many(
        ["research_lane"],
        holder_id="s0",
        task_id="t0",
        action="specialist",
        ttl_sec=120,
    )
    assert a.holder_id == b.holder_id
    cur = conn.raw.execute("SELECT COUNT(*) AS n FROM leases WHERE lane=?", ("research_lane",))
    assert int(cur.fetchone()["n"]) == 1
    await locks.release(b)


@pytest.mark.asyncio
async def test_research_lane_independent_of_benchmark_lane(conn, locks):
    """Inv-7.2: research_lane has no LANE_CONFLICTS, so a benchmark task and a specialist coexist."""
    set_lane_capacity(conn.raw, "research_lane", 6)
    bench = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="hb",
        task_id="tb",
        action="bench",
        ttl_sec=60,
    )
    spec = await locks.acquire_many(
        ["research_lane"],
        holder_id="hs",
        task_id="ts",
        action="specialist",
        ttl_sec=60,
    )
    assert "benchmark_lane" in bench.lanes
    assert "research_lane" in spec.lanes
    await locks.release(bench)
    await locks.release(spec)


def test_lane_conflicts_research_lane_isolated():
    assert LANE_CONFLICTS["research_lane"] == frozenset()
    for lane, conflicts in LANE_CONFLICTS.items():
        assert "research_lane" not in conflicts


@pytest.mark.asyncio
async def test_try_acquire_many_returns_lease_on_success(locks):
    lease = await locks.try_acquire_many(
        ["benchmark_lane"],
        holder_id="hb",
        task_id="tb",
        action="bench",
        ttl_sec=60,
    )
    assert lease is not None
    assert lease.holder_id == "hb"
    await locks.release(lease)


@pytest.mark.asyncio
async def test_try_acquire_many_returns_none_on_conflict(conn, locks):
    bench = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="hb",
        task_id="tb",
        action="bench",
        ttl_sec=60,
    )
    result = await locks.try_acquire_many(
        ["profile_lane"],
        holder_id="hp",
        task_id="tp",
        action="profile",
        ttl_sec=60,
    )
    assert result is None
    await locks.release(bench)


@pytest.mark.asyncio
async def test_try_acquire_many_returns_none_on_full(conn, locks):
    """A multi-holder lane at capacity → ``try_acquire_many`` returns None (LaneFull swallowed)."""
    set_lane_capacity(conn.raw, "research_lane", 2)
    leases = [
        await locks.acquire_many(
            ["research_lane"],
            holder_id=f"s{i}",
            task_id=f"t{i}",
            action="specialist",
            ttl_sec=60,
        )
        for i in range(2)
    ]
    result = await locks.try_acquire_many(
        ["research_lane"],
        holder_id="s2",
        task_id="t2",
        action="specialist",
        ttl_sec=60,
    )
    assert result is None
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_heartbeat_only_extends_own_holder_row(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"],
        holder_id="s0",
        task_id="t0",
        action="specialist",
        ttl_sec=30,
    )
    b = await locks.acquire_many(
        ["research_lane"],
        holder_id="s1",
        task_id="t1",
        action="specialist",
        ttl_sec=30,
    )
    await locks.heartbeat(a, ttl_sec=999)
    cur = conn.raw.execute(
        "SELECT holder_id, expires_at FROM leases WHERE lane=? ORDER BY holder_id",
        ("research_lane",),
    )
    rows = list(cur.fetchall())
    by_holder = {r["holder_id"]: r["expires_at"] for r in rows}
    assert by_holder["s0"] > by_holder["s1"]
    await locks.release(a)
    await locks.release(b)


@pytest.mark.asyncio
async def test_release_only_drops_own_holder_row(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"],
        holder_id="s0",
        task_id="t0",
        action="specialist",
        ttl_sec=60,
    )
    b = await locks.acquire_many(
        ["research_lane"],
        holder_id="s1",
        task_id="t1",
        action="specialist",
        ttl_sec=60,
    )
    n = await locks.release(a)
    assert n == 1
    holders = await locks.lane_holders()
    assert holders.get("research_lane") == 1
    await locks.release(b)


@pytest.mark.asyncio
async def test_old_rows_still_count_toward_lane_capacity(conn, locks):
    """Old timestamps cannot release an owner whose completion is unknown."""
    set_lane_capacity(conn.raw, "research_lane", 2)
    # Insert one expired and one live holder directly to bypass acquire_many's reap pass.
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane",
            "dead",
            "td",
            "specialist",
            1,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:01+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane",
            "live",
            "tl",
            "specialist",
            1,
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.commit()
    holders = await locks.lane_holders()
    assert holders.get("research_lane") == 2
    assert (
        await locks.try_acquire_many(
            ["research_lane"], holder_id="next", task_id="next", action="specialist", ttl_sec=60
        )
        is None
    )
    assert not hasattr(locks, "reap_expired")


_T0 = "2026-09-21T07:00:00+00:00"


def _seed_task(conn, task_id: str, state: str) -> None:
    """Give a lane holder the registry row reclamation judges it by."""
    conn.raw.execute(
        "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, created_at, updated_at) "
        "VALUES (?,?,?,'{}',?,?,?)",
        (task_id, "specialist", state, f"idem-{task_id}", _T0, _T0),
    )
    conn.raw.commit()


@pytest.mark.asyncio
async def test_terminal_holder_stops_blocking_its_lanes(conn, locks):
    """2026-09-21: six lanes held by already-terminal holders starved 19 queued tasks for two hours."""
    _seed_task(conn, "told", "failed")
    # Well inside its TTL: the holder's terminal state is what frees the lane.
    lease = await locks.acquire_many(
        ["gpu_research_lane"], holder_id="old", task_id="told", action="specialist", ttl_sec=3600
    )
    # The pid on the row is this very process, so liveness cannot refute it.
    assert await locks.reap_dead_holders() == []
    assert (
        await locks.try_acquire_many(
            ["benchmark_lane"], holder_id="next", task_id="tnext", action="baseline", ttl_sec=60
        )
        is not None
    )
    assert {r["holder_id"] for r in await conn.fetchall("SELECT holder_id FROM leases")} == {"next"}
    # Release is holder-keyed, so the old holder's late release misses the successor.
    assert await locks.release(lease) == 0


@pytest.mark.asyncio
async def test_running_holder_keeps_lanes_past_its_ttl(conn, locks):
    """A lane TTL is a static per-action budget and nothing ends the action when it lapses."""
    _seed_task(conn, "texplore", "running")
    lease = await locks.acquire_many(
        ["server_lifecycle"], holder_id="explore", task_id="texplore", action="explore", ttl_sec=-1
    )
    assert await locks.reap_finished_holders() == []
    assert (
        await locks.try_acquire_many(
            ["benchmark_lane"], holder_id="next", task_id="tnext", action="baseline", ttl_sec=60
        )
        is None
    )
    await locks.release(lease)


@pytest.mark.asyncio
async def test_terminal_holder_with_live_gpu_leases_keeps_its_lanes(conn, locks):
    """A lane records its coordinator, not the specialist GPU worker still holding cards."""
    _seed_task(conn, "tgpu", "succeeded")
    lease = await locks.acquire_many(
        ["gpu_research_lane"], holder_id="gpu", task_id="tgpu", action="specialist", ttl_sec=-1
    )
    conn.raw.execute(
        "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, expires_at, heartbeat_at) "
        "VALUES (0,?,?,?,?,?)",
        ("gpu", "tgpu", _T0, _T0, _T0),
    )
    conn.raw.commit()
    assert await locks.reap_finished_holders() == []
    assert (
        await locks.try_acquire_many(
            ["benchmark_lane"], holder_id="next", task_id="tnext", action="baseline", ttl_sec=60
        )
        is None
    )
    await locks.release(lease)


@pytest.mark.asyncio
async def test_terminal_holder_from_another_scope_is_left_alone(conn, locks):
    """Ownership is judgeable only inside the boot and PID namespace that recorded it."""
    _seed_task(conn, "tfar", "failed")
    await locks.acquire_many(["research_lane"], holder_id="far", task_id="tfar", action="specialist", ttl_sec=60)
    conn.raw.execute("UPDATE leases SET owner_scope='other-boot:4026531836'")
    conn.raw.commit()
    assert await locks.reap_finished_holders() == []
    assert (await locks.lane_holders())["research_lane"] == 1


@pytest.mark.asyncio
async def test_finished_holders_are_swept_without_an_acquire(conn, locks):
    """The dispatcher's lane gate reads holder counts before it ever attempts an acquire."""
    _seed_task(conn, "tdone", "cancelled")
    _seed_task(conn, "tlive", "running")
    _seed_task(conn, "tround", "failed")
    await locks.acquire_many(
        ["gpu_research_lane"], holder_id="done", task_id="tdone", action="specialist", ttl_sec=3600
    )
    await locks.acquire_many(["research_lane"], holder_id="live", task_id="tlive", action="specialist", ttl_sec=3600)
    hold_round_lane(
        conn.raw.cursor(),
        round_id="r1",
        holder_task_id="tround",
        expires_unix=time.time() + 600,
        now_unix=time.time(),
    )
    conn.raw.commit()
    reaped = await locks.reap_finished_holders()
    assert {r["holder_id"] for r in reaped} == {"done"}
    # A round ends in RoundStore alone, even once its holder task is terminal.
    assert await locks.lane_holders() == {"research_lane": 1, BRINGUP_ROUND_LANE: 1}


@pytest.mark.asyncio
async def test_reclaimed_holder_cannot_disturb_its_successor(conn, locks):
    """Release and heartbeat are holder-keyed, so a late one never lands on the successor."""
    _seed_task(conn, "tdone", "failed")
    stale = await locks.acquire_many(
        ["gpu_research_lane"], holder_id="done", task_id="tdone", action="specialist", ttl_sec=3600
    )
    successor = await locks.acquire_many(
        ["gpu_research_lane"], holder_id="next", task_id="tnext", action="specialist", ttl_sec=600
    )
    assert await locks.release(stale) == 0
    with pytest.raises(StaleLeaseError):
        await locks.heartbeat(stale, ttl_sec=600)
    assert (await locks.lane_holders())["gpu_research_lane"] == 1
    await locks.release(successor)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", ["2020-01-01T00:00:00+00:00", "2099-12-31T23:59:59+00:00"])
async def test_reap_dead_holders_releases_crashed_pid(conn, locks, monkeypatch, expires):
    """Confirmed-dead owners are scanned independently of their recorded TTL."""
    import os

    dead_pid = 2_147_483_646
    assert dead_pid != os.getpid()
    monkeypatch.setattr(SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: pid != dead_pid))
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    set_lane_capacity(conn.raw, "benchmark_lane", 1)
    # Long-lived (not expired) lease held by a dead PID.
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "benchmark_lane",
            "zombie",
            "tz",
            "explore",
            dead_pid,
            "2026-01-01T00:00:00+00:00",
            expires,
            "2026-01-01T00:00:00+00:00",
        ),
    )
    # A second lane held by a live PID (this process) must survive.
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "profile_lane",
            "alive",
            "ta",
            "profile",
            os.getpid(),
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.execute("UPDATE leases SET owner_scope='test-node'")
    conn.raw.commit()
    reaped = await locks.reap_dead_holders()
    assert any(r["holder_id"] == "zombie" for r in reaped)
    assert all(r["holder_id"] != "alive" for r in reaped)
    holders = await locks.lane_holders()
    assert "benchmark_lane" not in holders
    assert holders.get("profile_lane") == 1


@pytest.mark.asyncio
async def test_reap_dead_holders_skips_null_pid(conn, locks):
    """A lease with a null/zero pid is never reaped (cannot prove dead)."""
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "benchmark_lane",
            "nopid",
            "tn",
            "explore",
            0,
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.commit()
    reaped = await locks.reap_dead_holders()
    assert reaped == []
    holders = await locks.lane_holders()
    assert holders.get("benchmark_lane") == 1


@pytest.mark.asyncio
async def test_manager_counters_track_acquire_busy_full(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"],
        holder_id="s0",
        task_id="t0",
        action="specialist",
        ttl_sec=60,
    )
    a2 = await locks.acquire_many(
        ["research_lane"],
        holder_id="s1",
        task_id="t1",
        action="specialist",
        ttl_sec=60,
    )
    with pytest.raises(LaneFull):
        await locks.acquire_many(
            ["research_lane"],
            holder_id="s2",
            task_id="t2",
            action="specialist",
            ttl_sec=60,
        )
    # capacity-1 lanes still raise LaneBusy (not LaneFull).
    b = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="hb",
        task_id="tb",
        action="bench",
        ttl_sec=60,
    )
    with pytest.raises(LaneBusy):
        await locks.acquire_many(
            ["profile_lane"],
            holder_id="hp",
            task_id="tp",
            action="profile",
            ttl_sec=60,
        )
    await locks.release(a)
    await locks.release(a2)
    await locks.release(b)


@pytest.mark.asyncio
async def test_lane_holders_distinct(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 3)
    leases = [
        await locks.acquire_many(
            ["research_lane"],
            holder_id=f"s{i}",
            task_id=f"t{i}",
            action="specialist",
            ttl_sec=60,
        )
        for i in range(3)
    ]
    holders = await locks.lane_holders()
    assert holders == {"research_lane": 3}
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_lane_capacities_returns_full_table(conn, locks):
    caps = await locks.lane_capacities()
    for lane in KNOWN_LANES:
        assert lane in caps
    assert caps["research_lane"] == DEFAULT_LANE_CAPACITIES["research_lane"]


def _make_session_with_db(tmp_path: Path) -> tuple[Path, SqliteConnection]:
    session_dir = tmp_path / "session"
    (session_dir / "storage").mkdir(parents=True)
    db = SqliteConnection(session_dir / "storage" / "coordinator.db")
    ensure_schema(db.raw)
    return session_dir, db


@pytest.mark.asyncio
async def test_concurrent_acquires_respect_capacity(conn, locks):
    """Three async acquires racing for capacity=2; exactly one fails."""
    set_lane_capacity(conn.raw, "research_lane", 2)

    async def grab(holder: str):
        try:
            return await locks.acquire_many(
                ["research_lane"],
                holder_id=holder,
                task_id=f"t-{holder}",
                action="specialist",
                ttl_sec=60,
            )
        except LaneFull:
            return None

    results = await asyncio.gather(
        grab("a"),
        grab("b"),
        grab("c"),
    )
    succeeded = [r for r in results if r is not None]
    assert len(succeeded) == 2
    for lease in succeeded:
        await locks.release(lease)


def test_gpu_research_lane_known_and_conflicts_are_symmetric():
    """gpu_research_lane is a known lane, mutually exclusive with serving."""
    assert "gpu_research_lane" in KNOWN_LANES
    assert LANE_CONFLICTS["gpu_research_lane"] == frozenset({"benchmark_lane", "profile_lane", "server_lifecycle"})
    for serving in ("benchmark_lane", "profile_lane", "server_lifecycle"):
        assert "gpu_research_lane" in LANE_CONFLICTS[serving]
    assert "gpu_research_lane" not in LANE_CONFLICTS["gpu_research_lane"]


@pytest.mark.asyncio
async def test_gpu_research_lane_blocks_serving(locks):
    """Holding gpu_research_lane blocks every serving lane."""
    gpu = await locks.acquire_many(
        ["gpu_research_lane"],
        holder_id="g0",
        task_id="tg0",
        action="specialist",
        ttl_sec=60,
    )
    for serving in ("server_lifecycle", "benchmark_lane", "profile_lane"):
        with pytest.raises(LaneBusy) as exc:
            await locks.acquire_many(
                [serving],
                holder_id=f"h-{serving}",
                task_id=f"t-{serving}",
                action="serve",
                ttl_sec=60,
            )
        assert "gpu_research_lane" in exc.value.busy_lanes
    await locks.release(gpu)


@pytest.mark.asyncio
async def test_serving_blocks_gpu_research_lane(locks):
    """Symmetry: a live benchmark blocks a GPU specialist's gpu_research_lane."""
    bench = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="b0",
        task_id="tb0",
        action="bench",
        ttl_sec=60,
    )
    with pytest.raises(LaneBusy) as exc:
        await locks.acquire_many(
            ["gpu_research_lane"],
            holder_id="g0",
            task_id="tg0",
            action="specialist",
            ttl_sec=60,
        )
    assert "benchmark_lane" in exc.value.busy_lanes
    await locks.release(bench)


@pytest.mark.asyncio
async def test_gpu_research_lane_is_strictly_serial(locks):
    """A second GPU specialist is blocked while the first holds the lane."""
    first = await locks.acquire_many(
        ["gpu_research_lane"],
        holder_id="g0",
        task_id="tg0",
        action="specialist",
        ttl_sec=60,
    )
    with pytest.raises(LaneBusy):
        await locks.acquire_many(
            ["gpu_research_lane"],
            holder_id="g1",
            task_id="tg1",
            action="specialist",
            ttl_sec=60,
        )
    await locks.release(first)


@pytest.mark.asyncio
async def test_live_old_owner_keeps_conflicting_lanes_until_release(conn, locks):
    lease = await locks.acquire_many(
        ["gpu_research_lane"], holder_id="old", task_id="old", action="specialist", ttl_sec=-1
    )
    assert await locks.reap_dead_holders() == []
    assert (
        await locks.try_acquire_many(
            ["benchmark_lane"], holder_id="next", task_id="next", action="baseline", ttl_sec=60
        )
        is None
    )
    assert (await locks.lane_holders())["gpu_research_lane"] == 1
    await locks.release(lease)
    assert (
        await locks.try_acquire_many(
            ["benchmark_lane"], holder_id="next", task_id="next", action="baseline", ttl_sec=60
        )
        is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ray", [False, True])
async def test_old_gpu_owner_keeps_capacity_until_release(conn, ray):
    pool = SpecialistGpuPool(conn, gpu_ids=[0])

    async def acquire(holder):
        if ray:
            return await pool.try_acquire_ray_observation(holder_id=holder, task_id=holder, pending_limit=1)
        return await pool.try_acquire(count=1, holder_id=holder, task_id=holder)

    lease = await acquire("old")
    await conn.execute("UPDATE gpu_leases SET expires_at='2020-01-01T00:00:00+00:00'")
    assert await acquire("next") is None
    assert not hasattr(pool, "reap_expired")
    await pool.release(lease)
    assert await acquire("next") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["", "another-node"])
async def test_unknown_owner_scope_is_not_probed(conn, locks, monkeypatch, scope):
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    lease = await locks.acquire_many(["research_lane"], holder_id="old", task_id="old", action="specialist", ttl_sec=-1)
    await conn.execute("UPDATE leases SET owner_scope=?, pid=12345", (scope,))
    monkeypatch.setattr(locks.backend, "_pid_alive", lambda pid: pytest.fail("foreign PID must not be probed"))
    assert await locks.reap_dead_holders() == []
    assert await locks.lane_holders() == {"research_lane": 1}
    await locks.release(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_error", [PermissionError, OSError])
async def test_uncertain_pid_probe_retains_lane(conn, locks, monkeypatch, probe_error):
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    await locks.acquire_many(["research_lane"], holder_id="old", task_id="old", action="specialist", ttl_sec=-1)

    def probe(pid, signal):
        raise probe_error()

    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.os.kill", probe)
    assert await locks.reap_dead_holders() == []


@pytest.mark.asyncio
async def test_dead_coordinator_does_not_release_specialist_gpu_lanes(conn, locks, monkeypatch):
    monkeypatch.setattr("hyperloom.orchestrator.bus.resource_lock.local_owner_scope", lambda: "test-node")
    monkeypatch.setattr(SqliteLeaseBackend, "_pid_alive", staticmethod(lambda pid: False))
    await locks.acquire_many(
        ["gpu_research_lane"], holder_id="specialist", task_id="specialist", action="specialist", ttl_sec=-1
    )
    pool = SpecialistGpuPool(conn, gpu_ids=[0])
    await pool.try_acquire(count=1, holder_id="specialist", task_id="specialist")
    assert await locks.reap_dead_holders() == []
    assert (await locks.lane_holders())["gpu_research_lane"] == 1


def test_legacy_db_gains_unknown_owner_scope(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE leases (lane TEXT, holder_id TEXT, task_id TEXT, action TEXT, pid INTEGER, "
            "acquired_at TEXT, expires_at TEXT, heartbeat_at TEXT, PRIMARY KEY (lane, holder_id))"
        )
        raw.execute("INSERT INTO leases VALUES ('research_lane','old','old','specialist',123,'old','old','old')")
    db = SqliteConnection(path)
    try:
        ensure_schema(db.raw)
        row = db.raw.execute("SELECT owner_scope FROM leases").fetchone()
        assert row["owner_scope"] == ""
    finally:
        db.close()


def test_gpu_research_lane_seeded_capacity_one():
    """A fresh DB seeds gpu_research_lane at capacity 1 (strictly serial)."""
    assert DEFAULT_LANE_CAPACITIES.get("gpu_research_lane") == 1
