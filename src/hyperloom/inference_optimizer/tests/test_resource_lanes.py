# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""research_lane capacity + concurrent dispatcher tests."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
from hyperloom.orchestrator.bus.resource_lock import (
    KNOWN_LANES,
    LANE_CONFLICTS,
    LaneBusy,
    LaneFull,
    ResourceLockManager,
    SqliteLeaseBackend,
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


def test_schema_version_is_v3():
    """v3 adds the specialist GPU pool leases table."""
    assert SCHEMA_VERSION == 3


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


def test_v1_to_v2_migration_preserves_rows(tmp_path):
    """Spin up a v1 DB by hand and let ``ensure_schema`` migrate it."""
    p = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(p))
    raw.row_factory = sqlite3.Row
    raw.execute("""
        CREATE TABLE leases (
            lane TEXT PRIMARY KEY,
            holder_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            action TEXT NOT NULL,
            pid INTEGER NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL
        )
    """)
    raw.execute(
        "INSERT INTO leases VALUES (?,?,?,?,?,?,?,?)",
        (
            "benchmark_lane",
            "h1",
            "t1",
            "bench",
            12345,
            "2026-05-19T18:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-05-19T18:00:00+00:00",
        ),
    )
    raw.commit()
    raw.close()

    db = SqliteConnection(p)
    v = ensure_schema(db.raw)
    assert v == 3
    cur = db.raw.execute("PRAGMA table_info(leases)")
    pk_cols = sorted(row["name"] for row in cur.fetchall() if int(row["pk"] or 0) > 0)
    assert pk_cols == ["holder_id", "lane"]
    cur = db.raw.execute("SELECT * FROM leases")
    rows = [dict(r) for r in cur.fetchall()]
    assert len(rows) == 1
    assert rows[0]["lane"] == "benchmark_lane"
    assert rows[0]["holder_id"] == "h1"
    db.close()


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
async def test_reap_expired_keys_on_holder_id(conn, locks):
    """``reap_expired`` deletes only the expired ``(lane, holder_id)`` row, not the whole lane."""
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
    reaped = await locks.reap_expired()
    assert any(r["holder_id"] == "dead" for r in reaped)
    holders = await locks.lane_holders()
    assert holders.get("research_lane") == 1
    cur = conn.raw.execute(
        "SELECT holder_id FROM leases WHERE lane=?",
        ("research_lane",),
    )
    surviving = [r["holder_id"] for r in cur.fetchall()]
    assert surviving == ["live"]


@pytest.mark.asyncio
async def test_reap_dead_holders_releases_crashed_pid(conn, locks):
    """A not-yet-expired lease whose holder PID is dead is reaped immediately."""
    import os

    dead_pid = 2_147_483_646
    assert dead_pid != os.getpid()
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
            "2099-12-31T23:59:59+00:00",
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
async def test_collect_lane_timeline_summarises_capacity_and_holders(tmp_path):
    """lane_timeline row per known lane + __total__ aggregate."""
    from hyperloom.inference_optimizer.breakdown.collectors import _collect_lane_timeline

    session_dir, db = _make_session_with_db(tmp_path)
    set_lane_capacity(db.raw, "research_lane", 6)
    locks = ResourceLockManager(SqliteLeaseBackend(db))
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
    bench = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id="hb",
        task_id="tb",
        action="bench",
        ttl_sec=60,
    )
    warnings: list[str] = []
    rows = _collect_lane_timeline(session_dir, warnings)
    by_lane = {r["lane"]: r for r in rows}
    assert by_lane["research_lane"]["capacity"] == 6
    assert by_lane["research_lane"]["live_holders"] == 3
    assert by_lane["benchmark_lane"]["capacity"] == 1
    assert by_lane["benchmark_lane"]["live_holders"] == 1
    assert by_lane["__total__"]["live_holders"] >= 4
    assert warnings == []
    for l in leases:
        await locks.release(l)
    await locks.release(bench)
    db.close()


def test_collect_lane_timeline_missing_db_returns_empty(tmp_path):
    from hyperloom.inference_optimizer.breakdown.collectors import _collect_lane_timeline

    session_dir = tmp_path / "no_session"
    warnings: list[str] = []
    rows = _collect_lane_timeline(session_dir, warnings)
    assert rows == []
    assert warnings == []


def test_collect_lane_timeline_legacy_db_without_lane_capacity(tmp_path):
    """Legacy DBs without ``lane_capacity`` still produce a sensible lane_timeline using the default table."""
    from hyperloom.inference_optimizer.breakdown.collectors import _collect_lane_timeline

    session_dir = tmp_path / "legacy"
    (session_dir / "storage").mkdir(parents=True)
    raw = sqlite3.connect(str(session_dir / "storage" / "coordinator.db"))
    raw.execute("""
        CREATE TABLE leases (
            lane TEXT PRIMARY KEY,
            holder_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            action TEXT NOT NULL,
            pid INTEGER NOT NULL,
            acquired_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL
        )
    """)
    raw.commit()
    raw.close()
    warnings: list[str] = []
    rows = _collect_lane_timeline(session_dir, warnings)
    by_lane = {r["lane"]: r for r in rows if r["lane"] != "__total__"}
    for lane, cap in DEFAULT_LANE_CAPACITIES.items():
        assert by_lane[lane]["capacity"] == cap


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
    """gpu_research_lane is a known lane, mutually exclusive with serving.

    Conflicts must be declared symmetrically — each serving lane must list
    gpu_research_lane and vice versa.
    """
    assert "gpu_research_lane" in KNOWN_LANES
    assert LANE_CONFLICTS["gpu_research_lane"] == frozenset(
        {"benchmark_lane", "profile_lane", "server_lifecycle"}
    )
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
    """A second GPU specialist is blocked while the first holds the lane.

    gpu_research_lane is capacity-1 / strictly serial (one GPU specialist holds
    the whole machine at a time).
    """
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


def test_gpu_research_lane_seeded_capacity_one():
    """A fresh DB seeds gpu_research_lane at capacity 1 (strictly serial)."""
    assert DEFAULT_LANE_CAPACITIES.get("gpu_research_lane") == 1
