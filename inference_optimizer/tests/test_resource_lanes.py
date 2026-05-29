"""v0.8 M6 — research_lane capacity + concurrent dispatcher tests.

Covers KB_design §3.7 (resource lane redesign) + §3.13 M6:

* Schema migration: legacy v1 (PK=lane) → v2 (PK=(lane, holder_id))
  preserves row data.
* ``lane_capacity`` table is seeded with defaults; ``set_lane_capacity``
  upserts.
* Multi-holder ``research_lane`` (capacity > 1) admits the configured
  number of holders, then ``LaneFull`` (not ``LaneBusy``) on overflow.
* Single-holder serving lanes (capacity=1) still raise ``LaneBusy``
  for cross-lane conflicts (Inv-7.1).
* ``try_acquire_many`` is the non-blocking variant — returns ``None``
  on conflict / full.
* Same-holder idempotent retry refreshes the lease without raising.
* Heartbeat / release / reap_expired all key on (lane, holder_id) so a
  busy lane reaps only the expired holder.
* ``ResourceLockManager`` counters track per-lane acquire / busy /
  full counts.
* ``lane_holders`` / ``lane_capacities`` observability helpers.
* ``breakdown.telemetry.lane_timeline`` exposes per-lane summary.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.resource_lock import (
    KNOWN_LANES,
    LANE_CONFLICTS,
    LaneBusy,
    LaneFull,
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.storage import SqliteConnection
from inference_optimizer.storage.schema import (
    DEFAULT_LANE_CAPACITIES,
    SCHEMA_VERSION,
    ensure_schema,
    get_lane_capacity,
    set_lane_capacity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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


# ===========================================================================
# 1. Schema — v1 → v2 migration + defaults
# ===========================================================================
def test_schema_version_is_v2():
    """v0.8 M6 bumps schema_version 1 → 2 to mark the PK widening."""
    assert SCHEMA_VERSION == 2


def test_fresh_db_has_composite_pk(conn):
    cur = conn.raw.execute("PRAGMA table_info(leases)")
    pk_cols = sorted(
        row["name"] for row in cur.fetchall() if int(row["pk"] or 0) > 0
    )
    assert pk_cols == ["holder_id", "lane"]


def test_fresh_db_seeds_default_lane_capacity(conn):
    cur = conn.raw.execute(
        "SELECT lane, capacity FROM lane_capacity ORDER BY lane",
    )
    rows = {r["lane"]: int(r["capacity"]) for r in cur.fetchall()}
    for lane, cap in DEFAULT_LANE_CAPACITIES.items():
        assert rows[lane] == cap


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
            "benchmark_lane", "h1", "t1", "bench", 12345,
            "2026-05-19T18:00:00+00:00",
            "2099-12-31T23:59:59+00:00",  # very far future
            "2026-05-19T18:00:00+00:00",
        ),
    )
    raw.commit()
    raw.close()

    db = SqliteConnection(p)
    v = ensure_schema(db.raw)
    assert v == 2
    cur = db.raw.execute("PRAGMA table_info(leases)")
    pk_cols = sorted(
        row["name"] for row in cur.fetchall() if int(row["pk"] or 0) > 0
    )
    assert pk_cols == ["holder_id", "lane"]
    cur = db.raw.execute("SELECT * FROM leases")
    rows = [dict(r) for r in cur.fetchall()]
    assert len(rows) == 1
    assert rows[0]["lane"] == "benchmark_lane"
    assert rows[0]["holder_id"] == "h1"
    db.close()


def test_v2_ensure_schema_is_idempotent(conn):
    """Calling ensure_schema twice on the same DB doesn't lose data."""
    cur = conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane", "h1", "t1", "specialist", 1,
            "2026-05-19T18:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-05-19T18:00:00+00:00",
        ),
    )
    conn.raw.commit()
    ensure_schema(conn.raw)
    cur = conn.raw.execute("SELECT COUNT(*) AS n FROM leases")
    assert int(cur.fetchone()["n"]) == 1


# ===========================================================================
# 2. acquire_many: capacity, LaneFull vs LaneBusy
# ===========================================================================
@pytest.mark.asyncio
async def test_serving_lane_capacity_1_raises_LaneBusy(locks):
    a = await locks.acquire_many(
        ["benchmark_lane"], holder_id="ha", task_id="ta",
        action="bench", ttl_sec=60,
    )
    with pytest.raises(LaneBusy) as exc:
        await locks.acquire_many(
            ["benchmark_lane"], holder_id="hb", task_id="tb",
            action="bench", ttl_sec=60,
        )
    assert "benchmark_lane" in exc.value.busy_lanes
    await locks.release(a)


@pytest.mark.asyncio
async def test_research_lane_capacity_admits_multiple_holders(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 3)
    leases = []
    for i in range(3):
        l = await locks.acquire_many(
            ["research_lane"], holder_id=f"s{i}", task_id=f"t{i}",
            action="specialist", ttl_sec=60,
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
            ["research_lane"], holder_id=f"s{i}", task_id=f"t{i}",
            action="specialist", ttl_sec=60,
        )
        for i in range(2)
    ]
    with pytest.raises(LaneFull) as exc:
        await locks.acquire_many(
            ["research_lane"], holder_id="s2", task_id="t2",
            action="specialist", ttl_sec=60,
        )
    assert "research_lane" in exc.value.full_lanes
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_capacity_zero_means_lane_disabled(conn, locks):
    """``--research-lane-capacity 0`` semantics (KB_design §3.7 §4.4)."""
    set_lane_capacity(conn.raw, "research_lane", 0)
    with pytest.raises(LaneFull):
        await locks.acquire_many(
            ["research_lane"], holder_id="s0", task_id="t0",
            action="specialist", ttl_sec=60,
        )


@pytest.mark.asyncio
async def test_same_holder_retry_is_idempotent(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 1)
    a = await locks.acquire_many(
        ["research_lane"], holder_id="s0", task_id="t0",
        action="specialist", ttl_sec=30,
    )
    # Same holder retries with a fresh TTL → succeeds (refresh).
    b = await locks.acquire_many(
        ["research_lane"], holder_id="s0", task_id="t0",
        action="specialist", ttl_sec=120,
    )
    assert a.holder_id == b.holder_id
    # Only one row in DB.
    cur = conn.raw.execute("SELECT COUNT(*) AS n FROM leases WHERE lane=?",
                            ("research_lane",))
    assert int(cur.fetchone()["n"]) == 1
    await locks.release(b)


# ===========================================================================
# 3. Cross-lane conflicts (Inv-7.2): research_lane independent of serving
# ===========================================================================
@pytest.mark.asyncio
async def test_research_lane_independent_of_benchmark_lane(conn, locks):
    """KB_design §3.7 Inv-7.2: research_lane has no LANE_CONFLICTS so a
    benchmark task and a specialist can coexist on the same tick."""
    set_lane_capacity(conn.raw, "research_lane", 6)
    bench = await locks.acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb",
        action="bench", ttl_sec=60,
    )
    spec = await locks.acquire_many(
        ["research_lane"], holder_id="hs", task_id="ts",
        action="specialist", ttl_sec=60,
    )
    assert "benchmark_lane" in bench.lanes
    assert "research_lane" in spec.lanes
    await locks.release(bench)
    await locks.release(spec)


def test_lane_conflicts_research_lane_isolated():
    assert LANE_CONFLICTS["research_lane"] == frozenset()
    for lane, conflicts in LANE_CONFLICTS.items():
        assert "research_lane" not in conflicts


# ===========================================================================
# 4. try_acquire_many: non-blocking variant
# ===========================================================================
@pytest.mark.asyncio
async def test_try_acquire_many_returns_lease_on_success(locks):
    lease = await locks.try_acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb",
        action="bench", ttl_sec=60,
    )
    assert lease is not None
    assert lease.holder_id == "hb"
    await locks.release(lease)


@pytest.mark.asyncio
async def test_try_acquire_many_returns_none_on_conflict(conn, locks):
    bench = await locks.acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb",
        action="bench", ttl_sec=60,
    )
    # Cross-lane mutex.
    result = await locks.try_acquire_many(
        ["profile_lane"], holder_id="hp", task_id="tp",
        action="profile", ttl_sec=60,
    )
    assert result is None
    await locks.release(bench)


@pytest.mark.asyncio
async def test_try_acquire_many_returns_none_on_full(conn, locks):
    """Multi-holder lane reaching capacity → ``try_acquire_many`` returns
    None (LaneFull swallowed)."""
    set_lane_capacity(conn.raw, "research_lane", 2)
    leases = [
        await locks.acquire_many(
            ["research_lane"], holder_id=f"s{i}", task_id=f"t{i}",
            action="specialist", ttl_sec=60,
        )
        for i in range(2)
    ]
    result = await locks.try_acquire_many(
        ["research_lane"], holder_id="s2", task_id="t2",
        action="specialist", ttl_sec=60,
    )
    assert result is None
    for l in leases:
        await locks.release(l)


# ===========================================================================
# 5. Multi-holder heartbeat / release / reap_expired
# ===========================================================================
@pytest.mark.asyncio
async def test_heartbeat_only_extends_own_holder_row(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"], holder_id="s0", task_id="t0",
        action="specialist", ttl_sec=30,
    )
    b = await locks.acquire_many(
        ["research_lane"], holder_id="s1", task_id="t1",
        action="specialist", ttl_sec=30,
    )
    await locks.heartbeat(a, ttl_sec=999)
    cur = conn.raw.execute(
        "SELECT holder_id, expires_at FROM leases "
        "WHERE lane=? ORDER BY holder_id",
        ("research_lane",),
    )
    rows = list(cur.fetchall())
    by_holder = {r["holder_id"]: r["expires_at"] for r in rows}
    assert by_holder["s0"] > by_holder["s1"]   # only s0 refreshed
    await locks.release(a)
    await locks.release(b)


@pytest.mark.asyncio
async def test_release_only_drops_own_holder_row(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"], holder_id="s0", task_id="t0",
        action="specialist", ttl_sec=60,
    )
    b = await locks.acquire_many(
        ["research_lane"], holder_id="s1", task_id="t1",
        action="specialist", ttl_sec=60,
    )
    n = await locks.release(a)
    assert n == 1
    holders = await locks.lane_holders()
    assert holders.get("research_lane") == 1
    await locks.release(b)


@pytest.mark.asyncio
async def test_reap_expired_keys_on_holder_id(conn, locks):
    """``reap_expired`` deletes only the expired ``(lane, holder_id)``
    row, not the whole lane (KB_design §3.7 §4.1 multi-holder atomic
    release)."""
    set_lane_capacity(conn.raw, "research_lane", 2)
    # Insert one already-expired and one live holder directly so we
    # bypass acquire_many's own expired-row reap pass.
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane", "dead", "td", "specialist", 1,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:01+00:00",   # already expired
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
        "acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "research_lane", "live", "tl", "specialist", 1,
            "2026-01-01T00:00:00+00:00",
            "2099-12-31T23:59:59+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.raw.commit()
    reaped = await locks.reap_expired()
    assert any(r["holder_id"] == "dead" for r in reaped)
    holders = await locks.lane_holders()
    # The "live" holder survives the reap.
    assert holders.get("research_lane") == 1
    cur = conn.raw.execute(
        "SELECT holder_id FROM leases WHERE lane=?", ("research_lane",),
    )
    surviving = [r["holder_id"] for r in cur.fetchall()]
    assert surviving == ["live"]


# ===========================================================================
# 6. Manager counters + observability
# ===========================================================================
@pytest.mark.asyncio
async def test_manager_counters_track_acquire_busy_full(conn, locks):
    # Multi-holder lane (cap >= 2) raises LaneFull on overflow so the
    # dispatcher can distinguish "wait for capacity slot" from
    # "wait for cross-lane mutex".
    set_lane_capacity(conn.raw, "research_lane", 2)
    a = await locks.acquire_many(
        ["research_lane"], holder_id="s0", task_id="t0",
        action="specialist", ttl_sec=60,
    )
    a2 = await locks.acquire_many(
        ["research_lane"], holder_id="s1", task_id="t1",
        action="specialist", ttl_sec=60,
    )
    with pytest.raises(LaneFull):
        await locks.acquire_many(
            ["research_lane"], holder_id="s2", task_id="t2",
            action="specialist", ttl_sec=60,
        )
    # v0.6 single-holder semantics preserved: capacity-1 lanes raise
    # LaneBusy (not LaneFull) so existing v0.6 tests still match.
    b = await locks.acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb",
        action="bench", ttl_sec=60,
    )
    with pytest.raises(LaneBusy):
        await locks.acquire_many(
            ["profile_lane"], holder_id="hp", task_id="tp",
            action="profile", ttl_sec=60,
        )
    counters = locks.counters_snapshot()
    assert counters["research_lane"]["acquire_count"] == 2
    assert counters["research_lane"]["lane_full_count"] == 1
    assert counters["benchmark_lane"]["acquire_count"] >= 1
    # The cross-lane conflict shows up on profile_lane (the requested
    # lane that we couldn't acquire) AND on its conflict expansions.
    assert "lane_busy_count" in counters.get("profile_lane", {})
    await locks.release(a)
    await locks.release(a2)
    await locks.release(b)


@pytest.mark.asyncio
async def test_lane_holders_distinct(conn, locks):
    set_lane_capacity(conn.raw, "research_lane", 3)
    leases = [
        await locks.acquire_many(
            ["research_lane"], holder_id=f"s{i}", task_id=f"t{i}",
            action="specialist", ttl_sec=60,
        )
        for i in range(3)
    ]
    holders = await locks.lane_holders()
    assert holders == {"research_lane": 3}
    actives = await locks.active_lanes()
    assert actives == ["research_lane"]   # DISTINCT
    for l in leases:
        await locks.release(l)


@pytest.mark.asyncio
async def test_lane_capacities_returns_full_table(conn, locks):
    caps = await locks.lane_capacities()
    # Every known lane in the defaults.
    for lane in KNOWN_LANES:
        assert lane in caps
    assert caps["research_lane"] == DEFAULT_LANE_CAPACITIES["research_lane"]


# ===========================================================================
# 7. breakdown.telemetry.lane_timeline
# ===========================================================================
def _make_session_with_db(tmp_path: Path) -> tuple[Path, SqliteConnection]:
    session_dir = tmp_path / "session"
    (session_dir / "storage").mkdir(parents=True)
    db = SqliteConnection(session_dir / "storage" / "coordinator.db")
    ensure_schema(db.raw)
    return session_dir, db


@pytest.mark.asyncio
async def test_collect_lane_timeline_summarises_capacity_and_holders(tmp_path):
    """lane_timeline row per known lane + __total__ aggregate."""
    from inference_optimizer.breakdown.collectors import _collect_lane_timeline

    session_dir, db = _make_session_with_db(tmp_path)
    set_lane_capacity(db.raw, "research_lane", 6)
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    leases = [
        await locks.acquire_many(
            ["research_lane"], holder_id=f"s{i}", task_id=f"t{i}",
            action="specialist", ttl_sec=60,
        )
        for i in range(3)
    ]
    bench = await locks.acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb",
        action="bench", ttl_sec=60,
    )
    warnings: list[str] = []
    rows = _collect_lane_timeline(session_dir, warnings)
    by_lane = {r["lane"]: r for r in rows}
    assert by_lane["research_lane"]["capacity"] == 6
    assert by_lane["research_lane"]["live_holders"] == 3
    assert by_lane["benchmark_lane"]["capacity"] == 1
    assert by_lane["benchmark_lane"]["live_holders"] == 1
    # __total__ summary row.
    assert by_lane["__total__"]["live_holders"] >= 4
    assert warnings == []
    for l in leases:
        await locks.release(l)
    await locks.release(bench)
    db.close()


def test_collect_lane_timeline_missing_db_returns_empty(tmp_path):
    from inference_optimizer.breakdown.collectors import _collect_lane_timeline

    session_dir = tmp_path / "no_session"
    warnings: list[str] = []
    rows = _collect_lane_timeline(session_dir, warnings)
    assert rows == []
    assert warnings == []


def test_collect_lane_timeline_legacy_db_without_lane_capacity(tmp_path):
    """Legacy (pre-M6) DBs without ``lane_capacity`` still produce a
    sensible lane_timeline using the default capacity table."""
    from inference_optimizer.breakdown.collectors import _collect_lane_timeline

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
    # All defaults should be present (capacity 1 each).
    by_lane = {r["lane"]: r for r in rows if r["lane"] != "__total__"}
    for lane, cap in DEFAULT_LANE_CAPACITIES.items():
        assert by_lane[lane]["capacity"] == cap


# ===========================================================================
# 8. Concurrent acquire_many under asyncio.gather
# ===========================================================================
@pytest.mark.asyncio
async def test_concurrent_acquires_respect_capacity(conn, locks):
    """Three async acquires racing for capacity=2; exactly one fails."""
    set_lane_capacity(conn.raw, "research_lane", 2)

    async def grab(holder: str):
        try:
            return await locks.acquire_many(
                ["research_lane"], holder_id=holder, task_id=f"t-{holder}",
                action="specialist", ttl_sec=60,
            )
        except LaneFull:
            return None

    results = await asyncio.gather(
        grab("a"), grab("b"), grab("c"),
    )
    succeeded = [r for r in results if r is not None]
    assert len(succeeded) == 2
    for lease in succeeded:
        await locks.release(lease)
