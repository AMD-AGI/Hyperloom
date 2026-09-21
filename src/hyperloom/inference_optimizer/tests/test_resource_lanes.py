# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""research_lane capacity + concurrent dispatcher tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.bus import resource_lock
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

#: What the runner records when the release ran and teardown was acknowledged.
_CLEANED = {"cleanup_confirmed": True}


def _unconfirmed(tree_pgid: int | None = None) -> dict:
    """Build the cleanup evidence of a holder that retained its lane.

    What the runner records on the path that deliberately SKIPS the release,
    with the process group the raise site named when it could name one. Omitting
    it is not only a raise site with nothing to name: it is also the exact shape
    every row written before this code shipped carries, 2026-09-21's included.
    """
    evidence: dict = {"cleanup_confirmed": False, "cleanup_error": "physical cleanup unconfirmed"}
    if tree_pgid is not None:
        evidence["cleanup_tree_pgid"] = tree_pgid
    return evidence


@pytest.fixture(autouse=True)
def _forget_logged_remedies():
    """The remedy log de-duplicates for the life of the process, not of one test."""
    resource_lock._DIAGNOSED.clear()
    yield
    resource_lock._DIAGNOSED.clear()


def _seed_task(conn, task_id: str, state: str, *, evidence: dict | None = None) -> None:
    """Give a lane holder the registry row and cleanup account reclamation judges it by."""
    history = json.dumps([{"from": "running", "to": state, "ts": _T0, "evidence": evidence or _CLEANED}])
    conn.raw.execute(
        "INSERT INTO tasks(task_id, kind, state, params, idempotency_key, history, created_at, updated_at) "
        "VALUES (?,?,?,'{}',?,?,?,?)",
        (task_id, "specialist", state, f"idem-{task_id}", history, _T0, _T0),
    )
    conn.raw.commit()


def _dead_tree_pgid() -> int:
    """A process-group id whose group has certainly emptied.

    Spawned with ``start_new_session``, as both real launch sites do, so the
    child's pid is its group id -- and once it is reaped neither the process nor
    the group it led has a member left.
    """
    proc = subprocess.Popen([sys.executable, "-c", ""], start_new_session=True)  # nosec B603
    proc.wait()
    return proc.pid


@contextlib.contextmanager
def _live_tree(*, own_session: bool = True):
    """A real process, alive for the body of the test.

    Args:
        own_session: Whether it leads a session of its own, as a specialist
            launched with ``start_new_session`` does. False reproduces a root
            that shares its launcher's process group, where its own pid names
            no group at all and only walking the tree can see it.
    """
    proc = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=own_session
    )
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.asyncio
async def test_a_holder_that_confirmed_its_cleanup_stops_blocking_its_lanes(conn, locks):
    """Cleanup confirmed means the release ran: nothing is left on the lane to wait for."""
    _seed_task(conn, "told", "failed")
    # Well inside its TTL: the holder's own account of stopping frees the lane.
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
async def test_a_cleanup_unconfirmed_holder_keeps_its_lane_while_its_tree_runs(conn, locks):
    """The retention ExecutionCleanupUnconfirmed exists for: the tree may still be working.

    This is the case the ``TERMINAL_STATES`` rule got wrong on its own. The
    task row says failed and the release never ran, and that is exactly when
    the lane must stay taken -- a live tree is still using the machine.
    """
    with _live_tree() as tree:
        _seed_task(conn, "tlive", "failed", evidence=_unconfirmed(tree.pid))
        await locks.acquire_many(
            ["gpu_research_lane"], holder_id="live", task_id="tlive", action="specialist", ttl_sec=3600
        )
        assert await locks.reap_finished_holders() == []
        assert (
            await locks.try_acquire_many(
                ["benchmark_lane"], holder_id="next", task_id="tnext", action="baseline", ttl_sec=60
            )
            is None
        )
        assert (await locks.lane_holders())["gpu_research_lane"] == 1
        # Reclamation answers a question; it never ends the work it asked about.
        assert tree.poll() is None


@pytest.mark.asyncio
async def test_a_live_root_outside_a_group_of_its_own_keeps_its_lane(conn, locks):
    """A root that shares its launcher's group is invisible to a group probe.

    ``killpg`` on such a root's pid names no group and reads exactly like an
    empty one, so enumerating the tree from the root is the only thing standing
    between a running process and having its lane handed to somebody else.
    """
    with _live_tree(own_session=False) as tree:
        _seed_task(conn, "tnogroup", "failed", evidence=_unconfirmed(tree.pid))
        await locks.acquire_many(
            ["gpu_research_lane"], holder_id="nogroup", task_id="tnogroup", action="specialist", ttl_sec=3600
        )
        assert await locks.reap_finished_holders() == []
        assert (await locks.lane_holders())["gpu_research_lane"] == 1
        assert tree.poll() is None


@contextlib.contextmanager
def _orphaned_tree_root():
    """A root that exits leaving a live child behind in the group it created.

    The ``"exited root has no verifiable tree"`` case, built for real: the child
    re-parents away and no longer appears under the root, so walking the tree
    from the root finds nothing, while the process group the root started still
    has a member using the machine.
    """
    spawn = "import subprocess,sys;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);print(p.pid)"
    root = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", spawn], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    child_pid = int(root.stdout.readline())
    root.wait()
    try:
        yield root.pid, child_pid
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_holder_whose_root_died_but_left_the_group_running_keeps_its_lane(conn, locks):
    """The root being gone is not the tree being gone, which is why the group is probed too.

    A descendant that outlived its root is still doing whatever the lane
    serialises, and walking down from a dead root would report an empty tree and
    hand the lane away underneath it.
    """
    with _orphaned_tree_root() as (root_pid, child_pid):
        _seed_task(conn, "torphan", "failed", evidence=_unconfirmed(root_pid))
        await locks.acquire_many(
            ["gpu_research_lane"], holder_id="orphan", task_id="torphan", action="specialist", ttl_sec=3600
        )
        assert await locks.reap_finished_holders() == []
        assert (await locks.lane_holders())["gpu_research_lane"] == 1
        # And the survivor was only ever asked about, never ended.
        assert os.kill(child_pid, 0) is None


@pytest.mark.asyncio
async def test_a_cleanup_unconfirmed_holder_is_reclaimed_once_its_tree_is_gone(conn, locks):
    """A RECURRENCE of 2026-09-21 under code that records a group, and so heals it.

    Cleanup never confirmed, yet nothing of the execution is left. Reclaiming on
    ``cleanup_confirmed`` alone would hold these lanes for ever, which is the
    incident; probing the recorded process group is what tells them apart from
    the live holder above. The incident's own rows are NOT this shape -- they
    carry no group id at all; see
    ``test_an_incident_shaped_row_stays_held_and_hands_the_operator_the_remedy``.
    """
    _seed_task(conn, "tgone", "failed", evidence=_unconfirmed(_dead_tree_pgid()))
    await locks.acquire_many(
        ["gpu_research_lane"], holder_id="gone", task_id="tgone", action="specialist", ttl_sec=3600
    )
    reaped = await locks.reap_finished_holders()
    # The acquire took gpu_research_lane plus the three lanes it mutexes with.
    assert {r["lane"] for r in reaped} == {
        "gpu_research_lane",
        "benchmark_lane",
        "profile_lane",
        "server_lifecycle",
    }
    assert await locks.lane_holders() == {}


@pytest.mark.asyncio
async def test_a_holder_whose_group_was_never_recorded_keeps_its_lane(conn, locks):
    """Nothing to probe is not proof of anything, so the lane is never taken back.

    Silence reads the same whether the processes finished or are still running,
    and most raise sites hold no GPU lease, so nothing else speaks for the lane
    either. Held indefinitely is the deliberate answer -- no age, no TTL, no
    guess.
    """
    _seed_task(conn, "tmute", "failed", evidence=_unconfirmed())
    await locks.acquire_many(
        ["gpu_research_lane"], holder_id="mute", task_id="tmute", action="specialist", ttl_sec=3600
    )
    assert await locks.reap_finished_holders() == []
    assert (await locks.lane_holders())["gpu_research_lane"] == 1


@contextlib.contextmanager
def _survivor_that_left_the_group():
    """A root that exits leaving a child which gave itself a session of its own.

    The one survivor shape the probe is blind to, built for real: the child
    re-parents away from the dead root AND ``setsid``s out of the root's group,
    so neither walking the tree nor ``killpg`` on the recorded id finds it.
    """
    spawn = (
        "import subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True);"
        "print(p.pid)"
    )
    root = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", spawn], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    child_pid = int(root.stdout.readline())
    root.wait()
    try:
        yield root.pid, child_pid
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_survivor_that_calls_setsid_is_past_what_this_probe_can_see(conn, locks):
    """The known limitation, pinned so a reader finds it stated rather than missed.

    A descendant that gives itself a new session leaves the recorded group, and
    with its root already dead nothing in ``/proc`` ties it back to the recorded
    id. The lane is therefore released while that process is still running --
    asserted here deliberately, because the alternative is a reader assuming the
    group probe covers every survivor. A per-spawn cgroup would name it, and is
    a separate project.
    """
    with _survivor_that_left_the_group() as (pgid, survivor_pid):
        _seed_task(conn, "tsetsid", "failed", evidence=_unconfirmed(pgid))
        await locks.acquire_many(
            ["build_lane"], holder_id="setsid", task_id="tsetsid", action="specialist", ttl_sec=3600
        )

        assert [r["lane"] for r in await locks.reap_finished_holders()] == ["build_lane"]
        assert await locks.lane_holders() == {}
        # The blind spot in one line: still running, and its lane already gone.
        assert os.kill(survivor_pid, 0) is None


def _remedies(caplog) -> list[str]:
    """Every operator-facing warning the lane sweep logged."""
    return [r.getMessage() for r in caplog.records if r.name == "hyperloom.orchestrator.bus.resource_lock"]


@pytest.mark.asyncio
async def test_an_incident_shaped_row_stays_held_and_hands_the_operator_the_remedy(conn, locks, caplog):
    """2026-09-21's own rows are not auto-healed, and are not meant to be.

    The code that wrote them recorded ``cleanup_confirmed=False`` and no group
    id, so they take the unverifiable path and keep their lanes -- for the life
    of the session if need be, because no age or TTL may break the tie and a
    row like this reads exactly like a holder still on the machine.

    What the operator gets instead is this warning: the lane, the holder, why it
    could not be settled, and the statement that releases that one row, with the
    check that makes running it safe. On the day it took an hour of ``py-spy``
    and sqlite spelunking to arrive at the same command.
    """
    _seed_task(conn, "tincident", "failed", evidence={"cleanup_confirmed": False})
    await locks.acquire_many(
        ["build_lane"], holder_id="incident", task_id="tincident", action="specialist", ttl_sec=3600
    )
    assert await locks.reap_finished_holders() == []

    with caplog.at_level(logging.WARNING):
        stuck = await locks.diagnose_unverifiable_holders()
        # A sweep runs every maintenance tick, for the life of the session.
        assert len(await locks.diagnose_unverifiable_holders()) == 1

    assert [(r["lane"], r["task_id"], r["reason"]) for r in stuck] == [
        ("build_lane", "tincident", resource_lock.UNVERIFIABLE_NO_IDENTITY)
    ]
    assert (await locks.lane_holders())["build_lane"] == 1
    remedies = _remedies(caplog)
    assert len(remedies) == 1
    assert resource_lock.UNVERIFIABLE_NO_IDENTITY in remedies[0]
    assert "FIRST confirm no process of task tincident is still running" in remedies[0]
    # The remedy names this session's real database, not a $SESSION_DIR the
    # operator would have to resolve before they could paste it.
    assert f'sqlite3 "{conn.db_path}" ' in remedies[0]
    assert "\"DELETE FROM leases WHERE lane='build_lane' AND holder_id='incident';\"" in remedies[0]
    assert "$SESSION_DIR" not in remedies[0]


@pytest.mark.asyncio
async def test_a_lease_that_outlived_its_pruned_task_is_still_reported(conn, locks, caplog):
    """A holder pruned out from under its lease must not vanish from the report.

    ``prune_tasks`` deletes finished task rows without sparing one that still
    holds a lease, and reclamation joins through ``tasks`` — so such a row can
    be held with nothing able to judge it. Retaining unverifiable rows makes
    that MORE likely, not less, because they now outlive their task on purpose.
    Silence here would be the 2026-09-21 outcome with no log line at all.
    """
    await locks.acquire_many(["build_lane"], holder_id="orphan", task_id="torphan", action="specialist", ttl_sec=3600)
    _seed_task(conn, "torphan", "succeeded", evidence={resource_lock.CLEANUP_CONFIRMED_KEY: False})
    conn.raw.execute("DELETE FROM tasks WHERE task_id=?", ("torphan",))
    conn.raw.commit()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stuck = await locks.diagnose_unverifiable_holders()

    assert [(r["lane"], r["reason"]) for r in stuck] == [("build_lane", resource_lock.UNVERIFIABLE_HOLDER_PRUNED)]
    assert (await locks.lane_holders())["build_lane"] == 1
    assert len(_remedies(caplog)) == 1


@pytest.mark.asyncio
async def test_a_gpu_exempt_holder_is_retained_but_still_reported(conn, locks, caplog):
    """The exemption is a reason not to reclaim, never a reason not to report.

    A lane records its coordinator, not the specialist's GPU worker, so a holder
    with live ``gpu_leases`` rows is exempt from reclamation. But that exemption
    covers the GPU path — the one ``release_resources()`` fails on — which is
    precisely the shape that wedged 2026-09-21. Reclamation must respect it; the
    operator report must not.
    """
    await locks.acquire_many(["build_lane"], holder_id="gpuheld", task_id="tgpuheld", action="specialist", ttl_sec=3600)
    _seed_task(conn, "tgpuheld", "succeeded", evidence={resource_lock.CLEANUP_CONFIRMED_KEY: False})
    conn.raw.execute(
        "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, expires_at, heartbeat_at) "
        "VALUES (?,?,?,?,?,?)",
        (0, "gpuheld", "tgpuheld", _T0, _T0, _T0),
    )
    conn.raw.commit()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        stuck = await locks.diagnose_unverifiable_holders()
        reaped = await locks.reap_finished_holders()

    assert reaped == []  # the exemption still protects it from reclamation
    assert [(r["lane"], r["reason"]) for r in stuck] == [("build_lane", resource_lock.UNVERIFIABLE_HOLDS_GPU)]
    assert len(_remedies(caplog)) == 1


@pytest.mark.asyncio
async def test_a_holder_still_working_is_not_reported_as_stuck(conn, locks, caplog):
    """A lane in use is not a lane to be told about; only an unanswerable one is."""
    with _live_tree() as tree:
        _seed_task(conn, "tbusy", "failed", evidence=_unconfirmed(tree.pid))
        await locks.acquire_many(["build_lane"], holder_id="busy", task_id="tbusy", action="specialist", ttl_sec=3600)
        with caplog.at_level(logging.WARNING):
            assert await locks.diagnose_unverifiable_holders() == []
        assert _remedies(caplog) == []


@pytest.mark.asyncio
async def test_a_row_this_process_may_not_judge_is_still_reported(conn, locks, caplog):
    """No reaper here will ever take it back, so the operator is the only way out.

    The scope guard keeps the sweep from probing an id minted in another boot or
    PID namespace, where any answer would be about the wrong machine. That is
    the right call and also a dead end, so the diagnostic deliberately ignores
    scope where the sweep must not.
    """
    _seed_task(conn, "tfarstuck", "failed", evidence=_unconfirmed())
    await locks.acquire_many(
        ["build_lane"], holder_id="farstuck", task_id="tfarstuck", action="specialist", ttl_sec=3600
    )
    conn.raw.execute("UPDATE leases SET owner_scope='other-boot:4026531836'")
    conn.raw.commit()

    with caplog.at_level(logging.WARNING):
        stuck = await locks.diagnose_unverifiable_holders()

    assert [r["reason"] for r in stuck] == [resource_lock.UNVERIFIABLE_FOREIGN_SCOPE]
    assert resource_lock.UNVERIFIABLE_FOREIGN_SCOPE in _remedies(caplog)[0]


@pytest.mark.asyncio
async def test_a_group_that_cannot_be_enumerated_keeps_its_lane(conn, locks, monkeypatch):
    """An unreadable /proc is an unanswered question, and only an answer frees a lane."""
    _seed_task(conn, "tblind", "failed", evidence=_unconfirmed(_dead_tree_pgid()))
    await locks.acquire_many(
        ["gpu_research_lane"], holder_id="blind", task_id="tblind", action="specialist", ttl_sec=3600
    )

    def _unreadable(_pids):
        raise OSError(13, "Permission denied: /proc")

    monkeypatch.setattr(resource_lock, "collect_tree", _unreadable)
    assert await locks.reap_finished_holders() == []
    assert (await locks.lane_holders())["gpu_research_lane"] == 1
    # And it is reported as such, rather than sitting silently at capacity.
    assert {r["reason"] for r in await locks.diagnose_unverifiable_holders()} == {resource_lock.UNVERIFIABLE_UNREADABLE}


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
    """A pid means something only in the boot and PID namespace that recorded it."""
    _seed_task(conn, "tfar", "failed", evidence=_unconfirmed(_dead_tree_pgid()))
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
