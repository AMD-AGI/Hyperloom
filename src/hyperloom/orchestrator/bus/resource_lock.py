# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ResourceLockManager + SqliteLeaseBackend."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.state.task_states import TERMINAL_STATES

from .storage.connection import SqliteConnection
from .storage.schema import DEFAULT_LANE_CAPACITIES


log = logging.getLogger(__name__)


KNOWN_LANES = (
    "server_lifecycle",
    "workspace_mutation",
    "benchmark_lane",
    "profile_lane",
    # research_lane carries LLM specialist sub-agents; no serving-lane conflict and capacity may exceed 1.
    "research_lane",
    # gpu_research_lane carries GPU-holding specialists; mutually exclusive with the serving lanes and capacity-1 /
    # strictly serial (one GPU specialist holds the machine at a time; the GPU pool partitions cards within it).
    "gpu_research_lane",
    # build_lane serializes off-loop compile tasks; capacity-1 with no serving-lane conflict (the compile step needs
    # no GPU/server).
    "build_lane",
)

# Lane → lanes that must *also* be free or co-acquired.
LANE_CONFLICTS: dict[str, frozenset[str]] = {
    "benchmark_lane": frozenset({"profile_lane", "server_lifecycle", "gpu_research_lane"}),
    "profile_lane": frozenset({"benchmark_lane", "server_lifecycle", "gpu_research_lane"}),
    "server_lifecycle": frozenset({"benchmark_lane", "profile_lane", "gpu_research_lane"}),
    "workspace_mutation": frozenset(),
    # research_lane does not conflict with any serving-side lane.
    "research_lane": frozenset(),
    # gpu_research_lane ⊥ serving lanes; capacity-1 so GPU specialists serialize.
    "gpu_research_lane": frozenset({"benchmark_lane", "profile_lane", "server_lifecycle"}),
    # build_lane is a serialization/observability primitive only; no conflicts.
    "build_lane": frozenset(),
}


#: The lane an open bring-up round holds for as long as it is open.
#:
#: Deliberately absent from :data:`KNOWN_LANES`, because no task may request it.
#: A round is held by a task that still has to be dispatchable while the round
#: stands, and every serving lane mutexes against ``gpu_research_lane``, so a
#: round holding a serving lane under a holder id of its own would deny its own
#: holder the dispatch the round exists to cover. ``RoundStore`` writes and
#: drops this row inside the transaction that opens, renews and settles the
#: round; capacity readers and the breakdown lane timeline read it like any
#: other ownership row.
BRINGUP_ROUND_LANE = "bringup_round"

#: Recorded on a round's lane row in place of a pid. The round's holder is a
#: task, not this process, and only the task registry can prove a task's process
#: dead. The dead-holder pass skips non-positive pids, leaving these rows to
#: explicit round settlement.
ROUND_LEASE_PID = 0

#: Recorded in the lane row's ``action`` column.
_ROUND_LEASE_ACTION = "bringup_round"


_now_iso = now_iso


def local_owner_scope() -> str:
    """Identify this boot and PID namespace, or leave ownership unobservable."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        namespace = Path("/proc/self/ns/pid").stat().st_ino
    except OSError:
        return ""
    return f"{boot}:{namespace}" if boot else ""


def _lease_iso(unix_ts: float) -> str:
    """Render unix seconds the way the lease table's timestamps compare.

    Budget timestamps use the same offset and precision as :func:`now_iso`.

    Args:
        unix_ts: The instant to render.

    Returns:
        str: A fixed-width UTC ISO-8601 timestamp.
    """
    return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).isoformat(timespec="microseconds")


def hold_round_lane(
    cur: sqlite3.Cursor,
    *,
    round_id: str,
    holder_task_id: str,
    expires_unix: float,
    now_unix: float,
) -> None:
    """Write the lane row an open round holds, inside the caller's transaction.

    Keyed on ``(BRINGUP_ROUND_LANE, round_id)`` and idempotent, so the one call
    serves the acquire, every renewal, and a handoff that moves the round to a
    new holder. ``acquired_at`` survives a renewal; only the holder and the two
    clock columns move.

    Args:
        cur: Cursor of the transaction writing the round row, so the round and
            its lane land together or not at all.
        round_id: The round holding the lane; also the lease's holder id.
        holder_task_id: The task the round is held by right now.
        expires_unix: When the round's lease runs out.
        now_unix: Current wall time.
    """
    stamp = _lease_iso(now_unix)
    cur.execute(
        "INSERT INTO leases(lane, holder_id, task_id, action, pid,"
        "  acquired_at, expires_at, heartbeat_at) VALUES (?,?,?,?,?,?,?,?)"
        " ON CONFLICT(lane, holder_id) DO UPDATE SET"
        "  task_id = excluded.task_id,"
        "  expires_at = excluded.expires_at,"
        "  heartbeat_at = excluded.heartbeat_at",
        (
            BRINGUP_ROUND_LANE,
            round_id,
            holder_task_id,
            _ROUND_LEASE_ACTION,
            ROUND_LEASE_PID,
            stamp,
            _lease_iso(expires_unix),
            stamp,
        ),
    )


def drop_round_lane(cur: sqlite3.Cursor, *, round_id: str) -> None:
    """Release the lane row a round held, inside the caller's transaction.

    Args:
        cur: Cursor of the transaction settling the round.
        round_id: The round whose lane is released.
    """
    cur.execute(
        "DELETE FROM leases WHERE lane = ? AND holder_id = ?",
        (BRINGUP_ROUND_LANE, round_id),
    )


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Expand requested lanes by transitive conflicts; sorted deterministically."""
    out: set[str] = set()
    for lane in lanes:
        if lane not in KNOWN_LANES:
            raise ValueError(f"unknown lane: {lane!r}")
        out.add(lane)
        out.update(LANE_CONFLICTS.get(lane, frozenset()))
    return sorted(out)


def _reclaim_finished_holders(cur: sqlite3.Cursor, lanes: list[str], *, scope: str) -> list[dict]:
    """Drop lane rows whose holder task already ended, inside the caller's transaction.

    The rows leaked on 2026-09-21 all belonged to holders already recorded
    terminal: :meth:`SubAgentRunner._write_terminal` (``loop/sub_agent_runner``)
    runs even on the path that leaves cleanup unconfirmed, and it is exactly
    that path which skips the release. A terminal task cannot resume, so its
    lane row describes work nobody is doing -- which is why this pass reads the
    ``tasks`` state rather than ``expires_at``. A TTL is a static per-action budget that nothing
    enforces, so a lapse says only that a legitimately long run outlived its
    estimate (``explore`` budgets 7200s and holds ``server_lifecycle`` plus
    ``benchmark_lane`` across a benchmark that can exceed it); the terminal
    state is the holder's own account of having stopped.

    Sharing the caller's cursor keeps the sweep and the read that follows it in
    one ``BEGIN IMMEDIATE``, so no second acquirer can see a row this one has
    already reclaimed.

    Args:
        cur: Cursor of the transaction that acts on the surviving rows.
        lanes: Lanes to sweep; empty sweeps every lane but the round lane.
        scope: The boot/PID namespace whose rows this process may judge.

    Returns:
        list[dict]: The rows reclaimed.
    """
    # Same first act as :meth:`SqliteLeaseBackend.holder_is_dead`: with no
    # observable ownership domain, no row here is ours to judge.
    #
    # Pre-existing, inherited here, and deliberately NOT fixed by this change: a
    # row acquired while /proc is unreadable is stored with ``owner_scope=''``
    # (:func:`local_owner_scope` returns "" on OSError), and no reaper can ever
    # take it back. ``holder_is_dead`` rejects it on this same guard while /proc
    # stays unreadable and on ``row["owner_scope"] != scope`` once /proc
    # recovers; this pass rejects it here and again on the ``owner_scope = ?``
    # predicate below. :meth:`release` still drops such a row -- it keys on
    # ``(lane, holder_id)`` with no scope predicate -- so a holder that ends by
    # releasing is fine; what no reaper can do is take the row back for a holder
    # that ends without releasing, which is precisely the case this pass exists
    # for. Closing that means giving ownership a fallback identity
    # for the case where the kernel will not name one, which changes what a
    # scope asserts; it is its own change, not a widening smuggled into this
    # one, which only reuses the guard already in force.
    if not scope:
        return []
    terminal = sorted(TERMINAL_STATES)
    states = ",".join("?" * len(terminal))
    # A holder absent from ``tasks`` is deliberately left alone. Absence is
    # silence, not a finished holder, and the dead-holder pass already covers
    # the rows a crash leaves behind.
    finished = (
        "owner_scope = ? "
        # A lane records its coordinator, not the specialist's GPU worker: a
        # holder still on the cards is still working, whatever the coordinator
        # task row says. Same exemption reap_dead_holders makes.
        "AND task_id NOT IN (SELECT task_id FROM gpu_leases) "
        f"AND task_id IN (SELECT task_id FROM tasks WHERE state IN ({states}))"  # nosec B608 - generated placeholders only.
    )
    params: tuple = (scope, *terminal)
    if lanes:
        # Callers pass a ``_expand_lanes`` result, which is drawn from
        # KNOWN_LANES and so never names BRINGUP_ROUND_LANE.
        placeholders = ",".join("?" * len(lanes))
        where = f"lane IN ({placeholders}) AND {finished}"  # nosec B608 - generated placeholders only.
        params = (*lanes, *params)
    else:
        # A round outlives its holder on purpose: only RoundStore may say a
        # round is over, and its holder going terminal is the very input
        # ``Reconciler._advance_or_expire`` weighs before handing off.
        where = f"lane != ? AND {finished}"  # nosec B608 - generated placeholders only.
        params = (BRINGUP_ROUND_LANE, *params)
    cur.execute(f"SELECT * FROM leases WHERE {where}", params)  # nosec B608 - generated placeholders only.
    rows = [dict(r) for r in cur.fetchall()]
    if rows:
        cur.execute(f"DELETE FROM leases WHERE {where}", params)  # nosec B608 - generated placeholders only.
    return rows


@dataclass
class Lease:
    """Lease handle returned by ``acquire_many``."""

    holder_id: str
    task_id: str
    action: str
    lanes: tuple[str, ...]
    acquired_at: str
    expires_at: str
    pid: int = field(default_factory=os.getpid)


class LaneBusy(RuntimeError):
    """Raised by ``acquire_many`` on a cross-lane conflict (Inv-7.1); kept distinct from capacity."""

    def __init__(self, busy_lanes: list[str]):
        """Initialise with the lanes that triggered the cross-lane conflict."""
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class LaneFull(RuntimeError):
    """Raised by ``acquire_many`` when a lane hits its ``capacity`` cap (pure capacity decision, distinct from :class:`LaneBusy`)."""

    def __init__(self, full_lanes: list[str]):
        """Initialise with the lanes that were at capacity."""
        super().__init__(f"lanes full: {full_lanes!r}")
        self.full_lanes = full_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


class SqliteLeaseBackend:
    """Lease backend behind :class:`ResourceLockManager`; ``BEGIN IMMEDIATE`` + PK uniqueness gives atomic acquire-many."""

    def __init__(self, db: SqliteConnection):
        """Bind the backend to a SQLite connection."""
        self.db = db

    async def acquire_many(
        self,
        lanes: list[str],
        *,
        holder_id: str,
        task_id: str,
        action: str,
        ttl_sec: int,
    ) -> Lease:
        """Acquire ``lanes`` + transitive conflicts as one atomic batch."""
        if not lanes:
            raise ValueError("acquire_many called with no lanes")
        expanded = _expand_lanes(lanes)
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + ttl_sec
        expires_iso = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()

        async with self.db.transaction() as cur:
            # 2026-09-21: a dispatcher that deliberately retained capacity on an
            # unconfirmed cleanup left six lanes held by holders that
            # :meth:`SubAgentRunner._write_terminal` had already recorded
            # terminal. The rows still occupied the lanes two hours later, with
            # 19 queued tasks starved behind them. Reclaim them here as well as
            # in the sweep, so the capacity read below is never decided against
            # a holder that went terminal since the sweep ran.
            reclaimed = _reclaim_finished_holders(cur, expanded, scope=local_owner_scope())
            if reclaimed:
                # A lane that vanishes on the acquire path leaves no other trace
                # to attribute it to; the standalone sweep logs its own.
                log.warning(
                    "resource_lock: acquire reclaimed %d lease(s) from terminal holders: %s",
                    len(reclaimed),
                    ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(task={r['task_id'][:12]})" for r in reclaimed),
                )

            # Resolve capacity per lane (fallback for unseeded DBs).
            capacity_by_lane: dict[str, int] = {}
            placeholders = ",".join("?" * len(expanded))
            cur.execute(
                f"SELECT lane, capacity FROM lane_capacity WHERE lane IN ({placeholders})",  # nosec B608 - generated placeholders only.
                expanded,
            )
            for row in cur.fetchall():
                capacity_by_lane[row["lane"]] = int(row["capacity"])
            for lane in expanded:
                capacity_by_lane.setdefault(
                    lane,
                    int(DEFAULT_LANE_CAPACITIES.get(lane, 1)),
                )

            cur.execute(
                f"SELECT lane, holder_id FROM leases WHERE lane IN ({placeholders})",  # nosec B608 - generated placeholders only.
                expanded,
            )
            holders_per_lane: dict[str, set[str]] = {lane: set() for lane in expanded}
            for row in cur.fetchall():
                holders_per_lane[row["lane"]].add(row["holder_id"])

            # Distinguish capacity (LaneFull) from cross-lane mutex (LaneBusy).
            full: list[str] = []
            busy: list[str] = []
            for lane in expanded:
                live = holders_per_lane.get(lane, set())
                # Same-holder attempt is idempotent (acts as TTL refresh).
                if holder_id in live:
                    continue
                cap = capacity_by_lane.get(lane, 1)
                if cap <= 0:
                    # capacity=0 → lane disabled; LaneFull so dispatcher drops.
                    full.append(lane)
                    continue
                if len(live) >= cap:
                    # cap>1 full → LaneFull; cap==1 full → LaneBusy.
                    if cap > 1:
                        full.append(lane)
                    else:
                        busy.append(lane)

            if busy:
                raise LaneBusy(busy)
            if full:
                raise LaneFull(full)

            for lane in expanded:
                # INSERT OR REPLACE lets the same holder refresh its row.
                cur.execute(
                    "INSERT OR REPLACE INTO leases(lane, holder_id, "
                    "task_id, action, pid, acquired_at, expires_at, "
                    "heartbeat_at, owner_scope) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        lane,
                        holder_id,
                        task_id,
                        action,
                        os.getpid(),
                        now_iso,
                        expires_iso,
                        now_iso,
                        local_owner_scope(),
                    ),
                )

        return Lease(
            holder_id=holder_id,
            task_id=task_id,
            action=action,
            lanes=tuple(expanded),
            acquired_at=now_iso,
            expires_at=expires_iso,
        )

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        """Refresh ``expires_at`` for every lane this holder owns (keyed on ``(lane, holder_id)`` PK)."""
        new_expires_iso = datetime.fromtimestamp(time.time() + ttl_sec, tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"UPDATE leases SET expires_at=?, heartbeat_at=? WHERE lane IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                (new_expires_iso, now_iso, *lease.lanes, lease.holder_id),
            )
            if cur.rowcount != len(lease.lanes):
                raise StaleLeaseError(f"heartbeat mismatch: expected {len(lease.lanes)} rows, got {cur.rowcount}")

    async def heartbeat_by_task(self, task_id: str, *, ttl_sec: int) -> list[str]:
        """Refresh every lane row a task holds, whoever the holder is."""
        new_expires_iso = datetime.fromtimestamp(time.time() + ttl_sec, tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute("SELECT lane FROM leases WHERE task_id=?", (task_id,))
            lanes = sorted(str(r["lane"]) for r in cur.fetchall())
            if lanes:
                cur.execute(
                    "UPDATE leases SET expires_at=?, heartbeat_at=? WHERE task_id=?",
                    (new_expires_iso, now_iso, task_id),
                )
        return lanes

    async def release(self, lease: Lease) -> int:
        """Drop every (lane, holder_id) row this lease owns."""
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"DELETE FROM leases WHERE lane IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                (*lease.lanes, lease.holder_id),
            )
            return cur.rowcount

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Best-effort liveness probe for a lease-holder PID."""
        if pid <= 0:
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    @classmethod
    def holder_is_dead(cls, row) -> bool:
        """Accept only an absent PID observed in its recorded local PID domain."""
        scope = local_owner_scope()
        if not scope or row["owner_scope"] != scope:
            return False
        try:
            pid = int(row["pid"] or 0)
        except (TypeError, ValueError):
            return False
        return pid > 0 and not cls._pid_alive(pid)

    async def reap_dead_holders(self) -> list[dict]:
        """Release leases whose local holder process is confirmed absent."""
        reaped: list[dict] = []
        async with self.db.transaction() as cur:
            cur.execute("SELECT * FROM leases")
            live_rows = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT DISTINCT task_id FROM gpu_leases")
            gpu_tasks = {row["task_id"] for row in cur.fetchall()}
            for row in live_rows:
                # A lane records its coordinator, not the specialist's GPU worker.
                if row["task_id"] in gpu_tasks or not self.holder_is_dead(row):
                    continue
                cur.execute(
                    "DELETE FROM leases WHERE lane=? AND holder_id=?",
                    (row["lane"], row["holder_id"]),
                )
                reaped.append(row)
        if reaped:
            log.warning(
                "resource_lock: reaped %d lease(s) from dead holders: %s",
                len(reaped),
                ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(pid={r.get('pid')})" for r in reaped),
            )
        return reaped

    async def reap_finished_holders(self) -> list[dict]:
        """Release every lane still held by a task recorded terminal.

        A pass of its own because the dispatcher's lane gate reads
        :meth:`lane_holders` before it attempts an acquire: a leaked row starves
        the queue without any acquire ever running to reclaim it.

        Returns:
            list[dict]: The rows reclaimed.
        """
        async with self.db.transaction() as cur:
            reaped = _reclaim_finished_holders(cur, [], scope=local_owner_scope())
        if reaped:
            log.warning(
                "resource_lock: reclaimed %d lease(s) from terminal holders: %s",
                len(reaped),
                ", ".join(f"{r['lane']}<-{r['holder_id'][:12]}(task={r['task_id'][:12]})" for r in reaped),
            )
        return reaped

    async def bringup_round_holders(self, now_unix: float) -> set[str]:
        """Return round ids with a retained ownership row, irrespective of age."""
        rows = await self.db.fetchall(
            "SELECT holder_id FROM leases WHERE lane = ?",
            (BRINGUP_ROUND_LANE,),
        )
        return {str(r["holder_id"]) for r in rows}

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: holder_count}`` for every retained ownership row."""
        rows = await self.db.fetchall("SELECT lane, COUNT(*) AS n FROM leases GROUP BY lane")
        return {r["lane"]: int(r["n"]) for r in rows}

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` for every row in ``lane_capacity``."""
        try:
            rows = await self.db.fetchall("SELECT lane, capacity FROM lane_capacity")
        except sqlite3.OperationalError as exc:
            # Legacy DB never opened with v0.8 lacks the table; fall back to defaults.
            log.debug("lane_capacities: lane_capacity table unavailable: %s", exc)
            return dict(DEFAULT_LANE_CAPACITIES)
        out: dict[str, int] = dict(DEFAULT_LANE_CAPACITIES)
        for r in rows:
            out[r["lane"]] = int(r["capacity"])
        return out


class ResourceLockManager:
    """Coordinator-facing wrapper."""

    def __init__(self, backend: SqliteLeaseBackend):
        """Wrap a lease backend and initialise the per-process counters."""
        self.backend = backend
        # Per-process cumulative acquire / lane-full / lane-busy counters.
        self._counters: dict[str, dict[str, int]] = {}

    async def acquire_many(self, lanes: list[str], **kwargs) -> Lease:
        """Acquire lanes via the backend, updating lifetime counters."""
        try:
            lease = await self.backend.acquire_many(lanes, **kwargs)
        except LaneFull as exc:
            for lane in exc.full_lanes:
                self._bump_counter(lane, "lane_full_count")
            raise
        except LaneBusy as exc:
            for lane in exc.busy_lanes:
                self._bump_counter(lane, "lane_busy_count")
            raise
        for lane in lease.lanes:
            self._bump_counter(lane, "acquire_count")
        return lease

    async def try_acquire_many(self, lanes: list[str], **kwargs) -> Lease | None:
        """Non-blocking variant of :meth:`acquire_many`."""
        try:
            return await self.acquire_many(lanes, **kwargs)
        except (LaneBusy, LaneFull):
            return None

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        """Refresh a lease's TTL via the backend."""
        return await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def heartbeat_by_task(self, task_id: str, *, ttl_sec: int) -> list[str]:
        """Refresh every lane row a task holds."""
        return await self.backend.heartbeat_by_task(task_id, ttl_sec=ttl_sec)

    async def release(self, lease: Lease) -> int:
        """Release a lease and bump each lane's release counter."""
        n = await self.backend.release(lease)
        for lane in lease.lanes:
            self._bump_counter(lane, "release_count")
        return n

    async def reap_dead_holders(self) -> list[dict]:
        """Release leases whose holder process is dead via the backend."""
        fn = getattr(self.backend, "reap_dead_holders", None)
        if not callable(fn):
            return []
        return await fn()

    async def reap_finished_holders(self) -> list[dict]:
        """Release lanes held by tasks the registry wrote terminal, via the backend."""
        fn = getattr(self.backend, "reap_finished_holders", None)
        if not callable(fn):
            return []
        return await fn()

    async def bringup_round_holders(self, now_unix: float) -> set[str]:
        """Return the ids of rounds still holding their lane, via the backend.

        Args:
            now_unix: The instant to test.

        Returns:
            set[str]: Round ids holding :data:`BRINGUP_ROUND_LANE` then.
        """
        return await self.backend.bringup_round_holders(now_unix)

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: live_holder_count}`` via the backend."""
        return await self.backend.lane_holders()

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` via the backend."""
        return await self.backend.lane_capacities()

    def _bump_counter(self, lane: str, field: str) -> None:
        """Increment one per-lane lifetime counter by 1."""
        d = self._counters.setdefault(lane, {})
        d[field] = int(d.get(field, 0)) + 1


__all__ = [
    "BRINGUP_ROUND_LANE",
    "KNOWN_LANES",
    "LANE_CONFLICTS",
    "ROUND_LEASE_PID",
    "LaneBusy",
    "LaneFull",
    "Lease",
    "ResourceLockManager",
    "SqliteLeaseBackend",
    "StaleLeaseError",
    "drop_round_lane",
    "hold_round_lane",
]
