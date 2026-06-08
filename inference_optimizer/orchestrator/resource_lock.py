# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ResourceLockManager + SqliteLeaseBackend ( + KB_design §3.7).

All ``acquire_many`` writes happen inside a single ``BEGIN IMMEDIATE``
transaction, giving the "all or nothing" property the lane semantics
require: if any lane is busy or full, the entire INSERT batch ROLLBACKs.

v0.6 single-holder semantics:

Cross-lane mutual-exclusion rules (DESIGN §3.5.3) are enforced by adding
the conflicting lanes to the requested set before the SQL acquire — e.g.
``benchmark_lane`` conflicts with ``profile_lane``, so a ``bench_runner``
task transparently asks for ``[benchmark_lane, profile_lane]`` and lets
the per-lane capacity check enforce the mutual exclusion.

v0.8 M6 multi-holder semantics:

* ``leases`` PK widened to ``(lane, holder_id)`` so the same lane can
  carry multiple concurrent holders (e.g. ``research_lane`` with
  capacity=6 hosting six specialists).
* Per-lane capacity sits in the ``lane_capacity`` table. ``acquire_many``
  counts the current non-expired holders per lane and raises a new
  :class:`LaneFull` when ``count >= capacity`` — semantically distinct
  from :class:`LaneBusy` (capacity 0 / cross-lane conflict) so the
  dispatcher can decide whether to retry next tick (LaneFull) or
  permanently degrade (LaneBusy).
* :meth:`ResourceLockManager.try_acquire_many` is the non-blocking
  variant the Coordinator's concurrent dispatcher uses to fan out
  ready-to-run tasks per tick.

Inv-7.1 ``benchmark_lane.holders ≤ 1`` is preserved by the default
``capacity = 1`` for serving-side lanes (see
``storage.schema.DEFAULT_LANE_CAPACITIES``).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..storage.connection import SqliteConnection
from ..storage.schema import DEFAULT_LANE_CAPACITIES


log = logging.getLogger(__name__)


KNOWN_LANES = (
    "server_lifecycle",
    "workspace_mutation",
    "benchmark_lane",
    "profile_lane",
    # research_lane carries LLM specialist sub-agents. It has NO
    # LANE_CONFLICTS with the four serving lanes: a specialist reading
    # source / KB / PR can coexist with a benchmark / profile / server
    # restart on the same tick. Capacity may exceed 1 via a
    # (lane, holder_id) schema.
    "research_lane",
)

# Lane → lanes that must *also* be free or co-acquired (DESIGN §3.5.3).
LANE_CONFLICTS: dict[str, frozenset[str]] = {
    "benchmark_lane": frozenset({"profile_lane", "server_lifecycle"}),
    "profile_lane":   frozenset({"benchmark_lane", "server_lifecycle"}),
    "server_lifecycle": frozenset({"benchmark_lane", "profile_lane"}),
    "workspace_mutation": frozenset(),
    # research_lane does not conflict with any serving-side lane.
    # (Capacity caps come from a separate table.)
    "research_lane": frozenset(),
}


def _now_iso() -> str:
    """Return the current UTC time as a microsecond-precision ISO string.

    Returns:
        str: ``datetime.now(UTC)`` formatted with microsecond resolution.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Expand requested lanes by transitive conflicts; sort deterministically.

    Args:
        lanes (list[str]): Requested lane names; each must be in
            :data:`KNOWN_LANES`.

    Returns:
        list[str]: The requested lanes plus their :data:`LANE_CONFLICTS`,
            de-duplicated and sorted for deterministic acquire ordering.

    Raises:
        ValueError: If any requested lane is not a known lane.
    """
    out: set[str] = set()
    for lane in lanes:
        if lane not in KNOWN_LANES:
            raise ValueError(f"unknown lane: {lane!r}")
        out.add(lane)
        out.update(LANE_CONFLICTS.get(lane, frozenset()))
    return sorted(out)


# ---------------------------------------------------------------------------
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
    """Raised by ``acquire_many`` when at least one requested lane has
    a cross-lane conflict (Inv-7.1).

    KB_design §3.7 §5.1 keeps the *cross-lane mutex* and *capacity*
    failure modes semantically separate. ``LaneBusy`` covers the
    mutex case (e.g. benchmark vs. profile), so dispatchers know to
    leave the task queued and retry next tick once the conflicting
    holder releases.
    """

    def __init__(self, busy_lanes: list[str]):
        """Initialise with the lanes that triggered the cross-lane conflict.

        Args:
            busy_lanes (list[str]): Lanes whose cross-lane mutex blocked
                the acquire; stored on ``self.busy_lanes``.
        """
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class LaneFull(RuntimeError):
    """Raised by ``acquire_many`` when a requested lane has reached its
    per-lane ``capacity`` cap.

    Distinct from :class:`LaneBusy`: ``LaneFull`` is a pure capacity
    decision (e.g. research_lane with N holders, N == capacity); no
    cross-lane conflict is involved. Dispatcher leaves the task queued
    and tries again on the next tick when a holder releases.
    """

    def __init__(self, full_lanes: list[str]):
        """Initialise with the lanes that were at capacity.

        Args:
            full_lanes (list[str]): Lanes at their per-lane capacity cap;
                stored on ``self.full_lanes``.
        """
        super().__init__(f"lanes full: {full_lanes!r}")
        self.full_lanes = full_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


# ---------------------------------------------------------------------------
class SqliteLeaseBackend:
    """Default ``ResourceLockBackend`` (DESIGN §3.5.4 / ADR-42).

    Uses ``leases`` rows in the unified SQLite WAL DB. ``BEGIN IMMEDIATE``
    plus row-level uniqueness on the ``lane`` PK gives atomic acquire-many.
    """

    def __init__(self, db: SqliteConnection):
        """Bind the backend to a SQLite connection.

        Args:
            db (SqliteConnection): The unified WAL DB connection used for
                all lease reads / writes.
        """
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
        """Acquire every lane in ``lanes`` (plus their transitive
        cross-lane conflicts) as a single atomic batch.

        capacity-aware:

        * Counts *non-expired* holders per lane (skipping any row that
          shares ``holder_id`` so a retry from the same holder is a
          no-op rather than an immediate full).
        * Compares against the per-lane capacity (read from
          ``lane_capacity``; falls back to
          :data:`storage.schema.DEFAULT_LANE_CAPACITIES`).
        * Raises :class:`LaneFull` when at least one lane is at cap
          and :class:`LaneBusy` when an expanded conflict-lane shows
          a *different-holder* live row.

        Inv-7.1 is enforced via ``DEFAULT_LANE_CAPACITIES``: serving
        lanes default to capacity 1, so any second holder still raises
        LaneFull immediately.

        Args:
            lanes (list[str]): Requested lanes (expanded by transitive
                conflicts before acquire).
            holder_id (str): Unique holder id; a retry from the same holder
                is idempotent (acts as a TTL refresh).
            task_id (str): Task that owns the lease.
            action (str): Action name recorded on the lease.
            ttl_sec (int): Lease lifetime in seconds.

        Returns:
            Lease: The acquired lease covering all expanded lanes.

        Raises:
            ValueError: If ``lanes`` is empty.
            LaneFull: If a multi-holder lane is at capacity (or disabled).
            LaneBusy: If a single-holder lane has a different live holder.
        """
        if not lanes:
            raise ValueError("acquire_many called with no lanes")
        expanded = _expand_lanes(lanes)
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + ttl_sec
        expires_iso = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()

        async with self.db.transaction() as cur:
            # Resolve capacity per lane (defensive fallback for legacy
            # DBs where lane_capacity hasn't been seeded yet — should
            # never happen on a freshly-ensured schema).
            capacity_by_lane: dict[str, int] = {}
            placeholders = ",".join("?" * len(expanded))
            cur.execute(
                f"SELECT lane, capacity FROM lane_capacity "
                f"WHERE lane IN ({placeholders})",
                expanded,
            )
            for row in cur.fetchall():
                capacity_by_lane[row["lane"]] = int(row["capacity"])
            for lane in expanded:
                capacity_by_lane.setdefault(
                    lane, int(DEFAULT_LANE_CAPACITIES.get(lane, 1)),
                )

            # Pull *all* current holders so we can reap expired rows
            # and count surviving distinct holders per lane.
            cur.execute(
                f"SELECT lane, holder_id, expires_at FROM leases "
                f"WHERE lane IN ({placeholders})",
                expanded,
            )
            rows = [dict(r) for r in cur.fetchall()]

            holders_per_lane: dict[str, set[str]] = {
                lane: set() for lane in expanded
            }
            expired: list[tuple[str, str]] = []  # (lane, previous_holder)
            for row in rows:
                lane = row["lane"]
                row_holder = row["holder_id"]
                row_expires = datetime.fromisoformat(row["expires_at"]).timestamp()
                if row_expires > now_ts:
                    holders_per_lane.setdefault(lane, set()).add(row_holder)
                else:
                    expired.append((lane, row_holder))

            # Reap expired rows + emit lease_expired events.
            for lane, prev_holder in expired:
                cur.execute(
                    "DELETE FROM leases WHERE lane=? AND holder_id=?",
                    (lane, prev_holder),
                )
                cur.execute(
                    "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                    "in_reply_to, payload, priority, ts) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        "resource_lock",
                        "*",
                        "lease_expired",
                        None,
                        json.dumps(
                            {"lane": lane, "previous_holder": prev_holder},
                        ),
                        2,
                        now_iso,
                    ),
                )

            # distinguish capacity (LaneFull) from
            # cross-lane mutex (LaneBusy). For capacity == 1 lanes
            # (serving side) the two converge in behaviour; we still
            # surface the right exception class so dispatchers can
            # tell "I'm queued behind a long-running benchmark" from
            # "the lane is full of specialists".
            full: list[str] = []
            busy: list[str] = []
            for lane in expanded:
                live = holders_per_lane.get(lane, set())
                # An attempt by the same holder is idempotent; skip
                # all the gates so it acts as a TTL refresh.
                if holder_id in live:
                    continue
                cap = capacity_by_lane.get(lane, 1)
                if cap <= 0:
                    # capacity=0 → lane disabled (e.g. --research-lane-
                    # capacity 0). Treat as LaneFull so dispatcher
                    # knows to drop the task rather than spin.
                    full.append(lane)
                    continue
                if len(live) >= cap:
                    # Multi-holder lane (cap > 1) full → LaneFull.
                    # Single-holder lane (cap == 1) full → LaneBusy
                    # for back-compat with the legacy contract callers that pattern-
                    # match on ``LaneBusy``.
                    if cap > 1:
                        full.append(lane)
                    else:
                        busy.append(lane)

            if busy:
                raise LaneBusy(busy)
            if full:
                raise LaneFull(full)

            for lane in expanded:
                # ``INSERT OR REPLACE`` lets the same holder refresh
                # an existing lease row without violating the (lane,
                # holder_id) PK.
                cur.execute(
                    "INSERT OR REPLACE INTO leases(lane, holder_id, "
                    "task_id, action, pid, acquired_at, expires_at, "
                    "heartbeat_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        lane,
                        holder_id,
                        task_id,
                        action,
                        os.getpid(),
                        now_iso,
                        expires_iso,
                        now_iso,
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
        """Refresh ``expires_at`` for every lane this holder owns.

        keys on the composite ``(lane, holder_id)`` PK so a
        multi-holder lane (research_lane with capacity > 1) only
        bumps this holder's row, not the others.

        Args:
            lease (Lease): The lease whose lanes should be refreshed.
            ttl_sec (int): New lifetime in seconds from now.

        Raises:
            StaleLeaseError: If the number of updated rows does not match
                the lease's lane count (lease no longer ours).
        """
        new_expires_iso = datetime.fromtimestamp(
            time.time() + ttl_sec, tz=timezone.utc
        ).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"UPDATE leases SET expires_at=?, heartbeat_at=? "
                f"WHERE lane IN ({placeholders}) AND holder_id=?",
                (new_expires_iso, now_iso, *lease.lanes, lease.holder_id),
            )
            if cur.rowcount != len(lease.lanes):
                raise StaleLeaseError(
                    f"heartbeat mismatch: expected {len(lease.lanes)} rows, "
                    f"got {cur.rowcount}"
                )

    async def release(self, lease: Lease) -> int:
        """Drop every (lane, holder_id) row this lease owns.

        Other holders on the same lane are untouched (Inv-7.3 atomic
        release for one holder).

        Args:
            lease (Lease): The lease to release.

        Returns:
            int: Number of lease rows deleted.
        """
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"DELETE FROM leases WHERE lane IN ({placeholders}) "
                f"AND holder_id=?",
                (*lease.lanes, lease.holder_id),
            )
            return cur.rowcount

    async def reap_expired(self) -> list[dict]:
        """Sweep expired rows; emit one ``lease_expired`` event per stale row.

        v0.8 M6 keys deletion on both ``(lane, holder_id)`` columns so a
        multi-holder lane reaps only the holders whose TTL fired — live
        holders on the same lane keep their leases.

        Returns:
            list[dict]: The reaped lease rows (as dicts) that were deleted.
        """
        now_iso_str = _now_iso()
        reaped: list[dict] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT * FROM leases WHERE expires_at <= ?",
                (now_iso_str,),
            )
            stale = [dict(r) for r in cur.fetchall()]
            for row in stale:
                cur.execute(
                    "DELETE FROM leases WHERE lane=? AND holder_id=?",
                    (row["lane"], row["holder_id"]),
                )
                cur.execute(
                    "INSERT INTO events (msg_id, from_agent, to_agent, topic, "
                    "in_reply_to, payload, priority, ts) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        "resource_lock",
                        "*",
                        "lease_expired",
                        None,
                        json.dumps({
                            "lane": row["lane"],
                            "previous_holder": row["holder_id"],
                            "reap_pass": True,
                        }),
                        2,
                        now_iso_str,
                    ),
                )
                reaped.append(row)
        return reaped

    async def active_lanes(self) -> list[str]:
        """Return the distinct lane names with at least one live holder.

        Callers use this for capacity / breakdown observation;
        :meth:`lane_holders` returns the multi-holder count shape.

        Returns:
            list[str]: Distinct lane names that currently have a live row.
        """
        rows = await self.db.fetchall(
            "SELECT DISTINCT lane FROM leases WHERE expires_at > ?",
            (_now_iso(),),
        )
        return [r["lane"] for r in rows]

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: live_holder_count}`` for lanes with live rows.

        Used by the breakdown ``lane_timeline`` collector + by dispatchers
        to gauge research_lane occupancy.

        Returns:
            dict[str, int]: Map of lane name to its live holder count.
        """
        rows = await self.db.fetchall(
            "SELECT lane, COUNT(*) AS n FROM leases "
            "WHERE expires_at > ? GROUP BY lane",
            (_now_iso(),),
        )
        return {r["lane"]: int(r["n"]) for r in rows}

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` for every row in ``lane_capacity``.

        Falls back to :data:`storage.schema.DEFAULT_LANE_CAPACITIES`
        when the table is missing (legacy DB never opened with v0.8).

        Returns:
            dict[str, int]: Map of lane name to capacity, defaults merged
                with any rows present in the ``lane_capacity`` table.
        """
        try:
            rows = await self.db.fetchall(
                "SELECT lane, capacity FROM lane_capacity"
            )
        except Exception:  # noqa: BLE001 — best-effort observation
            return dict(DEFAULT_LANE_CAPACITIES)
        out: dict[str, int] = dict(DEFAULT_LANE_CAPACITIES)
        for r in rows:
            out[r["lane"]] = int(r["capacity"])
        return out


# ---------------------------------------------------------------------------
class ResourceLockManager:
    """Coordinator-facing wrapper.

    v0.8 M6 adds non-blocking acquire + multi-holder observability so
    the concurrent dispatcher can fan tasks out
    per tick without spinning on lane busy errors.
    """

    def __init__(self, backend: SqliteLeaseBackend):
        """Wrap a lease backend and initialise the per-process counters.

        Args:
            backend (SqliteLeaseBackend): The backend doing the actual
                lease reads / writes.
        """
        self.backend = backend
        # Per-process counters. The
        # leases DB is the authoritative source for current state;
        # these counters track *cumulative* acquire / lane-full /
        # lane-busy events for the lifetime of this process so the
        # breakdown can surface peak / total numbers without re-reading
        # the SQLite log.
        self._counters: dict[str, dict[str, int]] = {}

    async def acquire_many(self, lanes: list[str], **kwargs) -> Lease:
        """Acquire lanes via the backend, updating lifetime counters.

        Args:
            lanes (list[str]): Lanes to acquire.
            **kwargs: Forwarded to :meth:`SqliteLeaseBackend.acquire_many`
                (``holder_id`` / ``task_id`` / ``action`` / ``ttl_sec``).

        Returns:
            Lease: The acquired lease.

        Raises:
            LaneFull: Re-raised after bumping the lane's full counter.
            LaneBusy: Re-raised after bumping the lane's busy counter.
        """
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
        """Non-blocking variant of :meth:`acquire_many` (KB_design §3.7
        §4.3).

        Returns the :class:`Lease` on success, ``None`` when any
        requested lane is busy or full. The Coordinator's concurrent
        dispatcher uses this to fan out queued tasks per tick without
        having to wrap every dispatch in a try/except.

        Cross-lane mutex (``LaneBusy``) and capacity (``LaneFull``)
        both map to ``None``. Distinguishing the two is the caller's
        job via :meth:`acquire_many` when needed; the concurrent
        dispatch path treats them identically (retry next tick).

        Args:
            lanes (list[str]): Lanes to acquire.
            **kwargs: Forwarded to :meth:`acquire_many`.

        Returns:
            Lease | None: The acquired lease, or None when any requested
                lane is busy or full.
        """
        try:
            return await self.acquire_many(lanes, **kwargs)
        except (LaneBusy, LaneFull):
            return None

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        """Refresh a lease's TTL via the backend.

        Args:
            lease (Lease): The lease to refresh.
            ttl_sec (int): New lifetime in seconds.

        Returns:
            None: Delegates to :meth:`SqliteLeaseBackend.heartbeat`.
        """
        return await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def release(self, lease: Lease) -> int:
        """Release a lease and bump each lane's release counter.

        Args:
            lease (Lease): The lease to release.

        Returns:
            int: Number of lease rows deleted by the backend.
        """
        n = await self.backend.release(lease)
        for lane in lease.lanes:
            self._bump_counter(lane, "release_count")
        return n

    async def reap_expired(self) -> list[dict]:
        """Sweep expired leases via the backend.

        Returns:
            list[dict]: The reaped lease rows.
        """
        return await self.backend.reap_expired()

    async def active_lanes(self) -> list[str]:
        """Return the distinct lanes with at least one live holder.

        Returns:
            list[str]: Distinct active lane names.
        """
        return await self.backend.active_lanes()

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: live_holder_count}`` via the backend.

        Returns:
            dict[str, int]: Live holder count per lane.
        """
        return await self.backend.lane_holders()

    async def lane_capacities(self) -> dict[str, int]:
        """Return ``{lane: capacity}`` via the backend.

        Returns:
            dict[str, int]: Capacity per lane.
        """
        return await self.backend.lane_capacities()

    def counters_snapshot(self) -> dict[str, dict[str, int]]:
        """Return per-lane lifetime counters (acquire / release / busy / full).

        Cheap; returns an in-memory deep-ish copy so callers can't mutate
        the live counters.

        Returns:
            dict[str, dict[str, int]]: Map of lane name to its counter dict.
        """
        return {
            lane: dict(d) for lane, d in self._counters.items()
        }

    def _bump_counter(self, lane: str, field: str) -> None:
        """Increment one per-lane lifetime counter by 1.

        Args:
            lane (str): Lane whose counter dict is updated.
            field (str): Counter key to increment (e.g. ``"acquire_count"``).
        """
        d = self._counters.setdefault(lane, {})
        d[field] = int(d.get(field, 0)) + 1


__all__ = [
    "KNOWN_LANES",
    "LANE_CONFLICTS",
    "LaneBusy",
    "LaneFull",
    "Lease",
    "ResourceLockManager",
    "SqliteLeaseBackend",
    "StaleLeaseError",
]
