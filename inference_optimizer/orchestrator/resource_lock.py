# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ResourceLockManager + SqliteLeaseBackend (KB_design §3.7).

``acquire_many`` is a single ``BEGIN IMMEDIATE`` all-or-nothing batch acquire;
cross-lane mutual exclusion (DESIGN §3.5.3) co-acquires conflicting lanes.
v0.8 M6 multi-holder: ``leases`` PK widened to ``(lane, holder_id)``; raises
:class:`LaneFull` at capacity vs :class:`LaneBusy` cross-lane conflict.
Inv-7.1 ``benchmark_lane.holders ≤ 1`` preserved by default capacity=1 for
serving-side lanes.
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
    # research_lane carries LLM specialist sub-agents; no LANE_CONFLICTS
    # with the serving lanes and capacity may exceed 1.
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
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Expand requested lanes by transitive conflicts; sorted deterministically."""
    out: set[str] = set()
    for lane in lanes:
        if lane not in KNOWN_LANES:
            raise ValueError(f"unknown lane: {lane!r}")
        out.add(lane)
        out.update(LANE_CONFLICTS.get(lane, frozenset()))
    return sorted(out)


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
    """Raised by ``acquire_many`` on a cross-lane conflict (Inv-7.1, KB_design §3.7 §5.1); kept distinct from capacity."""

    def __init__(self, busy_lanes: list[str]):
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class LaneFull(RuntimeError):
    """Raised by ``acquire_many`` when a lane hits its ``capacity`` cap (pure capacity decision, distinct from :class:`LaneBusy`)."""

    def __init__(self, full_lanes: list[str]):
        super().__init__(f"lanes full: {full_lanes!r}")
        self.full_lanes = full_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


class SqliteLeaseBackend:
    """Default ``ResourceLockBackend`` (DESIGN §3.5.4 / ADR-42); ``BEGIN IMMEDIATE`` + PK uniqueness gives atomic acquire-many."""

    def __init__(self, db: SqliteConnection):
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
        """Acquire ``lanes`` + transitive conflicts as one atomic batch.

        Same-holder retries are idempotent; raises :class:`LaneFull` (at cap)
        or :class:`LaneBusy` (different-holder conflict). Inv-7.1: serving lanes default capacity 1.
        """
        if not lanes:
            raise ValueError("acquire_many called with no lanes")
        expanded = _expand_lanes(lanes)
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + ttl_sec
        expires_iso = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()

        async with self.db.transaction() as cur:
            # Resolve capacity per lane (defensive fallback for unseeded legacy DBs).
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

            # Pull holders to reap expired rows and count surviving distinct holders per lane.
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
                    # cap>1 full → LaneFull; cap==1 full → LaneBusy
                    # (back-compat for callers pattern-matching LaneBusy).
                    if cap > 1:
                        full.append(lane)
                    else:
                        busy.append(lane)

            if busy:
                raise LaneBusy(busy)
            if full:
                raise LaneFull(full)

            for lane in expanded:
                # INSERT OR REPLACE lets the same holder refresh its row
                # without violating the (lane, holder_id) PK.
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
        """Refresh ``expires_at`` for every lane this holder owns (keyed on ``(lane, holder_id)`` PK)."""
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
        """Drop every (lane, holder_id) row this lease owns. Other
        holders on the same lane are untouched (Inv-7.3 atomic
        release for one holder)."""
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"DELETE FROM leases WHERE lane IN ({placeholders}) "
                f"AND holder_id=?",
                (*lease.lanes, lease.holder_id),
            )
            return cur.rowcount

    async def reap_expired(self) -> list[dict]:
        """Sweep expired rows; emits one ``lease_expired`` event per stale
        (lane, holder_id) row. Reaps only TTL-fired holders, leaving live
        holders on a multi-holder lane untouched."""
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
        """Return the **distinct** lane names with at least one live
        holder. Callers use this for capacity / breakdown observation;
        ``lane_holders`` returns the multi-holder shape."""
        rows = await self.db.fetchall(
            "SELECT DISTINCT lane FROM leases WHERE expires_at > ?",
            (_now_iso(),),
        )
        return [r["lane"] for r in rows]

    async def lane_holders(self) -> dict[str, int]:
        """Return ``{lane: live_holder_count}`` for every lane with at
        least one live row. Used by the breakdown ``lane_timeline``
        collector + by dispatchers to gauge research_lane occupancy.
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
        when the table is missing (legacy DB never opened with v0.8)."""
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


class ResourceLockManager:
    """Coordinator-facing wrapper.

    v0.8 M6 adds non-blocking acquire + multi-holder observability so the
    concurrent dispatcher can fan tasks out without spinning on busy errors.
    """

    def __init__(self, backend: SqliteLeaseBackend):
        self.backend = backend
        # Per-process cumulative acquire / lane-full / lane-busy counters
        # so the breakdown can surface totals without re-reading SQLite.
        self._counters: dict[str, dict[str, int]] = {}

    async def acquire_many(self, lanes: list[str], **kwargs) -> Lease:
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
        """Non-blocking variant of :meth:`acquire_many` (KB_design §3.7 §4.3).

        Returns the :class:`Lease` on success, ``None`` when any lane is
        busy or full (both LaneBusy and LaneFull map to None; retry next tick).
        """
        try:
            return await self.acquire_many(lanes, **kwargs)
        except (LaneBusy, LaneFull):
            return None

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        return await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def release(self, lease: Lease) -> int:
        n = await self.backend.release(lease)
        for lane in lease.lanes:
            self._bump_counter(lane, "release_count")
        return n

    async def reap_expired(self) -> list[dict]:
        return await self.backend.reap_expired()

    async def active_lanes(self) -> list[str]:
        return await self.backend.active_lanes()

    async def lane_holders(self) -> dict[str, int]:
        return await self.backend.lane_holders()

    async def lane_capacities(self) -> dict[str, int]:
        return await self.backend.lane_capacities()

    def counters_snapshot(self) -> dict[str, dict[str, int]]:
        """Return per-lane lifetime counters (acquire / release / busy /
        full counts). Cheap; in-memory copy."""
        return {
            lane: dict(d) for lane, d in self._counters.items()
        }

    def _bump_counter(self, lane: str, field: str) -> None:
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
