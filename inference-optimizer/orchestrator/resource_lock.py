"""ResourceLockManager + SqliteLeaseBackend (DESIGN v0.6 §3.5).

All ``acquire_many`` writes happen inside a single ``BEGIN IMMEDIATE``
transaction, giving the "all or nothing" property the lane semantics
require: if any lane is busy, the entire INSERT batch ROLLBACKs.

Cross-lane mutual-exclusion rules (DESIGN §3.5.3) are enforced by adding
the conflicting lanes to the requested set before the SQL acquire — e.g.
``benchmark_lane`` conflicts with ``profile_lane``, so a ``bench_runner``
task transparently asks for ``[benchmark_lane, profile_lane]`` and lets
SQLite's PRIMARY KEY collision enforce the conflict.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..storage.connection import SqliteConnection


KNOWN_LANES = (
    "server_lifecycle",
    "workspace_mutation",
    "benchmark_lane",
    "profile_lane",
)

# Lane → lanes that must *also* be free or co-acquired (DESIGN §3.5.3).
LANE_CONFLICTS: dict[str, frozenset[str]] = {
    "benchmark_lane": frozenset({"profile_lane", "server_lifecycle"}),
    "profile_lane":   frozenset({"benchmark_lane", "server_lifecycle"}),
    "server_lifecycle": frozenset({"benchmark_lane", "profile_lane"}),
    "workspace_mutation": frozenset(),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Expand requested lanes by transitive conflicts; sort deterministically."""
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
    """Raised by ``acquire_many`` when at least one lane is currently held."""

    def __init__(self, busy_lanes: list[str]):
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


# ---------------------------------------------------------------------------
class SqliteLeaseBackend:
    """Default ``ResourceLockBackend`` (DESIGN §3.5.4 / ADR-42).

    Uses ``leases`` rows in the unified SQLite WAL DB. ``BEGIN IMMEDIATE``
    plus row-level uniqueness on the ``lane`` PK gives atomic acquire-many.
    """

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
        if not lanes:
            raise ValueError("acquire_many called with no lanes")
        expanded = _expand_lanes(lanes)
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + ttl_sec
        expires_iso = datetime.fromtimestamp(expires_ts, tz=timezone.utc).isoformat()

        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(expanded))
            cur.execute(
                f"SELECT lane, holder_id, expires_at FROM leases "
                f"WHERE lane IN ({placeholders})",
                expanded,
            )
            existing = {row["lane"]: dict(row) for row in cur.fetchall()}

            busy: list[str] = []
            expired: list[str] = []
            for lane, row in existing.items():
                row_expires = datetime.fromisoformat(row["expires_at"]).timestamp()
                if row_expires > now_ts:
                    busy.append(lane)
                else:
                    expired.append(lane)

            if busy:
                raise LaneBusy(busy)

            for lane in expired:
                cur.execute("DELETE FROM leases WHERE lane=?", (lane,))
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
                            {"lane": lane, "previous_holder": existing[lane]["holder_id"]}
                        ),
                        2,
                        now_iso,
                    ),
                )

            for lane in expanded:
                cur.execute(
                    "INSERT INTO leases(lane, holder_id, task_id, action, pid, "
                    "acquired_at, expires_at, heartbeat_at) "
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
        async with self.db.transaction() as cur:
            placeholders = ",".join("?" * len(lease.lanes))
            cur.execute(
                f"DELETE FROM leases WHERE lane IN ({placeholders}) "
                f"AND holder_id=?",
                (*lease.lanes, lease.holder_id),
            )
            return cur.rowcount

    async def reap_expired(self) -> list[dict]:
        """Sweep expired rows; emits one ``lease_expired`` event per stale lane."""
        now_iso_str = _now_iso()
        reaped: list[dict] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT * FROM leases WHERE expires_at <= ?",
                (now_iso_str,),
            )
            stale = [dict(r) for r in cur.fetchall()]
            for row in stale:
                cur.execute("DELETE FROM leases WHERE lane=?", (row["lane"],))
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
        rows = await self.db.fetchall(
            "SELECT lane FROM leases WHERE expires_at > ?", (_now_iso(),)
        )
        return [r["lane"] for r in rows]


# ---------------------------------------------------------------------------
class ResourceLockManager:
    """Coordinator-facing wrapper. Today it's a thin pass-through over the
    backend; future versions can attach scheduling hints / metrics here.
    """

    def __init__(self, backend: SqliteLeaseBackend):
        self.backend = backend

    async def acquire_many(self, lanes: list[str], **kwargs) -> Lease:
        return await self.backend.acquire_many(lanes, **kwargs)

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        return await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def release(self, lease: Lease) -> int:
        return await self.backend.release(lease)

    async def reap_expired(self) -> list[dict]:
        return await self.backend.reap_expired()

    async def active_lanes(self) -> list[str]:
        return await self.backend.active_lanes()


__all__ = [
    "KNOWN_LANES",
    "LANE_CONFLICTS",
    "LaneBusy",
    "Lease",
    "ResourceLockManager",
    "SqliteLeaseBackend",
    "StaleLeaseError",
]
