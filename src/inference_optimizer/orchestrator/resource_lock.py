"""ResourceLockManager + SqliteLeaseBackend (DESIGN §3.5).

All acquire-many writes happen inside a single ``BEGIN IMMEDIATE``
transaction. That guarantees the "all or nothing" property that the
lane semantics require — if any lane is busy or expired-but-still-needs-a
lease_expired event, the whole INSERT batch ROLLBACKs.

Cross-lane mutual-exclusion rules (DESIGN §3.5.3) are enforced by adding
extra "conflicting" lanes to the requested set before the SQL acquire.
For example, ``benchmark_lane`` conflicts with ``profile_lane`` so a
``bench_runner`` task asks for ``[benchmark_lane, profile_lane]`` even
though it only "uses" one. This way a single PRIMARY KEY collision in
SQLite enforces the conflict.
"""

from __future__ import annotations

import asyncio
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

# Lane → lanes that must *also* be free or co-acquired (one-directional
# forms). The acquire layer expands the request transitively.
LANE_CONFLICTS: dict[str, frozenset[str]] = {
    # bench cannot run while profile/server-restart is active
    "benchmark_lane": frozenset({"profile_lane", "server_lifecycle"}),
    # profile cannot run while bench/server-restart is active
    "profile_lane": frozenset({"benchmark_lane", "server_lifecycle"}),
    # server lifecycle blocks bench, profile, eval
    "server_lifecycle": frozenset({"benchmark_lane", "profile_lane"}),
    # workspace_mutation is a short global mutex; bench and profile see
    # it via their direct lanes too.
    "workspace_mutation": frozenset(),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _expand_lanes(lanes: list[str]) -> list[str]:
    """Add transitive conflicts. Returns deterministically-sorted unique list.

    Sorting is also what gives us a global lock-ordering.
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
    """Raised by acquire_many when at least one lane is currently held."""

    def __init__(self, busy_lanes: list[str]):
        super().__init__(f"lanes busy: {busy_lanes!r}")
        self.busy_lanes = busy_lanes


class StaleLeaseError(RuntimeError):
    """Heartbeat / release found that the lease no longer belongs to us."""


# ---------------------------------------------------------------------------
class SqliteLeaseBackend:
    """The default ``ResourceLockBackend`` (DESIGN §3.5.4 / ADR-33)."""

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
            # 1. snapshot any existing rows for these lanes
            placeholders = ",".join("?" * len(expanded))
            cur.execute(
                f"SELECT lane, holder_id, expires_at FROM leases WHERE lane IN ({placeholders})",
                expanded,
            )
            existing = {row["lane"]: row for row in cur.fetchall()}

            busy: list[str] = []
            expired: list[str] = []
            for lane, row in existing.items():
                row_expires = datetime.fromisoformat(row["expires_at"]).timestamp()
                if row_expires > now_ts:
                    busy.append(lane)
                else:
                    expired.append(lane)

            if busy:
                # leave the txn (will rollback)
                raise LaneBusy(busy)

            # 2. reap expired rows + emit lease_expired event in same txn
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

            # 3. insert the new lease rows for every requested lane
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
        new_expires_ts = time.time() + ttl_sec
        new_expires_iso = datetime.fromtimestamp(
            new_expires_ts, tz=timezone.utc
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
        """Sweep any expired rows (e.g. holder process died). Emits one
        lease_expired event per stale lane atomically."""
        now_ts = time.time()
        now_iso = _now_iso()
        reaped: list[dict] = []
        async with self.db.transaction() as cur:
            cur.execute(
                "SELECT * FROM leases WHERE expires_at <= ?",
                (datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),),
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
                        json.dumps(
                            {
                                "lane": row["lane"],
                                "previous_holder": row["holder_id"],
                                "reap_pass": True,
                            }
                        ),
                        2,
                        now_iso,
                    ),
                )
                reaped.append(row)
        return reaped

    async def active_lanes(self) -> list[str]:
        rows = await self.db.fetchall(
            "SELECT lane FROM leases WHERE expires_at > ?",
            (_now_iso(),),
        )
        return [r["lane"] for r in rows]


# ---------------------------------------------------------------------------
class ResourceLockManager:
    """High-level wrapper that adds back-off + dead-lock safe retry.

    Mirrors DESIGN §3.5.5 retry policy: exponential back-off 100ms → 1s →
    5s, total wait capped at ``ttl_sec * 2``. Returns a :class:`Lease`
    that the caller can ``await heartbeat / release`` on.
    """

    def __init__(self, backend: SqliteLeaseBackend):
        self.backend = backend

    async def acquire(
        self,
        lanes: list[str],
        *,
        holder_id: str | None = None,
        task_id: str,
        action: str,
        ttl_sec: int,
    ) -> Lease:
        holder_id = holder_id or uuid.uuid4().hex
        backoffs = (0.1, 1.0, 5.0)
        deadline = time.monotonic() + ttl_sec * 2
        attempt = 0
        last_err: LaneBusy | None = None
        while True:
            try:
                return await self.backend.acquire_many(
                    lanes,
                    holder_id=holder_id,
                    task_id=task_id,
                    action=action,
                    ttl_sec=ttl_sec,
                )
            except LaneBusy as err:
                last_err = err
                if time.monotonic() >= deadline:
                    raise
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                attempt += 1
                await asyncio.sleep(wait)
                continue
        # unreachable
        raise last_err  # pragma: no cover

    async def release(self, lease: Lease) -> None:
        await self.backend.release(lease)

    async def heartbeat(self, lease: Lease, *, ttl_sec: int) -> None:
        await self.backend.heartbeat(lease, ttl_sec=ttl_sec)

    async def summary(self) -> list[str]:
        return await self.backend.active_lanes()
