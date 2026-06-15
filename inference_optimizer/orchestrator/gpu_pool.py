# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SQLite-backed GPU pool for specialist sub-agents.

Separate from the serving lanes: only constrains specialists that request
``needs_gpu=true`` for short GPU experiments or microbenchmarks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..storage.connection import SqliteConnection


DEFAULT_GPU_LEASE_TTL_SEC = 1800


def _now_iso() -> str:
    """Return the current UTC time as a microsecond ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_gpu_list(raw: str) -> list[int]:
    """Parse a comma/semicolon-separated GPU id list.

    Args:
        raw: Raw string of GPU ids (``,`` or ``;`` separated).

    Returns:
        Unique non-negative GPU ids in first-seen order; malformed entries are
        skipped.
    """
    out: list[int] = []
    for part in (raw or "").replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            idx = int(p)
        except ValueError:
            continue
        if idx >= 0 and idx not in out:
            out.append(idx)
    return out


def resolve_gpu_specialist_devices(capacity: int) -> list[int]:
    """Resolve the GPU ids available to GPU specialists.

    ``INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES`` may name an explicit pool;
    absent → ``range(capacity)``. Capacity zero disables dispatch.
    """
    cap = max(0, int(capacity or 0))
    if cap <= 0:
        return []
    explicit = _parse_gpu_list(
        os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "")
    )
    if explicit:
        return explicit[:cap]
    return list(range(cap))


@dataclass(frozen=True)
class GpuLease:
    holder_id: str
    task_id: str
    gpu_ids: tuple[int, ...]
    acquired_at: str
    expires_at: str


class SpecialistGpuPool:
    """Capacity-limited GPU allocation for specialist tasks."""

    def __init__(
        self,
        db: SqliteConnection,
        *,
        gpu_ids: list[int] | tuple[int, ...],
    ):
        """Initialize the pool over a fixed set of GPU ids.

        Args:
            db: SQLite connection backing the lease table.
            gpu_ids: GPU ids managed by this pool; duplicates and negatives are
                dropped.
        """
        self.db = db
        self.gpu_ids = tuple(dict.fromkeys(int(g) for g in gpu_ids if int(g) >= 0))

    @property
    def capacity(self) -> int:
        """Return the number of GPUs the pool manages."""
        return len(self.gpu_ids)

    async def try_acquire(
        self,
        *,
        count: int,
        holder_id: str,
        task_id: str,
        ttl_sec: int = DEFAULT_GPU_LEASE_TTL_SEC,
    ) -> GpuLease | None:
        """Acquire ``count`` GPU ids or return ``None`` if the pool is full."""
        n = int(count or 0)
        if n <= 0 or n > self.capacity:
            return None
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + max(1, int(ttl_sec or DEFAULT_GPU_LEASE_TTL_SEC))
        expires_iso = datetime.fromtimestamp(
            expires_ts, tz=timezone.utc,
        ).isoformat(timespec="microseconds")

        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            placeholders = ",".join("?" * len(self.gpu_ids))
            cur.execute(
                f"SELECT gpu_id FROM gpu_leases WHERE gpu_id IN ({placeholders})",
                list(self.gpu_ids),
            )
            leased = {int(r["gpu_id"]) for r in cur.fetchall()}
            available = [g for g in self.gpu_ids if g not in leased]
            if len(available) < n:
                return None
            selected = available[:n]
            for gpu_id in selected:
                cur.execute(
                    """
                    INSERT INTO gpu_leases(
                        gpu_id, holder_id, task_id,
                        acquired_at, expires_at, heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (gpu_id, holder_id, task_id, now_iso, expires_iso, now_iso),
                )
        return GpuLease(
            holder_id=holder_id,
            task_id=task_id,
            gpu_ids=tuple(selected),
            acquired_at=now_iso,
            expires_at=expires_iso,
        )

    async def release(self, lease: GpuLease | None) -> None:
        """Release the GPUs held by a lease.

        Args:
            lease: The lease to release; ``None`` or an empty lease is a no-op.
        """
        if lease is None or not lease.gpu_ids:
            return
        placeholders = ",".join("?" * len(lease.gpu_ids))
        params = list(lease.gpu_ids) + [lease.holder_id]
        async with self.db.transaction() as cur:
            cur.execute(
                f"DELETE FROM gpu_leases "
                f"WHERE gpu_id IN ({placeholders}) AND holder_id=?",
                params,
            )

    async def heartbeat(self, lease: GpuLease | None) -> None:
        """Refresh the heartbeat timestamp for a lease's GPUs.

        Args:
            lease: The lease to refresh; ``None`` or an empty lease is a no-op.
        """
        if lease is None or not lease.gpu_ids:
            return
        now_iso = _now_iso()
        placeholders = ",".join("?" * len(lease.gpu_ids))
        params = [now_iso] + list(lease.gpu_ids) + [lease.holder_id]
        async with self.db.transaction() as cur:
            cur.execute(
                f"UPDATE gpu_leases SET heartbeat_at=? "
                f"WHERE gpu_id IN ({placeholders}) AND holder_id=?",
                params,
            )

    async def reap_expired(self) -> int:
        """Actively delete TTL-expired GPU leases; returns rows reaped.

        ``try_acquire`` already clears expired rows opportunistically, but a
        multi-day run may go long stretches without an acquire (e.g. an idle
        EXPLORE phase), leaking capacity. The reaper tick calls this so stale
        leases never pin a GPU id indefinitely."""
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?", (now_iso,),
            )
            return int(cur.rowcount or 0)


__all__ = [
    "DEFAULT_GPU_LEASE_TTL_SEC",
    "GpuLease",
    "SpecialistGpuPool",
    "resolve_gpu_specialist_devices",
]
