# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite-backed GPU pool for specialist sub-agents.

Separate from the serving lanes: constrains specialists that request
``needs_gpu=true`` for wall-budgeted on-GPU work (servers on non-8888 ports,
profiling, autotune, and real benchmark loops on their leased cards).

Ray-managed GPU execution (gated by
:func:`hyperloom.orchestrator.actions.executors._ray_backend.ray_gpu_specialist_exec_enabled`):
under single-node Ray execution the **physical** card assignment for a ``needs_gpu``
specialist is Ray's — the specialist subprocess runs inside a
``GpuSpecialistActor(num_gpus=…)`` that Ray masks to its assigned devices. This
pool is then only **capacity / TTL accounting**: ``try_acquire`` still gates
concurrency and records a lease (so the wall-budget / TTL views, prompt display,
and the existing dispatch tests keep working), but the ``GpuLease.gpu_ids`` it
returns are no longer the physical device pins (the dispatcher advertises the
logical ``0..N-1`` view Ray exposes instead). Off the Ray path (multi-node /
``INFERENCE_OPTIMIZER_RAY_EXEC`` off / tests) the ids remain the device pins.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from hyperloom.common.timeutil import now_iso
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS, parse_device_list

from .storage.connection import SqliteConnection


log = logging.getLogger(__name__)


DEFAULT_GPU_LEASE_TTL_SEC = 1800

# Reserved synthetic gpu_id base for single-node Ray pending-observation slots.
# Ray owns the physical card assignment, so under Ray this pool is a
# count-based admission ledger, not a physical-id allocator: admissions take a
# synthetic slot id in ``[_RAY_OBS_ID_BASE, _RAY_OBS_ID_BASE + pending_limit)``
# so multiple specialists can queue on a single physical GPU (Ray serializes
# them via ``num_gpus``) while TTL / reap / resume / release keep working on the
# same ``gpu_leases`` table. The base is far above any real device id.
_RAY_OBS_ID_BASE = 100000

# GPU-lease / gpu_research_lane TTL grace over the agent wall budget. Iron law:
# ``kill ≤ gpu_lease TTL ≤ gpu_research_lane TTL`` — the lease must outlive the
# agent's wall-budget kill so cards are not reclaimed mid-compute. TTL =
# wall_budget × (1 + grace).
GPU_LEASE_TTL_GRACE = 0.1


_now_iso = now_iso


def _parse_gpu_list(raw: str) -> list[int]:
    """Parse a comma/semicolon-separated GPU id list.

    Thin alias for :func:`hyperloom.common.visible_devices.parse_device_list`,
    the single definition of this parse; kept as a module-local name because
    call sites and tests already reference it.

    Args:
        raw: Raw string of GPU ids (``,`` or ``;`` separated).

    Returns:
        Unique non-negative GPU ids in first-seen order; malformed entries are
        skipped.
    """
    return parse_device_list(raw)


def _explicit_pool() -> list[int] | None:
    """Resolve the operator's explicit GPU pool, or ``None`` when unset.

    An unset or blank value is treated as absent so the caller falls back to
    the visible-device mask. A non-blank value that parses to no valid ids is
    treated as a misconfiguration and fails closed rather than widening to the
    pool the operator intended to override.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES")
    if raw is None or not raw.strip():
        return None
    ids = _parse_gpu_list(raw)
    if not ids:
        log.error("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES=%r has no valid GPU ids", raw)
    return ids


def _visible_device_mask() -> tuple[list[int], bool]:
    """Resolve the process's visible-GPU mask as absolute device ids.

    Checks ``ROCR_VISIBLE_DEVICES`` first (canonical ROCm pinning per the repo
    convention; the CLI preflight drops ``HIP_VISIBLE_DEVICES`` when ROCR is
    set), then ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``.

    Returns:
        ``(ids, present)`` where ``present`` is True when any of the masks is
        set (even to an empty string, which means "no visible GPUs" → ``[]``);
        ``present`` is False only when none of the masks is set.
    """
    for env_name in COUNTING_VISIBLE_DEVICE_VARS:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        return _parse_gpu_list(raw), True
    return [], False


def resolve_gpu_specialist_devices(
    capacity: int,
    *,
    serving_tp: int = 0,
) -> list[int]:
    """Resolve the absolute GPU ids available to GPU specialists.

    Precedence:

    1. ``INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES`` — explicit operator pool,
       capped to ``capacity``. The operator has already carved a
       specialist-only set here, so it is trusted verbatim and the serving
       cards are *not* subtracted again. Set but unparseable yields ``[]``
       rather than falling through to the wider pool it was meant to override.
    2. The process visible-device mask (``ROCR_VISIBLE_DEVICES``, then
       ``HIP``/``CUDA``), serving cards carved off the front, capped to
       ``capacity``. The leased ids are written verbatim into each specialist
       subprocess's ``ROCR_VISIBLE_DEVICES``, so scoping the pool to the mask
       keeps specialists on the operator's pinned cards and never hands them a
       card outside the serving/benchmark mask.
    3. No mask set → ``range(capacity)`` with serving cards carved off the
       front (whole-machine ids).

    Serving-disjoint physics invariant: the live serving process holds the
    first ``serving_tp`` cards of whatever pool we are scoped to. Handing a
    specialist one of those cards corrupts both the specialist's measurement
    and production serving (shared cards = garbage numbers for both), so they
    are subtracted from the resolvable pool. ``serving_tp=0`` (the default)
    preserves the legacy whole-pool behaviour.

    Capacity zero disables dispatch.

    Args:
        capacity: Maximum number of GPU ids to make available; values ``<= 0``
            disable dispatch.
        serving_tp: Number of cards the live serving process holds (its TP
            size). Those cards are carved off the front of the resolved pool
            for the mask / whole-machine cases. Ignored for the explicit
            operator pool (already carved by the operator).

    Returns:
        The absolute GPU ids available to specialists; ``[]`` when capacity is
        non-positive, the visible mask is set but empty, or serving claims the
        whole pool.
    """
    cap = max(0, int(capacity or 0))
    if cap <= 0:
        return []
    explicit = _explicit_pool()
    if explicit is not None:
        return explicit[:cap]
    serving = max(0, int(serving_tp or 0))
    mask_ids, mask_present = _visible_device_mask()
    if mask_present:
        return mask_ids[serving:][:cap]
    return list(range(cap))[serving:]


def resolve_whole_machine_devices() -> list[int]:
    """Resolve the full set of GPU ids on this node — **no** serving carve.

    Returns *every* visible card and, unlike
    :func:`resolve_gpu_specialist_devices`, does **not**:

    * subtract the ``serving_tp`` cards, nor
    * gate on ``gpu_specialist_capacity``.

    Id-source precedence mirrors :func:`resolve_gpu_specialist_devices`:

    1. ``INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES`` — explicit operator pool
       (used verbatim, uncapped here); set but unparseable yields ``[]``.
    2. The process visible-device mask (``ROCR_VISIBLE_DEVICES`` →
       ``HIP``/``CUDA``) — used verbatim.
    3. No mask set → ``range(detect_gpu_count())`` (the machine's detected GPU
       count, probed lazily via ``rocm-smi`` only in this branch).

    Returns:
        The absolute GPU ids available to a framework whole-machine lease;
        ``[]`` when the mask is set-but-empty or nothing can be resolved.
    """
    explicit = _explicit_pool()
    if explicit is not None:
        return explicit
    mask_ids, mask_present = _visible_device_mask()
    if mask_present:
        return mask_ids
    # No mask: fall back to the detected machine GPU count.
    from ..policy.gate import detect_gpu_count

    return list(range(max(0, int(detect_gpu_count() or 0))))


@dataclass(frozen=True)
class GpuLease:
    holder_id: str
    task_id: str
    gpu_ids: tuple[int, ...]
    acquired_at: str
    expires_at: str


class SpecialistGpuPool:
    """Capacity-limited GPU allocation for specialist tasks.

    Under single-node Ray execution this is capacity/TTL accounting only; Ray
    owns the physical card assignment (see the module docstring).
    """

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
        """Return the number of GPUs the pool manages.

        Returns:
            The count of GPU ids managed by this pool.
        """
        return len(self.gpu_ids)

    async def try_acquire(
        self,
        *,
        count: int,
        holder_id: str,
        task_id: str,
        ttl_sec: int = DEFAULT_GPU_LEASE_TTL_SEC,
    ) -> GpuLease | None:
        """Acquire ``count`` GPU ids or return ``None`` if the pool is full.

        Same holder_id + task_id is idempotent: a live lease is returned as-is
        (with its TTL refreshed) rather than allocating additional GPUs.

        Args:
            count: Number of GPU ids to acquire.
            holder_id: Identifier of the lease holder.
            task_id: Identifier of the task the lease is for.
            ttl_sec: Lease time-to-live in seconds.

        Returns:
            A :class:`GpuLease` for the acquired GPUs, or ``None`` when the
            request is invalid or insufficient GPUs are free.
        """
        n = int(count or 0)
        if n <= 0 or n > self.capacity:
            return None
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + max(1, int(ttl_sec or DEFAULT_GPU_LEASE_TTL_SEC))
        expires_iso = datetime.fromtimestamp(
            expires_ts,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds")

        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            # Same-holder/task acquire is idempotent when the existing lease
            # already satisfies the request: same count and every id still in
            # this pool. If either condition fails the old lease is stale and is
            # released so the request can be satisfied from the current pool.
            cur.execute(
                "SELECT gpu_id, acquired_at FROM gpu_leases WHERE holder_id=? AND task_id=?",
                (holder_id, task_id),
            )
            existing_rows = cur.fetchall()
            if existing_rows:
                existing_ids = tuple(int(r["gpu_id"]) for r in existing_rows)
                pool_set = set(self.gpu_ids)
                if len(existing_ids) == n and set(existing_ids) <= pool_set:
                    acquired_at = existing_rows[0]["acquired_at"]
                    cur.execute(
                        "UPDATE gpu_leases SET expires_at=?, heartbeat_at=? WHERE holder_id=? AND task_id=?",
                        (expires_iso, now_iso, holder_id, task_id),
                    )
                    return GpuLease(
                        holder_id=holder_id,
                        task_id=task_id,
                        gpu_ids=existing_ids,
                        acquired_at=acquired_at,
                        expires_at=expires_iso,
                    )
                # Stale lease — release it and fall through to a fresh acquire.
                cur.execute(
                    "DELETE FROM gpu_leases WHERE holder_id=? AND task_id=?",
                    (holder_id, task_id),
                )
            placeholders = ",".join("?" * len(self.gpu_ids))
            cur.execute(  # nosec B608 - placeholders string is generated from configured GPU id count.
                f"SELECT gpu_id FROM gpu_leases WHERE gpu_id IN ({placeholders})",  # nosec B608 - generated placeholders only.
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

    async def try_acquire_ray_observation(
        self,
        *,
        holder_id: str,
        task_id: str,
        pending_limit: int,
        ttl_sec: int = DEFAULT_GPU_LEASE_TTL_SEC,
    ) -> GpuLease | None:
        """Admit a GPU specialist under single-node Ray by COUNT, not physical id.

        Under single-node Ray execution the physical card is Ray's (the
        specialist runs inside a ``GpuSpecialistActor(num_gpus=…)``), so this
        pool becomes a count-based admission ledger: it hands out a
        synthetic slot id from ``[_RAY_OBS_ID_BASE, _RAY_OBS_ID_BASE +
        pending_limit)`` so up to ``pending_limit`` specialists can be in-flight
        (pending + running) at once — Ray then serializes them on ``num_gpus``.
        Returns ``None`` when the pending limit is already reached (backpressure;
        the task stays queued). TTL / :meth:`reap_expired` / :meth:`release` all
        work unchanged on the synthetic rows (same ``gpu_leases`` table).

        Args:
            holder_id: Identifier of the lease holder.
            task_id: Identifier of the task the lease is for.
            pending_limit: Max concurrent in-flight GPU specialists (>= 1).
            ttl_sec: Lease time-to-live in seconds.

        Returns:
            A :class:`GpuLease` over a synthetic slot id, or ``None`` when the
            pending limit is reached.
        """
        limit = max(1, int(pending_limit or 1))
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + max(1, int(ttl_sec or DEFAULT_GPU_LEASE_TTL_SEC))
        expires_iso = datetime.fromtimestamp(
            expires_ts,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds")

        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            cur.execute(
                "SELECT gpu_id FROM gpu_leases WHERE gpu_id >= ?",
                (_RAY_OBS_ID_BASE,),
            )
            used = {int(r["gpu_id"]) for r in cur.fetchall()}
            slot: int | None = None
            for i in range(limit):
                cand = _RAY_OBS_ID_BASE + i
                if cand not in used:
                    slot = cand
                    break
            if slot is None:
                return None
            cur.execute(
                """
                INSERT INTO gpu_leases(
                    gpu_id, holder_id, task_id,
                    acquired_at, expires_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (slot, holder_id, task_id, now_iso, expires_iso, now_iso),
            )
        return GpuLease(
            holder_id=holder_id,
            task_id=task_id,
            gpu_ids=(slot,),
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
            cur.execute(  # nosec B608 - placeholders string is generated from lease GPU id count.
                f"DELETE FROM gpu_leases WHERE gpu_id IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                params,
            )

    async def extend(self, task_id: str, ttl_sec: int) -> int:
        """Push a task's GPU rows out to ``ttl_sec`` from now.

        Keeps the ``kill <= gpu_lease TTL <= gpu_research_lane TTL`` ordering
        when a lane lease is extended.

        Args:
            task_id: The task whose GPU rows should be refreshed.
            ttl_sec: New lifetime in seconds from now.

        Returns:
            The number of GPU rows refreshed.
        """
        expires_iso = datetime.fromtimestamp(time.time() + max(0, int(ttl_sec)), tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "UPDATE gpu_leases SET expires_at=?, heartbeat_at=? WHERE task_id=?",
                (expires_iso, now_iso, task_id),
            )
            return int(cur.rowcount or 0)

    async def reap_expired(self) -> int:
        """Actively delete TTL-expired GPU leases; returns rows reaped.

        ``try_acquire`` already clears expired rows opportunistically, but a
        multi-day run may go long stretches without an acquire (e.g. an idle
        EXPLORE phase), leaking capacity. The reaper tick calls this so stale
        leases never pin a GPU id indefinitely.

        Returns:
            The number of expired lease rows deleted.
        """
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            return int(cur.rowcount or 0)


__all__ = [
    "DEFAULT_GPU_LEASE_TTL_SEC",
    "GPU_LEASE_TTL_GRACE",
    "GpuLease",
    "SpecialistGpuPool",
    "resolve_gpu_specialist_devices",
    "resolve_whole_machine_devices",
]
