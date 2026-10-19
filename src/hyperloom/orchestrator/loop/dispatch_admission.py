# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admission of a queued task: what it is allowed to hold before its action starts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from ..state.task_registry import Task


class LaneAdmission(NamedTuple):
    """Whether a queued task may start now, and the lane lease it holds while it runs."""

    admitted: bool
    lease: Any = None


class GpuAdmission(NamedTuple):
    """Whether a GPU specialist may start now, and the GPU leases it holds while it runs."""

    admitted: bool
    gpu_lease: Any = None
    gpu_specialist_lease: Any = None


class GpuPoolChoice(NamedTuple):
    """The pool a GPU specialist leases from and the card count it defaults to."""

    gpu_pool: Any
    default_gpu_count: int


def serving_slot_probe() -> Callable[[], bool] | None:
    """The serving-slot probe when serving priority is on; ``None`` leaves GPU specialists unpaused.

    Read once per dispatch pass (a pure env-var read, zero cost) and best-effort throughout: any
    failure here leaves dispatch ungated rather than blocked.
    """
    try:
        from ..actions.executors._ray_backend import ray_serving_priority_enabled, serving_slot_busy

        return serving_slot_busy if ray_serving_priority_enabled() else None
    except Exception:  # noqa: BLE001 — never block dispatch on the probe
        return None


async def acquire_gpu_lease(gpu_pool: Any, task: Task, *, gpu_count: int, ttl_sec: float) -> Any:
    """Take the pool lease for a GPU specialist, by Ray pending limit or by physical capacity.

    Under single-node Ray the physical GPU mutex is Ray's ``num_gpus``, not this SQLite pool, so
    admission is a count-based pending limit and several specialists may queue on one physical GPU
    (Ray time-multiplexes them). Off the Ray path (multi-node / RAY_EXEC off / pytest) the SQLite
    pool stays the physical mutex.
    """
    from ..actions.executors._ray_backend import ray_gpu_pending_limit, ray_gpu_specialist_exec_enabled

    if ray_gpu_specialist_exec_enabled():
        return await gpu_pool.try_acquire_ray_observation(
            holder_id=task.task_id,
            task_id=task.task_id,
            pending_limit=ray_gpu_pending_limit(),
            ttl_sec=ttl_sec,
        )
    return await gpu_pool.try_acquire(
        count=gpu_count,
        holder_id=task.task_id,
        task_id=task.task_id,
        ttl_sec=ttl_sec,
    )
