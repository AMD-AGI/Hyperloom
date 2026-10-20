# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admission of a queued task: what it is allowed to hold before its action starts."""

from __future__ import annotations

import logging as _logging
from typing import Any, NamedTuple

from hyperloom.common.env import is_truthy

from ..state.task_registry import Task

log = _logging.getLogger(__name__)


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


def gpu_pool_for_specialist(
    params: dict[str, Any],
    *,
    framework_pool: Any,
    specialist_pool: Any,
    serving_tp: int,
) -> GpuPoolChoice:
    """Pick the pool a GPU specialist leases from and the card count it defaults to."""
    from ..specialists.profile import is_authoring_specialist, uses_whole_machine_gpu_lane

    if not uses_whole_machine_gpu_lane(params):
        return GpuPoolChoice(specialist_pool, serving_tp or 1)
    if is_authoring_specialist(params):
        return GpuPoolChoice(framework_pool, framework_pool.capacity or 1)
    # Bench specialist: size to the serving TP.
    return GpuPoolChoice(framework_pool, serving_tp or framework_pool.capacity or 1)


def specialist_gpu_count(task: Task, default_gpu_count: int, serving_tp: int) -> int:
    """Cards to lease for a GPU specialist, floored up to the serving TP when it benches."""
    params = task.params or {}
    try:
        gpu_count = int(params.get("gpu_count", default_gpu_count) or default_gpu_count)
    except (TypeError, ValueError):
        gpu_count = default_gpu_count
    if is_truthy(params.get("bench")) and serving_tp > 0 and gpu_count < serving_tp:
        log.info(
            "specialist %s: bench=true with gpu_count=%d < serving "
            "TP=%d; flooring gpu_count to TP (a bench specialist "
            "starts a real TP-sharded server and cannot run on "
            "fewer cards).",
            task.task_id,
            gpu_count,
            serving_tp,
        )
        return serving_tp
    return gpu_count


async def acquire_gpu_lease(gpu_pool: Any, task: Task, *, gpu_count: int, ttl_sec: float) -> Any:
    """Take the pool lease for a GPU specialist, by Ray pending limit or by physical capacity.

    Under single-node Ray the physical GPU mutex is Ray's ``num_gpus``, not this SQLite pool, so
    admission there is a count-based pending limit and several specialists may share one GPU.
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
