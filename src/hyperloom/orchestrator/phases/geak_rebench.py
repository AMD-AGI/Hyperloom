# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK same-harness revalidation task identity and phase-boundary policy."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .machine_state import PHASE_CLOSE

if TYPE_CHECKING:
    from ..state.task_registry import Task, TaskRegistry

LEGACY_GEAK_REVALIDATE_PLACEHOLDER = "geak-revalidate"


def geak_revalidate_idempotency_key(macro_cycle: int) -> str:
    """Return the per-macro-cycle idempotency key for a GEAK 2b rebench task."""
    return f"geak-revalidate-c{macro_cycle}"


def geak_revalidation_placeholder_keys(macro_cycle: int) -> frozenset[str]:
    """Placeholder ids written to ``geak_pending`` before the task row exists."""
    return frozenset({LEGACY_GEAK_REVALIDATE_PLACEHOLDER, geak_revalidate_idempotency_key(macro_cycle)})


def is_geak_same_harness_rebench_task(kind: str, params: dict[str, Any] | None) -> bool:
    """True when a queued/running task is the orchestrator GEAK 2b revalidation explore."""
    payload = params if isinstance(params, dict) else {}
    return (
        str(kind or "").strip() == "explore"
        and str(payload.get("source") or "") == "resume_stack_revalidate"
        and bool(payload.get("geak_fallback"))
    )


def spare_geak_rebench_on_phase_transition(*, target_phase: str, kind: str, params: dict[str, Any]) -> bool:
    """Return True to leave a queued GEAK rebench alive across a phase boundary."""
    if (target_phase or "").strip().upper() == PHASE_CLOSE:
        return False
    return is_geak_same_harness_rebench_task(kind, params)


def geak_rebench_tracks_pending_task(
    pending_task_id: str,
    task: Task,
    *,
    macro_cycle: int,
) -> bool:
    """True when ``geak_pending.revalidation_task_id`` tracks this rebench task."""
    tracked = str(pending_task_id or "").strip()
    if not tracked:
        return False
    ids = {task.task_id, str(task.idempotency_key or "")}
    if tracked in ids:
        return True
    placeholders = geak_revalidation_placeholder_keys(macro_cycle)
    if tracked in placeholders and str(task.idempotency_key or "") in placeholders:
        return True
    if tracked == LEGACY_GEAK_REVALIDATE_PLACEHOLDER and str(task.idempotency_key or "").startswith(
        "geak-revalidate"
    ):
        return True
    return False


async def find_inflight_geak_rebench_task(tasks: TaskRegistry) -> Task | None:
    """Return the oldest queued/running GEAK same-harness rebench, if any."""
    queued_fn = getattr(tasks, "queued", None)
    running_fn = getattr(tasks, "running", None)
    if not callable(queued_fn) or not callable(running_fn):
        return None
    for pool in (await queued_fn(), await running_fn()):
        for task in pool:
            if is_geak_same_harness_rebench_task(task.kind, task.params):
                return task
    return None


async def cancel_queued_geak_rebench_tasks(tasks: TaskRegistry, *, reason: str) -> list[str]:
    """Cancel queued GEAK 2b rebench tasks (for example on CLOSE entry)."""
    cancelled: list[str] = []
    for task in await tasks.queued():
        if not is_geak_same_harness_rebench_task(task.kind, task.params):
            continue
        await tasks.transition(task.task_id, "cancelled", evidence={"reason": reason})
        cancelled.append(task.task_id)
    return cancelled


__all__ = [
    "LEGACY_GEAK_REVALIDATE_PLACEHOLDER",
    "cancel_queued_geak_rebench_tasks",
    "find_inflight_geak_rebench_task",
    "geak_rebench_tracks_pending_task",
    "geak_revalidate_idempotency_key",
    "geak_revalidation_placeholder_keys",
    "is_geak_same_harness_rebench_task",
    "spare_geak_rebench_on_phase_transition",
]
