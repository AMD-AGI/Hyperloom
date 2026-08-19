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


def geak_rebench_should_apply_result(
    pending_task_id: str,
    task: Task,
    *,
    macro_cycle: int,
) -> bool:
    """True when a finished 2b task may mutate ``geak_pending`` / ``geak_result``.

    Empty ``geak_pending.revalidation_task_id`` means resume-style revalidation
    with no live candidate slot; those results are still applied. Non-empty
    pending that does not track the finishing task is an orphan and is ignored.
    """
    pending_tid = str(pending_task_id or "").strip()
    if not pending_tid:
        return True
    return geak_rebench_tracks_pending_task(pending_tid, task, macro_cycle=macro_cycle)


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


async def settle_dangling_geak_pending(tasks: TaskRegistry, state: Any, *, reason: str) -> bool:
    """Settle ``geak_pending`` once no rebench can still land.

    Driven by state rather than by what a caller just cancelled: the phase
    boundary into CLOSE already cancels the queued rebench, so the CLOSE
    sequencer finds nothing left to cancel yet still has to close the slot. A
    rebench still queued or running is left alone so its result can arrive.
    """
    pending = getattr(state, "geak_pending", None) or {}
    if not isinstance(pending, dict):
        return False
    if str(pending.get("status") or "").strip().lower() != "awaiting_rebench":
        return False
    if await find_inflight_geak_rebench_task(tasks) is not None:
        return False
    state.geak_pending = {
        "status": "rebench_cancelled",
        "revalidation_error": str(reason)[:500],
    }
    state.resume_pending_revalidation = False
    return True


__all__ = [
    "LEGACY_GEAK_REVALIDATE_PLACEHOLDER",
    "cancel_queued_geak_rebench_tasks",
    "find_inflight_geak_rebench_task",
    "geak_rebench_should_apply_result",
    "geak_rebench_tracks_pending_task",
    "geak_revalidate_idempotency_key",
    "geak_revalidation_placeholder_keys",
    "is_geak_same_harness_rebench_task",
    "settle_dangling_geak_pending",
    "spare_geak_rebench_on_phase_transition",
]
