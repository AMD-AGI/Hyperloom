# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK same-harness revalidation task identity and phase-boundary policy."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .machine_state import PHASE_CLOSE

if TYPE_CHECKING:
    from ..state.task_registry import Task, TaskRegistry

LEGACY_GEAK_REVALIDATE_PLACEHOLDER = "geak-revalidate"

# ``geak_pending.status`` values that record a closed verdict. A result arriving
# against one of these is late or orphaned and must not reopen the slot.
SETTLED_PENDING_STATUSES: frozenset[str] = frozenset({"rebench_cancelled", "rebench_unavailable"})

# Fresh keys a single macro-cycle may mint. Each cancelled attempt burns one, so
# this bounds how often a prune/cancel loop can re-dispatch the same rebench.
MAX_REBENCH_ATTEMPTS_PER_CYCLE = 4


def geak_revalidate_idempotency_key(macro_cycle: int, attempt: int = 0) -> str:
    """Return the idempotency key for a GEAK 2b rebench task.

    ``attempt`` distinguishes retries within one macro-cycle: reusing the key of
    a settled row would hand that row back from
    ``create_or_return_existing`` and be read as ``rebench_unavailable``.
    """
    base = f"geak-revalidate-c{macro_cycle}"
    return base if attempt <= 0 else f"{base}-r{attempt}"


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
    """Return True to leave a queued GEAK rebench alive across a phase boundary.

    Deny-list rather than allow-list: only ``CLOSE`` kills the rebench, every
    other target spares it. The window that actually matters is KERNEL through
    SWEEP (plus the SWEEP re-loop back into FRAMEWORK_AGENT), so an allow-list would be
    tighter — but the phase set changes over time and a missing entry silently
    reintroduces the audit-only bug this whole path exists to prevent, whereas a
    surplus entry costs at most one wasted bench.

    Correctness of the surplus is owned elsewhere: a rebench that outlives its
    macro-cycle is refused by :func:`geak_rebench_should_apply_result`, because
    the slot it would have to be tracked in has moved on. So this predicate only
    decides whether the task keeps running, never whether its result counts.
    """
    if (target_phase or "").strip().upper() == PHASE_CLOSE:
        return False
    return is_geak_same_harness_rebench_task(kind, params)


def geak_rebench_tracks_pending_task(
    pending_task_id: str,
    task: Task,
    *,
    macro_cycle: int,
) -> bool:
    """True when ``geak_pending.revalidation_task_id`` tracks this rebench task.

    The slot normally holds a task id. It holds a key instead only inside the
    reservation window — the slot is published before the task row exists so the
    phase guard can see a pending revalidation — and, on state written before
    keys were cycle-scoped, the bare legacy key. Both are matched through
    :func:`geak_revalidation_placeholder_keys`, which requires the task to carry
    that same key, so a rebench from another macro-cycle cannot claim the slot.
    """
    tracked = str(pending_task_id or "").strip()
    if not tracked:
        return False
    key = str(task.idempotency_key or "")
    if tracked in {task.task_id, key}:
        return True
    placeholders = geak_revalidation_placeholder_keys(macro_cycle)
    return tracked in placeholders and key in placeholders


def geak_rebench_should_apply_result(state: Any, task: Task, *, macro_cycle: int) -> bool:
    """True when a finished 2b task may mutate ``geak_pending`` / ``geak_result``.

    Ordered so the closed verdicts win first:

    * a settled slot rejects everything — the verdict is already recorded;
    * a slot naming a task accepts only that task, so orphans are ignored;
    * ``awaiting_rebench`` with no id yet is the window between recording the
      candidate and publishing the task id, so it accepts;
    * an empty slot accepts only while ``resume_pending_revalidation`` marks a
      resume revalidation, which owns no candidate slot of its own.
    """
    pending = getattr(state, "geak_pending", None) or {}
    if not isinstance(pending, dict):
        pending = {}
    status = str(pending.get("status") or "").strip().lower()
    if status in SETTLED_PENDING_STATUSES:
        return False
    tracked = str(pending.get("revalidation_task_id") or "").strip()
    if tracked:
        return geak_rebench_tracks_pending_task(tracked, task, macro_cycle=macro_cycle)
    if status == "awaiting_rebench":
        return True
    return bool(getattr(state, "resume_pending_revalidation", False))


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


async def cancel_geak_rebench_tasks(
    tasks: TaskRegistry,
    *,
    reason: str,
    include_running: bool = False,
) -> list[str]:
    """Cancel in-flight GEAK 2b rebench tasks.

    Args:
        tasks: The task registry.
        reason: Stamped onto each cancellation's history evidence.
        include_running: Also cancel a rebench already executing. CLOSE sets
            this: the phase only writes reports, so a running rebench holds the
            GPU lane against post-opt roofline and its result can no longer be
            consumed. A backlog drain (prune) leaves running work alone.

    Returns:
        The cancelled task ids.
    """
    queued_fn = getattr(tasks, "queued", None)
    running_fn = getattr(tasks, "running", None)
    if not callable(queued_fn):
        return []
    pools = [await queued_fn()]
    if include_running and callable(running_fn):
        pools.append(await running_fn())
    cancelled: list[str] = []
    for pool in pools:
        for task in pool:
            if not is_geak_same_harness_rebench_task(task.kind, task.params):
                continue
            await tasks.transition(task.task_id, "cancelled", evidence={"reason": reason})
            cancelled.append(task.task_id)
    return cancelled


async def resolve_geak_revalidate_idempotency_key(tasks: TaskRegistry, macro_cycle: int) -> str:
    """Pick the key for the next 2b rebench in ``macro_cycle``.

    Steps past attempts whose row already settled, because reusing their key
    returns that terminal row instead of dispatching. Stops at
    ``MAX_REBENCH_ATTEMPTS_PER_CYCLE`` so a cancel loop cannot dispatch forever.
    """
    lookup = getattr(tasks, "find_by_idempotency_key", None)
    if not callable(lookup):
        return geak_revalidate_idempotency_key(macro_cycle)
    last = geak_revalidate_idempotency_key(macro_cycle)
    for attempt in range(MAX_REBENCH_ATTEMPTS_PER_CYCLE):
        last = geak_revalidate_idempotency_key(macro_cycle, attempt)
        row = await lookup(last)
        if row is None or row.state in {"queued", "running"}:
            return last
    return last


async def settle_dangling_geak_pending(tasks: TaskRegistry, state: Any, *, reason: str) -> bool:
    """Settle ``geak_pending`` once no rebench can still land.

    Driven by state rather than by what a caller just cancelled: the phase
    boundary into CLOSE already cancels the queued rebench, so the CLOSE
    sequencer finds nothing left to cancel yet still has to close the slot. A
    rebench still in flight is left alone so its result can arrive.

    Only the verdict fields change. The candidate's self-reported numbers are
    what the report uses to say *what* was dropped, so they are kept; the id of
    a task that will never land is not.
    """
    pending = getattr(state, "geak_pending", None) or {}
    if not isinstance(pending, dict):
        return False
    if str(pending.get("status") or "").strip().lower() != "awaiting_rebench":
        return False
    if await find_inflight_geak_rebench_task(tasks) is not None:
        return False
    settled = dict(pending)
    settled["status"] = "rebench_cancelled"
    settled["revalidation_error"] = str(reason)[:500]
    settled.pop("revalidation_task_id", None)
    state.geak_pending = settled
    state.resume_pending_revalidation = False
    return True


__all__ = [
    "LEGACY_GEAK_REVALIDATE_PLACEHOLDER",
    "MAX_REBENCH_ATTEMPTS_PER_CYCLE",
    "SETTLED_PENDING_STATUSES",
    "cancel_geak_rebench_tasks",
    "find_inflight_geak_rebench_task",
    "geak_rebench_should_apply_result",
    "geak_rebench_tracks_pending_task",
    "geak_revalidate_idempotency_key",
    "geak_revalidation_placeholder_keys",
    "is_geak_same_harness_rebench_task",
    "resolve_geak_revalidate_idempotency_key",
    "settle_dangling_geak_pending",
    "spare_geak_rebench_on_phase_transition",
]
