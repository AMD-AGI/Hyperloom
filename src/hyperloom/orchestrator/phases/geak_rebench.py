# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK same-harness revalidation task identity and phase-boundary policy."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .machine_state import PHASE_CLOSE

if TYPE_CHECKING:
    from ..state.task_registry import Task, TaskRegistry

LEGACY_GEAK_REVALIDATE_PLACEHOLDER = "geak-revalidate"

INCOMPARABLE_REVALIDATION = "incomparable"

# Verdict annotations are not candidate identity.
_REVALIDATION_ANNOTATION_KEYS: frozenset[str] = frozenset(
    {
        "kernel_event_id",
        "revalidation_status",
        "revalidation_error",
        "revalidation_error_class",
        "revalidation_blocked_overlay",
    }
)

# Failed measurements remain retryable.
_TERMINAL_REVALIDATION_STATUSES: frozenset[str] = frozenset({"no_material", "no_promote"})

# ``geak_pending.status`` values that record a closed verdict.
SETTLED_PENDING_STATUSES: frozenset[str] = frozenset({"rebench_cancelled", "rebench_unavailable"})

# Fresh keys a single macro-cycle may mint.
MAX_REBENCH_ATTEMPTS_PER_CYCLE = 4


def geak_revalidate_idempotency_key(macro_cycle: int, attempt: int = 0) -> str:
    """Return the idempotency key for a GEAK 2b rebench task."""
    base = f"geak-revalidate-c{macro_cycle}"
    return base if attempt <= 0 else f"{base}-r{attempt}"


def geak_revalidation_placeholder_keys(macro_cycle: int) -> frozenset[str]:
    """Placeholder ids written to ``geak_pending`` before the task row exists."""
    return frozenset({LEGACY_GEAK_REVALIDATE_PLACEHOLDER, geak_revalidate_idempotency_key(macro_cycle)})


def geak_harness_replays_workload(state: Any) -> bool:
    """Whether GEAK's own harness can replay this session's workload.

    The canonical AgentX workload it cannot: ``_validate_geak_via_geak_harness``
    refuses before launch, which is what makes a refusal there structural rather
    than a run that might land next time. Persisted mode wins over the ambient
    switch, matching that refusal, so a session recorded as synthetic keeps
    replaying.
    """
    from hyperloom.common.perf_metric import is_agentx_mode
    from ..actions.executors._workload_envs import agentx_enabled

    mode = str(getattr(state, "benchmark_mode", "") or "").strip()
    return not (is_agentx_mode(mode) if mode else agentx_enabled())


def geak_verdict_is_terminal(persisted: Any) -> bool:
    """True when a replay of the same candidate could not change the verdict.

    A refusal counts only with its typed class (:data:`INCOMPARABLE_REVALIDATION`),
    so state written before the class was recorded stays retryable instead of
    being read out of a reason string.
    """
    prev = persisted if isinstance(persisted, dict) else {}
    status = str(prev.get("revalidation_status") or "")
    return status in _TERMINAL_REVALIDATION_STATUSES or (
        status == "fallback_failed" and str(prev.get("revalidation_error_class") or "") == INCOMPARABLE_REVALIDATION
    )


def geak_candidate_is_adjudicated(persisted: Any, recovered: Any, *, harness_can_replay: bool) -> bool:
    """True when ``persisted``'s verdict already settles the ``recovered`` result.

    Guards KERNEL crash-recovery, whose job is to promote a ``result.json`` whose
    handback was lost. Where the harness can replay the workload the verdict
    alone decides, as it always has. Where it cannot, the verdict must also have
    been reached on the result now on disk: identity is the runner's own result
    content minus this module's verdict stamps, so any field a rerun moves reads
    as new evidence. A pre-dispatch overlay refusal also checks whether the
    blocked overlay has become loadable; other artifact contents are not read.

    Args:
        persisted: ``shared_state.geak_result`` — the last adjudicated result.
        recovered: The result parsed from ``result.json``.
        harness_can_replay: See :func:`geak_harness_replays_workload`.

    Returns:
        Whether recovery must leave the recovered result alone.
    """
    prev = persisted if isinstance(persisted, dict) else {}
    if harness_can_replay:
        return str(prev.get("revalidation_status") or "") in _TERMINAL_REVALIDATION_STATUSES
    if not geak_verdict_is_terminal(prev):
        return False
    blocked_overlay = str(prev.get("revalidation_blocked_overlay") or "")
    if blocked_overlay:
        from ..loop.coordinator_helpers import _geak_overlay_is_loadable, _normalize_geak_overlay_dir

        if _geak_overlay_is_loadable(_normalize_geak_overlay_dir(blocked_overlay)):
            return False
    raw = recovered if isinstance(recovered, dict) else {}
    # The phase stamps the runner's exit code onto state with ``setdefault``, so
    # ``returncode`` is an annotation only where the file carries none: it is a
    # property of the process, not of the product. A value GEAK wrote itself
    # stays part of the product and is compared.
    stamped = _REVALIDATION_ANNOTATION_KEYS if "returncode" in raw else _REVALIDATION_ANNOTATION_KEYS | {"returncode"}
    return _geak_candidate_identity(prev, stamped) == _geak_candidate_identity(raw, stamped)


def _geak_candidate_identity(result: Any, stamped: frozenset[str]) -> dict[str, Any]:
    """The GEAK result content that identifies a candidate, ``stamped`` keys aside."""
    payload = result if isinstance(result, dict) else {}
    return {key: value for key, value in payload.items() if key not in stamped}


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
    key = str(task.idempotency_key or "")
    if tracked in {task.task_id, key}:
        return True
    placeholders = geak_revalidation_placeholder_keys(macro_cycle)
    return tracked in placeholders and key in placeholders


def geak_rebench_should_apply_result(state: Any, task: Task, *, macro_cycle: int) -> bool:
    """True when a finished 2b task may mutate ``geak_pending`` / ``geak_result``."""
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
    """Cancel in-flight GEAK 2b rebench tasks."""
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
    """Pick the key for the next 2b rebench in ``macro_cycle``."""
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
    """Settle ``geak_pending`` once no rebench can still land."""
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
    "INCOMPARABLE_REVALIDATION",
    "LEGACY_GEAK_REVALIDATE_PLACEHOLDER",
    "MAX_REBENCH_ATTEMPTS_PER_CYCLE",
    "SETTLED_PENDING_STATUSES",
    "cancel_geak_rebench_tasks",
    "find_inflight_geak_rebench_task",
    "geak_candidate_is_adjudicated",
    "geak_harness_replays_workload",
    "geak_rebench_should_apply_result",
    "geak_rebench_tracks_pending_task",
    "geak_revalidate_idempotency_key",
    "geak_revalidation_placeholder_keys",
    "geak_verdict_is_terminal",
    "is_geak_same_harness_rebench_task",
    "resolve_geak_revalidate_idempotency_key",
    "settle_dangling_geak_pending",
    "spare_geak_rebench_on_phase_transition",
]
