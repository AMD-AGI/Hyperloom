# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The order the dispatcher visits queued tasks in: operator work, then the kinds the phase dispatches first, then the rest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..specialists.domains import CUSTOM_SPECIALIST_KEY
from .machine_state import PHASE_ENABLEMENT, PHASE_FRAMEWORK_AGENT

if TYPE_CHECKING:
    from ..state.task_registry import Task

__all__ = ["LEVER_BUNDLE_REASON", "PHASE_DISPATCH_FIRST", "in_dispatch_order", "is_operator_work"]

#: The ``reason`` of the explore that benches every dormant rewrite lever together. It goes ahead of queued integrates:
#: its patches are already written and in the tree, each missed the KEEP threshold alone, and benched together is the
#: only way they clear it. Behind the integrates it never ran -- they filled the phase first.
LEVER_BUNDLE_REASON: str = "dormant_lever_bundle"

# In FRAMEWORK_AGENT a reviewed patch and a configuration grid hold the same serving lanes for a whole round: a grid
# can be proposed again, but a patch still queued when the phase exits is cancelled unmeasured.
# ENABLEMENT holds its exit for the initial roofline its promoted baseline enqueued, so a roofline queued behind
# integrates would keep the phase open for as long as they take.
PHASE_DISPATCH_FIRST: dict[str, frozenset[str]] = {
    PHASE_ENABLEMENT: frozenset({"roofline"}),
    PHASE_FRAMEWORK_AGENT: frozenset({"integrate_patch"}),
}


def is_operator_work(kind: str, params: dict[str, Any] | None) -> bool:
    """Whether a task is the operator-defined specialist or the bench of its proposals."""
    params = params or {}
    if kind == "specialist":
        return str(params.get("domain") or "") == CUSTOM_SPECIALIST_KEY
    return kind == "explore" and bool(params.get("operator_priority"))


def _is_lever_bundle(kind: str, params: dict[str, Any] | None) -> bool:
    """Whether a task is the round benching every dormant rewrite lever together."""
    return kind == "explore" and str((params or {}).get("reason") or "") == LEVER_BUNDLE_REASON


def _is_cycle_batch(kind: str, params: dict[str, Any] | None) -> bool:
    """Whether a task is a cycle-bench batch; the integrates queued behind it are admitted dormant only once it ran."""
    from ..actions.executors.cycle_bench_arm import CYCLE_ARMS_PARAM

    return kind == "explore" and CYCLE_ARMS_PARAM in (params or {})


def _is_host_patch(params: dict[str, Any] | None) -> bool:
    """Whether a patch touches a host path the cycle bench cannot see, so its replay round is its only measurement."""
    from ..actions.executors._integrate_vocabulary import SCREEN_HOST, SCREEN_PARAM

    return (params or {}).get(SCREEN_PARAM) == SCREEN_HOST


def in_dispatch_order(queued: list[Task], phase: str) -> list[Task]:
    """Return ``queued`` grouped as operator work, the dormant-lever bundle, cycle batches and host-side patches, then
    ``phase``'s dispatch-first kinds, then the rest, each in order."""
    first = PHASE_DISPATCH_FIRST.get(phase, frozenset())

    def tier(task: Task) -> int:
        if (
            is_operator_work(task.kind, task.params)
            or _is_lever_bundle(task.kind, task.params)
            or _is_cycle_batch(task.kind, task.params)
        ):
            return 0
        if task.kind in first:
            return 0 if _is_host_patch(task.params) else 1
        return 2

    return sorted(queued, key=tier)
