# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The order the dispatcher visits queued tasks in: operator work, then the kinds the phase dispatches first, then the rest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..specialists.domains import CUSTOM_SPECIALIST_KEY
from .machine_state import PHASE_FRAMEWORK_AGENT

if TYPE_CHECKING:
    from ..state.task_registry import Task

__all__ = ["PHASE_DISPATCH_FIRST", "in_dispatch_order", "is_operator_work"]

# In FRAMEWORK_AGENT a reviewed patch and a configuration grid hold the same serving lanes for a whole round: a grid
# can be proposed again, but a patch still queued when the phase exits is cancelled unmeasured.
PHASE_DISPATCH_FIRST: dict[str, frozenset[str]] = {
    PHASE_FRAMEWORK_AGENT: frozenset({"integrate_patch"}),
}


def is_operator_work(kind: str, params: dict[str, Any] | None) -> bool:
    """Whether a task is the operator-defined specialist or the bench of its proposals."""
    params = params or {}
    if kind == "specialist":
        return str(params.get("domain") or "") == CUSTOM_SPECIALIST_KEY
    return kind == "explore" and bool(params.get("operator_priority"))


def in_dispatch_order(queued: list[Task], phase: str) -> list[Task]:
    """Return ``queued`` grouped as operator work, then ``phase``'s dispatch-first kinds, then the rest, each in order."""
    first = PHASE_DISPATCH_FIRST.get(phase, frozenset())

    def tier(task: Task) -> int:
        if is_operator_work(task.kind, task.params):
            return 0
        return 1 if task.kind in first else 2

    return sorted(queued, key=tier)
