# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The order the dispatcher visits a phase's queued tasks in: the kinds the phase dispatches first, then the rest."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .machine_state import PHASE_ENABLEMENT, PHASE_FRAMEWORK_AGENT

if TYPE_CHECKING:
    from ..state.task_registry import Task

__all__ = ["PHASE_DISPATCH_FIRST", "in_dispatch_order"]

# In FRAMEWORK_AGENT a reviewed patch and a configuration grid hold the same serving lanes for a whole round: a grid
# can be proposed again, but a patch still queued when the phase exits is cancelled unmeasured.
# ENABLEMENT holds its exit for the initial roofline its promoted baseline enqueued, so a roofline queued behind
# integrates would keep the phase open for as long as they take.
PHASE_DISPATCH_FIRST: dict[str, frozenset[str]] = {
    PHASE_ENABLEMENT: frozenset({"roofline"}),
    PHASE_FRAMEWORK_AGENT: frozenset({"integrate_patch"}),
}


def in_dispatch_order(queued: list[Task], phase: str) -> list[Task]:
    """Return ``queued`` with the kinds ``phase`` dispatches first moved ahead, each group keeping its order."""
    first = PHASE_DISPATCH_FIRST.get(phase, frozenset())
    return sorted(queued, key=lambda task: task.kind not in first)
