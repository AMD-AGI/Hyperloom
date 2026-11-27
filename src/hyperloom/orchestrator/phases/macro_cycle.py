# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle engine: soft-restart and the cycle-start reprofile."""

from __future__ import annotations

import logging as _logging
from typing import Any

from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state
from ..loop.maintenance import report_lease_cleanup

log = _logging.getLogger(__name__)

__all__ = ["MacroCycleCollaborator"]


class MacroCycleCollaborator(CoordinatorCollaborator):
    """Soft-restart and reprofile at a macro-cycle boundary."""

    async def run_cycle_soft_restart(
        self,
        *,
        prior_cycle: int,
        new_cycle: int,
    ) -> None:
        """Medium-intensity soft restart at a macro-cycle boundary.

        Recycles transient/per-cycle resources (fresh leases, cleared
        caches, re-scoped system prompt) without losing accumulated optimization state;
        ``current_best`` / ``optimization_stack`` / ``explore_search`` are
        preserved. Idempotent.

        Args:
            prior_cycle: The macro-cycle number that just finished.
            new_cycle: The macro-cycle number being entered.
        """
        summary: dict[str, Any] = {
            "prior_cycle": int(prior_cycle),
            "new_cycle": int(new_cycle),
        }
        await report_lease_cleanup(self._coord, summary, reason="cycle_soft_restart")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "cycle_soft_restart", **summary},
        )

    async def on_cycle_start_reprofile(self, *, from_phase: str) -> None:
        """Force a fresh analysis at the start of a reopened macro-cycle.

        Reached on every cycle start, whichever phase the reloop targets, so a
        new cycle never re-targets the bottleneck the *previous* cycle measured.

        Args:
            from_phase: The phase being left; only a CYCLE_END origin starts a cycle.
        """
        if (from_phase or "").upper() == _phase_state.PHASE_CYCLE_END and int(self.shared_state.macro_cycle or 0) > 0:
            task = await self._coord.phase_prelude.enqueue_internal_analysis_task(
                reason="cycle_start",
            )
            if task is None:
                return
            self.shared_state.auto_roofline_pending_task_id = task.task_id
            log.info(
                "cycle %d start: forced reprofile task=%s",
                int(self.shared_state.macro_cycle or 0),
                task.task_id,
            )
