# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Settling the tasks reclaimed from a dead lease holder."""

from __future__ import annotations

import logging

from ..specialists.failure_class import DEAD_HOLDER_ERROR_CLASS
from ..state.task_registry import TaskNotFound
from .sub_agent_runner import SubAgentResult

log = logging.getLogger(__name__)


class DeadHolderAccountingMixin:
    """Accounting for the tasks whose holder process died before reporting a result."""

    async def _account_dead_holder_failures(
        self,
        task_ids: list[str],
        *,
        reason: str,
    ) -> None:
        """Retry or charge a failure for each task reclaimed from a dead lease holder.

        A holder killed from outside this process (its lease reaped by
        ``reap_dead_holders``) never returns a result, so without this the
        per-action streak counters never see the death and the streak-based
        auto-terminate stays blind to the whole failure mode. A gap's
        specialist first takes the bounded infra retry a crashed run gets.

        Args:
            task_ids: Task ids just transitioned ``running -> failed`` by the
                dead-holder reclaim.
            reason: Reclaim reason label, echoed into the failure result.
        """
        for task_id in task_ids:
            if task_id in self._dead_holder_accounted:
                continue
            self._dead_holder_accounted.add(task_id)
            try:
                task = await self.tasks.get(task_id)
            except TaskNotFound:
                continue
            error = f"lease holder process died before reporting a result; task reclaimed by {reason}"
            died = SubAgentResult(task_id, "failed", {}, error=error, error_class=DEAD_HOLDER_ERROR_CLASS)
            settles_specialist = task.kind == "specialist" and not self.owes_result(task_id)
            # A gapless discovery round settles instead: the FRAMEWORK pump keys its next round on that settle.
            if (
                settles_specialist
                and task.params.get("gap_canonical_id")
                and await self._coord.specialist_dispatch.maybe_auto_retry_specialist(task, died)
            ):
                continue
            await self._coord.writeback.handle_unpromotable_result(
                task,
                {"status": "failed", "error_class": DEAD_HOLDER_ERROR_CLASS, "error": error},
            )
            if settles_specialist:
                await self._record_specialist_settlement(task, died)
            log.warning(
                "dispatcher: counted dead-holder death of task=%s kind=%s as an action failure",
                task_id,
                task.kind,
            )
