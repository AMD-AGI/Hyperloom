# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reclaiming the running tasks nothing will settle, and settling them as failed."""

from __future__ import annotations

import logging
import time

from hyperloom.inference_optimizer.session.session_paths import runs_dir

from ..specialists.failure_class import DEAD_HOLDER_ERROR_CLASS
from ..specialists.reap_watch import went_quiet
from ..state.task_registry import Task, TaskNotFound
from .sub_agent_runner import SubAgentResult

log = logging.getLogger(__name__)

_SPECIALIST_KIND = "specialist"


class DeadHolderAccountingMixin:
    """Accounting for the tasks whose holder ended before reporting a result."""

    async def _reclaim_ended_running(self) -> list[str]:
        """Fail the running tasks nothing will settle, and return their ids.

        A holder in another process is judged by its pid (:meth:`TaskRegistry.reclaim_dead_running`). A holder that
        is this process cannot be, so an execution of its own is judged by the lease it was granted and by its
        executor: one whose lease ran out unrenewed while its executor went quiet is failed and its lanes freed
        (:meth:`TaskRegistry.reclaim_lapsed_running`), and its cancel scope is cancelled for any part still polling it.
        """
        dead = await self.tasks.reclaim_dead_running(reason="dead_holder_pump")
        lapsed = await self.tasks.reclaim_lapsed_running(
            quiet=self._executor_went_quiet, now_unix=time.time(), reason="lapsed_lease_pump"
        )
        for task_id in lapsed:
            if entry := self._inflight_actions.get(task_id):
                entry.scope.cancel(reason="lease_lapsed")
        if dead:
            log.warning(
                "dispatcher: reclaimed %d running task(s) with dead holders: %s",
                len(dead),
                ", ".join(t[:12] for t in dead),
            )
        if lapsed:
            log.warning(
                "dispatcher: reclaimed %d running task(s) whose lease ran out while their executor went quiet: %s; "
                "lanes released, each booked failed, a late result is not counted",
                len(lapsed),
                ", ".join(t[:12] for t in lapsed),
            )
        return [*dead, *lapsed]

    def _executor_went_quiet(self, task_id: str, kind: str) -> bool:
        """Whether an execution's executor has stopped showing life; only a specialist's (its CLI's output) is read,
        so a task of any other kind never counts as quiet."""
        workspace = runs_dir(self.session_dir, _SPECIALIST_KIND, task_id)
        return kind == _SPECIALIST_KIND and went_quiet(workspace, now_unix=time.time())

    async def _book_completion(self, task: Task, result: SubAgentResult) -> None:
        """Book a finished execution, unless its task was already reclaimed and charged as failed."""
        if task.task_id in self._dead_holder_accounted:
            log.warning(
                "dispatcher: dropping the late %s result of task=%s, already reclaimed and booked failed",
                result.state,
                task.task_id,
            )
            return
        await self.reap_dispatched_task(task, result)

    async def _account_dead_holder_failures(
        self,
        task_ids: list[str],
        *,
        reason: str,
    ) -> None:
        """Retry or charge a failure for each task reclaimed by :meth:`_reclaim_ended_running` or the reconciler.

        A reclaimed task never returns a result to book -- one that arrives late is dropped by
        :meth:`_book_completion` -- so without this the per-action streak counters never see the death and the
        streak-based auto-terminate stays blind to the whole failure mode. A gap's specialist first takes the bounded
        infra retry a crashed run gets.

        Args:
            task_ids: Task ids just transitioned ``running -> failed`` by a reclaim.
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
            settles_specialist = task.kind == _SPECIALIST_KIND
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
