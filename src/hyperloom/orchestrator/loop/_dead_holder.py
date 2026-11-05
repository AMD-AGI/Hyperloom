# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Failure accounting for the tasks reclaimed from a dead lease holder."""

from __future__ import annotations

import logging
from typing import Any

from ..state.task_registry import TaskNotFound
from .sub_agent_runner import SubAgentResult

log = logging.getLogger(__name__)


async def account_dead_holder_failures(dispatcher: Any, task_ids: list[str], *, reason: str) -> None:
    """Charge a failure for each task reclaimed from a dead lease holder.

    A holder killed from outside this process (its lease reaped by
    ``reap_dead_holders``) never returns a result, so without this the
    per-action streak counters never see the death and the streak-based
    auto-terminate stays blind to the whole failure mode.

    Args:
        dispatcher: The dispatcher that reclaimed the tasks.
        task_ids: Task ids just transitioned ``running -> failed`` by the
            dead-holder reclaim.
        reason: Reclaim reason label, echoed into the failure result.
    """
    for task_id in task_ids:
        if task_id in dispatcher._dead_holder_accounted:
            continue
        dispatcher._dead_holder_accounted.add(task_id)
        try:
            task = await dispatcher.tasks.get(task_id)
        except TaskNotFound:
            continue
        error = f"lease holder process died before reporting a result; task reclaimed by {reason}"
        settles_specialist = task.kind == "specialist" and not dispatcher.owes_result(task_id)
        died = SubAgentResult(task_id, "failed", {"runner_status": "stale"}, error=error)
        # A gap's specialist is retried within its cap; a gapless discovery round is re-armed by its pool check.
        gap = settles_specialist and (task.params or {}).get("gap_canonical_id")
        if gap and await dispatcher._coord.specialist_dispatch.maybe_auto_retry_specialist(task, died):
            continue
        await dispatcher._coord.writeback.handle_unpromotable_result(
            task,
            {"status": "failed", "error_class": "dead_holder_reaped", "error": error},
        )
        if settles_specialist:
            await dispatcher._record_specialist_settlement(task, died)
        log.warning(
            "dispatcher: counted dead-holder death of task=%s kind=%s as an action failure",
            task_id,
            task.kind,
        )
