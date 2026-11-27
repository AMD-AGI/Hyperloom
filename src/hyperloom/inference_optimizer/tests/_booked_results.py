# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Put a finished task's result where the session keeps it: the outcome on the task row, the announcement on the bus."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
from hyperloom.orchestrator.state.task_registry import Task, TaskOutcome, TaskSpec

if TYPE_CHECKING:
    from hyperloom.orchestrator.loop.coordinator import Coordinator


def announced(event: dict[str, Any], result: dict[str, Any], *, error: str | None = None, seq: int = 7):
    """A ``delegated_result`` message and the outcomes its renderer joins: ``result`` and ``error`` on the task's row."""
    message = Message.new("coordinator", "*", "delegated_result", event)
    message.seq = seq
    return message, {event["task_id"]: TaskOutcome(result=result, error=error, cleanup_confirmed=True)}


async def record_outcome(
    coord: "Coordinator", task_id: str, result: dict[str, Any], *, state: str = "succeeded", error: str | None = None
) -> None:
    """End a queued or running task the way its runner does: the outcome and a confirmed cleanup on the terminal row."""
    row = await coord.tasks.get(task_id)
    if row.state == "queued":
        await coord.tasks.transition(task_id, "running")
    outcome = SubAgentResult(task_id=task_id, state=state, result=result, error=error)
    await coord.tasks.transition(task_id, state, evidence={"outcome": asdict(outcome), "cleanup_confirmed": True})


async def announce(coord: "Coordinator", task: Task, *, state: str = "succeeded") -> None:
    """Append the event the dispatcher writes when it books ``task``'s result."""
    event = {"task_id": task.task_id, "kind": task.kind, "state": state}
    await coord.bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", event))


async def book_result(
    coord: "Coordinator",
    kind: str,
    result: dict[str, Any],
    *,
    state: str = "succeeded",
    error: str | None = None,
    params: dict[str, Any] | None = None,
) -> Task:
    """Create a task of ``kind``, end it with ``result`` and announce it; returns the terminal row."""
    spec = TaskSpec(kind=kind, params=params or {}, idempotency_key=f"{kind}:{uuid.uuid4().hex}")
    task, _ = await coord.tasks.create_or_return_existing(spec)
    await record_outcome(coord, task.task_id, result, state=state, error=error)
    await announce(coord, task, state=state)
    return await coord.tasks.get(task.task_id)
