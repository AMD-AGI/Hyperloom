"""SubAgentRunner — DESIGN v0.6 §15.

Receives ``delegate{action_name, params}`` intents (after PolicyGate),
materializes them into ``tasks`` rows, and dispatches the work.

v0.6 §15.2 distinguishes two sub-agent forms:

* **ActionExecutor** (Python class, no LLM) — fast, deterministic shell
  wrappers (``BaselineExecutor`` / ``BenchRunnerExecutor`` / ...). Looked
  up via ``EXECUTOR_REGISTRY[task.kind]``.
* **OOB sub-agent** (LLM) — fallback path: spawn a fresh ``backend.run()``.

P0-3 ships only the routing skeleton: enqueue + run hook + a stub
"echo" executor that tests can plug in. Real shell-out executors land
in P0-6+.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .resource_lock import Lease, ResourceLockManager
from .task_registry import Task, TaskRegistry


@dataclass
class ExecutorContext:
    task: Task
    lease: Lease | None
    extra: dict = field(default_factory=dict)


# Executor signature: async fn(ctx) -> result_payload (dict)
ExecutorFn = Callable[[ExecutorContext], Awaitable[dict]]


@dataclass
class SubAgentResult:
    task_id: str
    state: str   # "succeeded" / "failed" / "needs_manual_review"
    result: dict
    error: str | None = None


class SubAgentRunner:
    """Routes ``delegate`` work to the matching ActionExecutor + holds leases.

    ``executor_registry`` maps ``task.kind`` → :data:`ExecutorFn`. Tests
    register stubs here. Production registers shell-wrapping executors
    from ``inference_optimizer.orchestrator.action_executors`` (P0-6).
    """

    def __init__(
        self,
        locks: ResourceLockManager,
        tasks: TaskRegistry,
        *,
        executor_registry: dict[str, ExecutorFn] | None = None,
    ):
        self.locks = locks
        self.tasks = tasks
        self.executor_registry: dict[str, ExecutorFn] = dict(executor_registry or {})

    def register_executor(self, kind: str, fn: ExecutorFn) -> None:
        self.executor_registry[kind] = fn

    async def run_task(self, task: Task) -> SubAgentResult:
        """Acquire required lanes, transition queued→running, execute, transition out.

        Note: task state machine only allows ``queued → running`` then
        ``running → failed/succeeded/...``, so we always transition to
        ``running`` first — even on the "no executor" failure path —
        otherwise IllegalTransition fires.
        """
        # queued → running first (state machine constraint)
        await self.tasks.transition(task.task_id, "running")

        executor = self.executor_registry.get(task.kind)
        if executor is None:
            await self.tasks.transition(
                task.task_id, "failed",
                evidence={"reason": "no_executor", "kind": task.kind},
            )
            return SubAgentResult(
                task_id=task.task_id, state="failed",
                result={}, error=f"no executor registered for kind={task.kind!r}",
            )

        lease: Lease | None = None
        if task.requires_lanes:
            lease = await self.locks.acquire_many(
                list(task.requires_lanes),
                holder_id=task.task_id,
                task_id=task.task_id,
                action=task.kind,
                ttl_sec=task.lease_ttl_sec or 60,
            )
        try:
            ctx = ExecutorContext(task=task, lease=lease)
            try:
                result_payload = await executor(ctx)
            except Exception as exc:  # noqa: BLE001 — surface to task.history
                await self.tasks.transition(
                    task.task_id, "failed", evidence={"error": repr(exc)},
                )
                return SubAgentResult(
                    task_id=task.task_id, state="failed",
                    result={}, error=repr(exc),
                )
            await self.tasks.transition(
                task.task_id, "succeeded", evidence={"result_keys": sorted(result_payload.keys())},
            )
            return SubAgentResult(
                task_id=task.task_id, state="succeeded", result=result_payload,
            )
        finally:
            if lease is not None:
                await self.locks.release(lease)


__all__ = ["ExecutorContext", "ExecutorFn", "SubAgentResult", "SubAgentRunner"]
