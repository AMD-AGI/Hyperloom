# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""SubAgentRunner

Receives ``delegate{action_name, params}`` intents (after PolicyGate),
materializes them into ``tasks`` rows, and dispatches the work.

v0.6 §15.2 distinguishes two sub-agent forms: deterministic Python
``ActionRunner`` executors (looked up via ``EXECUTOR_REGISTRY[task.kind]``)
and the LLM OOB sub-agent fallback (``backend.run()``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import logging

from ..session_paths import _runs_actions, runs_dir
from .resource_lock import Lease, ResourceLockManager
from .task_registry import Task, TaskNotFound, TaskRegistry

log = logging.getLogger(__name__)


@dataclass
class RunnerContext:
    """Per-task context handed to an :data:`ExecutorFn`.

    Attributes:
        task (Task): The task being executed.
        lease (Lease | None): The resource lease held for this task, or
            None when the task requires no lanes.
        extra (dict): Optional extras (e.g. ``workspace`` / ``session_dir``
            paths) the runner stashes for the executor.
    """

    task: Task
    lease: Lease | None
    extra: dict = field(default_factory=dict)


# Runner signature: async fn(ctx) -> result_payload (dict)
ExecutorFn = Callable[[RunnerContext], Awaitable[dict]]


@dataclass
class SubAgentResult:
    """Outcome of a single :meth:`SubAgentRunner.run_task` call.

    Attributes:
        task_id (str): Id of the task that ran.
        state (str): Terminal state — ``"succeeded"`` / ``"failed"`` /
            ``"needs_manual_review"``.
        result (dict): Executor result payload (empty on failure).
        error (str | None): Error string when the task failed, else None.
    """

    task_id: str
    state: str   # "succeeded" / "failed" / "needs_manual_review"
    result: dict
    error: str | None = None


class SubAgentRunner:
    """Routes ``delegate`` work to the matching ActionRunner + holds leases.

    ``executor_registry`` maps ``task.kind`` → :data:`ExecutorFn` (tests
    register stubs; production registers shell-wrapping executors, P0-6).
    """

    def __init__(
        self,
        locks: ResourceLockManager,
        tasks: TaskRegistry,
        *,
        executor_registry: dict[str, ExecutorFn] | None = None,
        session_dir: Path | None = None,
        shared_state: object | None = None,
    ):
        """Initialise the runner with its lock manager + task registry.

        Args:
            locks (ResourceLockManager): Lane lease manager used to gate
                task execution.
            tasks (TaskRegistry): Registry the runner transitions task
                state through.
            executor_registry (dict[str, ExecutorFn] | None): Optional
                initial map of ``task.kind`` to executor function (copied).
            session_dir (Path | None): Session root used to pre-create
                per-action workspaces; None disables workspace pre-mkdir.
        """
        self.locks = locks
        self.tasks = tasks
        self.executor_registry: dict[str, ExecutorFn] = dict(executor_registry or {})
        self.session_dir = Path(session_dir) if session_dir else None
        # Live SharedState threaded into each executor's ctx.extra so
        # gain-computing executors can recover a baseline anchor when
        # params['base_tput'] is absent.
        self.shared_state = shared_state

    def register_executor(self, kind: str, fn: ExecutorFn) -> None:
        """Register (or replace) the executor for a task kind.

        Args:
            kind (str): The ``task.kind`` this executor handles.
            fn (ExecutorFn): Async callable invoked with a
                :class:`RunnerContext`.
        """
        self.executor_registry[kind] = fn

    def _pre_mkdir_workspace(self, task: Task) -> Path | None:
        """Pre-create ``runs/<action>/<task_id>/`` for actions that have one.

        Returns the path (stashed on ``RunnerContext.extra``) or None when
        the task kind is not a known runs/ action.
        """
        if self.session_dir is None:
            return None
        kind = str(task.kind or "").strip()
        if kind not in _runs_actions():
            return None
        ws = runs_dir(self.session_dir, kind, task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    async def _transition_resilient(
        self,
        task_id: str,
        new_state: str,
        *,
        evidence: dict | None = None,
        context: str,
    ) -> bool:
        """Transition a task to ``new_state`` but tolerate ``TaskNotFound``.

        Bug-fix N34: a long-running task's row can vanish before its terminal
        transition; swallowing TaskNotFound keeps the pipeline running.
        Returns True on success, False on the swallowed-TaskNotFound branch.
        """
        try:
            await self.tasks.transition(task_id, new_state, evidence=evidence or {})
            return True
        except TaskNotFound:
            log.warning(
                "sub_agent_runner: tasks row for task_id=%s vanished before "
                "transition→%s (context=%s); continuing so the executor "
                "result is not lost. See sub_agent_runner._transition_"
                "resilient docstring for the disappearing-row hypothesis.",
                task_id, new_state, context,
            )
            return False

    async def run_task(
        self,
        task: Task,
        *,
        prebound_lease: Lease | None = None,
        extra_context: dict | None = None,
    ) -> SubAgentResult:
        """Acquire required lanes, transition queued→running, execute, transition out.

        Always transitions to ``running`` first (state machine constraint).
        With ``prebound_lease`` the runner skips its own acquire but still
        owns the release in its finally block.
        """
        # queued → running first (state machine constraint).
        await self._transition_resilient(
            task.task_id, "running", context="enter_running",
        )

        runner = self.executor_registry.get(task.kind)
        if runner is None:
            await self._transition_resilient(
                task.task_id, "failed",
                evidence={"reason": "no_executor", "kind": task.kind},
                context="no_executor",
            )
            if prebound_lease is not None:
                await self.locks.release(prebound_lease)
            return SubAgentResult(
                task_id=task.task_id, state="failed",
                result={}, error=f"no runner registered for kind={task.kind!r}",
            )

        lease: Lease | None = prebound_lease
        owned_lease = prebound_lease is None
        if owned_lease and task.requires_lanes:
            lease = await self.locks.acquire_many(
                list(task.requires_lanes),
                holder_id=task.task_id,
                task_id=task.task_id,
                action=task.kind,
                ttl_sec=task.lease_ttl_sec or 60,
            )
        try:
            workspace = self._pre_mkdir_workspace(task)
            extra: dict = {}
            if workspace is not None:
                extra["workspace"] = str(workspace)
            if self.session_dir is not None:
                extra["session_dir"] = str(self.session_dir)
            if self.shared_state is not None:
                extra["shared_state"] = self.shared_state
            if extra_context:
                extra.update(dict(extra_context))
            ctx = RunnerContext(task=task, lease=lease, extra=extra)
            try:
                result_payload = await runner(ctx)
            except Exception as exc:  # noqa: BLE001 — surface to task.history
                await self._transition_resilient(
                    task.task_id, "failed",
                    evidence={"error": repr(exc)},
                    context="executor_exception",
                )
                return SubAgentResult(
                    task_id=task.task_id, state="failed",
                    result={}, error=repr(exc),
                )
            await self._transition_resilient(
                task.task_id, "succeeded",
                evidence={"result_keys": sorted(result_payload.keys())},
                context="executor_success",
            )
            return SubAgentResult(
                task_id=task.task_id, state="succeeded", result=result_payload,
            )
        finally:
            # Always release whoever acquired the lease — pre-bound or owned
            # (atomic release).
            if lease is not None:
                await self.locks.release(lease)


__all__ = ["RunnerContext", "ExecutorFn", "SubAgentResult", "SubAgentRunner"]
