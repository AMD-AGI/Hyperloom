"""SubAgentRunner

Receives ``delegate{action_name, params}`` intents (after PolicyGate),
materializes them into ``tasks`` rows, and dispatches the work.

v0.6 §15.2 distinguishes two sub-agent forms:

* **ActionRunner** (Python class, no LLM) — fast, deterministic shell
  wrappers (``BaselineExecutor`` / ``BenchRunnerExecutor`` / ...). Looked
  up via ``EXECUTOR_REGISTRY[task.kind]``.
* **OOB sub-agent** (LLM) — fallback path: spawn a fresh ``backend.run()``.

P0-3 ships only the routing skeleton: enqueue + run hook + a stub
"echo" runner that tests can plug in. Real shell-out executors land
in P0-6+.
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
    task: Task
    lease: Lease | None
    extra: dict = field(default_factory=dict)


# Runner signature: async fn(ctx) -> result_payload (dict)
ExecutorFn = Callable[[RunnerContext], Awaitable[dict]]


@dataclass
class SubAgentResult:
    task_id: str
    state: str   # "succeeded" / "failed" / "needs_manual_review"
    result: dict
    error: str | None = None


class SubAgentRunner:
    """Routes ``delegate`` work to the matching ActionRunner + holds leases.

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
        session_dir: Path | None = None,
    ):
        self.locks = locks
        self.tasks = tasks
        self.executor_registry: dict[str, ExecutorFn] = dict(executor_registry or {})
        self.session_dir = Path(session_dir) if session_dir else None

    def register_executor(self, kind: str, fn: ExecutorFn) -> None:
        self.executor_registry[kind] = fn

    def _pre_mkdir_workspace(self, task: Task) -> Path | None:
        """Pre-create ``runs/<action>/<task_id>/`` for actions that have one.

        Returns the path so the caller can stash it on ``RunnerContext.extra``.
        Returns None when the task kind is not one of the known runs/
        actions (e.g. kernel-owned actions which use their own
        kernel-agent-workspace tree).
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

        Bug-fix (N34, May 2026): empirically the ``tasks`` row for a
        long-running grid task can disappear from the SQLite registry
        between the moment the dispatcher pulls it out of
        ``tasks.queued()`` and the moment the executor's terminal
        transition runs. The transition then raises ``TaskNotFound``,
        the dispatcher's ``except Exception ... continue`` drops the
        executor's result on the floor, and downstream gates never
        fire — the whole optimization loop silently stalls. Treat
        ``TaskNotFound`` on a terminal transition as a warning so the
        rest of the dispatcher pipeline (bus event + promotion to
        SharedState) still runs. Returns ``True`` on a successful
        transition, ``False`` on the swallowed-TaskNotFound branch.
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

        Note: task state machine only allows ``queued → running`` then
        ``running → failed/succeeded/...``, so we always transition to
        ``running`` first — even on the "no runner" failure path —
        otherwise IllegalTransition fires.

        v0.8 M6: when the Coordinator's concurrent
        dispatcher pre-acquires the lease via ``try_acquire_many``
        (non-blocking), it passes the resulting :class:`Lease` via
        ``prebound_lease`` and the runner skips its own acquire step.
        The runner still owns the release in its finally block — so
        the dispatcher doesn't have to thread the release path.
        """
        # queued → running first (state machine constraint). Use the
        # resilient variant so a missing row doesn't kill the runner
        # before the executor has even started -- see
        # _transition_resilient for the rationale (Bug N34 #1/#2).
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
            # Always release whoever acquired the lease — pre-bound or
            # owned. The dispatcher passes the lease in but trusts the
            # runner's finally to release it (Inv-7.3 atomic release).
            if lease is not None:
                await self.locks.release(lease)


__all__ = ["RunnerContext", "ExecutorFn", "SubAgentResult", "SubAgentRunner"]
