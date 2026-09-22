# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SubAgentRunner

Runs one already-queued ``tasks`` row under the lease its caller won: replays
PolicyGate at dispatch, invokes the deterministic Python ``ActionRunner``
executor registered for ``task.kind`` in ``self.executor_registry``, and
transitions the row to its terminal state.
"""

from __future__ import annotations

import asyncio
import json
import sys
from concurrent.futures import CancelledError as FuturesCancelledError
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import logging

from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import _RUNS_ACTIONS, runs_dir
from ..actions.cancel_channel import current_cancel_scope
from ..bus.resource_lock import (
    CLEANUP_CONFIRMED_KEY,
    CLEANUP_TREE_PGID_KEY,
    Lease,
    ResourceLockManager,
)
from ..policy.gate import PolicyDenied
from ..state.task_registry import IllegalTransition, Task, TaskRegistry
from ..trace.task_progress import ProgressReporter, progress_scope

if TYPE_CHECKING:
    from ..policy.gate import PolicyGate

log = logging.getLogger(__name__)


# Every ``tasks`` row is dispatched and awaited by the Coordinator's
# orchestration loop, and the table carries no requester column, so a heartbeat
# can only ever attest to this one agent. Widening that — letting any running
# task vouch for whoever happens to be quiet — is what allowed a single busy
# task to silence stall detection for the whole session.
# Only work that becomes a row is covered here; inline kernel requests are kept
# visible by a bus heartbeat around the handler instead.
PROGRESS_OWNER_AGENT = "orchestration"


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


# Runner signature: async fn(ctx) -> result_payload (dict).
ExecutorFn = Callable[[RunnerContext], Awaitable[dict]]


@dataclass
class SubAgentResult:
    """Outcome of a single :meth:`SubAgentRunner.run_task` call.

    Attributes:
        task_id (str): Id of the task that ran.
        state (str): Terminal state — ``"succeeded"`` / ``"failed"`` / ``"cancelled"``.
        result (dict): Executor result payload (empty on failure).
        error (str | None): Error string when the task failed, else None.
        error_class (str): Machine-readable failure category. Not a closed
            enum — each producer mints its own values, so a new prefix
            family here is discoverable only via its consumers, not via a
            central registry:

            * ``"policy_{rule}"`` (e.g.
              ``"policy_path_outside_session_dir"``): a
              ``PolicyDenied`` dispatch rejection, keyed on
              :attr:`PolicyDenied.rule <..policy.gate.PolicyDenied.rule>`.
              Falls through any exact-match bucket below by design — a
              policy denial isn't a runtime crash/oom/hang, so
              :meth:`writeback._pitfall_severity_for` correctly excludes it
              from ``SEVERITY_CRASH``. Still lands in the gap ledger as its
              own ``(action, error_class)`` key
              (:meth:`explore._extract_gaps_from_attempts`), which is enough
              to group repeat denials without a dedicated bucket.
            * ``"crash"`` / ``"oom"`` / ``"hang"`` / ``"detokenizer_stall"``:
              exact-matched by :meth:`writeback._pitfall_severity_for` to
              classify a failure as crash-severity for the KB.
            * ``"no_executor"``: no runner registered for the task's
              ``kind`` — set directly on this dataclass, same site as
              ``policy_{rule}``, so this exit no longer collapses into
              ``"unknown_error"`` either.
            * The raised exception's ``__class__.__name__`` (e.g.
              ``"TimeoutError"``): an executor raised instead of returning a
              result. Same reasoning — a real class beats the generic
              bucket, even though the exact name isn't enumerable up front.
            * Anything else (including empty): executors set their own
              ``error_class`` inside ``result``, or leave it unset, in which
              case the gap ledger buckets it as ``"unknown_error"``.
    """

    task_id: str
    state: str  # "succeeded" / "failed" / "cancelled"
    result: dict
    error: str | None = None
    error_class: str = ""


class ExecutionCleanupUnconfirmed(RuntimeError):
    """Physical cleanup did not acknowledge release of an execution's resources.
    Recorded on the terminal row as a LEAD FOR AN OPERATOR, not as something a
    reaper acts on. Nothing probes it: a served process is setsid'd by design,
    so it leaves the group its spawn created, and three attempts to prove a lane
    free from identities like this one were refuted in review. What it is still
    good for is telling a human where to start looking when a lane is reported
    held -- see :func:`~hyperloom.orchestrator.bus.resource_lock._report_unverifiable`,
    which prints it with exactly that caveat.

    Attributes:
        result: The executor's own outcome, for the terminal row.
        tree_pgid: Process group whose teardown went unconfirmed, or None when
            the raise site has no local group to name -- a cleanup that failed
            before any process existed, or a Ray actor whose ids belong to
            another node.
    """

    def __init__(
        self,
        message: str,
        *,
        result: SubAgentResult | None = None,
        tree_pgid: int | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.tree_pgid = tree_pgid


class SubAgentRunner:
    """Routes ``delegate`` work to the matching ActionRunner, and releases its lease.

    ``executor_registry`` maps ``task.kind`` → :data:`ExecutorFn` (tests
    register stubs; production registers shell-wrapping executors).
    """

    def __init__(
        self,
        locks: ResourceLockManager,
        tasks: TaskRegistry,
        *,
        executor_registry: dict[str, ExecutorFn] | None = None,
        session_dir: Path | None = None,
        shared_state: object | None = None,
        policy: PolicyGate | None = None,
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
            shared_state (object | None): Live session state forwarded to
                executors via ``ctx.extra``.
            policy (PolicyGate | None): Optional gate replayed at dispatch
                time for defense against forged queued task rows.
        """
        self.locks = locks
        self.tasks = tasks
        self.executor_registry: dict[str, ExecutorFn] = dict(executor_registry or {})
        self.session_dir = Path(session_dir) if session_dir else None
        # Live SharedState threaded into each executor's ctx.extra so gain-computing
        # executors can recover a baseline anchor when params['base_tput'] is absent.
        self.shared_state = shared_state
        self.policy = policy

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

        Args:
            task: The task whose workspace directory should be pre-created.

        Returns:
            The created workspace path, or ``None`` when there is no session
            dir or the task kind is not a known runs/ action.
        """
        if self.session_dir is None:
            return None
        kind = str(task.kind or "").strip()
        if kind not in _RUNS_ACTIONS:
            return None
        ws = runs_dir(self.session_dir, kind, task.task_id)
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    async def _write_terminal(
        self,
        task_id: str,
        new_state: str,
        *,
        evidence: dict | None = None,
        context: str,
    ) -> None:
        """Record a task's terminal state, tolerating a row already terminal.

        The one place this runner leaves a task. A watchdog can reclaim the row
        first, and losing that race means the outcome is already recorded. The
        ``queued -> running`` claim does not come through here: its rejection is
        the double-spawn guard and has to reach the caller.

        Args:
            task_id: The task to transition.
            new_state: The terminal state to record.
            evidence: Optional evidence dict recorded with the transition.
            context: Short label describing the call site.
        """
        try:
            await self.tasks.transition(task_id, new_state, evidence=evidence or {})
        except IllegalTransition:
            # An outcome is durable evidence, not a prunable progress heartbeat.
            async with self.tasks.db.transaction() as cur:
                cur.execute("SELECT history FROM tasks WHERE task_id=?", (task_id,))
                history = json.loads(cur.fetchone()["history"])
                history.append({"ts": now_iso(), "evidence": evidence or {}})
                cur.execute("UPDATE tasks SET history=? WHERE task_id=?", (json.dumps(history), task_id))
            log.warning(
                "sub_agent_runner: task_id=%s already terminal before "
                "transition→%s (context=%s); keeping the executor result",
                task_id,
                new_state,
                context,
            )

    async def run_task(
        self,
        task: Task,
        *,
        prebound_lease: Lease | None = None,
        extra_context: dict | None = None,
        release_resources: Callable[[], Awaitable[bool]] | None = None,
    ) -> SubAgentResult:
        """Claim the row, execute it, record the outcome.

        Normal completion records a terminal state and releases resources.
        A hard-cancelled await is not proof its worker ended, so that path keeps
        the running row and ownership for explicit cleanup.

        Args:
            task: The task to execute.
            prebound_lease: The lease the caller won for this task's lanes;
                released here. ``None`` only for a task that needs no lanes.
            extra_context: Optional extra values merged into the
                :class:`RunnerContext`.
            release_resources: Confirm worker cleanup before releasing lanes.

        Returns:
            The :class:`SubAgentResult` capturing terminal state and payload.
        """
        runner = self.executor_registry.get(task.kind)
        lease: Lease | None = prebound_lease
        outcome: SubAgentResult | None = None
        terminal_state: str | None = None
        evidence: dict[str, Any] = {}
        context = "executor_success"
        try:
            if task.kind == "recover" and runner is None and task.state == "queued":
                evidence = {"reason": "unsupported_action"}
                outcome = SubAgentResult(
                    task_id=task.task_id,
                    state="cancelled",
                    result={"status": "cancelled", "error_class": "unsupported_action"},
                    error="recover is no longer supported",
                    error_class="unsupported_action",
                )
                return outcome
            if self.policy is not None:
                try:
                    self.policy.validate_dispatched_task(
                        task.kind,
                        dict(task.params or {}),
                        task_id=task.task_id,
                    )
                except PolicyDenied as denied:
                    terminal_state = "cancelled"
                    context = "dispatch_policy_denied"
                    evidence = {"reason": "policy_denied", "rule": denied.rule, "error": str(denied)}
                    rule = denied.rule or "denied"
                    outcome = SubAgentResult(
                        task_id=task.task_id,
                        state="failed",
                        result={},
                        error=str(denied),
                        error_class=f"policy_{rule}",
                    )
                    return outcome

            # Running an action whose lanes nobody holds would run it
            # unserialised, so a missing lease is a caller bug, not a fallback.
            if lease is None and task.requires_lanes:
                raise ValueError(
                    f"task {task.task_id} ({task.kind}) requires lanes "
                    f"{list(task.requires_lanes)} but was dispatched without a lease"
                )

            # Rejection here is the double-spawn guard; it belongs to the caller.
            await self.tasks.transition(task.task_id, "running")

            if runner is None:
                context = "no_executor"
                evidence = {"reason": "no_executor", "kind": task.kind}
                outcome = SubAgentResult(
                    task_id=task.task_id,
                    state="failed",
                    result={},
                    error=f"no runner registered for kind={task.kind!r}",
                    error_class="no_executor",
                )
                return outcome

            # Workspace prep is inside the terminal-writing block: an ENOSPC
            # there is a task that failed, not a task still running.
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
                with progress_scope(self._progress_reporter(task.task_id)):
                    result_payload = await runner(ctx)
            except asyncio.CancelledError:
                log.warning("sub_agent_runner: task=%s cleanup unconfirmed; retaining ownership", task.task_id)
                raise
            except ExecutionCleanupUnconfirmed as exc:
                outcome = exc.result
                raise
            except FuturesCancelledError as exc:
                context = "executor_cancelled"
                evidence = {"reason": str(exc)}
                outcome = SubAgentResult(
                    task_id=task.task_id,
                    state="cancelled",
                    result={"status": "cancelled", "reason": str(exc)},
                    error=str(exc),
                    error_class="cancelled",
                )
                return outcome
            except Exception as exc:  # noqa: BLE001 — surface to task.history
                context = "executor_exception"
                evidence = {"error": repr(exc)}
                outcome = SubAgentResult(
                    task_id=task.task_id,
                    state="failed",
                    result={},
                    error=repr(exc),
                    error_class=exc.__class__.__name__,
                )
                return outcome
            evidence = {"result_keys": sorted(result_payload.keys())}
            outcome = SubAgentResult(task_id=task.task_id, state="succeeded", result=result_payload)
            return outcome
        finally:
            execution_error = sys.exc_info()[1]
            # Cancellation of an await does not establish that its worker stopped.
            if isinstance(execution_error, asyncio.CancelledError):
                scope = current_cancel_scope()
                if scope is not None:
                    scope.cancel(reason="execution_cancelled")
            else:
                cleanup_error = execution_error if isinstance(execution_error, ExecutionCleanupUnconfirmed) else None
                released = False
                try:
                    if cleanup_error is None:
                        try:
                            released = release_resources is None or await release_resources() is True
                            if released and lease is not None:
                                await self.locks.release(lease)
                        finally:
                            if sys.exc_info()[1] is not execution_error:
                                cleanup_error = sys.exc_info()[1]
                finally:
                    cleanup_confirmed = released and cleanup_error is None
                    if outcome is not None:
                        evidence.update({"outcome": asdict(outcome), CLEANUP_CONFIRMED_KEY: cleanup_confirmed})
                        if not cleanup_confirmed:
                            evidence["cleanup_error"] = (
                                repr(cleanup_error) if cleanup_error else "physical cleanup unconfirmed"
                            )
                            # The lane this path deliberately retains is not
                            # taken back by anything: no reaper inspects this
                            # number, because a served process setsid's out of
                            # the group it names. It is recorded so the operator
                            # who has to clear that lane by hand has somewhere
                            # to start -- as a number, rather than left to be
                            # dug back out of the repr above.
                            pgid = getattr(cleanup_error, "tree_pgid", None)
                            if isinstance(pgid, int) and not isinstance(pgid, bool) and pgid > 0:
                                evidence[CLEANUP_TREE_PGID_KEY] = pgid
                        await self._write_terminal(
                            task.task_id,
                            terminal_state or outcome.state,
                            evidence=evidence,
                            context=context,
                        )
                    if isinstance(cleanup_error, asyncio.CancelledError):
                        scope = current_cancel_scope()
                        if scope is not None:
                            scope.cancel(reason="execution_cleanup_cancelled")
                        log.warning(
                            "sub_agent_runner: task=%s cleanup cancelled; retaining unconfirmed capacity", task.task_id
                        )
                    elif isinstance(cleanup_error, ExecutionCleanupUnconfirmed):
                        cleanup_error.result = outcome
                    elif not cleanup_confirmed:
                        raise ExecutionCleanupUnconfirmed(
                            f"task={task.task_id}: physical cleanup unconfirmed", result=outcome
                        ) from cleanup_error

    def _progress_reporter(self, task_id: str) -> ProgressReporter:
        """Build the ambient progress sink for one task's executor.

        Composite actions call :func:`~...trace.task_progress.report_progress`
        as each internal unit lands, which reaches this sink and lands on the
        task row as a heartbeat instead of the row going dark for the whole
        run. Each note is stamped with :data:`PROGRESS_OWNER_AGENT` so a
        consumer can tell whose silence the heartbeat actually excuses.

        Args:
            task_id (str): Task the returned sink reports for.

        Returns:
            ProgressReporter: ``async (**note) -> None``.
        """

        async def sink(**note: Any) -> None:
            note.setdefault("agent", PROGRESS_OWNER_AGENT)
            log.info("task_progress: task=%s %s", task_id, _format_progress(note))
            await self.tasks.record_progress(task_id, note)

        return sink


def _format_progress(note: dict[str, Any]) -> str:
    """Render a progress note as one greppable ``key=value`` log line.

    Args:
        note (dict[str, Any]): The reported note; ``index``/``total`` collapse
            into a single ``3/12`` counter and empty values are dropped.

    Returns:
        str: Space-separated ``key=value`` pairs.
    """
    index, total = note.get("index"), note.get("total")
    parts = [f"{k}={v}" for k, v in note.items() if k not in ("index", "total") and v not in (None, "")]
    if index is not None:
        parts.append(f"progress={index}/{total}" if total is not None else f"progress={index}")
    return " ".join(parts)


__all__ = [
    "PROGRESS_OWNER_AGENT",
    "RunnerContext",
    "ExecutorFn",
    "SubAgentResult",
    "SubAgentRunner",
]
