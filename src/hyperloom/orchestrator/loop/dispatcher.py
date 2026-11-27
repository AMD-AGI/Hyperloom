# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator action dispatch: admission, launch, reaping and cancellation of lane tasks."""

from __future__ import annotations
import asyncio
import collections
import math
import sqlite3
import time
from collections.abc import Awaitable, Callable, Collection
from functools import partial
from typing import Any, NamedTuple
from hyperloom_common.coerce import to_int
from hyperloom.common.deadline import Deadline
from hyperloom_common.env import is_truthy
from hyperloom_common.llm.llm_attribution import current_action_scope
from hyperloom_common.proctree import TERM_GRACE_SEC
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from hyperloom.inference_optimizer.trace.trajectory_trace import trajectory_scope
from ..actions.cancel_channel import CancelScope, use_cancel_scope
from ..actions.executors._subprocess_kill import (
    COOPERATIVE_REAP_BUDGET_SEC,
    STOP_GATE_POLL_SECONDS,
)
from ..phases import dispatch_order
from ..bus.message_bus import Message
from ..policy.gate import SPECIALIST_FROM_AGENT_PREFIX
from ..policy.projection import serving_tp_for_policy
from ..bus.gpu_pool import GPU_LEASE_TTL_GRACE
from ..bus.resource_lock import KNOWN_LANES
from ._dead_holder import DeadHolderAccountingMixin
from .sub_agent_runner import SubAgentResult
from ..specialists.wall_budget import limit_wall_budget_extension
from ..state.task_registry import Task
from .time_budget import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    gated_phase_remaining_sec,
    lease_ttl_sec,
    measured_baseline_runtime_sec,
    phase_deadline,
)

from ..collaborator import CoordinatorCollaborator
import logging as _logging
from ._admission import AdmissionMixin

log = _logging.getLogger(__name__)

# How long a cancel waits for work that is listening on its cancel scope to stop
# itself: stopping the round in flight (:data:`COOPERATIVE_REAP_BUDGET_SEC`) and
# then tearing down the server it left behind (:data:`TERM_GRACE_SEC`). A sequence,
# not alternatives -- one unwind pays both, so a serial step the unwind takes and
# this sum does not name has to be removed from the unwind instead. Falling short
# discards the sentinel the round was about to return for a hard
# ``CancelledError``; the wait ends the moment the last victim is done, so a round
# that stops promptly pays nothing for it.
_COOPERATIVE_CANCEL_GRACE_SEC: float = COOPERATIVE_REAP_BUDGET_SEC + TERM_GRACE_SEC

# How long a cancel waits for anything to start listening before deciding
# nothing will. Work that is already blocking is listening before the cancel is
# raised; the window is for the gap between an action starting and reaching the
# call that watches the scope, so it is the poll that call checks the scope at
# rather than anything about how long the work takes.
_CANCEL_NOTICE_SEC: float = STOP_GATE_POLL_SECONDS


#: Kinds whose execution manages its own completion: the pump does not book
#: their result. Admission is unchanged — same budget, lane and lease gates.
_SELF_SETTLING_KINDS: frozenset[str] = frozenset({"targeted_build", "kernel_agent"})
#: Warm replay settles through its own promote handler even when dispatch failed.
_REPLAY_WARM_RECIPE = "replay_warm_recipe"


class HeldLeases(NamedTuple):
    """Leases a caller already holds for a task, released when the task's run ends.

    ``prebound_lease`` is the lanes; ``gpu_lease`` the SQLite GPU-specialist lease.
    """

    prebound_lease: Any = None
    gpu_lease: Any = None


class _InflightAction(NamedTuple):
    """A running action's handle: what it is, its task, and how to ask it to stop."""

    kind: str
    atask: asyncio.Task[Any]
    scope: CancelScope


class DispatcherCollaborator(AdmissionMixin, DeadHolderAccountingMixin, CoordinatorCollaborator):
    """Handles specialist dispatch: budget gating, action routing, and task enqueue."""

    def __init__(self, coordinator, *, refresh_admission_freeze: Callable[[], Awaitable[None]] | None = None) -> None:
        super().__init__(coordinator)
        # Re-derives :attr:`admission_frozen` once the pump has booked; unwired, the flag is left as set.
        self._refresh_admission_freeze = refresh_admission_freeze
        # Handles on the actions currently running, ``task_id -> _InflightAction``:
        # the only reach into a dispatched action once the pump has returned.
        # Entries remove themselves in :meth:`run_task_registered`.
        self._inflight_actions: dict[str, _InflightAction] = {}
        self._executions: set[asyncio.Task[Any]] = set()
        # The shielded execution behind each dispatched task, ``task_id -> execution``.
        # The registered handle can finish early (a cancelled caller stops waiting)
        # while the execution still writes the terminal row and queues the result;
        # :meth:`owes_result` asks this one. Entries remove themselves when done.
        self._execution_of: dict[str, asyncio.Task[Any]] = {}
        # Poll cadence of the waits on running dispatched work (closing grace, close steps).
        self.poll_sec = 10.0
        # Finished executions awaiting bookkeeping; drained only by the pump so
        # bookkeeping never interleaves with a reactor turn.
        self._completion_queue: collections.deque[tuple[Task, SubAgentResult]] = collections.deque()
        # Set by the phase barrier while a transition is pending: no new spawns.
        self.admission_frozen: bool = False

    async def close_db_after_executions(self) -> None:
        """Drain physical cleanup and completion before the entry-point loop exits.

        An execution that has not settled keeps its database: it may still write
        through it. In particular, cancelling an asyncio task cannot establish
        that its worker thread stopped.
        """
        for entry in self._inflight_actions.values():
            entry.scope.cancel(reason="dispatcher_shutdown")
        executions = set(self._executions)
        pending = {execution for execution in executions if not execution.done()}
        if pending:
            log.warning("dispatcher: draining %d execution cleanup(s) before database close", len(pending))
            _done, pending = await asyncio.wait(pending, timeout=_COOPERATIVE_CANCEL_GRACE_SEC)
        unconfirmed = {
            execution
            for execution in (executions - pending) & self._executions
            if execution.cancelled() or execution.exception() is not None
        }
        if pending or unconfirmed:
            log.error(
                "dispatcher: shutdown cleanup unconfirmed (%d pending, %d interrupted); retaining database and ownership",
                len(pending),
                len(unconfirmed),
            )
            return
        await self._drain_completions()
        self.db.close()

    async def _queue_completion(self, task: Task, result: SubAgentResult) -> None:
        """Queue a finished execution for bookkeeping by the pump."""
        self._completion_queue.append((task, result))

    def has_unbooked_completions(self) -> bool:
        """Whether a finished execution still awaits its bookkeeping."""
        return bool(self._completion_queue)

    def _forget_execution(self, task_id: str) -> Callable[[asyncio.Task[Any]], None]:
        """Done callback dropping ``task_id``'s execution, unless a newer one replaced it."""

        def _forget(execution: asyncio.Task[Any]) -> None:
            if self._execution_of.get(task_id) is execution:
                del self._execution_of[task_id]

        return _forget

    def owes_result(self, task_id: str) -> bool:
        """Whether ``task_id``'s result has yet to be booked by the pump.

        A dispatched task's row goes terminal inside its execution, but its
        result is only routed -- ``delegated_result``, promotion, the
        enablement revalidation verdict -- when a later pump drains the
        completion. Between the two, the row reads as finished while nothing
        has acted on what it returned.

        Args:
            task_id: The task to ask about.

        Returns:
            bool: True while the execution is still running -- however early
            its caller stopped waiting -- or its completion waits in the
            queue. An execution that ended by raising queues nothing, so it
            owes nothing.
        """
        tid = str(task_id or "")
        if not tid:
            return False
        execution = self._execution_of.get(tid)
        if execution is not None and not execution.done():
            return True
        return any(str(getattr(task, "task_id", "") or "") == tid for task, _result in self._completion_queue)

    async def _drain_completions(self) -> int:
        """Book every queued completion in order; returns how many were booked."""
        count = 0
        while self._completion_queue:
            task, result = self._completion_queue.popleft()
            count += 1
            await self.reap_dispatched_task(task, result)
        return count

    def registry_lanes_ttl(self, kind: str) -> tuple[list[str], int]:
        """Resolve an action's lanes, filtered to KNOWN_LANES, and the lease it is granted.

        The lease is the catalogue ``lease_ttl_sec`` floored by the cost this session admits the action at
        (``time_budget.lease_ttl_sec``), so it can exceed the catalogue value once a baseline round is measured.

        Args:
            kind: The action name to resolve.

        Returns:
            A ``(requires_lanes, lease_ttl_sec)`` tuple; ``([], 0)`` for an
            action the catalogue does not know.
        """
        meta = self.action_registry.get(kind)
        if meta is None:
            return [], 0
        lanes = [lane for lane in meta.requires_lanes if lane in KNOWN_LANES]
        return lanes, lease_ttl_sec(meta, measured_baseline_sec=measured_baseline_runtime_sec(self.shared_state))

    def cycle_idem_suffix(self) -> str:
        """Idempotency-key suffix scoping a per-cycle internal singleton to the
        current macro-cycle. Empty for cycle 0 (the first macro-cycle).

        Returns:
            ``"-c<cycle>"`` for macro-cycle > 0, else an empty string.
        """
        cycle = int(self.shared_state.macro_cycle or 0)
        return f"-c{cycle}" if cycle > 0 else ""

    def dispatch_paused_for_phase_budget(self) -> bool:
        """True when the current phase's budget is spent, so the dispatcher should stop launching NEW phase-scoped variants.

        In-flight tasks keep running; the phase advance decides whether they are cancelled. Scoped to the phases whose
        own budget bounds their search. Applies to every session budget: the pause is driven purely by the phase budget
        remaining (charge-back for short bounded runs, the per-cycle window for long/unbounded runs), keeping dispatch
        consistent with the phase-advance gates that consume the same helper.

        Returns:
            ``True`` when new phase-scoped dispatch should pause for budget.
        """
        remaining = gated_phase_remaining_sec(self.shared_state)
        return remaining is not None and remaining <= 0.0

    async def cancel_inflight_actions(
        self,
        *,
        reason: str,
        exempt: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Stop the running dispatched actions and wait for them to unwind.

        The last of the wall-clock defences, and the only one that reaches work
        already under way: admission refuses what cannot fit, the timeout clamp
        bounds what does, and the subprocess reaper stops the child trees that
        were handed a session deadline.

        Cancelling the action's task is not by itself enough to stop it. Every
        benchmark executor spends its time inside ``asyncio.to_thread``, and a
        thread that has started cannot be cancelled: the ``await`` raises here
        while the subprocess runs on, so the lanes and the GPU lease would be
        released, and the database closed, under a benchmark still holding the
        card. So the cancel goes out on the action's :class:`CancelScope` first
        -- the channel the blocking side polls -- and work that is listening on
        it is given :data:`_COOPERATIVE_CANCEL_GRACE_SEC` to stop itself and
        return through its own ``finally`` blocks. Past that bound only the
        caller is cancelled; the shielded execution runs on and keeps its
        capacity until it settles a terminal row of its own.

        Every scope is cancelled before the first await, so a caller that is
        itself being cancelled still leaves no action running unattended: the
        work that can hear the channel stops on it even if this coroutine never
        reaches the wait.

        Args:
            reason: Short cause, logged, used as evidence, and carried to the
                blocking side so it can attribute its own stop.
            exempt: Action kinds to leave running -- the closing actions, when
                the trigger is a budget that already reserved time for them.

        Returns:
            list[str]: Task ids asked to stop; cleanup may still be pending.
        """
        victims = [
            (task_id, entry)
            for task_id, entry in self._inflight_actions.items()
            if entry.kind not in exempt and not entry.atask.done()
        ]
        if not victims:
            return []
        log.warning(
            "dispatcher: cancelling %d in-flight action(s) [%s]: %s",
            len(victims),
            reason,
            ", ".join(f"{entry.kind}/{task_id[:12]}" for task_id, entry in victims),
        )
        for _task_id, entry in victims:
            entry.scope.cancel(reason=reason)
        # A caller that gave up on its action stays registered; when that caller is running
        # this cancel, its action is reached through the scope and the execution drain only.
        current = asyncio.current_task()
        callers = [(task_id, entry) for task_id, entry in victims if entry.atask is not current]
        try:
            await self._wait_for_cooperative_stop(callers)
        finally:
            for _task_id, entry in callers:
                if not entry.atask.done():
                    entry.atask.cancel()
        await asyncio.gather(*(entry.atask for _task_id, entry in callers), return_exceptions=True)
        return [task_id for task_id, _entry in victims]

    async def _wait_for_cooperative_stop(self, victims: list[tuple[str, _InflightAction]]) -> None:
        """Give already-cancelled work the chance to stop itself and return.

        Only work watching its scope can be waited for: waiting on the rest
        would trade a thread that outlives the cancel for a shutdown that blocks
        on one, and the second is the worse failure. So the wait ends at
        whichever comes first -- every victim unwound, the grace spent, or the
        notice window closing with nothing listening.

        Args:
            victims: The ``(task_id, handle)`` pairs whose scopes were just
                cancelled.
        """
        loop = asyncio.get_running_loop()
        notice_deadline = loop.time() + _CANCEL_NOTICE_SEC
        grace_deadline = loop.time() + _COOPERATIVE_CANCEL_GRACE_SEC
        listening = False
        while True:
            alive = [entry.atask for _task_id, entry in victims if not entry.atask.done()]
            if not alive:
                return
            listening = listening or any(entry.scope.has_listeners for _task_id, entry in victims)
            now = loop.time()
            deadline = grace_deadline if listening else notice_deadline
            if now >= deadline:
                return
            await asyncio.wait(alive, timeout=deadline - now, return_when=asyncio.FIRST_COMPLETED)

    async def _cancel_inflight_that_outlived_the_session(self) -> bool:
        """Stop in-flight actions the session can no longer wait for.

        Two causes, both of which mean no result is coming: the process was
        asked to shut down, or the wall-clock budget is spent. The budget case
        spares :data:`TIME_BUDGET_EXEMPT_ACTIONS` because the closing reserve
        this gate trips on exists precisely so those actions can run.

        Returns:
            bool: ``True`` when the pump must not start anything new. Only a
            shutdown says that; a spent budget does not, because the queue scan
            is what cancels the rows it can no longer fit, and the closing
            actions it exempts still have their reserve to run in.
        """
        if self._coord.stop_requested():
            await self.cancel_inflight_actions(reason="shutdown_requested")
            return True
        usable_sec = self.shared_state.session_budget_usable_sec()
        if usable_sec is not None and usable_sec <= 0.0:
            await self.cancel_inflight_actions(
                reason="session_time_exhausted",
                exempt=TIME_BUDGET_EXEMPT_ACTIONS,
            )
        return False

    async def _account_reconciled_failures(self) -> None:
        """Charge, once each, the tasks the reconciler failed because their holder process is gone.

        The reconciler reclaims those rows and sweeps their leases before every tick's pump; what is left to the
        pump is the failure each reclaim implies. The accounting absorbs its store failures, because a pump that
        raised here would stop booking results.
        """
        dead_tasks = self._coord.reconciler.take_failed_tasks()
        if not dead_tasks:
            return
        try:
            await self._account_dead_holder_failures(dead_tasks, reason="dead_holder_pump")
        except (OSError, sqlite3.Error):
            log.exception("dispatcher: dead-holder failure accounting failed")

    async def pump_dispatcher_once(self) -> None:
        """Book finished executions and spawn the queued tasks that now fit; never waits on running work.

        Dispatched tasks outlive the pump and queue their completion for a later
        one. The pump repeats while a pass books something, so lanes freed by
        completions already queued are refilled in the same call. New spawns stop
        while the phase budget is spent or :attr:`admission_frozen` is set, which
        is re-derived after a drain that booked so a hold that booking settled is not read stale.
        """
        await self._account_reconciled_failures()
        while True:
            booked = await self._drain_completions()
            if booked and self._refresh_admission_freeze is not None:
                await self._refresh_admission_freeze()
            shutting_down = await self._cancel_inflight_that_outlived_the_session()
            if not shutting_down and not self.admission_frozen and not self.dispatch_paused_for_phase_budget():
                await self._spawn_fitting_queued()
            if not booked:
                return

    async def wait_for_running_work(self, *, timeout: float) -> None:
        """Wait until a running action finishes or ``timeout`` elapses; returns at once when none runs."""
        running = [entry.atask for entry in self._inflight_actions.values() if not entry.atask.done()]
        if running:
            await asyncio.wait(running, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    async def _admit_lanes(self, task: Task) -> Any:
        """Take the lane lease a queued task needs; ``None`` when it needs none or a lane it needs is at capacity."""
        lanes_needed = task.requires_lanes
        if not lanes_needed:
            return None
        try:
            return await self.locks.acquire_many(
                lanes_needed,
                holder_id=task.task_id,
                task_id=task.task_id,
                action=task.kind,
                ttl_sec=task.lease_ttl_sec or 60,
            )
        except ValueError:
            log.warning(
                "dispatcher: task %s has unknown lane in %r; skipping until resolved",
                task.task_id,
                lanes_needed,
            )
            return None

    async def _admit_gpu_specialist(self, task: Task, lease: Any, extra_context: dict[str, Any]) -> HeldLeases | None:
        """Take the GPU leases a specialist needs, recording the devices it may use in ``extra_context``."""
        from ..specialists.profile import is_authoring_specialist, requires_gpu
        from ..specialists.profile import uses_whole_machine_gpu_lane

        params = task.params
        if not requires_gpu(params):
            return HeldLeases(prebound_lease=lease)
        serving_tp = serving_tp_for_policy(self.shared_state)
        gpu_pool = self.framework_gpu_pool
        if not uses_whole_machine_gpu_lane(params):
            gpu_pool = self.gpu_specialist_pool
            default_gpu_count = serving_tp or 1
        elif is_authoring_specialist(params):
            default_gpu_count = gpu_pool.capacity or 1
        else:
            # Bench specialist: size to the serving TP.
            default_gpu_count = serving_tp or gpu_pool.capacity or 1
        gpu_count = to_int(params.get("gpu_count")) or default_gpu_count
        if is_truthy(params.get("bench")) and serving_tp > 0 and gpu_count < serving_tp:
            log.info(
                "specialist %s: bench=true with gpu_count=%d < serving TP=%d; flooring gpu_count to TP "
                "(a bench specialist starts a real TP-sharded server and cannot run on fewer cards).",
                task.task_id,
                gpu_count,
                serving_tp,
            )
            gpu_count = serving_tp
        gpu_lease = await gpu_pool.try_acquire(
            count=gpu_count,
            holder_id=task.task_id,
            task_id=task.task_id,
        )
        if gpu_lease is None:
            return None
        extra_context["gpu_ids"] = list(gpu_lease.gpu_ids)
        return HeldLeases(lease, gpu_lease)

    async def _spawn_fitting_queued(self) -> None:
        """Spawn every currently lane-fitting queued task not already in flight.

        Pure dispatch — per-task completion bookkeeping is handled by
        :meth:`reap_dispatched_task`. Applies the capacity /
        GPU-specialist-lease gating; each lease is bound to its task_id.
        Tasks already in ``_inflight_actions`` are skipped so a task is never
        dispatched twice across pump calls.
        """
        queued = dispatch_order.in_dispatch_order(await self.tasks.queued(), self.shared_state.phase)
        if not queued:
            return
        for task in queued:
            if task.task_id in self._inflight_actions:
                continue
            if await self._cancel_queued_task_over_budget(task):
                continue
            # No lease means refused, unless the task asked for no lanes in the first place.
            lease = await self._admit_lanes(task)
            if lease is None and task.requires_lanes:
                continue
            extra_context = self._dispatch_context(task)
            held = HeldLeases(prebound_lease=lease)
            if task.kind == "specialist":
                held = await self._admit_gpu_specialist(task, lease, extra_context)
            if held is None:
                if lease is not None:
                    await self.locks.release(lease)
                continue
            cancel_scope = CancelScope()
            atask = asyncio.create_task(
                self.run_task_registered(
                    task,
                    held,
                    extra_context=extra_context,
                    cancel_scope=cancel_scope,
                    on_complete=None if task.kind in _SELF_SETTLING_KINDS else partial(self._queue_completion, task),
                ),
            )
            self._inflight_actions[task.task_id] = _InflightAction(task.kind, atask, cancel_scope)
            atask.add_done_callback(self._report_spawned_failure(task))

    @staticmethod
    def _report_spawned_failure(task: Task) -> "Callable[[asyncio.Task[Any]], None]":
        """Build the done-callback for a handle no caller awaits."""

        def _report(atask: "asyncio.Task[Any]") -> None:
            # A cancelled task has no exception to read, and cancelling is how
            # shutdown reaches it, so it is not a failure worth reporting.
            if atask.cancelled():
                return
            exc = atask.exception()
            if exc is not None:
                log.error(
                    "dispatcher: %s (%s) raised: %r",
                    task.task_id,
                    task.kind,
                    exc,
                    exc_info=exc,
                )

        return _report

    async def _record_dispatch_evidence(
        self, task: Task, lease: Any, release_resources: Callable[[], Awaitable[None]]
    ) -> None:
        """Record a task's dispatch evidence, or give back what it holds and refuse.

        Args:
            task: The task row about to run.
            lease: The lane lease the dispatch holds, if any.
            release_resources: The execution's GPU slot release.

        Raises:
            RuntimeError: The evidence was not durably recorded, in which case
                nothing ran and the row is left queued.
        """
        from hyperloom.orchestrator.state.task_registry import record_task_dispatch

        try:
            record_task_dispatch(task)
        except Exception as exc:
            # Nothing ran under this lease and the row stays queued, so its
            # lanes go back whatever cleanup says: holding them would make
            # the task's own next dispatch find them busy, for good.
            await release_resources()
            if lease is not None:
                await self.locks.release(lease)
            self._inflight_actions.pop(task.task_id, None)
            raise RuntimeError(f"refusing to start task {task.task_id!r}: dispatch evidence write failed") from exc

    async def run_task_registered(
        self,
        task: Task,
        held: HeldLeases = HeldLeases(),
        *,
        extra_context: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        on_complete: Callable[[SubAgentResult], Awaitable[None]] | None = None,
    ) -> "SubAgentResult | None":
        """Run one task under the wall-clock defences, and hand back what it held.

        The only way an action runs. Registering the handle is what lets
        :meth:`cancel_inflight_actions` reach the work at shutdown or on a spent
        budget, and binding the teardown to this coroutine frees the lanes and
        cards on every exit, including one the reap never sees.

        Args:
            task: The task row to run.
            held: Leases the caller already holds and this call releases; when
                no lanes are held and the task needs lanes they are taken
                here, non-blocking.
            extra_context: Per-task extras (wall budget, gpu ids, …).
            cancel_scope: The caller's cancel channel. The pump passes one
                because it registers its handle before this coroutine starts;
                other callers leave it ``None`` and are registered here.
            on_complete: Completion owned by the execution, independent of its
                caller's lifetime. Self-settling tasks pass ``None``.

        Returns:
            The runner's result, or ``None`` when the task's lanes were busy, in
            which case the row is untouched and still queued.
        """
        lease = held.prebound_lease
        if lease is None and task.requires_lanes:
            lease = await self.locks.acquire_many(
                list(task.requires_lanes),
                holder_id=task.task_id,
                task_id=task.task_id,
                action=task.kind,
                ttl_sec=task.lease_ttl_sec or 60,
            )
            if lease is None:
                log.info(
                    "dispatcher: %s (%s) not started; lanes %s are busy",
                    task.task_id,
                    task.kind,
                    list(task.requires_lanes),
                )
                return None
        if cancel_scope is None:
            cancel_scope = CancelScope()
            handle = asyncio.current_task()
            if handle is not None:
                self._inflight_actions[task.task_id] = _InflightAction(task.kind, handle, cancel_scope)

        async def release_resources() -> None:
            if held.gpu_lease is not None:
                await self.gpu_specialist_pool.release(held.gpu_lease)

        await self._record_dispatch_evidence(task, lease, release_resources)

        async def execute_and_complete() -> SubAgentResult:
            result = await self.sub.run_task(
                task,
                prebound_lease=lease,
                extra_context=extra_context,
                release_resources=release_resources,
            )
            try:
                if on_complete is not None:
                    await on_complete(result)
                return result
            finally:
                self._inflight_actions.pop(task.task_id, None)
                self._executions.discard(asyncio.current_task())

        with (
            use_cancel_scope(cancel_scope),
            current_action_scope(task.kind),
            trajectory_scope(task_id=task.task_id, parent_span_id=task.task_id),
        ):
            execution = asyncio.create_task(execute_and_complete())
        self._executions.add(execution)
        self._execution_of[task.task_id] = execution
        execution.add_done_callback(self._report_spawned_failure(task))
        execution.add_done_callback(self._forget_execution(task.task_id))
        try:
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            cancel_scope.cancel(reason="caller_cancelled")
            grace = _COOPERATIVE_CANCEL_GRACE_SEC if cancel_scope.has_listeners else _CANCEL_NOTICE_SEC
            await asyncio.wait({execution}, timeout=grace)
            if not execution.done():
                log.warning(
                    "dispatcher: task=%s execution cleanup still pending; retaining capacity and running state",
                    task.task_id,
                )
            raise

    def _specialist_progress_publisher(self, task: Task) -> Any:
        """Build the callback that turns partial checkpoints into observations.

        Args:
            task: The specialist task the callback reports for.

        Returns:
            An async callable taking ``(payload, elapsed_sec)``.
        """

        async def _publish(payload: dict[str, Any], elapsed: float) -> None:
            """Append one specialist_progress observation.

            Args:
                payload: The checkpoint payload the specialist wrote.
                elapsed: Seconds since the subprocess was spawned.
            """
            params = task.params or {}
            proposals = payload.get("proposal_set")
            findings = payload.get("new_findings")
            questions = payload.get("residual_questions")
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "observation",
                    {
                        "kind": "specialist_progress",
                        "task_id": task.task_id,
                        "domain": str(params.get("domain") or ""),
                        "gap_canonical_id": str(params.get("gap_canonical_id") or ""),
                        "elapsed_sec": int(elapsed),
                        "summary": str(payload.get("summary") or "")[:400],
                        "proposals_so_far": len(proposals) if isinstance(proposals, list) else 0,
                        "new_findings": [str(f)[:200] for f in (findings or [])[:3]]
                        if isinstance(findings, list)
                        else [],
                        "residual_questions": [str(q)[:200] for q in (questions or [])[:3]]
                        if isinstance(questions, list)
                        else [],
                    },
                )
            )

        return _publish

    def _specialist_wall_budget_sec(self, *, params: dict[str, Any] | None = None) -> float:
        """How long a specialist of this shape is worth running, ignoring the session.

        A mode-tiered base (research 10min / patch 60min) amplified by the macro-cycle
        count and hard-capped at 4h; a no-lease-extension dispatch gets the cap. Bench-capable patch specialists are floored
        at the rebench helper's timeout plus a 10-minute startup allowance, since
        a budget under that guarantees the kill lands mid-benchmark::

            budget_sec = min(base × (macro_cycle + 1), 240 min)
            if profile.bench:
                budget_sec = max(budget_sec, rebench_timeout + 10 min)

        What the session can still afford is applied separately by
        :meth:`_specialist_deadline`, so session exhaustion cannot be read here
        as an absent budget.

        Args:
            params: Specialist dispatch parameters; the mode picks the base and
                bench-capable patch specialists get the rebench floor.

        Returns:
            float: A positive wall-clock budget in seconds.
        """
        from ..specialists.profile import NO_LEASE_EXTENSION, resolve_specialist_profile, wall_budget_base_min
        from ..actions.executors._subprocess_kill import resolve_benchmark_timeouts

        macro_cycle = int(self.shared_state.macro_cycle or 0)
        cap_min = 240.0
        base_min = wall_budget_base_min(params) * (macro_cycle + 1)
        budget_sec = (cap_min if (params or {}).get(NO_LEASE_EXTENSION) else min(base_min, cap_min)) * 60.0
        if resolve_specialist_profile(params).reserves_benchmark_lane:
            budget_sec = max(budget_sec, resolve_benchmark_timeouts()[1] + 10 * 60)
        return budget_sec

    def _specialist_deadline(self, *, params: dict[str, Any] | None = None) -> Deadline:
        """The absolute instant this specialist must have stopped by.

        An exhausted session yields an already-expired instant, which the reaper
        kills on immediately -- unlike a zero-second duration, which a lower
        layer could read as "unbudgeted".

        Args:
            params: Specialist dispatch parameters.

        Returns:
            Deadline: The earliest of the shape budget, the session bound and the gated phase's end.
        """
        shape = Deadline.after(self._specialist_wall_budget_sec(params=params))
        return shape.tightened_to(self.shared_state.session_deadline()).tightened_to(phase_deadline(self.shared_state))

    def _dispatch_context(self, task: Task) -> dict[str, Any]:
        """The bounds a dispatched task runs under: its phase's end and, for a specialist, its own stop instant."""
        context: dict[str, Any] = {"phase_deadline": phase_deadline(self.shared_state)}
        if task.kind == "specialist":
            stop = self._specialist_deadline(params=task.params or {})
            ceiling = context["phase_deadline"]
            limit_wall_budget_extension(task.task_id, None if ceiling is None else ceiling.at - stop.at)
            context.update(specialist_deadline=stop, specialist_progress_cb=self._specialist_progress_publisher(task))
        return context

    def gpu_lease_ttl_sec(
        self,
        floor_ttl_sec: int = 0,
        *,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Single source for the ``gpu_research_lane`` TTL.

        Nothing releases a lease when this TTL passes; the reaper stops the
        specialist at its deadline. The lane expiry feeds the in-flight
        ``lease_expires_in_sec`` countdown and ``recover --confirm-scope-gone``'s
        liveness check, so covering the deadline keeps both from calling a lease
        lapsed while its specialist can still run.

        Args:
            floor_ttl_sec: A lower bound (e.g. the registry / existing
                ``lease_ttl_sec``) the computed TTL is raised to.
            params: Specialist dispatch parameters, used to derive the deadline.

        Returns:
            int: ``max(floor_ttl_sec, time_to_deadline × (1 + grace))``.
        """
        stop_at = self._specialist_deadline(params=params)
        return max(
            int(floor_ttl_sec or 0),
            math.ceil(max(0.0, stop_at.remaining()) * (1.0 + GPU_LEASE_TTL_GRACE)),
        )

    async def reap_dispatched_task(self, task: Task, result: SubAgentResult) -> None:
        """Run completion bookkeeping for one finished dispatched task.

        Performs post-completion bookkeeping: specialist auto-retry,
        ``delegated_result`` emission, ledgers, shared-state promotion,
        fact-write and explore-gap refresh. Execution owns resource cleanup.
        A promoted result's verdict comes from its promoter; cancelled and
        unpromotable results settle FAILED (including cancelled tasks).
        """
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event
        from hyperloom.inference_optimizer.breakdown.recorder.phase_records import ActionSettlement

        # Bounded transient-failure auto-retry (infra only): on a subprocess
        # timeout / crash / stale-heartbeat, re-enqueue a fresh specialist
        # task and skip this attempt's bookkeeping. Semantic empties fall
        # through and are recorded.
        if task.kind == "specialist" and result.state != "cancelled":
            if await self._coord.specialist_dispatch.maybe_auto_retry_specialist(task, result):
                return
        reauthor_attempt = task.params.get("reauthor_attempt")
        if reauthor_attempt not in (None, ""):
            result.result.setdefault("reauthor_attempt", reauthor_attempt)
        announcement = {
            "task_id": task.task_id,
            "kind": task.kind,
            "state": result.state,
            "result": result.result,
            "error": result.error,
        }
        try:
            await self.bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", announcement))
        except sqlite3.Error as exc:
            log.exception("dispatcher: failed to append delegated_result for task=%s", task.task_id)
            self._coord.record_exception(stage="dispatcher_result", exc=exc)
            return
        if task.kind == "specialist":
            await self._record_specialist_settlement(task, result)
        payload = _promotion_payload(task, result)
        promotion = await self._promote_settled_result(task, result, payload)
        verdict, adopted = promotion or (Verdict.FAILED, ())
        # Settle the dispatch row on the phase that ordered it, whichever way promotion ended.
        phase_event.record_settle(
            ActionSettlement(
                task_id=task.task_id,
                status=result.state,
                decision=verdict.value,
                error_class=payload.get("error_class") or result.error_class,
                workspace=payload.get("workspace"),
                settled_unix=time.time(),
            )
        )
        if task.kind in ("explore", "integrate_patch"):
            self._coord.writeback.record_intervention_for_task(task, result.result, verdict)
        if promotion is None:
            return
        if task.kind == "integrate_patch":
            await self._coord.phase_framework.on_integrate_patch_settled(task, result, verdict)
        # replay_warm_recipe is excluded from the fact-write: it is verification, not a fact.
        if task.kind != _REPLAY_WARM_RECIPE:
            try:
                await self._coord.recipe_journal.fact_write_hook(
                    task=task, result=result, verdict=verdict, adopted_variants=adopted
                )
            except OSError as exc:
                log.exception("dispatcher: fact-write hook failed for task=%s", task.task_id)
                self._coord.record_exception(stage="dispatcher_fact_write", exc=exc)
        if task.kind == "explore":
            # Append this explore round's per-variant KEEP/REVERT, then re-run the global gap refresh.
            workload_id = self._coord.recipe_journal.workload_canonical_id()
            gap_refresh = self._coord.gap_refresh
            gap_refresh.record_explore_round_gaps(task=task, result=result.result, workload_id=workload_id)
            gap_refresh.record_explore_variant_failures(task=task, result=result.result)
            await gap_refresh.refresh_gaps(reason="explore_round", workload_id=workload_id)

    async def _record_specialist_settlement(self, task: Task, result: SubAgentResult) -> None:
        """Record an ended specialist round once; the FRAMEWORK pump keys its next discovery round on this settle."""
        done_payload = result.result.get("specialist_done") or {}
        run_error = str(result.error or "")
        await self._coord.specialist_dispatch.record_specialist_result(
            task=task,
            done_payload=done_payload,
            source=(f"{SPECIALIST_FROM_AGENT_PREFIX}{task.task_id}"),
            run_error=run_error,
        )
        self._coord.phase_framework.on_specialist_settled(task, done_payload, run_error=run_error)

    async def _promote_settled_result(
        self,
        task: Task,
        result: SubAgentResult,
        payload: dict[str, Any],
    ) -> tuple[Verdict, Collection[str]] | None:
        """Promote a succeeded result into shared state (Coordinator-only writer), or record why not.

        ``None`` says the result was cancelled or promotion itself raised, leaving the rest of the
        bookkeeping unowed. Warm replay is deliberately routed through its promote handler even when
        dispatch itself failed: that handler owns rollback of pre-applied framework patches and
        clears the PRELUDE ``in_flight`` gate.
        """
        if result.state == "cancelled":
            return None
        writeback = self._coord.writeback
        try:
            promotable = result.state == "succeeded" or (task.kind == _REPLAY_WARM_RECIPE and result.state == "failed")
            if promotable and writeback.is_promotable_result(task.kind, payload):
                promoted = await writeback.promote_to_shared_state(task.kind, payload, task=task)
                return promoted.verdict, promoted.adopted_variants
            unpromotable = dict(result.result)
            if result.error_class and not unpromotable.get("error_class"):
                unpromotable["error_class"] = result.error_class
            await writeback.handle_unpromotable_result(task, unpromotable)
        except Exception as exc:  # isolation boundary: one settled task's promotion
            log.exception("dispatcher: promotion/unpromotable handling failed for task=%s", task.task_id)
            self._coord.record_exception(stage="dispatcher_promote", exc=exc)
            return None
        return Verdict.FAILED, ()


def _promotion_payload(task: Task, result: SubAgentResult) -> dict[str, Any]:
    """The settled result's payload as its promote handler reads it."""
    payload = dict(result.result)
    # A warm replay that failed in dispatch still owes its promote handler a cleanup pass.
    if task.kind == _REPLAY_WARM_RECIPE and result.state == "failed":
        payload.setdefault("status", "failed")
        payload.setdefault("error_class", "dispatch_failed")
        if result.error:
            payload.setdefault("error", str(result.error))
    return payload
