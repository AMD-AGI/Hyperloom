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
from hyperloom.common.coerce import to_int
from hyperloom.common.deadline import Deadline
from hyperloom.common.env import is_truthy
from hyperloom.common.framework_arm import verdict_subject
from hyperloom.common.llm_attribution import current_action_scope
from hyperloom.common.proctree import TERM_GRACE_SEC
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    KERNEL_AGENT_OWNED_ACTIONS,
    REQUEST_KIND_TO_OWNED_ACTION,
)
from hyperloom.inference_optimizer.session.optimization_journal import Verdict
from hyperloom.inference_optimizer.trace.trajectory_trace import trajectory_scope
from ..actions.cancel_channel import CancelScope, use_cancel_scope
from ..actions.executors._ray_serving import (
    CANCEL_ROUND_GRACE_SEC,
    CLOSE_STOP_TIMEOUT_SEC,
)
from ..actions.executors._subprocess_kill import (
    COOPERATIVE_REAP_BUDGET_SEC,
    STOP_GATE_POLL_SECONDS,
)
from ..phases import dispatch_order, machine_state as _phase_state
from ..bus.message_bus import Message
from ..kernel.request_handlers import get_handler
from ..policy.gate import (
    INTEGRATE_PATCH_PERMISSIVE_VERDICTS,
    PolicyDenied,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from ..policy.projection import serving_tp_for_policy
from ..policy.specialist_in_flight import specialist_in_flight_denial
from ..bus.gpu_pool import (
    GPU_LEASE_TTL_GRACE,
)
from ..bus.resource_lock import (
    KNOWN_LANES,
    _expand_lanes,
)
from .sub_agent_runner import SubAgentResult
from ..specialists.wall_budget import limit_wall_budget_extension
from ..state.task_registry import IllegalTransition, Task, TaskNotFound, TaskSpec
from .time_budget import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    expected_action_cost_minutes,
    gated_phase_remaining_sec,
    lease_ttl_sec,
    measured_baseline_runtime_sec,
    phase_deadline,
    specialist_admission_minutes,
    time_budget_denial,
)

from ..collaborator import CoordinatorCollaborator
import logging as _logging

log = _logging.getLogger(__name__)

# How long a cancel waits for work that is listening on its cancel scope to stop
# itself, composed from the two things an action still has to do after it is
# asked:
#
# * stop the round in flight -- locally that is
#   :data:`COOPERATIVE_REAP_BUDGET_SEC`, through a Ray actor it is
#   :data:`CANCEL_ROUND_GRACE_SEC`, and whichever path this action took, only the
#   longer of the two bounds it;
# * release what the round held -- the driver-side teardown of the server it left
#   behind (:data:`TERM_GRACE_SEC`) and then the release of the Ray lease it
#   ran in (:data:`CLOSE_STOP_TIMEOUT_SEC`). A sequence, not alternatives: the
#   server is reaped BEFORE the lease is dropped so that no GPU process outlives
#   it (§4.2), and on the Ray path a single unwind pays both -- in the explore
#   executor the per-variant ``finally`` tears the server down and the enclosing
#   one then closes the round's lease; the baseline executor does the same two
#   calls in one ``finally``.
#
# Derived rather than picked, because these three windows only mean anything
# together. Each was plausible on its own at ten, eight and five seconds, and
# composed they said the dispatcher gives up a good five seconds before the work
# it is waiting for can finish -- so the honest sentinel the round was about to
# return was discarded for a hard ``CancelledError`` every time, which is exactly
# what the cooperative channel exists to avoid. Taking the longer of the two
# release terms instead of both reproduced that shortfall exactly, on the path
# that pays the most: the teardown a Ray round owes is not an alternative to
# closing its lease, it is what it does first.
#
# The enumeration above has to stay exhaustive, so a serial step the unwind takes
# and this sum does not name has to be removed from the unwind instead.
#
# Past this window the coroutine is cancelled anyway, and the window is only ever
# spent when work is still unwinding: the wait ends the moment the last victim is
# done, so covering the teardown term costs a round that stops promptly nothing.
# What the budget case pays for it is five more seconds before the run crosses its
# deadline, because this wait runs inside the reserve the admission gate holds
# back. It does not come out of the closing phase, whose grace window is measured
# from the moment it starts, and five seconds of overshoot is cheaper than the
# attributed sentinel a hard cancel destroys.
_COOPERATIVE_CANCEL_GRACE_SEC: float = (
    max(COOPERATIVE_REAP_BUDGET_SEC, CANCEL_ROUND_GRACE_SEC) + TERM_GRACE_SEC + CLOSE_STOP_TIMEOUT_SEC
)

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

    ``prebound_lease`` is the lanes; ``gpu_lease`` the SQLite GPU-specialist accounting lease; ``gpu_specialist_lease``
    the Ray ``GpuSpecialistLease`` to close.
    """

    prebound_lease: Any = None
    gpu_lease: Any = None
    gpu_specialist_lease: Any = None


class _InflightAction(NamedTuple):
    """A running action's handle: what it is, its task, and how to ask it to stop."""

    kind: str
    atask: asyncio.Task[Any]
    scope: CancelScope


class DispatcherCollaborator(CoordinatorCollaborator):
    """Handles specialist dispatch: budget gating, action routing, and task enqueue."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        # Task ids already charged a failure by the dead-holder reclaim path, so
        # a late normal result for the same task cannot double-count it.
        self._dead_holder_accounted: set[str] = set()
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

    async def _reclaim_stale_dispatch_state(self) -> None:
        """Free rows and leases a previous tick left stuck, before scanning the queue.

        Runs every tick and is idempotent. Four independent claims a crashed or
        vanished worker can leave behind, each reclaimed on its own so one
        failure does not hide the next -- and every one of them best-effort,
        because a self-heal that raises would stop the pump it exists to keep
        running:

        * a running row whose holder PID is dead, so its lanes free and the task
          fails while it is still retry-eligible, this same tick;
        * the failure that reclaim implies, charged once;
        * the lane leases that holder still held;
        * an ``integrate_patch`` row cancelled at dispatch whose critic verdict
          was restored afterwards by a resume.
        """
        dead_tasks: list[str] = []
        try:
            dead_tasks = await self.tasks.reclaim_dead_running(reason="dead_holder_pump")
            if dead_tasks:
                log.warning(
                    "dispatcher: reclaimed %d running task(s) with dead holders: %s",
                    len(dead_tasks),
                    ", ".join(t[:12] for t in dead_tasks),
                )
        except Exception:
            log.exception("dispatcher: dead-running task reclaim failed")
        dead_tasks.extend(self._coord.reconciler.last_report.failed_tasks)
        if dead_tasks:
            try:
                await self._account_dead_holder_failures(dead_tasks, reason="dead_holder_pump")
            except Exception:
                log.exception("dispatcher: dead-holder failure accounting failed")
        try:
            await self.locks.reap_dead_holders()
        except Exception:
            log.exception("dispatcher: dead-holder lease reap failed")
        try:
            await self._reconcile_cancelled_policy_denied_integrate_tasks()
        except Exception:
            log.exception("dispatcher: cancelled policy-denied integrate_patch reconcile failed")

    async def pump_dispatcher_once(self) -> None:
        """Book finished executions and spawn the queued tasks that now fit; never waits on running work.

        Dispatched tasks outlive the pump and queue their completion for a later
        one. The pump repeats while a pass books something, so lanes freed by
        completions already queued are refilled in the same call. New spawns stop
        while the phase budget is spent or :attr:`admission_frozen` is set.
        """
        await self._reclaim_stale_dispatch_state()
        while True:
            booked = await self._drain_completions()
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

    async def _reconcile_cancelled_policy_denied_integrate_tasks(self) -> list[str]:
        """Re-queue integrate_patch rows cancelled at dispatch when policy now passes.

        Covers the resume gap where ``coordinator.db`` retained a cancelled task
        but ``SharedState.specialist_patch_verdicts`` was restored later. The
        row is re-keyed by its review subject, so a pre-screened upstream-PR
        candidate reconciles the same way an authored patch does. Only
        ``integrate_patch_requires_critic_verdict`` denials are retried; forged
        params that still fail :meth:`PolicyGate.validate_dispatched_task` are
        left terminal.

        Returns:
            list[str]: New queued task ids created by reconcile (may be empty).
        """
        gate = self.sub.policy
        if gate is None:
            return []
        get_verdict = self.shared_state.get_specialist_patch_verdict

        created: list[str] = []
        for task in await self.tasks.by_state("cancelled"):
            if task.kind != "integrate_patch":
                continue
            evidence = _dispatch_policy_denied_evidence(task)
            if not evidence:
                continue
            if str(evidence.get("rule") or "") != "integrate_patch_requires_critic_verdict":
                continue
            params = dict(task.params or {})
            sid = verdict_subject(params)
            if not sid:
                continue
            verdict = str(get_verdict(sid) or "").strip().lower()
            if verdict not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
                continue
            try:
                gate.validate_dispatched_task(task.kind, params)
            except PolicyDenied:
                continue
            base_key = str(task.idempotency_key or f"integrate-{task.task_id}").strip()
            child_key_prefix = f"{base_key}-reconcile"
            if await self.tasks.exists_with_key_prefix(
                task.kind,
                child_key_prefix,
                states=("succeeded", "queued", "running"),
            ):
                continue
            for attempt in range(1, 6):
                new_key = f"{child_key_prefix}{attempt}"
                new_task, was_existing = await self.tasks.create_or_return_existing(
                    TaskSpec(
                        kind=task.kind,
                        params=params,
                        idempotency_key=new_key,
                        requires_lanes=list(task.requires_lanes or []),
                        lease_ttl_sec=int(task.lease_ttl_sec or 0),
                        dispatch_class="coordinator",
                    )
                )
                if not was_existing:
                    created.append(new_task.task_id)
                    log.info(
                        "dispatcher: reconciled cancelled integrate_patch %s -> %s (key=%s)",
                        task.task_id,
                        new_task.task_id,
                        new_key,
                    )
                    break
        return created

    async def _admit_lanes(self, task: Task, holders: dict[str, int], capacities: dict[str, int]) -> Any:
        """Take the lane lease a queued task needs, bumping ``holders`` for the rest of this pass.

        Under single-node Ray execution the authoritative GPU mutex is Ray's custom resources, not
        this SQLite gate, which is kept as a resume-safe view feeding the lane timeline.
        """
        lanes_needed = task.requires_lanes
        if not lanes_needed:
            return None
        try:
            expanded = _expand_lanes(lanes_needed)
        except ValueError:
            log.warning(
                "dispatcher: task %s has unknown lane in %r; skipping until resolved",
                task.task_id,
                lanes_needed,
            )
            return None
        if not self._lanes_fit(expanded, holders, capacities):
            return None
        # ``None`` here is a race: another holder grabbed the lane; leave queued.
        lease = await self.locks.acquire_many(
            lanes_needed,
            holder_id=task.task_id,
            task_id=task.task_id,
            action=task.kind,
            ttl_sec=task.lease_ttl_sec or 60,
        )
        if lease is None:
            return None
        for lane in lease.lanes:
            holders[lane] = holders.get(lane, 0) + 1
        return lease

    async def _admit_gpu_specialist(self, task: Task, lease: Any, extra_context: dict[str, Any]) -> HeldLeases | None:
        """Take the GPU leases a specialist needs, recording the devices it may use in ``extra_context``."""
        from ..actions.executors._ray_backend import (
            ray_gpu_pending_limit,
            ray_exec_enabled,
            ray_serving_priority_enabled,
            serving_slot_busy,
        )
        from ..actions.executors._ray_serving import maybe_gpu_specialist_lease
        from ..specialists.profile import holds_serving_slot, is_authoring_specialist, requires_gpu
        from ..specialists.profile import uses_whole_machine_gpu_lane

        params = task.params
        if not requires_gpu(params):
            return HeldLeases(prebound_lease=lease)
        # Probed immediately before admitting this GPU specialist, not once per pass, so a serving
        # start that races the pass does not slip through.
        if ray_serving_priority_enabled() and serving_slot_busy():
            # Serving is active — keep the task queued rather than piling onto the contended GPU.
            return None
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
        # Under single-node Ray the physical GPU mutex is Ray's ``num_gpus``, not this SQLite pool, so
        # admission there is a count-based pending limit and several specialists may share one GPU.
        if ray_exec_enabled():
            gpu_lease = await gpu_pool.try_acquire_ray_observation(
                holder_id=task.task_id,
                task_id=task.task_id,
                pending_limit=ray_gpu_pending_limit(),
            )
        else:
            gpu_lease = await gpu_pool.try_acquire(
                count=gpu_count,
                holder_id=task.task_id,
                task_id=task.task_id,
            )
        if gpu_lease is None:
            return None
        extra_context["gpu_ids"] = list(gpu_lease.gpu_ids)
        # Only bench-capable specialists that start their OWN serving loop hold the whole-machine
        # serving_slot mutex; every other GPU specialist takes ``num_gpus`` only, and Ray's
        # ``num_gpus`` accounting still keeps it off serving's cards.
        gpu_specialist_lease = maybe_gpu_specialist_lease(
            num_gpus=gpu_count,
            serving_slot=holds_serving_slot(params),
        )
        if gpu_specialist_lease is not None:
            # Ray assigns and masks the physical cards, so the specialist sees a logical 0..N-1
            # view and the SQLite ids are capacity accounting only. ``None`` off the Ray path
            # (multi-node / RAY_EXEC off / tests) keeps the SQLite-gpu-id device path.
            extra_context["gpu_ids"] = list(range(gpu_count))
            extra_context["gpu_specialist_lease"] = gpu_specialist_lease
        return HeldLeases(lease, gpu_lease, gpu_specialist_lease)

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
        holders = await self.locks.lane_holders()
        capacities = await self.locks.lane_capacities()
        execs = self.sub.executor_registry
        for task in queued:
            if task.task_id in self._inflight_actions:
                continue
            if await self._cancel_queued_task_over_budget(task):
                continue
            # No lease means refused, unless the task asked for no lanes in the first place.
            lease = await self._admit_lanes(task, holders, capacities)
            if lease is None and task.requires_lanes:
                continue
            extra_context = self._dispatch_context(task)
            held = HeldLeases(prebound_lease=lease)
            if task.kind == "specialist":
                held = await self._admit_gpu_specialist(task, lease, extra_context)
            if held is None:
                if lease is not None:
                    await self.locks.release(lease)
                    for lane in lease.lanes:
                        holders[lane] = max(0, holders.get(lane, 0) - 1)
                continue
            # Kernel-owned kinds are legitimately unregistered under ``--no-kernel``.
            if execs and task.kind not in (*execs, "specialist", *KERNEL_AGENT_OWNED_ACTIONS):
                log.warning(
                    "dispatch audit: queued task_id=%s kind=%r has no registered executor (dispatch unchanged)",
                    task.task_id,
                    task.kind,
                )
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
            if task.kind == "specialist":
                self.shared_state.note_specialist_dispatched(str(task.params.get("domain") or ""))

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
        self, task: Task, lease: Any, release_resources: Callable[[], Awaitable[bool]]
    ) -> None:
        """Record a task's dispatch evidence, or give back what it holds and refuse.

        Args:
            task: The task row about to run.
            lease: The lane lease the dispatch holds, if any.
            release_resources: The execution's slot release, which reports
                whether the worker confirmed its physical cleanup.

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
            if not await release_resources():
                log.warning("dispatcher: task=%s refused with cleanup unconfirmed", task.task_id)
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

        async def release_resources() -> bool:
            if held.gpu_lease is None and held.gpu_specialist_lease is None:
                return True
            return await self.gpu_specialist_pool.release_execution(
                held.gpu_lease, held.gpu_specialist_lease, task.task_id
            )

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

    async def _account_dead_holder_failures(
        self,
        task_ids: list[str],
        *,
        reason: str,
    ) -> None:
        """Charge a failure for each task reclaimed from a dead lease holder.

        A holder killed from outside this process (its lease reaped by
        ``reap_dead_holders``) never returns a result, so without this the
        per-action streak counters never see the death and the streak-based
        auto-terminate stays blind to the whole failure mode.

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
            await self._coord.writeback.handle_unpromotable_result(
                task,
                {"status": "failed", "error_class": "dead_holder_reaped", "error": error},
            )
            if task.kind == "specialist" and not self.owes_result(task_id):
                await self._record_specialist_settlement(task, SubAgentResult(task_id, "failed", {}, error=error))
            log.warning(
                "dispatcher: counted dead-holder death of task=%s kind=%s as an action failure",
                task_id,
                task.kind,
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
        except Exception as exc:  # isolation boundary: one settled task's delegated_result append
            log.exception("dispatcher: failed to append delegated_result for task=%s", task.task_id)
            self._coord.record_exception(stage="dispatcher_result", exc=exc)
            return
        if task.kind == "specialist":
            await self._record_specialist_settlement(task, result)
        payload = dict(result.result)
        # A warm replay that failed in dispatch still owes its promote handler a cleanup pass.
        if task.kind == _REPLAY_WARM_RECIPE and result.state == "failed":
            payload.setdefault("status", "failed")
            payload.setdefault("error_class", "dispatch_failed")
            if result.error:
                payload.setdefault("error", str(result.error))
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
            except Exception as exc:  # isolation boundary: one settled task's fact-write hook
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
            if task.task_id not in self._dead_holder_accounted:
                unpromotable = dict(result.result)
                if result.error_class and not unpromotable.get("error_class"):
                    unpromotable["error_class"] = result.error_class
                await writeback.handle_unpromotable_result(task, unpromotable)
        except Exception as exc:  # isolation boundary: one settled task's promotion
            log.exception("dispatcher: promotion/unpromotable handling failed for task=%s", task.task_id)
            self._coord.record_exception(stage="dispatcher_promote", exc=exc)
            return None
        return Verdict.FAILED, ()

    @staticmethod
    def _lanes_fit(
        expanded_lanes: list[str],
        holders: dict[str, int],
        capacities: dict[str, int],
    ) -> bool:
        """Local-view headroom hint for the concurrent dispatcher (authoritative gate is acquire_many).

        Args:
            expanded_lanes: The fully-expanded lanes the task requires.
            holders: Current per-lane holder counts (local view).
            capacities: Per-lane capacities.

        Returns:
            ``True`` when every requested lane has local headroom, else
            ``False``.
        """
        for lane in expanded_lanes:
            cap = int(capacities.get(lane, 1))
            used = int(holders.get(lane, 0))
            if cap <= 0 or used >= cap:
                return False
        return True

    def _phase_denial_for_action(self, action_name: str) -> PolicyDenied | None:
        """Refuse an action the current phase does not let an agent propose.

        Before the phase machine has entered its first phase there is no
        contract to hold the action to, so the gate abstains.

        Args:
            action_name: The proposed/delegated action name.

        Returns:
            A :class:`PolicyDenied` when the phase does not offer the action, else ``None``.
        """
        action = str(action_name or "").strip()
        phase = str(self.shared_state.phase or "").strip().upper()
        proposable = _phase_state.allowed_actions_for(phase)
        if not phase or action in proposable:
            return None
        return PolicyDenied(
            f"action={action!r} is not proposable while the phase is {phase!r}",
            rule="phase_incompatible",
            hint=f"{phase} proposes: {', '.join(proposable) or '(none)'}",
        )

    def _sequence_denial_for_action(
        self,
        action_name: str,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts before baseline. Only invariant: nothing runs until baseline_tput > 0 (a data-dependency).

        Args:
            action_name: The proposed/delegated action name.

        Returns:
            A :class:`PolicyDenied` when the action must wait for baseline, else
            ``None``.
        """
        action = str(action_name or "").strip()
        sequence_actions = {
            "target_analysis",
            "baseline",
            "report",
            "integrate",
            "explore",
        }
        if action not in sequence_actions:
            return None
        if self.shared_state.stop_reason:
            return None
        if self.shared_state.baseline_tput <= 0 and action not in {"baseline", "target_analysis"}:
            return PolicyDenied(
                f"action={action!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` until baseline_tput > 0",
            )
        return None

    def admission_cost_minutes(self, action_name: str) -> float:
        """What a catalogued action is admitted at, priced on this session's measured baseline round once one exists."""
        measured_sec = measured_baseline_runtime_sec(self.shared_state)
        return expected_action_cost_minutes(self.action_registry.get(action_name), measured_baseline_sec=measured_sec)

    def time_budget_denial_for_action(
        self,
        action_name: str,
        *,
        fallback_cost_minutes: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> PolicyDenied | None:
        """Refuse an action whose expected cost cannot fit the remaining session budget or gated phase budget.

        The first of the wall-clock defences: cheaper to never start a 60-minute
        action with 20 minutes left than to reap it half-done, because a reaped
        action spends the budget and yields no measurement. Denying here also
        keeps the refusal out of the failure ledgers — no task row is created, so
        nothing teaches the KB that the action failed.

        What the action costs is :meth:`admission_cost_minutes`.

        Args:
            action_name: The proposed/delegated action name.
            fallback_cost_minutes: What to price the action at when the
                catalogue does not carry it. Without it, a kind deliberately
                kept out of the catalogue is admitted at any remaining budget.
            params: The action's parameters; a specialist is priced at no less than the shortest run worth starting.

        Returns:
            A :class:`PolicyDenied` when the budget cannot fit the action, else
            ``None`` (unbounded budget, exempt action, or no cost on record).
        """
        action = str(action_name or "").strip()
        if not action or action in TIME_BUDGET_EXEMPT_ACTIONS:
            return None
        if self.shared_state.stop_reason:
            return None
        if self.action_registry.get(action) is not None:
            expected_min = self.admission_cost_minutes(action)
        elif fallback_cost_minutes is not None:
            expected_min = float(fallback_cost_minutes)
        else:
            return None
        if action == "specialist":
            expected_min = max(expected_min, specialist_admission_minutes(self.shared_state, params))
        return time_budget_denial(action, expected_min, self.shared_state)

    def admission_denial_for_action(
        self,
        action_name: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> PolicyDenied | None:
        """Run every pre-dispatch gate for an action name, first denial wins.

        The single entry point the intent handlers share, so a new gate
        reaches all paths at once.

        Args:
            action_name: The proposed/delegated action name.
            params: The proposal's parameters, which price a specialist.

        Returns:
            The first :class:`PolicyDenied` that fires, else ``None``.
        """
        return (
            self._phase_denial_for_action(action_name)
            or self._sequence_denial_for_action(action_name)
            or specialist_in_flight_denial(action_name, params, self.tasks, self.shared_state.find_gap)
            or self.time_budget_denial_for_action(action_name, params=params)
        )

    async def _cancel_queued_task_over_budget(self, task: Task) -> bool:
        """Drop a queued task the budget can no longer fit, before it takes a lane.

        The admission gate runs when the action is proposed, but a task can wait
        for a busy lane long enough for the budget to drain underneath it. This
        is the same gate re-applied at the last moment it is still free to say
        no. The row is cancelled rather than left queued so the pump does not
        re-examine it every tick, and it stays out of the failure ledgers: a task
        that never ran is not evidence about the action.

        Args:
            task: The queued task about to be considered for dispatch.

        Returns:
            ``True`` when the task was cancelled and must be skipped this pass.
        """
        # An uncatalogued kind is priced by the lease its enqueue sized, the
        # only cost estimate it has.
        ttl_sec = int(getattr(task, "lease_ttl_sec", 0) or 0)
        denied = self.time_budget_denial_for_action(
            task.kind,
            fallback_cost_minutes=(ttl_sec / 60.0) if ttl_sec > 0 else None,
            params=task.params,
        )
        if denied is None:
            return False
        try:
            await self.tasks.transition(
                task.task_id,
                "cancelled",
                evidence={"reason": denied.rule, "error": str(denied)},
            )
        except (sqlite3.Error, TaskNotFound, IllegalTransition):
            log.exception(
                "dispatcher: could not cancel over-budget task=%s kind=%s",
                task.task_id,
                task.kind,
            )
            return True
        log.warning(
            "dispatcher: dropped queued task=%s kind=%s before dispatch: %s",
            task.task_id,
            task.kind,
            denied,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "dispatch_denied_time_budget",
                "task_id": task.task_id,
                "action": task.kind,
                "error": str(denied),
                "hint": getattr(denied, "hint", ""),
            },
        )
        # A cancelled conc_sweep never writes last_conc_sweep on its own, so SWEEP
        # would idle. Stamp the skip here so SWEEP can exit.
        if str(task.kind or "") == "conc_sweep":
            self._coord.phase_sweep.record_session_budget_conc_sweep_skip(denied=denied)
        return True

    def admission_denial_for_request(
        self,
        target_agent: str,
        kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip the baseline prerequisite or that the wall-clock budget cannot fit.

        Nothing kernel-side runs before baseline_tput > 0, and a request runs its handler inline, so the time gate of
        the action it gates as is its only chance to be refused.

        Args:
            target_agent: The request's target agent; only ``"kernel_agent"`` is
                gated.
            kind: The kernel request kind; ``trace_analyze`` and unknown kinds
                are exempt.

        Returns:
            A :class:`PolicyDenied` when the kernel request must wait for
            baseline or does not fit the time left, else ``None``.
        """
        target = str(target_agent or "").strip()
        req_kind = str(kind or "").strip()
        if target != "kernel_agent" or self.shared_state.stop_reason:
            return None
        if req_kind == "trace_analyze":
            return None
        if get_handler(req_kind) is None:
            return None
        if self.shared_state.baseline_tput <= 0:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` before kernel requests",
            )
        return self.time_budget_denial_for_action(REQUEST_KIND_TO_OWNED_ACTION.get(req_kind, req_kind))


def _dispatch_policy_denied_evidence(task: Task) -> dict[str, Any]:
    """Return policy-denied evidence from a queued→cancelled dispatch transition."""
    for entry in reversed(task.history or []):
        if entry.get("from") != "queued" or entry.get("to") != "cancelled":
            continue
        evidence = entry.get("evidence") or {}
        if evidence.get("reason") == "policy_denied":
            return dict(evidence)
    return {}
