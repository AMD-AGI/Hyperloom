# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Collection
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, NamedTuple
from hyperloom.common.llm_attribution import current_action_scope
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    KERNEL_AGENT_OWNED_ACTIONS,
)
from ..actions.cancel_channel import CancelScope, use_cancel_scope
from ..actions.executors._ray_serving import (
    CANCEL_ROUND_GRACE_SEC,
    CLOSE_STOP_TIMEOUT_SEC,
)
from ..actions.executors._subprocess_kill import (
    COOPERATIVE_REAP_BUDGET_SEC,
    STOP_GATE_POLL_SECONDS,
    TERM_GRACE_SECONDS,
)
from ..phases import machine_state as _phase_state
from ..bus.message_bus import Message
from ..kernel.request_handlers import get_handler
from ..policy.gate import (
    INTEGRATE_PATCH_PERMISSIVE_VERDICTS,
    patch_verdict_subject,
    PolicyDenied,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from ..bus.gpu_pool import (
    GPU_LEASE_TTL_GRACE,
)
from ..bus.resource_lock import (
    KNOWN_LANES,
    _expand_lanes,
)
from .sub_agent_runner import SubAgentResult
from ..state.task_registry import Task
from .coordinator_helpers import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    action_fits_time_budget,
    coerce_needs_gpu,
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
)

from .coordinator import (
    _format_inbox_event,
)
import logging as _logging

log = _logging.getLogger(__name__)

# How long a cancel waits for work that is listening on its cancel scope to stop itself, composed from the two things
# an action still has to do after it is asked:
_COOPERATIVE_CANCEL_GRACE_SEC: float = (
    max(COOPERATIVE_REAP_BUDGET_SEC, CANCEL_ROUND_GRACE_SEC) + TERM_GRACE_SECONDS + CLOSE_STOP_TIMEOUT_SEC
)

# How long a cancel waits for anything to start listening before deciding nothing will.
_CANCEL_NOTICE_SEC: float = STOP_GATE_POLL_SECONDS


#: Kinds the pump dispatches but does not join: it drains what it joins before
#: returning, and an off-loop compile there would hold every reactor turn for
#: its duration. Admission is unchanged — same budget, lane and lease gates.
_NOT_JOINED_KINDS: frozenset[str] = frozenset({"targeted_build"})


class _InflightAction(NamedTuple):
    """A running action's handle: what it is, its task, and how to ask it to stop."""

    kind: str
    atask: asyncio.Task[Any]
    scope: CancelScope


class DispatcherCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator
        # Task ids already charged a failure by the dead-holder reclaim path, so a late normal result for the same
        # task cannot double-count it.
        self._dead_holder_accounted: set[str] = set()
        # Handles on the actions currently running, ``task_id -> _InflightAction``.
        self._inflight_actions: dict[str, _InflightAction] = {}

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _registry_lanes_ttl(self, kind: str) -> tuple[list[str], int]:
        """Resolve ``(requires_lanes, lease_ttl_sec)`` from the action catalogue; lanes filtered to KNOWN_LANES."""
        meta = self.action_registry.get(kind)
        if meta is None:
            return [], 0
        lanes = [lane for lane in meta.requires_lanes if lane in KNOWN_LANES]
        return lanes, meta.lease_ttl_sec

    def _cycle_idem_suffix(self) -> str:
        """Idempotency-key suffix scoping a per-cycle internal singleton to the"""
        cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        return f"-c{cycle}" if cycle > 0 else ""

    def _dispatch_paused_for_phase_budget(self) -> bool:
        """True when the current phase's budget is spent, so the dispatcher should stop launching NEW phase-scoped variants."""
        state = self.shared_state
        phase = (getattr(state, "phase", "") or "").upper()
        if phase not in self._BUDGET_GATED_DISPATCH_PHASES:
            return False
        try:
            remaining = _phase_state.phase_budget_remaining_seconds(
                state,
                budget_pct=self._phase_budget_pct,
            )
        except Exception:  # noqa: BLE001 — never let the guard wedge dispatch
            return False
        return remaining is not None and remaining <= 0.0

    async def cancel_inflight_actions(
        self,
        *,
        reason: str,
        exempt: frozenset[str] = frozenset(),
        only_task_ids: Collection[str] | None = None,
    ) -> list[str]:
        """Stop the running dispatched actions and wait for them to unwind."""
        victims = [
            (task_id, entry)
            for task_id, entry in self._inflight_actions.items()
            if entry.kind not in exempt
            and not entry.atask.done()
            and (only_task_ids is None or task_id in only_task_ids)
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
        try:
            await self._wait_for_cooperative_stop(victims)
        finally:
            for _task_id, entry in victims:
                if not entry.atask.done():
                    entry.atask.cancel()
        await asyncio.gather(*(entry.atask for _task_id, entry in victims), return_exceptions=True)
        return [task_id for task_id, _entry in victims]

    async def _wait_for_cooperative_stop(self, victims: list[tuple[str, _InflightAction]]) -> None:
        """Give already-cancelled work the chance to stop itself and return."""
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
        """Stop in-flight actions the session can no longer wait for."""
        stop_event = getattr(self, "_stop", None)
        if stop_event is not None and stop_event.is_set():
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
        """Free rows and leases a previous tick left stuck, before scanning the queue."""
        dead_tasks: list[str] = []
        try:
            dead_tasks = await self.tasks.reclaim_dead_running(reason="dead_holder_pump")
            if dead_tasks:
                log.warning(
                    "dispatcher: reclaimed %d running task(s) with dead holders: %s",
                    len(dead_tasks),
                    ", ".join(t[:12] for t in dead_tasks),
                )
        except Exception:  # noqa: BLE001 — self-heal never aborts the pump
            log.exception("dispatcher: dead-running task reclaim failed")
        if dead_tasks:
            try:
                await self._account_dead_holder_failures(dead_tasks, reason="dead_holder_pump")
            except Exception:  # noqa: BLE001 — bookkeeping never aborts the pump
                log.exception("dispatcher: dead-holder failure accounting failed")
        try:
            await self.locks.reap_dead_holders()
        except Exception:  # noqa: BLE001
            log.exception("dispatcher: dead-holder lease reap failed")
        try:
            expired_tasks = await self.tasks.reclaim_expired_running(reason="pump_watchdog")
            if expired_tasks:
                log.warning(
                    "dispatcher: reclaimed %d expired-running task(s): %s",
                    len(expired_tasks),
                    ", ".join(t[:12] for t in expired_tasks),
                )
        except Exception:  # noqa: BLE001 — self-heal never aborts the pump
            log.exception("dispatcher: expired-running task reclaim failed")
        try:
            await self._reconcile_cancelled_policy_denied_integrate_tasks()
        except Exception:  # noqa: BLE001 — reconcile must not abort the pump
            log.exception("dispatcher: cancelled policy-denied integrate_patch reconcile failed")

    async def _pump_dispatcher_once(self) -> None:
        """Dispatch queued tasks respecting per-lane capacity, re-scanning for"""
        await self._reclaim_stale_dispatch_state()
        inflight: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
        # Cumulative across the whole pump, not just the live in-flight set, so a fast task reaped before its
        # queued->running transition is visible is not re-dispatched.
        dispatched_ids: set[str] = set()
        try:
            while True:
                # Wall-clock guard: a spent session budget (or a shutdown request) stops the actions already running,
                # because waiting for them is what the budget no longer allows.
                shutting_down = await self._cancel_inflight_that_outlived_the_session()
                # Budget guard: stop launching NEW phase-scoped variants once the phase's cyclic budget is spent;
                # drain in-flight then return.
                if not shutting_down and not self._dispatch_paused_for_phase_budget():
                    spawned = await self._spawn_fitting_queued(exclude_ids=dispatched_ids)
                    dispatched_ids.update(t.task_id for t, _, _ in spawned)
                    inflight.extend(spawned)
                if not inflight:
                    return
                done, _pending = await asyncio.wait(
                    [atask for _, atask, _ in inflight],
                    timeout=self._dispatcher_poll_sec,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    # Poll elapsed with no completion; re-scan in case a lane freed.
                    continue
                remaining: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
                completed: list[tuple[Task, Any, Any]] = []
                for entry in inflight:
                    task, atask, gpu_lease = entry
                    if atask in done:
                        try:
                            maybe_result: Any = atask.result()
                        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 — mirror gather(return_exceptions=True); capture task error + cancellation, never KeyboardInterrupt/SystemExit
                            maybe_result = exc
                        completed.append((task, maybe_result, gpu_lease))
                    else:
                        remaining.append(entry)
                inflight = remaining
                for task, maybe_result, gpu_lease in completed:
                    await self._reap_dispatched_task(task, maybe_result, gpu_lease)
        finally:
            # ``_inflight_actions`` is dispatcher-wide: the inline path registers a handle there too, and that action
            # is meant to outlive the caller that started it.
            await self.cancel_inflight_actions(
                reason="dispatcher_pump_exit",
                only_task_ids={task.task_id for task, _atask, _gpu_lease in inflight},
            )

    async def _reconcile_cancelled_policy_denied_integrate_tasks(self) -> list[str]:
        """Re-queue integrate_patch rows cancelled at dispatch when policy now passes."""
        gate = getattr(self.sub, "policy", None)
        state = getattr(self, "shared_state", None)
        if gate is None or state is None:
            return []
        get_verdict = getattr(state, "get_specialist_patch_verdict", None)
        if get_verdict is None:
            return []

        created: list[str] = []
        try:
            cancelled = await self.tasks.by_state("cancelled")
        except Exception:  # noqa: BLE001 — defensive
            log.exception("dispatcher: reconcile could not list cancelled tasks")
            return []

        for task in cancelled:
            if task.kind != "integrate_patch":
                continue
            evidence = _dispatch_policy_denied_evidence(task)
            if not evidence:
                continue
            if str(evidence.get("rule") or "") != "integrate_patch_requires_critic_verdict":
                continue
            params = dict(task.params or {})
            sid = patch_verdict_subject(params)
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
            if await self._integrate_reconcile_child_exists(
                base_key,
                states=("succeeded",),
            ):
                continue
            if await self._integrate_reconcile_child_exists(
                base_key,
                states=("queued", "running"),
            ):
                continue
            for attempt in range(1, 6):
                new_key = f"{base_key}-reconcile{attempt}"
                new_task, was_existing = await self.tasks.create_or_return_existing(
                    kind=task.kind,
                    params=params,
                    idempotency_key=new_key,
                    requires_lanes=list(task.requires_lanes or []),
                    lease_ttl_sec=int(task.lease_ttl_sec or 0),
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
                if new_task.state in ("queued", "running", "succeeded"):
                    break
        return created

    async def _integrate_reconcile_child_exists(
        self,
        base_key: str,
        *,
        states: tuple[str, ...],
    ) -> bool:
        """Return whether a reconcile child idempotency key exists in any of ``states``."""
        if not states:
            return False
        prefix = f"{base_key}-reconcile%"
        placeholders = ",".join("?" for _ in states)
        row = await self.tasks.db.fetchone(
            "SELECT 1 FROM tasks WHERE kind='integrate_patch' "
            f"AND idempotency_key LIKE ? AND state IN ({placeholders}) LIMIT 1",
            (prefix, *states),
        )
        return row is not None

    async def _spawn_fitting_queued(
        self,
        *,
        exclude_ids: set[str],
    ) -> list[tuple[Task, "asyncio.Task[SubAgentResult]", Any]]:
        """Spawn every currently lane-fitting queued task not already in flight."""
        queued = await self.tasks.queued()
        if not queued:
            return []
        holders = await self.locks.lane_holders()
        capacities = await self.locks.lane_capacities()
        spawned: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
        # Serving priority: pre-compute whether serving-priority is enabled once per pass (a pure env-var read, zero
        # cost).
        _ray_serving_priority_enabled = False
        _serving_slot_busy_fn = None
        try:
            from ..actions.executors._ray_backend import (
                ray_serving_priority_enabled as _rsp_enabled,
                serving_slot_busy as _ssb,
            )

            if _rsp_enabled():
                _ray_serving_priority_enabled = True
                _serving_slot_busy_fn = _ssb
        except Exception:  # noqa: BLE001 — never block dispatch on the probe
            pass
        for task in queued:
            if task.task_id in exclude_ids:
                # Already dispatched in a prior pass of this pump.
                continue
            join_in_pump = task.kind not in _NOT_JOINED_KINDS
            if not join_in_pump and task.task_id in self._inflight_actions:
                # Still running from an earlier pump that returned without it.
                continue
            if await self._cancel_queued_task_over_budget(task):
                continue
            lanes_needed = list(task.requires_lanes or [])
            if lanes_needed:
                # SQLite lane gate.
                try:
                    expanded = _expand_lanes(lanes_needed)
                except ValueError:
                    log.warning(
                        "dispatcher: task %s has unknown lane in %r; skipping until resolved",
                        task.task_id,
                        lanes_needed,
                    )
                    continue
                if not self._lanes_fit(expanded, holders, capacities):
                    continue
                lease = await self.locks.try_acquire_many(
                    lanes_needed,
                    holder_id=task.task_id,
                    task_id=task.task_id,
                    action=task.kind,
                    ttl_sec=task.lease_ttl_sec or 60,
                )
                if lease is None:
                    # Race: another holder grabbed the lane; leave queued.
                    continue
                # Reflect the bump in our local view for the next task in this tick.
                for lane in lease.lanes:
                    holders[lane] = int(holders.get(lane, 0)) + 1
            else:
                lease = None
            gpu_lease = None
            gpu_specialist_lease: Any = None
            extra_context: dict[str, Any] = {}
            if task.kind == "specialist":
                params = task.params or {}
                needs_gpu = coerce_needs_gpu(params.get("needs_gpu", False))
                # Explicit wall-clock budget (lane-tiered base × macro_cycle, with a benchmark-profile floor and
                # capped by session time).
                extra_context["wall_budget_sec"] = self._specialist_wall_budget_sec(
                    needs_gpu=needs_gpu,
                    params=params,
                )
                extra_context["specialist_progress_cb"] = self._specialist_progress_publisher(task)
                if needs_gpu:
                    # Probe serving-slot state immediately before admitting this GPU specialist (not once per pass) so
                    # a serving start that races the pass does not slip through.
                    _immediate_pause = False
                    if _ray_serving_priority_enabled and _serving_slot_busy_fn is not None:
                        try:
                            _immediate_pause = _serving_slot_busy_fn()
                        except Exception:  # noqa: BLE001 — never block dispatch
                            _immediate_pause = False
                    if _immediate_pause:
                        # Serving is active — defer this GPU specialist to a later pass (keep it queued) rather than
                        # piling onto the contended GPU.
                        if lease is not None:
                            await self.locks.release(lease)
                            for lane in lease.lanes:
                                holders[lane] = max(0, int(holders.get(lane, 0)) - 1)
                        continue
                    # Whole-machine, time-shared lane vs serving-disjoint pool: framework-authoring and bench-capable
                    # specialists lease the whole machine from ``framework_gpu_pool``; every other GPU specialist
                    # leases from ``gpu_specialist_pool``.
                    from ..specialists.profile import (
                        holds_serving_slot,
                        uses_whole_machine_gpu_lane,
                    )

                    whole_machine_lane = uses_whole_machine_gpu_lane(params)
                    is_framework_authoring = bool(params.get("framework_agent_authoring"))
                    if whole_machine_lane:
                        gpu_pool = self.framework_gpu_pool
                        if is_framework_authoring:
                            # Default to the whole machine; explicit gpu_count wins.
                            default_gpu_count = gpu_pool.capacity or 1
                        else:
                            # Bench specialist: size to the serving TP.
                            default_gpu_count = self._resolve_serving_tp() or gpu_pool.capacity or 1
                    else:
                        gpu_pool = self.gpu_specialist_pool
                        # Default gpu_count to the serving TP; explicit wins.
                        default_gpu_count = self._resolve_serving_tp() or 1
                    try:
                        gpu_count = int(params.get("gpu_count", default_gpu_count) or default_gpu_count)
                    except (TypeError, ValueError):
                        gpu_count = default_gpu_count
                    # A bench-capable specialist floors gpu_count up to the serving TP; others keep their explicit
                    # count.
                    bench_raw = params.get("bench", False)
                    bench = (
                        bench_raw.strip().lower() in ("1", "true", "yes", "on")
                        if isinstance(bench_raw, str)
                        else bool(bench_raw)
                    )
                    serving_tp = self._resolve_serving_tp() or 0
                    if bench and serving_tp > 0 and gpu_count < serving_tp:
                        log.info(
                            "specialist %s: bench=true with gpu_count=%d < serving "
                            "TP=%d; flooring gpu_count to TP (a bench specialist "
                            "starts a real TP-sharded server and cannot run on "
                            "fewer cards).",
                            task.task_id,
                            gpu_count,
                            serving_tp,
                        )
                        gpu_count = serving_tp
                    # TTL re-sourced to the wall budget.
                    gpu_ttl_sec = self._gpu_lease_ttl_sec(
                        int(task.lease_ttl_sec or 0),
                        params=params,
                    )
                    # Under single-node Ray the physical GPU mutex is Ray's ``num_gpus``, not this SQLite pool.
                    from ..actions.executors._ray_backend import (
                        ray_gpu_pending_limit,
                        ray_gpu_specialist_exec_enabled,
                    )

                    if ray_gpu_specialist_exec_enabled():
                        gpu_lease = await gpu_pool.try_acquire_ray_observation(
                            holder_id=task.task_id,
                            task_id=task.task_id,
                            pending_limit=ray_gpu_pending_limit(),
                            ttl_sec=gpu_ttl_sec,
                        )
                    else:
                        gpu_lease = await gpu_pool.try_acquire(
                            count=gpu_count,
                            holder_id=task.task_id,
                            task_id=task.task_id,
                            ttl_sec=gpu_ttl_sec,
                        )
                    if gpu_lease is None:
                        if lease is not None:
                            await self.locks.release(lease)
                            for lane in lease.lanes:
                                holders[lane] = max(
                                    0,
                                    int(holders.get(lane, 0)) - 1,
                                )
                        continue
                    extra_context["gpu_ids"] = list(gpu_lease.gpu_ids)
                    # Ray-managed GPU execution: route the whole specialist subprocess into a ``num_gpus`` actor.
                    from ..actions.executors._ray_serving import (
                        maybe_gpu_specialist_lease,
                    )

                    # serving_slot: only bench-capable specialists that start their OWN serving loop hold the
                    # whole-machine serving_slot mutex.
                    gpu_specialist_lease = maybe_gpu_specialist_lease(
                        num_gpus=gpu_count,
                        serving_slot=holds_serving_slot(params),
                    )
                    if gpu_specialist_lease is not None:
                        extra_context["gpu_ids"] = list(range(gpu_count))
                        extra_context["gpu_specialist_lease"] = gpu_specialist_lease
            # Defensive audit (log-only): flag a queued task whose kind has no registered executor.
            try:
                _coord = object.__getattribute__(self, "_coord")
                _execs = getattr(getattr(_coord, "sub", None), "executor_registry", None)
                if (
                    isinstance(_execs, dict)
                    and _execs
                    and task.kind not in _execs
                    and task.kind != "specialist"
                    and task.kind not in KERNEL_AGENT_OWNED_ACTIONS
                ):
                    log.warning(
                        "dispatch audit: queued task_id=%s kind=%r has no registered executor (dispatch unchanged)",
                        task.task_id,
                        task.kind,
                    )
            except Exception:  # noqa: BLE001 - audit must never affect dispatch
                pass
            cancel_scope = CancelScope()
            atask = asyncio.create_task(
                self.run_task_registered(
                    task,
                    prebound_lease=lease,
                    extra_context=extra_context,
                    gpu_lease=gpu_lease,
                    gpu_specialist_lease=gpu_specialist_lease,
                    cancel_scope=cancel_scope,
                ),
            )
            self._inflight_actions[task.task_id] = _InflightAction(task.kind, atask, cancel_scope)
            if join_in_pump:
                spawned.append((task, atask, gpu_lease))
            else:
                # Nothing retrieves this handle's exception, so report it here.
                atask.add_done_callback(self._report_unjoined_failure(task))
                exclude_ids.add(task.task_id)
        return spawned

    @staticmethod
    def _report_unjoined_failure(task: Task) -> "Callable[[asyncio.Task[Any]], None]":
        """Build the done-callback for a handle no caller awaits."""

        def _report(atask: "asyncio.Task[Any]") -> None:
            # A cancelled task has no exception to read, and cancelling is how shutdown reaches it, so it is not a
            # failure worth reporting.
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

    async def run_task_registered(
        self,
        task: Task,
        *,
        prebound_lease: Any = None,
        extra_context: dict[str, Any] | None = None,
        gpu_lease: Any = None,
        gpu_specialist_lease: Any = None,
        cancel_scope: CancelScope | None = None,
    ) -> "SubAgentResult | None":
        """Run one task under the wall-clock defences, and hand back what it held."""
        lease = prebound_lease
        if lease is None and task.requires_lanes:
            lease = await self.locks.try_acquire_many(
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
        try:
            # Every LLM call this action makes, in-process or in a child, is labelled with its kind from here.
            with use_cancel_scope(cancel_scope), current_action_scope(task.kind):
                return await self.sub.run_task(
                    task,
                    prebound_lease=lease,
                    extra_context=extra_context,
                )
        finally:
            self._inflight_actions.pop(task.task_id, None)
            if gpu_lease is not None:
                try:
                    await self.gpu_specialist_pool.release(gpu_lease)
                except Exception:  # noqa: BLE001 — defensive cleanup; TTL backstops
                    log.exception(
                        "dispatcher: finally GPU-lease release failed for task=%s",
                        task.task_id,
                    )
            if gpu_specialist_lease is not None:
                try:
                    gpu_specialist_lease.close()
                except Exception:  # noqa: BLE001 — teardown must not raise
                    log.exception(
                        "dispatcher: finally GpuSpecialistLease close failed for task=%s",
                        task.task_id,
                    )

    def _specialist_progress_publisher(self, task: Task) -> Any:
        """Build the callback that turns partial checkpoints into observations."""

        async def _publish(payload: dict[str, Any], elapsed: float) -> None:
            """Append one specialist_progress observation."""
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

    def _specialist_wall_budget_sec(
        self,
        *,
        needs_gpu: bool,
        params: dict[str, Any] | None = None,
    ) -> float:
        """Compute the explicit wall-clock budget for a specialist task."""
        base_min = 60.0 if needs_gpu else 10.0
        macro_cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        budget_min = min(base_min * (macro_cycle + 1), 240.0)
        budget_sec = budget_min * 60.0
        from ..specialists.profile import resolve_specialist_profile
        from ..specialists.rebench import DEFAULT_REBENCH_TIMEOUT_SEC

        profile = resolve_specialist_profile(params or {})
        if profile.reserves_benchmark_lane:
            budget_sec = max(budget_sec, float(DEFAULT_REBENCH_TIMEOUT_SEC + 10 * 60))
        session_remaining_min = self.shared_state.remaining_minutes()
        if session_remaining_min is not None:
            budget_sec = min(budget_sec, session_remaining_min * 60.0)
        return max(0.0, budget_sec)

    def _resolve_serving_tp(self) -> int:
        """Resolve the live serving process's TP size (cards it holds)."""
        tp = int(getattr(self.shared_state, "tp", 0) or 0)
        if tp > 0:
            return tp
        try:
            return max(0, int(os.environ.get("TP", "0") or 0))
        except ValueError:
            return 0

    def _gpu_lease_ttl_sec(
        self,
        floor_ttl_sec: int = 0,
        *,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Single source for the GPU-specialist lease / ``gpu_research_lane`` TTL."""
        return max(
            int(floor_ttl_sec or 0),
            int(
                self._specialist_wall_budget_sec(
                    needs_gpu=True,
                    params=params,
                )
                * (1.0 + GPU_LEASE_TTL_GRACE)
            ),
        )

    async def _account_dead_holder_failures(
        self,
        task_ids: list[str],
        *,
        reason: str,
    ) -> None:
        """Charge a failure for each task reclaimed from a dead lease holder."""
        for task_id in task_ids:
            if task_id in self._dead_holder_accounted:
                continue
            self._dead_holder_accounted.add(task_id)
            try:
                task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — a missing row must not abort the pump
                log.exception(
                    "dispatcher: dead-holder accounting could not load task=%s",
                    task_id,
                )
                continue
            await self._handle_unpromotable_result(
                task,
                {
                    "status": "failed",
                    "error_class": "dead_holder_reaped",
                    "error": (f"lease holder process died before reporting a result; task reclaimed by {reason}"),
                },
            )
            log.warning(
                "dispatcher: counted dead-holder death of task=%s kind=%s as an action failure",
                task_id,
                task.kind,
            )

    async def _reap_dispatched_task(
        self,
        task: Task,
        maybe_result: Any,
        gpu_lease: Any,
    ) -> None:
        """Run completion bookkeeping for one finished dispatched task."""
        for (task, _, gpu_lease), maybe_result in zip(
            [(task, None, gpu_lease)],
            [maybe_result],
        ):
            if gpu_lease is not None:
                try:
                    await self.gpu_specialist_pool.release(gpu_lease)
                except Exception:  # noqa: BLE001 — defensive cleanup
                    log.exception(
                        "dispatcher: failed to release GPU specialist lease for task=%s",
                        task.task_id,
                    )
            if isinstance(maybe_result, asyncio.CancelledError):
                # Asked for, not gone wrong: the wall-clock defences stop in-flight actions on purpose.
                log.warning(
                    "dispatcher: in-flight action task=%s kind=%s was cancelled",
                    task.task_id,
                    task.kind,
                )
                continue
            if isinstance(maybe_result, BaseException):
                log.exception(
                    "dispatcher: spawned task %s raised: %r",
                    task.task_id,
                    maybe_result,
                )
                continue
            result: SubAgentResult = maybe_result
            # Bounded transient-failure auto-retry (infra only): on a subprocess timeout / crash / stale-heartbeat,
            # re-enqueue a fresh specialist task and skip this attempt's bookkeeping.
            if task.kind == "specialist":
                try:
                    if await self._maybe_auto_retry_specialist(task, result):
                        continue
                except Exception:  # noqa: BLE001 — never block the dispatch loop
                    log.exception(
                        "specialist auto-retry hook failed for task=%s",
                        task.task_id,
                    )
            if isinstance(result.result, dict):
                reauthor_attempt = (getattr(task, "params", None) or {}).get("reauthor_attempt")
                if reauthor_attempt not in (None, ""):
                    result.result.setdefault("reauthor_attempt", reauthor_attempt)
            try:
                await self.bus.append_and_seq(
                    Message.new(
                        "coordinator",
                        "*",
                        "delegated_result",
                        {
                            "task_id": task.task_id,
                            "kind": task.kind,
                            "state": result.state,
                            "result": result.result,
                            "error": result.error,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "dispatcher: failed to append delegated_result for task=%s",
                    task.task_id,
                )
                self._record_coordinator_exception(
                    stage="dispatcher_result",
                    exc=exc,
                )
                continue
            # Specialist bookkeeping: done payload under result.result['specialist_done']; always runs to keep the ledgers coherent.
            if task.kind == "specialist":
                result_dict = result.result if isinstance(result.result, dict) else {}
                done_payload = result_dict.get("specialist_done") or {}
                if isinstance(done_payload, dict):
                    try:
                        await self._record_specialist_result(
                            task=task,
                            done_payload=done_payload,
                            source=(f"{SPECIALIST_FROM_AGENT_PREFIX}{task.task_id}"),
                            run_error=str(result.error or ""),
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "specialist bookkeeping hook failed for task=%s",
                            task.task_id,
                        )
                    # FRAMEWORK authoring bridge for an EMPTY deliverable: a specialist that authored no patch never
                    # spawns an integrate_patch; stamp the terminal progress row here to avoid a pump livelock.
                    try:
                        self._record_framework_agent_authoring_empty_outcome(
                            task=task,
                            done_payload=done_payload,
                            run_error=str(result.error or ""),
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "FRAMEWORK authoring empty-outcome bridge failed for task=%s",
                            task.task_id,
                        )
                    # Harvest a discovery specialist's candidates into the source arm's batch.
                    try:
                        self._ingest_candidate_discovery(
                            task=task,
                            done_payload=done_payload,
                            run_error=str(result.error or ""),
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "FRAMEWORK: candidate discovery ingest failed for task=%s",
                            task.task_id,
                        )
            # intervention-mix ledger: log change_type for explore/integrate_patch.
            if task.kind in ("explore", "integrate_patch"):
                try:
                    self._record_intervention_for_task(task, result.result)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "intervention ledger update failed for task=%s",
                        task.task_id,
                    )
            # integrate_patch completion handling.
            if task.kind == "integrate_patch":
                # FRAMEWORK authoring bridge: record authored-patch KEEP/REVERT.
                if bool((getattr(task, "params", None) or {}).get("framework_agent_authoring")):
                    try:
                        self._record_framework_agent_authored_outcome(
                            task=task,
                            result=result,
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "FRAMEWORK authored-outcome bridge failed for task=%s",
                            task.task_id,
                        )
                # Unified rearm: handles enablement and apply_failed perf-lane results (schedules retry or stamps
                # terminal).
                res_dict = getattr(result, "result", None)
                try:
                    self._maybe_rearm_authored_lane(res_dict)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "AUTHORED_LANE rearm failed for task=%s",
                        task.task_id,
                    )
                # Drain pending apply-failure retries queued by _maybe_rearm_authored_lane.
                try:
                    await self._drain_apply_fail_retry_pending()
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "apply_fail retry drain failed for task=%s",
                        task.task_id,
                    )
            # Auto-promote succeeded results into CORE_STATE_FIELDS (Coordinator-only writer).
            result_payload = dict(result.result or {})
            replay_needs_cleanup = task.kind == "replay_warm_recipe" and result.state != "succeeded"
            if replay_needs_cleanup:
                result_payload.setdefault("status", "failed")
                result_payload.setdefault("error_class", "dispatch_failed")
                if result.error:
                    result_payload.setdefault("error", str(result.error))
            kept = (result.state == "succeeded" or replay_needs_cleanup) and self._is_promotable_result(
                task.kind, result_payload
            )
            try:
                if kept:
                    await self._promote_to_shared_state(
                        task.kind,
                        result_payload,
                        task=task,
                    )
                elif task.task_id not in self._dead_holder_accounted:
                    unpromotable_result = dict(result.result or {})
                    # Surface a PolicyGate dispatch rejection's specific rule (e.g.
                    # "policy_source_file_outside_trusted_scope") into the gap ledger instead of letting it default to
                    # "unknown_error" — result.result is {} for these (rejected before the executor ever ran), so
                    # error_class would otherwise be silently dropped here.
                    if result.error_class and not unpromotable_result.get("error_class"):
                        unpromotable_result["error_class"] = result.error_class
                    await self._handle_unpromotable_result(task, unpromotable_result)
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "dispatcher: promotion/unpromotable handling failed for task=%s",
                    task.task_id,
                )
                self._record_coordinator_exception(
                    stage="dispatcher_promote",
                    exc=exc,
                )
                continue
            # Fact-write hook: lands KEEP/REVERT in the journal + optional KB write. replay_warm_recipe is excluded
            # (verification, not a fact).
            if task.kind != "replay_warm_recipe":
                try:
                    await self._fact_write_hook(task=task, result=result, kept=kept)
                except Exception as exc:  # noqa: BLE001
                    log.exception(
                        "dispatcher: fact-write hook failed for task=%s",
                        task.task_id,
                    )
                    self._record_coordinator_exception(
                        stage="dispatcher_fact_write",
                        exc=exc,
                    )
            # explore-round gap update: append per-variant KEEP/REVERT, then re-run the global refresh.
            if task.kind == "explore":
                result_dict = result.result if isinstance(result.result, dict) else {}
                try:
                    self._record_explore_round_gaps(
                        task=task,
                        result=result_dict,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "gaps refresh: explore-round update failed for task=%s",
                        task.task_id,
                    )
                try:
                    self._record_explore_variant_failures(
                        task=task,
                        result=result_dict,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "explore: per-variant failure recording failed for task=%s",
                        task.task_id,
                    )
                try:
                    await self._refresh_gaps(reason="explore_round")
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "gaps refresh: _refresh_gaps after explore failed for task=%s",
                        task.task_id,
                    )

    @staticmethod
    def _lanes_fit(
        expanded_lanes: list[str],
        holders: dict[str, int],
        capacities: dict[str, int],
    ) -> bool:
        """Local-view headroom hint for the concurrent dispatcher (authoritative gate is try_acquire_many)."""
        for lane in expanded_lanes:
            cap = int(capacities.get(lane, 1))
            used = int(holders.get(lane, 0))
            if cap <= 0 or used >= cap:
                return False
        return True

    def _sequence_denial_for_action(
        self,
        action_name: str,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts before baseline. Only invariant: nothing runs until baseline_tput > 0 (a data-dependency)."""
        action = str(action_name or "").strip()
        sequence_actions = {
            "target_analysis",
            "baseline",
            "profile",
            "roofline",
            "sweep",
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

    def _time_budget_denial_for_action(
        self,
        action_name: str,
        *,
        fallback_cost_minutes: float | None = None,
    ) -> PolicyDenied | None:
        """Refuse an action whose expected cost cannot fit the remaining session budget."""
        action = str(action_name or "").strip()
        if not action or action in TIME_BUDGET_EXEMPT_ACTIONS:
            return None
        if self.shared_state.stop_reason:
            return None
        reg = getattr(self, "action_registry", None)
        meta = reg.get(action) if reg is not None else None
        if meta is None:
            if fallback_cost_minutes is None:
                return None
            expected_min = float(fallback_cost_minutes)
        else:
            expected_min = expected_action_cost_minutes(
                meta,
                measured_baseline_sec=measured_baseline_runtime_sec(self.shared_state),
            )
        usable_sec = self.shared_state.session_budget_usable_sec()
        if action_fits_time_budget(
            usable_sec=usable_sec,
            expected_cost_minutes=expected_min,
        ):
            return None
        remaining_min = (usable_sec or 0.0) / 60.0
        return PolicyDenied(
            f"action={action!r} denied: needs ~{expected_min:.0f} min but only "
            f"{remaining_min:.0f} min of the session budget is left",
            rule="time_budget",
            hint=(
                "the wall-clock budget cannot fit this action; delegate `report` "
                "to close the session, or pick an action that fits the time left"
            ),
        )

    def _admission_denial_for_action(
        self,
        action_name: str,
    ) -> PolicyDenied | None:
        """Run every pre-dispatch gate for an action name, first denial wins."""
        denied = self._sequence_denial_for_action(action_name)
        if denied is not None:
            return denied
        return self._time_budget_denial_for_action(action_name)

    async def _cancel_queued_task_over_budget(self, task: Task) -> bool:
        """Drop a queued task the budget can no longer fit, before it takes a lane."""
        # An uncatalogued kind is priced by the lease its enqueue sized, the only cost estimate it has.
        ttl_sec = int(getattr(task, "lease_ttl_sec", 0) or 0)
        denied = self._time_budget_denial_for_action(
            task.kind,
            fallback_cost_minutes=(ttl_sec / 60.0) if ttl_sec > 0 else None,
        )
        if denied is None:
            return False
        try:
            await self.tasks.transition(
                task.task_id,
                "cancelled",
                evidence={"reason": "time_budget", "error": str(denied)},
            )
        except Exception:  # noqa: BLE001 — a lost row must not abort the pump
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
        try:
            await self._record_observation(
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
        except Exception:  # noqa: BLE001 — observability must not block dispatch
            log.exception(
                "dispatcher: could not record time-budget denial for task=%s",
                task.task_id,
            )
        # A cancelled conc_sweep never writes last_conc_sweep on its own, so SWEEP would idle until the LLM emits
        # skip_to_close and CI would read that as robustness_escalated.
        if str(task.kind or "") == "conc_sweep":
            try:
                self._record_session_budget_conc_sweep_skip(denied=denied)
            except Exception:  # noqa: BLE001 — a stamp miss must not abort the pump
                log.exception(
                    "dispatcher: could not record conc_sweep time-budget skip for task=%s",
                    task.task_id,
                )
        return True

    def _sequence_denial_for_request(
        self,
        target_agent: str,
        kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip the baseline prerequisite (invariant: nothing kernel-side runs before baseline_tput > 0)."""
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
        return None

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        """Report whether GEMM tuning is disabled via the env escape hatch."""
        return os.environ.get(
            "INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Decide whether GEMM tuning must run before kernel_opt."""
        if self._skip_gemm_tuning():
            return False
        ss = self.shared_state
        precision = str(getattr(ss, "precision", "") or "").strip().lower()
        framework = str(getattr(ss, "framework", "") or "").strip().lower()

        from ..kernel.request_handlers import _resolve_gemm_tuning_backend

        backend = _resolve_gemm_tuning_backend({})

        if backend == "forge":
            # kernelforge gemm-tune handles any precision (bf16/fp16/fp8/fp4/mxfp4), dense or MoE, on sglang/vllm.
            eligible = framework in ("sglang", "vllm", "vllm-aiter")
        else:
            # GEAK: legacy FP8 + SGLang only.
            eligible = precision == "fp8" and framework == "sglang"

        if not eligible:
            return False
        last = getattr(ss, "last_gemm_tuning", {}) or {}
        status = str(last.get("status") or "").strip().lower()
        # The fp8 -> bf16 dense retry now runs inside a single gemm call (the tuner router selects the bf16 pass as a
        # fallback), so a completed run's status is terminal -- there is no pending second attempt to re-trigger.
        return status not in {
            "ok",
            "succeeded",
            "success",
            "complete",
            "completed",
            "skipped",
            "failed",
        }

    # Inline fast-action execution (folded in from the former InlineActionsCollaborator).
    def _inline_action_whitelist(self) -> frozenset[str]:
        """Derive the set of actions safe to run inline (A3): lane-light, registered executor, not in _INLINE_ACTION_DENY. PolicyGate remains the real security boundary."""
        coord = object.__getattribute__(self, "_coord")
        executors = getattr(coord.sub, "executor_registry", {}) or {}
        allowed: set[str] = set()
        for name in coord.action_registry:
            if name in self._INLINE_ACTION_DENY:
                continue
            if name not in executors:
                continue
            lanes, _ttl = self._registry_lanes_ttl(name)
            if lanes:
                continue
            allowed.add(name)
        return frozenset(allowed)

    def _run_action_now_sync(
        self,
        action_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Bridge callable for the ``run_action_now`` context tool (A3): marshals the executor coroutine onto the Coordinator loop and blocks with a timeout."""
        if not self._inline_fast_actions_enabled:
            return (
                "(run_action_now disabled: set "
                "INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS to a non-off value "
                "to enable; use emit_intent delegate for async execution)"
            )
        name = (action_name or "").strip()
        if not name:
            return "(run_action_now: action_name required)"
        whitelist = self._inline_action_whitelist()
        if name not in whitelist:
            return (
                f"(run_action_now: {name!r} is not inline-eligible — only "
                f"cheap, lane-light actions may run inline: "
                f"{sorted(whitelist)}. Use emit_intent delegate to run it "
                f"asynchronously.)"
            )
        loop = self._coordinator_loop
        if loop is None or loop.is_closed():
            return "(run_action_now unavailable: coordinator loop not running)"
        # Defensive audit (log-only): detect and log if this sync bridge is invoked on the coordinator loop thread.
        try:
            _running = asyncio.get_running_loop()
            if _running is loop:
                log.warning(
                    "run_action_now: invoked on the coordinator loop thread (action=%r)",
                    name,
                )
        except RuntimeError:
            pass
        except Exception:  # noqa: BLE001 - audit must never affect flow
            pass
        coro = self._run_action_now(name, dict(params or {}))
        # Cap inline wait under backend timeout so a slow action can't wedge the turn.
        try:
            timeout_s = float(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S",
                    "120",
                )
                or 120
            )
        except (TypeError, ValueError):
            timeout_s = 120.0
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            return f"(run_action_now: could not schedule on coordinator loop: {exc!r})"
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return (
                f"(run_action_now: {name!r} still running after "
                f"{timeout_s:.0f}s; it keeps running asynchronously — check "
                "get_recent_outcomes or the next-tick inbox for its "
                "delegated_result)"
            )
        except (FuturesCancelledError, asyncio.CancelledError):
            # The wall-clock defences stopped it on purpose.
            return (
                f"(run_action_now: {name!r} was cancelled — the session is shutting down or out of wall-clock budget)"
            )
        except Exception as exc:  # noqa: BLE001 — never crash the turn
            log.exception("run_action_now: inline run of %r failed", name)
            return f"(run_action_now: {name!r} errored: {exc!r})"

    async def _inline_action_denial(
        self,
        action_name: str,
        params: dict[str, Any],
    ) -> str | None:
        """Run the admission gates an inline action shares with a dispatched one."""
        intent = Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": action_name, "params": dict(params or {})},
        )
        try:
            self.policy.validate_intent("orchestration", intent)
        except PolicyDenied as denied:
            await self._record_policy_denied("orchestration", intent, denied)
            return (
                f"(run_action_now: {action_name!r} denied by policy: "
                f"{getattr(denied, 'rule', '')!s} — "
                f"{str(getattr(denied, 'hint', denied))[:200]})"
            )
        seq_denied = self._admission_denial_for_action(action_name)
        if seq_denied is None:
            return None
        await self._record_policy_denied(
            "orchestration",
            intent,
            seq_denied,
            action_name=action_name,
        )
        return f"(run_action_now: {action_name!r} denied: {str(getattr(seq_denied, 'hint', seq_denied))[:200]})"

    async def _run_action_now(
        self,
        action_name: str,
        params: dict[str, Any],
    ) -> str:
        """Coordinator-loop coroutine that runs a whitelisted action inline through PolicyGate + SubAgentRunner, publishing a delegated_result for audit/inbox parity."""
        denial = await self._inline_action_denial(action_name, params)
        if denial is not None:
            return denial
        lanes, ttl = self._registry_lanes_ttl(action_name)
        content_fp = hashlib.sha1(
            json.dumps(params or {}, sort_keys=True, default=str).encode(),
            usedforsecurity=False,
        ).hexdigest()[:10]
        key = f"inline:orchestration:{action_name}:t{int(self.shared_state.tick or 0)}:{content_fp}"
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=action_name,
            params=dict(params or {}),
            idempotency_key=key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing and task.state not in (
            "queued",
            "succeeded",
            "failed",
            "cancelled",
        ):
            return (
                f"(run_action_now: an identical {action_name!r} task is "
                f"already {task.state!r}; wait for its delegated_result)"
            )
        # This path abandons its future once the caller's inline wait elapses, so the handle registered below is the
        # only thing that can still stop the action.
        result = await self.run_task_registered(task)
        result_payload = {
            "task_id": task.task_id,
            "kind": task.kind,
            "state": result.state,
            "result": result.result,
            "error": result.error,
        }
        try:
            await self.bus.append_and_seq(
                Message.new(
                    "coordinator",
                    "*",
                    "delegated_result",
                    {**result_payload, "inline": True},
                )
            )
        except Exception:  # noqa: BLE001 — audit best-effort
            log.exception(
                "run_action_now: failed to append delegated_result for %s",
                task.task_id,
            )
        rendered = _format_inbox_event(
            Message.new(
                "coordinator",
                "orchestration",
                "delegated_result",
                result_payload,
            )
        )
        return f"inline run complete: {rendered}"


def _dispatch_policy_denied_evidence(task: Task) -> dict[str, Any]:
    """Return policy-denied evidence from a queued→cancelled dispatch transition."""
    for entry in reversed(task.history or []):
        if entry.get("from") != "queued" or entry.get("to") != "cancelled":
            continue
        evidence = entry.get("evidence") or {}
        if evidence.get("reason") == "policy_denied":
            return dict(evidence)
    return {}
