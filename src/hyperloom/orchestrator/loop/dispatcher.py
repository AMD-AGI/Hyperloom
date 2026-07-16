# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import asyncio
import hashlib
import json
import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    KERNEL_AGENT_OWNED_ACTIONS,
)
from ..phases import machine_state as _phase_state
from ..bus.message_bus import Message
from ..kernel.request_handlers import get_handler
from ..policy.gate import (
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
from .coordinator_helpers import coerce_needs_gpu

from .coordinator import (
    _format_inbox_event,
)
import logging as _logging
log = _logging.getLogger(__name__)


class DispatcherCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _registry_lanes_ttl(self, kind: str) -> tuple[list[str], int]:
        """Resolve ``(requires_lanes, lease_ttl_sec)`` from the ActionRegistry; lanes filtered to KNOWN_LANES, returns ([], 0) for unknown actions.

        Args:
            kind: The action name to resolve.

        Returns:
            A ``(requires_lanes, lease_ttl_sec)`` tuple; ``([], 0)`` when the
            action is unknown or no registry is loaded.
        """
        reg = getattr(self, "action_registry", None)
        if reg is None:
            return [], 0
        meta = reg.get(kind)
        if meta is None:
            return [], 0
        lanes = [lane for lane in (getattr(meta, "requires_lanes", ()) or ()) if lane in KNOWN_LANES]
        return lanes, int(getattr(meta, "lease_ttl_sec", 0) or 0)

    def _cycle_idem_suffix(self) -> str:
        """Idempotency-key suffix scoping a per-cycle internal singleton to the
        current macro-cycle. Empty for cycle 0 / non-cyclic runs.

        Returns:
            ``"-c<cycle>"`` for macro-cycle > 0, else an empty string.
        """
        cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        return f"-c{cycle}" if cycle > 0 else ""

    async def _cursor_advance_to_latest(self, agent_name: str) -> None:
        """Advance an agent's read cursor to the latest message addressed to it.

        Args:
            agent_name (str): The agent whose inbox cursor to advance.
        """
        latest = await self.bus.tail(n=1, to_agent=agent_name)
        if latest:
            top = latest[0]
            await self.cursors.advance(agent_name, seq=top.seq, msg_id=top.msg_id)

    def _dispatch_paused_for_phase_budget(self) -> bool:
        """True when the current phase's cyclic budget is spent, so the dispatcher should stop launching NEW phase-scoped variants.

        Pausing new spawns lets in-flight tasks finish and the pump return so
        the tick can advance the phase. Scoped to cyclic long-runs and the
        discretionary search phases.

        Returns:
            ``True`` when new phase-scoped dispatch should pause for budget.
        """
        state = self.shared_state
        phase = (getattr(state, "phase", "") or "").upper()
        if phase not in self._BUDGET_GATED_DISPATCH_PHASES:
            return False
        try:
            if not _phase_state.is_cyclic_phases_enabled() or not _phase_state.is_long_run(state):
                return False
            remaining = _phase_state.phase_budget_remaining_seconds(
                state,
                budget_pct=self._phase_budget_pct,
            )
        except Exception:  # noqa: BLE001 — never let the guard wedge dispatch
            return False
        return remaining is not None and remaining <= 0.0

    async def _pump_dispatcher_once(self) -> None:
        """Dispatch queued tasks respecting per-lane capacity, re-scanning for
        newly-fittable tasks while in-flight tasks run.

        Re-scans the queue whenever an in-flight task completes
        (FIRST_COMPLETED) or a short poll elapses, so a queued GPU task starts
        the moment its lane frees. The pump still fully drains all currently
        dispatchable work before returning. Each GPU lease is bound to its
        task_id and released by the runner.

        Budget guard: once the phase's cyclic budget is spent
        (:meth:`_dispatch_paused_for_phase_budget`), stop spawning NEW
        phase-scoped variants — drain in-flight, then return so the tick can
        advance the phase.
        """
        # Dead-holder self-heal (runs every tick, before scanning the queue):
        # detect a crashed worker's dead PID so its leased lanes free and the
        # stuck task fails (retry-eligible) this same tick.
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
        try:
            await self.locks.reap_dead_holders()
        except Exception:  # noqa: BLE001
            log.exception("dispatcher: dead-holder lease reap failed")
        # TTL-expiry self-heal (runs every tick): covers tasks whose holder PID
        # was recycled or whose holder record is missing. Idempotent.
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
        inflight: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
        # Cumulative across the whole pump, not just the live in-flight set, so a
        # fast task reaped before its queued->running transition is visible is
        # not re-dispatched. A task is dispatched at most once per pump.
        dispatched_ids: set[str] = set()
        while True:
            # Budget guard: stop launching NEW phase-scoped variants once the
            # phase's cyclic budget is spent; drain in-flight then return.
            if not self._dispatch_paused_for_phase_budget():
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

    async def _spawn_fitting_queued(
        self,
        *,
        exclude_ids: set[str],
    ) -> list[tuple[Task, "asyncio.Task[SubAgentResult]", Any]]:
        """Spawn every currently lane-fitting queued task not already in flight.

        Returns the ``(task, asyncio_task, gpu_lease)`` tuples spawned this pass
        (possibly empty). Pure dispatch — per-task completion bookkeeping is
        handled by :meth:`_reap_dispatched_task`. Applies the capacity /
        GPU-specialist-lease gating; each lease is bound to its task_id.

        Args:
            exclude_ids: Task ids already dispatched this pump pass; skipped so
                a task is never dispatched twice.

        Returns:
            The ``(task, asyncio_task, gpu_lease)`` tuples spawned this pass
            (possibly empty).
        """
        queued = await self.tasks.queued()
        if not queued:
            return []
        holders = await self.locks.lane_holders()
        capacities = await self.locks.lane_capacities()
        spawned: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
        for task in queued:
            if task.task_id in exclude_ids:
                # Already dispatched in a prior pass of this pump.
                continue
            lanes_needed = list(task.requires_lanes or [])
            if lanes_needed:
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
            extra_context: dict[str, Any] = {}
            if task.kind == "specialist":
                params = task.params or {}
                needs_gpu = coerce_needs_gpu(params.get("needs_gpu", False))
                # Explicit wall-clock budget (lane-tiered base × macro_cycle,
                # capped at 4h).
                extra_context["wall_budget_sec"] = self._specialist_wall_budget_sec(
                    needs_gpu=needs_gpu,
                )
                if needs_gpu:
                    # Whole-machine, time-shared lane vs serving-disjoint pool:
                    # framework-authoring and bench-capable specialists lease the
                    # whole machine from ``framework_gpu_pool``; every other GPU
                    # specialist leases from ``gpu_specialist_pool``.
                    from ..specialists.profile import (
                        uses_whole_machine_gpu_lane,
                    )

                    whole_machine_lane = uses_whole_machine_gpu_lane(params)
                    is_framework_authoring = bool(
                        params.get("framework_agent_authoring")
                    )
                    if whole_machine_lane:
                        gpu_pool = self.framework_gpu_pool
                        if is_framework_authoring:
                            # Default to the whole machine; explicit gpu_count wins.
                            default_gpu_count = gpu_pool.capacity or 1
                        else:
                            # Bench specialist: size to the serving TP.
                            default_gpu_count = (
                                self._resolve_serving_tp() or gpu_pool.capacity or 1
                            )
                    else:
                        gpu_pool = self.gpu_specialist_pool
                        # Default gpu_count to the serving TP; explicit wins.
                        default_gpu_count = self._resolve_serving_tp() or 1
                    try:
                        gpu_count = int(params.get("gpu_count", default_gpu_count) or default_gpu_count)
                    except (TypeError, ValueError):
                        gpu_count = default_gpu_count
                    # A bench-capable specialist floors gpu_count up to the
                    # serving TP; others keep their explicit count.
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
                    # TTL re-sourced to the wall budget. Iron law:
                    # kill <= gpu_lease TTL <= gpu_research_lane TTL. Both TTLs
                    # come from ``_gpu_lease_ttl_sec`` so they never drift apart.
                    gpu_ttl_sec = self._gpu_lease_ttl_sec(int(task.lease_ttl_sec or 0))
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
            # Defensive audit (log-only): flag a queued task whose kind has
            # no registered executor. Kernel-owned kinds are legitimately
            # unregistered under --no-kernel, so they are excluded to avoid a
            # false positive. Dispatch is unchanged.
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
                        "dispatch audit: queued task_id=%s kind=%r "
                        "has no registered executor (dispatch unchanged)",
                        task.task_id, task.kind,
                    )
            except Exception:  # noqa: BLE001 - audit must never affect dispatch
                pass
            spawned.append(
                (
                    task,
                    asyncio.create_task(
                        self._run_dispatched_with_gpu_release(
                            task,
                            prebound_lease=lease,
                            extra_context=extra_context,
                            gpu_lease=gpu_lease,
                        ),
                    ),
                    gpu_lease,
                )
            )
        return spawned

    async def _run_dispatched_with_gpu_release(
        self,
        task: Task,
        *,
        prebound_lease: Any,
        extra_context: dict[str, Any],
        gpu_lease: Any,
    ) -> "SubAgentResult":
        """Run a dispatched task, releasing its GPU lease in a structured finally.

        Binding the GPU-lease release to the asyncio task's own lifecycle
        guarantees the cards are freed on completion, error, or cancellation
        even if the pump coroutine is cancelled or the reap never runs.
        ``release`` is idempotent, so the release in
        :meth:`_reap_dispatched_task` remains harmless.

        Args:
            task: The dispatched task.
            prebound_lease: The already-acquired resource-lane lease (or None).
            extra_context: Per-task context (wall budget, gpu ids, …).
            gpu_lease: The GPU specialist lease to release, or None.

        Returns:
            SubAgentResult: The result from ``sub.run_task``.
        """
        try:
            return await self.sub.run_task(
                task,
                prebound_lease=prebound_lease,
                extra_context=extra_context,
            )
        finally:
            if gpu_lease is not None:
                try:
                    await self.gpu_specialist_pool.release(gpu_lease)
                except Exception:  # noqa: BLE001 — defensive cleanup; TTL backstops
                    log.exception(
                        "dispatcher: finally GPU-lease release failed for task=%s",
                        task.task_id,
                    )

    def _specialist_wall_budget_sec(self, *, needs_gpu: bool) -> float:
        """Compute the explicit wall-clock budget for a specialist task.

        The budget is a lane-tiered base (cpu 10min / gpu 60min) amplified by the
        macro-cycle count and hard-capped at 4h::

            budget_min = min(base × (macro_cycle + 1), 240)

        ``macro_cycle`` only grows on long/unbounded runs (``is_long_run`` >=24h
        gate), so <24h bounded runs always get the base value (cpu 10 / gpu 60)
        and never degrade.

        Args:
            needs_gpu: Whether the specialist holds a GPU lease (selects the
                60min GPU lane base vs the 10min cpu base).

        Returns:
            float: The wall-clock budget in seconds.
        """
        base_min = 60.0 if needs_gpu else 10.0
        macro_cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        budget_min = min(base_min * (macro_cycle + 1), 240.0)
        return budget_min * 60.0

    def _resolve_serving_tp(self) -> int:
        """Resolve the live serving process's TP size (cards it holds).

        Used for the serving-disjoint specialist pool (B1) and as the default
        ``gpu_count`` for TP-coupled GPU specialists (B2). Prefers the
        resume-safe ``shared_state.tp``; falls back to the ``TP`` env the CLI
        exports before construction. Returns ``0`` when neither is set (the
        legacy whole-pool / single-card behaviour).

        Returns:
            int: The serving TP size, or ``0`` when unknown.
        """
        tp = int(getattr(self.shared_state, "tp", 0) or 0)
        if tp > 0:
            return tp
        try:
            return max(0, int(os.environ.get("TP", "0") or 0))
        except ValueError:
            return 0

    def _gpu_lease_ttl_sec(self, floor_ttl_sec: int = 0) -> int:
        """Single source for the GPU-specialist lease / ``gpu_research_lane`` TTL.

        The iron law is ``kill ≤ gpu_lease TTL ≤ gpu_research_lane TTL`` — both the
        GPU-pool lease (dispatch) and the lane lease (intent_router) must outlive
        the agent's WS1 wall-budget kill, so both are sourced from the same
        ``wall_budget × (1 + GPU_LEASE_TTL_GRACE)`` here to keep them from
        drifting apart.

        Args:
            floor_ttl_sec: A lower bound (e.g. the registry / existing
                ``lease_ttl_sec``) the computed TTL is raised to.

        Returns:
            int: ``max(floor_ttl_sec, wall_budget × (1 + grace))``.
        """
        return max(
            int(floor_ttl_sec or 0),
            int(self._specialist_wall_budget_sec(needs_gpu=True) * (1.0 + GPU_LEASE_TTL_GRACE)),
        )

    async def _reap_dispatched_task(
        self,
        task: Task,
        maybe_result: Any,
        gpu_lease: Any,
    ) -> None:
        """Run completion bookkeeping for one finished dispatched task.

        Performs per-task post-completion handling (GPU-lease
        release, specialist auto-retry, ``delegated_result`` emission, ledgers,
        shared-state promotion, fact-write, explore-gap refresh).

        Args:
            task: The finished dispatched task.
            maybe_result: The task's result, or the exception it raised.
            gpu_lease: The GPU specialist lease to release, or ``None``.
        """
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
            if isinstance(maybe_result, BaseException):
                log.exception(
                    "dispatcher: spawned task %s raised: %r",
                    task.task_id,
                    maybe_result,
                )
                continue
            result: SubAgentResult = maybe_result
            # Bounded transient-failure auto-retry (infra only): on a subprocess
            # timeout / crash / stale-heartbeat, re-enqueue a fresh specialist
            # task and skip this attempt's bookkeeping. Semantic empties fall
            # through and are recorded.
            if task.kind == "specialist":
                try:
                    if await self._maybe_auto_retry_specialist(task, result):
                        continue
                except Exception:  # noqa: BLE001 — never block the dispatch loop
                    log.exception(
                        "specialist auto-retry hook failed for task=%s",
                        task.task_id,
                    )
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
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "specialist bookkeeping hook failed for task=%s",
                            task.task_id,
                        )
                    # FRAMEWORK authoring bridge for an EMPTY deliverable: a
                    # specialist that authored no patch never spawns an
                    # integrate_patch; stamp the terminal progress row here to
                    # avoid a pump livelock.
                    try:
                        self._record_framework_agent_authoring_empty_outcome(
                            task=task,
                            done_payload=done_payload,
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "FRAMEWORK authoring empty-outcome bridge failed for task=%s",
                            task.task_id,
                        )
                    # FRAMEWORK config-exploration: harvest a generation
                    # specialist's config proposal_set into the pending grid.
                    try:
                        self._ingest_framework_config_generation(
                            task=task,
                            done_payload=done_payload,
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "framework_config: generation ingest failed for task=%s",
                            task.task_id,
                        )
                # Bump the per-EXPLORE specialist dispatch counter.
                try:
                    self.shared_state.bump_specialist_dispatched()
                except Exception:  # noqa: BLE001
                    log.exception("bump_specialist_dispatched failed")
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
                if (
                    getattr(
                        self.shared_state,
                        "framework_agent_authoring_enabled",
                        False,
                    )
                    and (self.shared_state.phase or "").strip().upper() == _phase_state.PHASE_FRAMEWORK_AGENT
                ):
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
                # Unified rearm: handles enablement and apply_failed perf-lane
                # results (schedules retry or stamps terminal).
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
            kept = result.state == "succeeded" and self._is_promotable_result(task.kind, result.result or {})
            try:
                if kept:
                    await self._promote_to_shared_state(
                        task.kind,
                        result.result,
                        task=task,
                    )
                else:
                    await self._handle_unpromotable_result(task, result.result)
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
            # Fact-write hook: lands KEEP/REVERT in the journal + optional KB
            # write. replay_warm_recipe is excluded (verification, not a fact).
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
            # Framework prs_tested write-back: record KEEP/REVERT patches.
            if task.kind == "framework_agent":
                try:
                    self._write_prs_tested_from_framework_agent(task=task, result=result, kept=kept)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "dispatcher: prs_tested write-back failed for task=%s",
                        task.task_id,
                    )
                    continue
            # explore-round gap update: append per-variant KEEP/REVERT, then re-run the global refresh.
            if task.kind == "explore":
                result_dict = result.result if isinstance(result.result, dict) else {}
                if str((task.params or {}).get("source") or "") == "framework_config_exploration":
                    try:
                        self._record_framework_config_exploration_result(
                            task=task,
                            result=result_dict,
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "framework_config: result bookkeeping failed for task=%s",
                            task.task_id,
                        )
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
        """Local-view headroom hint for the concurrent dispatcher (authoritative gate is try_acquire_many).

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

    def _sequence_denial_for_request(
        self,
        target_agent: str,
        kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip the baseline prerequisite (invariant: nothing kernel-side runs before baseline_tput > 0).

        Args:
            target_agent: The request's target agent; only ``"kernel_agent"`` is
                gated.
            kind: The kernel request kind; ``trace_analyze`` and unknown kinds
                are exempt.

        Returns:
            A :class:`PolicyDenied` when the kernel request must wait for
            baseline, else ``None``.
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
        return None

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        """Report whether GEMM tuning is disabled via the env escape hatch.

        Returns:
            bool: ``True`` when ``INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING`` is set.
        """
        return os.environ.get(
            "INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Decide whether GEMM tuning must run before kernel_opt.

        When using forge-gemm-tune backend: eligible for any framework
        (sglang/vllm) and any precision with a MoE model or FP8 dense.
        When using GEAK backend: only FP8 + SGLang (legacy behavior).

        Returns:
            bool: ``True`` when GEMM tuning should run before source-level
                ``kernel_opt``.
        """
        if self._skip_gemm_tuning():
            return False
        ss = self.shared_state
        precision = str(getattr(ss, "precision", "") or "").strip().lower()
        framework = str(getattr(ss, "framework", "") or "").strip().lower()

        from ..kernel.request_handlers import _resolve_gemm_tuning_backend

        backend = _resolve_gemm_tuning_backend({})

        if backend == "forge":
            # forge-gemm-tune handles any precision (bf16/fp16/fp8/fp4/mxfp4),
            # dense or MoE, on sglang/vllm. Real e2e KEEPs span all of these —
            # including bf16 *dense* (+11.1%) — so we must NOT pre-filter on
            # precision/MoE here, or a category that can optimize gets silently
            # blocked. Gate only on a supported framework and let forge itself
            # return no_improvement when a shape can't be beaten.
            eligible = framework in ("sglang", "vllm", "vllm-aiter")
        else:
            # GEAK: legacy FP8 + SGLang only.
            eligible = (precision == "fp8" and framework == "sglang")

        if not eligible:
            return False
        last = getattr(ss, "last_gemm_tuning", {}) or {}
        status = str(last.get("status") or "").strip().lower()
        if self._bf16_dense_gemm_fallback_pending():
            return True
        return status not in {
            "ok",
            "succeeded",
            "success",
            "complete",
            "completed",
            "skipped",
            "failed",
        }

    # Inline fast-action execution (folded in from the former
    # InlineActionsCollaborator). ``_run_action_now_sync`` is the ``run_action_now``
    # context-tool bridge used by ConversationCollaborator.
    def _inline_action_whitelist(self) -> frozenset[str]:
        """Derive the set of actions safe to run inline (A3): lane-light, registered executor, not in _INLINE_ACTION_DENY. PolicyGate remains the real security boundary.

        Returns:
            A frozenset of action names eligible for inline execution; empty
            when no action registry is loaded.
        """
        coord = object.__getattribute__(self, "_coord")
        reg = getattr(coord, "action_registry", None)
        if reg is None:
            return frozenset()
        executors = getattr(coord.sub, "executor_registry", {}) or {}
        names_fn = getattr(reg, "names", None)
        try:
            if callable(names_fn):
                names = list(names_fn())
            else:
                all_fn = getattr(reg, "all", None)
                metas = list(all_fn()) if callable(all_fn) else []
                names = [str(getattr(meta, "name", "") or "") for meta in metas]
        except Exception:  # noqa: BLE001 — defensive
            names = []
        allowed: set[str] = set()
        for name in names:
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
        """Bridge callable for the ``run_action_now`` context tool (A3): marshals the executor coroutine onto the Coordinator loop and blocks with a timeout.

        Args:
            action_name: Name of the action to run inline; must be
                inline-eligible per :meth:`_inline_action_whitelist`.
            params: Optional parameter mapping forwarded to the executor.

        Returns:
            A human-readable status string describing the inline run outcome,
            disablement, ineligibility, timeout, or error.
        """
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
        # Defensive audit (log-only): detect and log if this sync bridge is
        # invoked on the coordinator loop thread. Behaviour is unchanged.
        try:
            _running = asyncio.get_running_loop()
            if _running is loop:
                log.warning(
                    "run_action_now: invoked on the coordinator "
                    "loop thread (action=%r)",
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
        except Exception as exc:  # noqa: BLE001 — never crash the turn
            log.exception("run_action_now: inline run of %r failed", name)
            return f"(run_action_now: {name!r} errored: {exc!r})"

    async def _run_action_now(
        self,
        action_name: str,
        params: dict[str, Any],
    ) -> str:
        """Coordinator-loop coroutine that runs a whitelisted action inline through PolicyGate + SubAgentRunner, publishing a delegated_result for audit/inbox parity.

        Args:
            action_name: Name of the action to execute.
            params: Parameter mapping forwarded to the task/executor.

        Returns:
            A status string: a policy/sequence denial message, an
            already-in-flight notice, or the rendered delegated_result line.
        """

        # PolicyGate parity: validate synthetic delegate intent so phase/role/paths/red-line gates apply.
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
        seq_denied = self._sequence_denial_for_action(action_name)
        if seq_denied is not None:
            await self._record_policy_denied(
                "orchestration",
                intent,
                seq_denied,
                action_name=action_name,
            )
            return f"(run_action_now: {action_name!r} denied: {str(getattr(seq_denied, 'hint', seq_denied))[:200]})"
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
            "needs_manual_review",
        ):
            return (
                f"(run_action_now: an identical {action_name!r} task is "
                f"already {task.state!r}; wait for its delegated_result)"
            )
        result = await self.sub.run_task(task)
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
