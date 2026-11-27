# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Admission control: the GPU-pool, phase and time-budget gates on an action, the baseline and time-budget gates on a kernel request."""

from __future__ import annotations

import sqlite3
from typing import Any, NamedTuple
from hyperloom_common.coerce import to_int
from hyperloom_common.env import is_truthy
from hyperloom.inference_optimizer.protocol.action_surfaces import REQUEST_KIND_TO_OWNED_ACTION
from ..bus.gpu_pool import SpecialistGpuPool
from ..phases import machine_state as _phase_state
from ..kernel.request_handlers import KERNEL_REQUEST_HANDLERS
from ..policy.denial import PolicyDenied
from ..policy.gate import SPECIALIST_ACTION_NAME
from ..policy.projection import gpu_specialist_ceiling, serving_tp_for_policy
from ..policy.specialist_in_flight import specialist_in_flight_denial
from ..specialists.profile import is_authoring_specialist, requires_gpu, uses_whole_machine_gpu_lane
from ._explore_grid import explore_grid_denial
from ._untested_backlog import untested_backlog_denial
from ..state.task_registry import IllegalTransition, Task, TaskNotFound
from .time_budget import (
    TIME_BUDGET_EXEMPT_ACTIONS,
    expected_action_cost_minutes,
    measured_baseline_runtime_sec,
    specialist_admission_minutes,
    time_budget_denial,
)
import logging as _logging

log = _logging.getLogger(__name__)

SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY = "specialist_gpu_request_exceeds_capacity"


class GpuRequest(NamedTuple):
    """The cards a GPU specialist leases.

    Attributes:
        pool: The pool they come from.
        count: How many it takes.
        whole_machine: Whether ``pool`` is the whole-machine pool rather than the serving-disjoint one.
    """

    pool: SpecialistGpuPool
    count: int
    whole_machine: bool


class AdmissionMixin:
    """Decides whether an action may be admitted, and cancels queued tasks the time budget no longer covers."""

    def specialist_gpu_request(self, params: dict[str, Any]) -> GpuRequest | None:
        """Size the GPU lease of a specialist dispatch; ``None`` when it needs no card.

        The one sizing: admission judges the request it returns and the dispatcher leases it.

        Args:
            params: The specialist dispatch ``params``.
        """
        if not requires_gpu(params):
            return None
        serving_tp = serving_tp_for_policy(self.shared_state)
        whole_machine = uses_whole_machine_gpu_lane(params)
        pool = self.framework_gpu_pool if whole_machine else self.gpu_specialist_pool
        if not whole_machine:
            default_count = serving_tp or 1
        elif is_authoring_specialist(params):
            default_count = pool.capacity or 1
        else:
            # Bench specialist: size to the serving TP.
            default_count = serving_tp or pool.capacity or 1
        count = to_int(params.get("gpu_count")) or default_count
        if is_truthy(params.get("bench")) and serving_tp > 0 and count < serving_tp:
            log.info(
                "specialist: bench=true with gpu_count=%d < serving TP=%d; flooring gpu_count to TP "
                "(a bench specialist starts a real TP-sharded server and cannot run on fewer cards).",
                count,
                serving_tp,
            )
            count = serving_tp
        return GpuRequest(pool, count, whole_machine)

    def _gpu_denial_for_action(self, action_name: str, params: dict[str, Any] | None) -> PolicyDenied | None:
        """Refuse a GPU specialist whose request its pool can never fund.

        The pool answers a busy acquire and an impossible one alike, so a task
        queued for the latter would wait for cards no release brings.

        Args:
            action_name: The delegated action name.
            params: The specialist dispatch ``params``.

        Returns:
            A :class:`PolicyDenied` when the sized request exceeds the pool it leases from, else ``None``.
        """
        if str(action_name or "").strip() != SPECIALIST_ACTION_NAME:
            return None
        request = self.specialist_gpu_request(dict(params or {}))
        if request is None or request.pool.can_fund(request.count):
            return None
        pool_name = "whole-machine GPU pool" if request.whole_machine else "serving-disjoint GPU specialist pool"
        sizes = (
            f"(configured capacity={gpu_specialist_ceiling(self.shared_state)}, "
            f"serving_tp={serving_tp_for_policy(self.shared_state)})"
        )
        if request.pool.capacity == 0:
            return PolicyDenied(
                f"delegate{{action='specialist'}}: the {pool_name} holds no card {sizes}",
                rule=SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY,
                hint=(
                    "No gpu_count fits an empty pool. Dispatch the work without a GPU request; the "
                    "'=== Resource pools ===' block shows which pool holds cards."
                ),
            )
        return PolicyDenied(
            f"delegate{{action='specialist'}}: effective gpu_count={request.count} cannot be leased from the "
            f"{pool_name} size={request.pool.capacity} {sizes}",
            rule=SPECIALIST_GPU_REQUEST_EXCEEDS_CAPACITY,
            hint=(
                f"Pass a gpu_count between 1 and {request.pool.capacity}; a bench specialist takes at least "
                "serving TP cards whatever it asks for."
            ),
        )

    def _phase_denial_for_action(self, action_name: str) -> PolicyDenied | None:
        """Refuse an action the current phase does not let an agent propose.

        Args:
            action_name: The delegated action name.

        Returns:
            A :class:`PolicyDenied` when the phase does not offer the action, else ``None``.
        """
        action = str(action_name or "").strip()
        phase = str(self.shared_state.phase or "").strip().upper()
        proposable = _phase_state.allowed_actions_for(phase)
        if action in proposable:
            return None
        return PolicyDenied(
            f"action={action!r} is not proposable while the phase is {phase!r}",
            rule="phase_incompatible",
            hint=f"{phase} proposes: {', '.join(proposable) or '(none)'}",
        )

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
            action_name: The delegated action name.
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
        self, action_name: str, *, params: dict[str, Any] | None = None
    ) -> PolicyDenied | None:
        """Run every pre-dispatch gate for an action name, first denial wins.

        The single entry point the intent handlers share, so a new gate
        reaches all paths at once.

        Args:
            action_name: The delegated action name.
            params: The proposal's parameters: a specialist's price, an explore's grid.

        Returns:
            The first :class:`PolicyDenied` that fires, else ``None``.
        """
        return (
            self._gpu_denial_for_action(action_name, params)
            or self._phase_denial_for_action(action_name)
            or specialist_in_flight_denial(action_name, params, self.tasks, self.shared_state.find_gap)
            or self.time_budget_denial_for_action(action_name, params=params)
            or untested_backlog_denial(action_name, params, self.shared_state, self.tasks)
            # Last: a session out of time is told so before an explore is told its grid is missing.
            or explore_grid_denial(action_name, params, self.shared_state)
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
        if req_kind not in KERNEL_REQUEST_HANDLERS:
            return None
        if self.shared_state.baseline_tput <= 0:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: baseline must run first",
                rule="execution_order",
                hint="delegate `baseline` before kernel requests",
            )
        return self.time_budget_denial_for_action(REQUEST_KIND_TO_OWNED_ACTION.get(req_kind, req_kind))
