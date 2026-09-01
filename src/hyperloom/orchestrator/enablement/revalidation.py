# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Genuine-baseline revalidation that finalises an eval-origin enablement KEEP."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from ..collaborator import CoordinatorCollaborator
from .params import _enablement_carrier_params

if TYPE_CHECKING:
    from ..state.task_registry import Task

import logging as _logging

log = _logging.getLogger(__name__)


class EnablementRevalidation(CoordinatorCollaborator):
    """Re-measures a kept enablement round against a real baseline."""

    async def _maybe_enqueue_enablement_baseline_revalidation(self) -> str:
        """Enqueue one genuine baseline to revalidate a KEEP'd eval-origin patch.

        Uses the accepted config from the KEEP'd candidate bench (preferred) or
        falls back to the original probe config, plus that bench's env/arg layers,
        which the YAML does not carry and without which a different configuration
        would be graded. The frozen eval controls from the carrier params ensure
        RUN_EVAL and eval task/limit match the trigger contract. Idempotent and
        one-at-a-time.
        """
        state = self.shared_state
        if not bool(state.enablement.validation_pending):
            return ""
        # If we already have a tracked revalidation task that is still alive, do
        # not create another one.
        tracked_tid = str(state.enablement.revalidation_task_id or "").strip()
        if tracked_tid:
            try:
                for t in (*await self.tasks.queued(), *await self.tasks.running()):
                    if str(getattr(t, "task_id", "") or "") == tracked_tid:
                        return tracked_tid
            except Exception:  # noqa: BLE001 — defensive
                pass
        # Do not open a row the dispatcher would cancel on sight. A revalidation
        # is a full baseline, and the queue scan drops a queued one the session
        # budget can no longer fit -- which leaves a cancelled row owning this
        # window's idempotency key, and a row cancelled at dispatch never
        # produces a result to route, so nothing would advance the generation
        # past it. Holding the window shut for now costs nothing: it stays open,
        # and the resume that has budget again enqueues it.
        denied = self._time_budget_denial_for_action("baseline")
        if denied is not None:
            log.info("ENABLEMENT revalidation: window held open, not enqueued -- %s", denied)
            return ""
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": ENABLEMENT_REVALIDATION_REASON,
            "disable_run_eval": False,
            **_enablement_carrier_params(state),
        }
        accepted_cfg = str(state.enablement.accepted_config_path or "").strip()
        probe_cfg = str(state.enablement.probe_config_path or "").strip()
        cfg = accepted_cfg or probe_cfg
        if cfg:
            params["config_path"] = cfg
        effective = state.enablement.accepted_config
        for key in ("extra_envs", "extra_server_args", "remove_args", "unset_envs", "args_mode"):
            if effective.get(key):
                params[key] = effective[key]
        # Carry the active runtime override so the revalidation baseline runs
        # under the same framework runtime as the KEEP'd candidate.
        active_rt = state.enablement.active_runtime or {}
        if isinstance(active_rt, dict) and active_rt:
            from ..framework.stack_actions import FrameworkRuntime

            rt_obj = FrameworkRuntime.from_state(active_rt)
            rt_override = rt_obj.to_runtime_override()
            if rt_override:
                params["runtime_override"] = rt_override
        task = await self._open_revalidation_row(params)
        if task is None:
            return ""
        task_id = str(getattr(task, "task_id", "") or "")
        # Persist the task_id so _promote_baseline can verify identity.
        if task_id and task_id != str(state.enablement.revalidation_task_id or ""):
            state.enablement.revalidation_task_id = task_id
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.debug("enablement revalidation: save of task_id failed", exc_info=True)
        return task_id

    async def _open_revalidation_row(self, params: dict[str, Any]) -> "Task | None":
        """Resolve this revalidation window's task row, on a generation it can use.

        Args:
            params: The baseline params for the revalidation row.

        Returns:
            The row to track, or ``None`` when this tick found only spent
            generations -- the window stays open and the next tick tries again.
        """
        state = self.shared_state
        _baseline_lanes, _baseline_ttl = self._registry_lanes_ttl("baseline")
        task, generation = await self._open_row_past_spent_generations(
            kind="baseline",
            params=params,
            key_for=lambda gen: f"enablement_revalidation:gen{gen}",
            generation=int(state.enablement.revalidation_generation or 0),
            label="revalidation",
            # Both halves of the catalogue contract: a baseline re-launches the
            # server, so it must hold the same lanes any other baseline does.
            requires_lanes=_baseline_lanes,
            lease_ttl_sec=_baseline_ttl,
        )
        state.enablement.revalidation_generation = generation
        return task

    async def _open_row_past_spent_generations(
        self,
        *,
        kind: str,
        params: dict[str, Any],
        key_for: Callable[[int], str],
        generation: int,
        label: str,
        attempts: int = 2,
        **create_kwargs: Any,
    ) -> tuple["Task | None", int]:
        """Create or re-use a task row, skipping generations already spent.

        A generation in the idempotency key is what lets one piece of work get a
        fresh row after an earlier attempt at it went terminal. That only holds if
        a key resolving to a terminal row is recognised as a spent generation
        rather than an enqueue: a row cancelled at dispatch -- which is what the
        queue scan does to work the wall-clock budget can no longer fit -- never
        produces a result to route, so nothing downstream advances the generation
        past it, and every later attempt resolves to a row that measured nothing.

        Args:
            kind: The task kind to create.
            params: The task params.
            key_for: Builds the idempotency key for a generation number.
            generation: The generation to try first.
            label: How this work is named in the log when a generation is spent.
            attempts: How many generations to try before giving up this pass.
            **create_kwargs: Passed through to
                :meth:`TaskRegistry.create_or_return_existing` (lanes, TTL).

        Returns:
            The row to use and the generation it sits on, or ``None`` with the
            generation to try next when every attempt this pass found a spent
            one. The caller persists the generation, since only the caller knows
            where it lives.
        """
        from ..state.task_registry import TERMINAL_STATES

        for _attempt in range(max(1, attempts)):
            task, _existing = await self.tasks.create_or_return_existing(
                kind=kind,
                params=params,
                idempotency_key=key_for(generation),
                **create_kwargs,
            )
            if str(getattr(task, "state", "") or "") not in TERMINAL_STATES:
                return task, generation
            log.warning(
                "ENABLEMENT %s: gen%d resolves to terminal task %s (%s); opening "
                "generation %d so the work is not stuck on it",
                label,
                generation,
                getattr(task, "task_id", ""),
                getattr(task, "state", ""),
                generation + 1,
            )
            generation += 1
        return None, generation
