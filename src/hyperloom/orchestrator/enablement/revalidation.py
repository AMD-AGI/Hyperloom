# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Genuine-baseline revalidation that finalises an eval-origin enablement KEEP."""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.breakdown.recorder import enablement_event

from ..actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from ..collaborator import CoordinatorCollaborator
from ..state.round_store import RoundRequest
from ..state.task_registry import TaskSpec, TerminalTaskReuse, create_in_cursor, task_dispatch_origin
from .params import _enablement_carrier_params

if TYPE_CHECKING:
    from ..state.task_registry import Task

import logging as _logging

log = _logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationScan:
    """Which idempotency-key generations a row opening tries.

    Attributes:
        key_for: Builds the idempotency key for a generation number.
        generation: The generation to try first.
        attempts: How many generations to try before giving up this pass.
    """

    key_for: Callable[[int], str]
    generation: int
    attempts: int = 2


class _RowAlreadyLive(Exception):
    """The generation's key already names a live row, so no round is taken."""


class EnablementRevalidation(CoordinatorCollaborator):
    """Re-measures a kept enablement round against a real baseline."""

    async def maybe_enqueue_enablement_baseline_revalidation(self) -> str:
        """Enqueue one genuine baseline to revalidate a KEEP'd eval-origin patch."""
        state = self.shared_state
        if not bool(state.enablement.validation_pending):
            return ""
        # If we already have a tracked revalidation task that is still alive, do not create another one.
        tracked_tid = str(state.enablement.revalidation_task_id or "").strip()
        if tracked_tid:
            for t in (*await self.tasks.queued(), *await self.tasks.running()):
                if str(getattr(t, "task_id", "") or "") == tracked_tid:
                    return tracked_tid
        # Do not open a row the dispatcher would cancel on sight.
        denied = self._coord.dispatcher.time_budget_denial_for_action("baseline")
        if denied is not None:
            log.info("ENABLEMENT revalidation: window held open, not enqueued -- %s", denied)
            return ""
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": ENABLEMENT_REVALIDATION_REASON,
            "disable_run_eval": False,
            **_enablement_carrier_params(state),
        }
        benchmark_script = state.accepted_baseline_script()
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        params.update(self.baseline_launch_params())
        task_id = await self._open_revalidation_row(params)
        if not task_id:
            return ""
        enablement_event.record_revalidation(
            generation=int(state.enablement.revalidation_generation or 0),
            task_id=task_id,
            config_path=params.get("config_path", ""),
        )
        # Persist the task_id so _promote_baseline can verify identity.
        if task_id and task_id != str(state.enablement.revalidation_task_id or ""):
            state.enablement.revalidation_task_id = task_id
            try:
                state.save(self.session_dir)
            except Exception:
                log.debug("enablement revalidation: save of task_id failed", exc_info=True)
        return task_id

    def baseline_launch_params(self) -> dict[str, Any]:
        """The config, launch controls and framework runtime every baseline-arm launch boots under."""
        enablement = self.shared_state.enablement
        params: dict[str, Any] = {}
        cfg = enablement.accepted_config_path or enablement.probe_config_path
        if cfg:
            params["config_path"] = cfg
        for key in ("extra_envs", "extra_server_args", "remove_args", "unset_envs", "args_mode"):
            if enablement.accepted_config.get(key):
                params[key] = enablement.accepted_config[key]
        if enablement.active_runtime:
            from .runtime.stack_actions import FrameworkRuntime

            rt_override = FrameworkRuntime.from_state(enablement.active_runtime).to_runtime_override()
            if rt_override:
                params["runtime_override"] = rt_override
        return params

    async def _open_revalidation_row(self, params: dict[str, Any]) -> str:
        """Open this revalidation window's round, on a generation it can use.

        Args:
            params: The baseline params for the revalidation row.

        Returns:
            str: The holder task id, or ``""`` when this tick found only spent
            generations or a machine something else still holds -- the window
            stays open and the next tick tries again.
        """
        state = self.shared_state
        baseline_lanes, baseline_ttl = self._coord.dispatcher.registry_lanes_ttl("baseline")
        task_id, generation = await self._open_round_past_spent_generations(
            params=params,
            scan=GenerationScan(
                key_for=lambda gen: f"enablement_revalidation:gen{gen}",
                generation=int(state.enablement.revalidation_generation or 0),
            ),
            # Both halves of the catalogue contract: a baseline re-launches the
            # server, so it must hold the same lanes any other baseline does.
            requires_lanes=baseline_lanes,
            lease_ttl_sec=int(baseline_ttl),
        )
        state.enablement.revalidation_generation = generation
        return task_id

    async def open_row_past_spent_generations(
        self,
        *,
        kind: str,
        params: dict[str, Any],
        scan: GenerationScan,
        label: str,
        **create_kwargs: Any,
    ) -> tuple["Task | None", int]:
        """Create or re-use a task row, skipping generations already spent."""
        from ..state.task_registry import TERMINAL_STATES

        generation = scan.generation
        for _attempt in range(max(1, scan.attempts)):
            task, _existing = await self.tasks.create_or_return_existing(
                kind=kind,
                params=params,
                idempotency_key=scan.key_for(generation),
                **create_kwargs,
                dispatch_class="coordinator",
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

    async def _open_round_past_spent_generations(
        self,
        *,
        params: dict[str, Any],
        scan: GenerationScan,
        requires_lanes: list[str],
        lease_ttl_sec: int,
    ) -> tuple[str, int]:
        """Acquire a bring-up round for a baseline row, skipping spent generations.

        A revalidation launches the server again, so it takes the machine the
        same way an authoring round does. The row and the acquire commit
        together, so the store names this task the round's holder and the round
        it opened is not read as one standing in its way.

        A generation in the idempotency key is what lets one piece of work get a
        fresh row after an earlier attempt went terminal, so a key resolving to a
        terminal row is recognised as a spent generation rather than an enqueue.

        Args:
            params: The baseline task params.
            scan: The generations to try, from the first one.
            requires_lanes: Lanes the baseline must hold while running.
            lease_ttl_sec: The baseline's lease, and the round's first one.

        Returns:
            tuple[str, int]: The holder task id and the generation it sits on,
            or ``""`` with the generation to try next. The caller persists the
            generation.
        """
        generation = scan.generation
        for _attempt in range(max(1, scan.attempts)):
            key = scan.key_for(generation)
            holder = uuid.uuid4().hex

            def _join(cur: sqlite3.Cursor, _key: str = key, _holder: str = holder) -> None:
                _task, existing = create_in_cursor(
                    cur,
                    TaskSpec(
                        kind="baseline",
                        params=params,
                        idempotency_key=_key,
                        requires_lanes=requires_lanes,
                        lease_ttl_sec=lease_ttl_sec,
                        task_id=_holder,
                        dispatch_class="coordinator",
                        dispatch_origin=task_dispatch_origin(self.shared_state),
                    ),
                )
                if existing:
                    # A live row already runs this generation, under the round
                    # it opened.
                    raise _RowAlreadyLive(_key)

            try:
                acquired = await self.rounds.open(
                    f"revalidation-{holder}",
                    RoundRequest(now_unix=time.time(), request_id=key, evidence={"idempotency_key": key}),
                    holder_task_id=holder,
                    lease_sec=float(lease_ttl_sec),
                    join=_join,
                )
            except _RowAlreadyLive:
                return "", generation
            except TerminalTaskReuse as exc:
                log.warning(
                    "ENABLEMENT revalidation: gen%d is spent (%s); opening generation %d",
                    generation,
                    exc,
                    generation + 1,
                )
                generation += 1
                continue
            if not acquired.ok:
                log.info("ENABLEMENT revalidation: round not opened (%s)", acquired.reason)
                return "", generation
            return holder, generation
        return "", generation
