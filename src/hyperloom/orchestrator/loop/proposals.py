# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Queuing the task an action or a specialist's deliverable asks for, and the framework record of a grid."""

from __future__ import annotations
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping
from hyperloom_common.coerce import to_float
from hyperloom.orchestrator.lever import LEVER_CONFIG
from ..actions.executors.benchmark_result import is_valid_measurement
from ..actions.executors.integrate._integrate_vocabulary import RESULT_BENCHED
from ..phases import machine_state as _phase_state
from ..policy.gate import SPECIALIST_ACTION_NAME
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import (
    TERMINAL_STATES,
    Task,
    TaskSpec,
    UnspentKey,
    confirmed_result,
    first_unspent_key,
)
from ..collaborator import CoordinatorCollaborator

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import FrameworkEventRecorder

import logging as _logging

log = _logging.getLogger(__name__)


def record_config_proposal(
    recorder: FrameworkEventRecorder | None,
    proposal_id: str,
    action_name: str,
    payload: Mapping[str, Any],
    *,
    outcome: str,
) -> None:
    """Record one config-arm grid on the framework event ``recorder`` writes, as its task is queued; ``None`` when no
    FRAMEWORK entry is open.

    One row per grid, not per variant: the measured attempts point back at the grid
    through their ``proposal_ref``.
    """
    if not proposal_id or recorder is None:
        return
    kb_read_id = str(payload.get("kb_read_id") or "")
    rendered_refs = payload.get("kb_rendered_refs") or []
    if kb_read_id or rendered_refs:
        recorder.record_proposal(proposal_id, kb_read_id=kb_read_id, rendered_refs=rendered_refs)
    if action_name != "explore":
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        ARM_CONFIG,
        PRODUCER_ORCHESTRATION,
        STEP_PROPOSED,
        producer_for_provenance,
    )

    params = payload.get("params") or {}
    grid = [row for row in (params.get("grid") or []) if isinstance(row, dict)]
    labels = {str(row.get("provenance") or "").strip() for row in grid}
    if len(labels) == 1:
        producer, producer_ref = producer_for_provenance(next(iter(labels)))
    else:
        # A grid mixing provenances was assembled by the orchestration agent.
        # Each variant keeps its own label on its attempt, so naming the
        # assembler here loses nothing.
        producer, producer_ref = PRODUCER_ORCHESTRATION, ""
    scopes = {str(row.get("scope") or "").strip() for row in grid if str(row.get("scope") or "").strip()}
    recorder.record_proposal(
        proposal_id,
        arm=ARM_CONFIG,
        producer=producer,
        producer_ref=producer_ref,
        lever_kind=LEVER_CONFIG,
        scope=scopes.pop() if len(scopes) == 1 else "",
        kb_read_id=kb_read_id,
        rendered_refs=rendered_refs,
    )
    recorder.record_proposal_step(proposal_id, step=STEP_PROPOSED, outcome=outcome)


def integrate_patch_key(specialist_task_id: str) -> str:
    """The idempotency key of the ``integrate_patch`` that measures one specialist's deliverable, whoever asks for it."""
    return f"integrate_patch:{specialist_task_id}"


def twin_answers(action_name: str, twin: Task) -> bool:
    """Whether an existing task under an action's key answers a repeat of it, rather than leaving a retry key to it."""
    if twin.state not in TERMINAL_STATES:
        return True
    # A twin that measured answers a repeat; the row state is ``succeeded`` for any result.
    outcome = confirmed_result(twin)
    if action_name == "baseline":
        return is_valid_measurement(outcome)
    return action_name == "integrate_patch" and bool((outcome or {}).get(RESULT_BENCHED))


class ProposalsCollaborator(CoordinatorCollaborator):
    """Queues the task a delegated action or a finished specialist's deliverable asks for."""

    def action_key(self, action_name: str, params: dict[str, Any]) -> str:
        """One key per piece of work: a specialist's deliverable, a baseline's launch, else the params themselves."""
        specialist_task_id = str(params.get("specialist_task_id") or "")
        if action_name == "integrate_patch" and specialist_task_id:
            return integrate_patch_key(specialist_task_id)
        payload: Any = (
            self._coord.writeback.baseline_params_fingerprint(params) if action_name == "baseline" else params
        )
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode(), usedforsecurity=False
        ).hexdigest()[:16]
        return f"{action_name}:{digest}"

    def credit_predictor_proposals(self, params: dict) -> None:
        """Credit the predictor for an orchestration grid's copies of its rows (see ``predictor.attribution``)."""
        grid = params.get("grid")
        if not isinstance(grid, list) or not grid:
            return
        from ..predictor.attribution import stamp_predictor_provenance

        stamped = stamp_predictor_provenance(grid, self.shared_state.specialist_rounds or [])
        if stamped:
            log.info("explore grid: %d/%d variant(s) credited to the predictor", stamped, len(grid))

    def floor_keep_threshold(self, params: dict[str, Any]) -> None:
        """Set ``keep_threshold_pct`` to the session's bar, or to the agent's value when that is a finite higher one.

        An agent may only raise the bar: a lower value would KEEP gains the session refuses, a negative one would KEEP
        regressions, and the stamped value also decides which earlier REVERTs a lower bar reopens.
        """
        session_pct = _phase_state.resolve_keep_threshold(self.shared_state)
        proposed = to_float(params.get("keep_threshold_pct"))
        params["keep_threshold_pct"] = session_pct if proposed is None else max(proposed, session_pct)

    def inject_explore_runtime_params(self, params: dict) -> None:
        """Inject explore-task operational knobs from SharedState into ``params``. setdefault preserves LLM overrides, except the KEEP bar, which an agent may only raise."""
        br = float(self.shared_state.baseline_runtime_sec or 0.0)
        if br > 0:
            params.setdefault("baseline_runtime_sec", br)
        baseline_accuracy = float(self.shared_state.baseline_accuracy or 0.0)
        if baseline_accuracy > 0:
            params.setdefault("accuracy_baseline", baseline_accuracy)
        # Warm measure-round anchor for admission costing.
        bwr = float(self.shared_state.baseline_warm_runtime_sec or 0.0)
        if bwr > 0:
            params.setdefault("baseline_warm_runtime_sec", bwr)
        self.floor_keep_threshold(params)
        # The round-id seed: the executor holds no cross-round state, so the round
        # it labels itself with has to come from the durable cursor.
        cursor = int((self.shared_state.explore_search or {}).get("cursor") or 0)
        params.setdefault("explore_search_cursor", cursor)

    async def enqueue_integrate_patch(self, params: dict[str, Any]) -> Task | None:
        """Queue the ``integrate_patch`` that measures a finished specialist's deliverable.

        ``None`` when a task under the deliverable's key already holds it or has measured it.
        """
        self.seed_integrate_patch_params(params)
        unspent = await self.unspent_key("integrate_patch", integrate_patch_key(params["specialist_task_id"]))
        if unspent is None or unspent.holder is not None:
            return None
        task = await self.create_task("integrate_patch", params, unspent.key)
        # The round the authoring specialist opened runs on under this task id.
        await self._coord.enablement_lane.handoff_enablement_round(task)
        return task

    async def unspent_key(self, action_name: str, key: str) -> UnspentKey | None:
        """Where the work under ``key`` stands: the free key its task takes, or the twin that answers it.

        Requests that would launch identical work collapse to one task. An ended twin leaves a fresh key to a
        retry, unless it measured what is asked for; ``None`` once every retry key is spent.
        """
        return await first_unspent_key(self.tasks, key, spent=lambda twin: not twin_answers(action_name, twin))

    async def create_task(self, action_name: str, params: dict[str, Any], key: str) -> Task:
        """Create the task under ``key``, a free key of :meth:`unspent_key`.

        Intents are routed and completions booked one after the other, so nothing takes the key between the two.
        """
        if action_name == SPECIALIST_ACTION_NAME:
            lanes, ttl = self._coord.specialist_dispatch.lanes_ttl(params)
        else:
            lanes, ttl = self._coord.dispatcher.registry_lanes_ttl(action_name)
        task, _existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind=action_name,
                params=params,
                idempotency_key=key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                # One class for every route to a key: a key holds one class.
                dispatch_class="llm",
            )
        )
        return task

    def seed_integrate_patch_params(self, params: dict[str, Any]) -> None:
        """Default an integrate_patch's grading context and its patched-eval server."""
        # ``source_phase`` is stamped where the specialist is created and carried from there; a
        # second derivation here would be a second decision, and could write an empty owner.
        self.floor_keep_threshold(params)
        # Seed the patched-eval server with the same base args/config every other eval server uses, else it
        # launches on bare framework defaults and crashes at startup regardless of the patch.
        inject_stack_base_params(params, self.shared_state, anchor=True)
        if self.shared_state.baseline_config_path:
            params.setdefault("config_path", self.shared_state.baseline_config_path)
