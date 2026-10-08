# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PendingProposal and the Critic-approved path: materializing an approved proposal into a dispatched task."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping
from hyperloom.common.framework_arm import is_upstream_pr_prescreen
from hyperloom.orchestrator.lever import LEVER_CONFIG
from hyperloom.inference_optimizer.trace.trajectory_trace import EVENT_PROPOSAL, STATUS_QUEUED, record_event
from ..phases import machine_state as _phase_state
from ..bus.message_bus import Message
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import TERMINAL_STATES
from ..collaborator import CoordinatorCollaborator

if TYPE_CHECKING:
    from .coordinator import Coordinator

import logging as _logging

log = _logging.getLogger(__name__)

_MAX_IDEMPOTENCY_ATTEMPTS: int = 6


@dataclass
class PendingProposal:
    """A propose_action intent waiting for Critic Review."""

    proposal_msg_id: str
    from_agent: str
    action_name: str
    predicted_gain_pct: float
    payload: dict[str, Any]
    #: The task this proposal materialized into, once the Critic approved it.
    task_id: str | None = None


async def record_proposal(
    coord: "Coordinator",
    *,
    from_agent: str,
    action_name: str,
    predicted_gain_pct: float,
    payload: dict[str, Any],
) -> PendingProposal:
    """Create one PendingProposal, write it to the bus, and record it on the phase timeline.

    This is the single path for both LLM-originated and coordinator-originated
    proposals. Every proposal that goes through here gets a phase timeline row
    so the Critic's ruling is always filed.

    Returns the PendingProposal that was inserted into ``coord.state.pending_proposals``.
    """
    msg = Message.new(
        from_agent,
        "*",
        "proposal",
        {**payload, "needs_review": True},
    )
    await coord.bus.append_and_seq(msg)
    pending = PendingProposal(
        proposal_msg_id=msg.msg_id,
        from_agent=from_agent,
        action_name=action_name,
        predicted_gain_pct=predicted_gain_pct,
        payload=payload,
    )
    coord.state.pending_proposals[msg.msg_id] = pending
    record_event(
        EVENT_PROPOSAL,
        status=STATUS_QUEUED,
        span_id=msg.msg_id,
        attributes={
            "name": action_name,
            "action_name": action_name,
            "from_agent": from_agent,
            "predicted_gain_pct": pending.predicted_gain_pct,
        },
    )
    _record_phase_proposal(coord, pending)
    record_config_proposal(coord, pending.proposal_msg_id, pending.action_name, pending.payload)
    return pending


def _record_phase_proposal(coord: "Coordinator", pending: PendingProposal) -> None:
    """Record one proposal against the phase that raised it.

    Every proposal, not only the ones a framework arm claims: this is the row
    the Critic's ruling is filed on.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event

    state = coord.shared_state
    if not pending.proposal_msg_id or not state.phase:
        return
    params = pending.payload.get("params") if isinstance(pending.payload.get("params"), dict) else {}
    phase_event.record_proposal(
        proposal_msg_id=pending.proposal_msg_id,
        action=pending.action_name,
        phase=state.phase,
        macro_cycle=state.macro_cycle,
        from_agent=pending.from_agent,
        tick=state.tick,
        predicted_gain_pct=pending.predicted_gain_pct,
        candidate_id=pending.payload.get("framework_agent_candidate_id") or params.get("framework_agent_candidate_id"),
        variant_name=pending.payload.get("variant_name") or params.get("variant_name"),
    )


def record_config_proposal(
    coord: "Coordinator",
    proposal_id: str,
    action_name: str,
    payload: Mapping[str, Any],
    *,
    outcome: str = "submitted",
) -> None:
    """Record one config-arm grid on the framework event, as it is proposed or delegated.

    Recorded at proposal time rather than at approval, so a grid the Critic
    denies is still on record as a thing the phase pursued and dropped. One row
    per grid, not per variant: the measured attempts point back at the grid
    through their ``proposal_ref``.
    """
    if not proposal_id:
        return
    recorder = coord.phase_framework.timeline()
    if recorder is None:
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


def apply_critic_grid_filter(
    params: dict[str, Any],
    *,
    original_grid: list[Any],
    approved_variant_names: set[str] | None,
) -> bool:
    """Restrict ``params['grid']`` to Critic-approved variant names."""
    stamped_grid: list[Any] = []
    for variant in original_grid:
        if not isinstance(variant, dict):
            if approved_variant_names is None:
                stamped_grid.append(variant)
            continue
        vname = str(variant.get("name") or "").strip()
        if approved_variant_names is not None and vname not in approved_variant_names:
            continue
        stamped_grid.append(dict(variant))
    params["grid"] = stamped_grid
    if approved_variant_names is None:
        return True
    original_grid_len = len([v for v in original_grid if isinstance(v, dict)])
    params["critic_filtered_count"] = max(0, original_grid_len - len(stamped_grid))
    return bool(stamped_grid)


def _framework_recorder(coll: Any, pending: Any) -> Any:
    """The framework recorder to write one config-arm proposal's step onto.

    ``None`` whenever the step is not a config-arm one to record: another
    action, or a phase whose event is not open.
    """
    if str(getattr(pending, "action_name", "") or "") != "explore":
        return None
    if not str(getattr(pending, "proposal_msg_id", "") or ""):
        return None
    return coll._coord.phase_framework.timeline()


def _record_proposal_materialized(proposal_msg_id: str, task_id: str) -> None:
    """Name the task a proposal became, on the proposal's own row.

    This is what joins the two halves of one decision: the proposal row says
    what was asked for and what the Critic said about it, the dispatch row
    beside it on the same phase event says what was run. Without the task id
    they sit on one event with nothing connecting them, and the join has to be
    rebuilt from a sidecar map at export.
    """
    if not proposal_msg_id or not task_id:
        return
    try:
        from hyperloom.inference_optimizer.breakdown.recorder import phase_event

        phase_event.record_proposal_outcome(
            proposal_msg_id=str(proposal_msg_id),
            materialized=True,
            task_id=str(task_id),
        )
    except Exception:
        log.debug("phase timeline: proposal task link failed for %s", proposal_msg_id, exc_info=True)


def _record_config_routed(coll: Any, pending: Any, *, task_id: str) -> None:
    """Record that one config-arm grid reached a bench.

    ``task_id`` is the task it became, which its attempts also carry.
    """
    recorder = _framework_recorder(coll, pending)
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import STEP_ROUTED

    recorder.record_proposal_step(
        pending.proposal_msg_id,
        step=STEP_ROUTED,
        outcome="materialized",
        reason=str(task_id or ""),
    )


def _record_config_dropped(coll: Any, pending: Any, *, reason: str) -> None:
    """Settle one config-arm grid that never reached a bench.

    The empty-grid case is the one that most needs saying: the Critic approved
    the proposal and then named no variant that survived the filter, so the
    arm spent a review and benched nothing. Without a settled row that reads
    as a proposal still under way.
    """
    recorder = _framework_recorder(coll, pending)
    if recorder is None:
        return
    from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
        DISPOSITION_DROPPED,
        STEP_DROPPED,
    )

    recorder.record_proposal_step(pending.proposal_msg_id, step=STEP_DROPPED, reason=reason)
    recorder.settle_proposal(
        pending.proposal_msg_id,
        disposition=DISPOSITION_DROPPED,
        reason=reason,
    )


class ProposalsCollaborator(CoordinatorCollaborator):
    """Turns approved proposals into queued tasks."""

    def _approved_idempotency_key(self, action_name: str, params: dict[str, Any]) -> str:
        """Content-addressed idempotency key for an approved proposal."""
        payload: Any = (
            self._coord.writeback.baseline_params_fingerprint(params) if action_name == "baseline" else params
        )
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True, default=str).encode(), usedforsecurity=False
        ).hexdigest()[:16]
        return f"approved:{action_name}:{digest}"

    def inject_explore_runtime_params(self, params: dict) -> None:
        """Inject explore-task operational knobs from SharedState into ``params`` (single source of truth for both propose/Critic and direct-delegate paths). setdefault preserves LLM overrides."""
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
        keep = _phase_state.resolve_keep_threshold(self.shared_state)
        params.setdefault("keep_threshold_pct", keep)
        # The round-id seed: the executor holds no cross-round state, so the round
        # it labels itself with has to come from the durable cursor.
        cursor = int((self.shared_state.explore_search or {}).get("cursor") or 0)
        params.setdefault("explore_search_cursor", cursor)

    async def materialize_approved_proposal(
        self,
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        """Promote an approved proposal into a TaskRegistry entry. Stack-aware actions get current_best's anchor and the base config it was measured on; approved_variant_names filters the explore grid (None keeps full)."""
        if is_upstream_pr_prescreen(pending.action_name, pending.payload):
            await self._coord.phase_framework.materialize_candidate(pending)
            return
        params = dict(pending.payload.get("params") or {})
        # Carry the proposer's predicted gain onto the task for predicted-vs-realized calibration.
        if pending.predicted_gain_pct:
            params.setdefault(
                "predicted_gain_pct",
                float(pending.predicted_gain_pct),
            )
        # Filter the grid to the Critic-approved subset.
        if pending.action_name == "explore" and isinstance(params.get("grid"), list):
            original_grid = list(params["grid"])
            if not apply_critic_grid_filter(
                params,
                original_grid=original_grid,
                approved_variant_names=approved_variant_names,
            ):
                await self._coord.writeback.record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_materialize_skipped",
                        "reason": "critic_filter_empty_grid",
                        "proposal_msg_id": pending.proposal_msg_id,
                        "action_name": pending.action_name,
                        "from_agent": pending.from_agent,
                    },
                )
                _record_config_dropped(self, pending, reason="critic_filter_empty_grid")
                return
        if pending.action_name == "profile":
            # Stamp the server config that produced this trace.
            inject_stack_base_params(params, self.shared_state)
        if pending.action_name == "sweep":
            inject_stack_base_params(params, self.shared_state, anchor=True)
            if self.shared_state.baseline_config_path:
                params.setdefault("config_path", self.shared_state.baseline_config_path)
        if pending.action_name == "explore":
            self.inject_explore_runtime_params(params)
            inject_stack_base_params(params, self.shared_state, anchor=True)
        if pending.action_name == "integrate_patch":
            # ``source_phase`` is stamped where the specialist is created and carried from there; a
            # second derivation here would be a second decision, and could write an empty owner.
            params.setdefault("keep_threshold_pct", _phase_state.resolve_keep_threshold(self.shared_state))
            # Seed the patched-eval server with the same base args/config every other eval server uses, else it
            # launches on bare framework defaults and crashes at startup regardless of the patch.
            inject_stack_base_params(params, self.shared_state, anchor=True)
            if self.shared_state.baseline_config_path:
                params.setdefault("config_path", self.shared_state.baseline_config_path)
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl(pending.action_name)
        # Content-addressed so a batch of proposals that would launch identical work collapses to one task; a
        # terminated twin still gets a fresh key so a legitimate retry after failure is never locked out.
        raw_key = self._approved_idempotency_key(pending.action_name, params)
        # Preserve the authoritative config-proposal join on the materialized task. Keep this out of the
        # content-addressed idempotency key above so two proposals for identical grids still collapse to one task.
        if pending.action_name == "explore" and pending.proposal_msg_id:
            params["proposal_msg_id"] = str(pending.proposal_msg_id)
        task = None
        was_existing = False
        for attempt in range(_MAX_IDEMPOTENCY_ATTEMPTS):
            idempotency_key = raw_key if attempt == 0 else f"{raw_key}-retry{attempt}"
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=pending.action_name,
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="llm",
            )
            if not was_existing:
                break
            if task.state not in TERMINAL_STATES:
                await self._coord.writeback.record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "proposal_materialize_skipped",
                        "reason": "duplicate_proposal_content",
                        "proposal_msg_id": pending.proposal_msg_id,
                        "task_id": task.task_id,
                        "task_state": task.state,
                        "action_name": pending.action_name,
                        "from_agent": pending.from_agent,
                    },
                )
                return
        else:
            await self._coord.writeback.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "proposal_materialize_skipped",
                    "reason": "idempotency_key_exhausted",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "task_id": task.task_id if task is not None else "",
                    "task_state": task.state if task is not None else "",
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                },
            )
            return
        # The round the authoring specialist opened runs on under this task id.
        await self._coord.enablement_lane.handoff_enablement_round(task)
        # proposal_msg_id is the resume contract for the deferred queue (see replay_for_resume).
        await self.bus.append_and_seq(
            Message.new(
                "coordinator",
                "*",
                "decision",
                {
                    "kind": "approved_proposal",
                    "task_id": task.task_id,
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                    "proposal_msg_id": pending.proposal_msg_id,
                },
            )
        )
        # Trace attribution: record proposal_msg_id -> task_id for the decision-trace collector.
        self._record_proposal_task_map(pending.proposal_msg_id, task.task_id)
        pending.task_id = task.task_id
        _record_proposal_materialized(pending.proposal_msg_id, task.task_id)
        _record_config_routed(self, pending, task_id=task.task_id)

    def _record_proposal_task_map(self, proposal_msg_id: str, task_id: str) -> None:
        """Append one ``{proposal_msg_id -> task_id}`` row to the trace map."""
        if not proposal_msg_id or not task_id:
            return
        try:
            from hyperloom.common.timeutil import now_iso
            from hyperloom.common.io import append_jsonl
            from hyperloom.inference_optimizer.session.session_paths import proposal_task_map_path

            path = proposal_task_map_path(self.session_dir)
            row = {
                "ts": now_iso(),
                "proposal_msg_id": str(proposal_msg_id),
                "task_id": str(task_id),
            }
            append_jsonl(path, row, make_parents=True, sort_keys=True)
        except Exception:
            log.debug(
                "full-trace: proposal_task_map append failed for msg_id=%s task_id=%s",
                proposal_msg_id,
                task_id,
                exc_info=True,
            )
