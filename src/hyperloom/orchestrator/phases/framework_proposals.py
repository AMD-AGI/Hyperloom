# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT proposals: the integrate_patch a finished specialist's deliverable is queued as, and untested custom proposals benched as a grid."""

from __future__ import annotations

import logging as _logging
import uuid
from collections.abc import Mapping
from typing import Any, NamedTuple
from . import machine_state as _phase_state
from ..state.task_registry import Task, TaskSpec
from ..state.shared_state import inject_stack_base_params
from ..loop.proposals import record_config_proposal
from hyperloom.orchestrator.lever import patch_owner_phase
from .proposal_bench import CUSTOM_PROPOSAL_DOMAIN, kb_exposure_of, waiting_proposal_rows

log = _logging.getLogger(__name__)


def _forward_enablement_carriers(src: dict[str, Any], dst: dict[str, Any]) -> None:
    """Copy the round's enablement context from specialist params to the integrate task.

    The pre-patch boot observation travels on every enablement round; the
    eval-origin trigger context only on an eval-origin one.
    """
    before_path = str(src.get("enablement_before_observation_path") or "")
    if before_path:
        dst["enablement_before_observation_path"] = before_path
    origin = str(src.get("enablement_origin") or "")
    if not origin:
        return
    dst["enablement_origin"] = origin
    dst["enablement_accuracy_floor"] = float(src.get("enablement_accuracy_floor") or 0.0)
    cfg = str(src.get("enablement_probe_config_path") or "")
    if cfg:
        dst["enablement_probe_config_path"] = cfg
        # Bench the candidate against the original workload/eval contract rather
        # than the shipped default config.
        dst.setdefault("config_path", cfg)


def _forward_integrate_source(
    src: dict[str, Any],
    dst: dict[str, Any],
    done_payload: Mapping[str, Any],
) -> None:
    """Preserve proposal ownership across delayed ``integrate_patch`` execution."""

    domain = str(src.get("domain") or "").strip()
    source_phase = patch_owner_phase(src)
    if source_phase:
        dst["source_phase"] = source_phase
    if domain:
        dst["domain"] = domain
        dst["provenance"] = f"specialist:{domain}"
    elif src.get("provenance"):
        # A domain-less dispatch such as a predictor mandate keeps the label it was dispatched under.
        dst["provenance"] = str(src["provenance"])
    # ``framework`` is intentionally not forwarded: integrate_patch consumes
    # that parameter when selecting accuracy parsing/gating behavior, whereas
    # proposal ownership only needs the gap metadata below.
    # ``lever_kind`` travels with the proposal: the patch that lands moved the
    # same lever the specialist was dispatched against, and re-deriving it at
    # writeback time is how attribution drifts. The KB exposure does too: it is
    # what the authoring specialist was shown, and the attempt is recorded later.
    for key in (
        "gap_canonical_id",
        "gap_layer",
        "lever_kind",
        "reauthor_attempt",
        "apply_retry_attempt",
        "kb_read_id",
        "kb_rendered_refs",
    ):
        value = src.get(key)
        if value not in (None, "", [], {}):
            dst[key] = value
    # What the specialist says those Experiences did to the patch it wrote, already checked against them.
    citations = done_payload.get("experience_citations")
    if citations:
        dst["experience_citations"] = list(citations)


class Deliverable(NamedTuple):
    """The integrate_patch that measures what a finished specialist delivered."""

    params: dict[str, Any]
    #: Recorded when the Coordinator queues it.
    observation: dict[str, Any]


def _deliverable_integrate_params(
    sid: str,
    spec_params: dict[str, Any],
    done_payload: dict[str, Any],
    deliverable: dict[str, Any],
) -> dict[str, Any]:
    """Build the integrate_patch params that land a specialist's deliverable."""
    proposals = done_payload.get("proposal_set")
    first = proposals[0] if isinstance(proposals, list) and proposals else None
    params: dict[str, Any] = {
        "specialist_task_id": sid,
        "provenance": "specialist",
        "patch_name": str(first.get("name") or "") if isinstance(first, dict) else "",
        **deliverable,
    }
    _forward_integrate_source(spec_params, params, done_payload)
    # The authored-outcome bridge keys the progress row on the real candidate id.
    for key in ("framework_agent_candidate_id", "framework_batch_id"):
        value = str(spec_params.get(key) or "")
        if value:
            params[key] = value
    if spec_params.get("enablement"):
        # Without the marker the integrate result never carries
        # ``enablement=True``, so the round is never rearmed, settled or charged.
        params["enablement"] = True
        _forward_enablement_carriers(spec_params, params)
        # Earlier rounds' setup commands plus this deliverable's own, so the
        # round replays its install step before booting.
        setup_commands = [
            *(spec_params.get("enablement_setup_commands") or []),
            *(done_payload.get("setup_commands") or []),
        ]
        merged_setup = list(dict.fromkeys(str(c) for c in setup_commands if str(c)))
        if merged_setup:
            params["enablement_setup_commands"] = merged_setup
    return params


class _FrameworkProposalsMixin:
    """Queues the integrate_patch for a finished specialist's deliverable and benches the phase's untested proposals."""

    async def _enqueue_deliverable_integrate(self, deliverable: Deliverable) -> Task | None:
        """Queue the integrate_patch of a finished specialist's deliverable; idempotent per specialist.

        Returns:
            The queued task, or ``None`` when a task already holds or has measured this deliverable.
        """
        params = deliverable.params
        task = await self._coord.proposals.enqueue_integrate_patch(params)
        if task is None:
            return None
        sid = params["specialist_task_id"]
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                **deliverable.observation,
                "specialist_task_id": sid,
                "task_id": task.task_id,
                "patch_name": params["patch_name"],
            },
        )
        log.info("specialist %s: deliverable queued as integrate_patch %s", sid, task.task_id)
        return task

    async def _maybe_bench_untested_proposals(self) -> None:
        """Enqueue explore grids from ``waiting_proposal_rows``: the custom specialist's alone, ahead of every gate;
        the rest once no explore is queued or running, whether or not the candidate track is still open."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        if self._coord.dispatcher.dispatch_paused_for_phase_budget():
            return
        explores = [t for t in (*await self.tasks.queued(), *await self.tasks.running()) if t.kind == "explore"]
        custom = self._custom_proposal_rows()
        if custom and not self._has_operator_work(explores):
            if await self._enqueue_bench_grid(custom, operator_priority=True):
                return
        if state.custom_specialist_only or explores:
            return
        rows = [r for r in waiting_proposal_rows(state, self.tasks) if r["domain"] != CUSTOM_PROPOSAL_DOMAIN]
        if rows:
            await self._enqueue_bench_grid(rows[: self._AUTO_EXPLORE_GRID_CAP], operator_priority=False)

    async def _enqueue_bench_grid(self, rows: list[dict[str, Any]], *, operator_priority: bool) -> bool:
        """Enqueue one explore grid benching ``rows``; returns whether a new task was created."""
        from ..actions.executors._proposal_identity import controls_of

        state = self.shared_state
        grid = [
            {
                "name": row["name"],
                "extra_args": row["extra_args"],
                "extra_envs": dict(row["extra_envs"]),
                **controls_of(row),
                "provenance": row["provenance"],
                # The proposing specialist's whole reasoning and checked citations reach the measured Experience.
                "reasoning": row["reason"],
                **({"experience_citations": row["experience_citations"]} if row["experience_citations"] else {}),
            }
            for row in rows
        ]
        # The grid's attempts join their proposal row through this id, as a proposed or delegated grid's do.
        proposal_id = uuid.uuid4().hex
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "auto_bench_untested_proposals",
            "grid": grid,
            "proposal_msg_id": proposal_id,
        }
        if operator_priority:
            params["operator_priority"] = True
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        inject_stack_base_params(params, state, anchor=True)
        benchmark_script = state.accepted_baseline_script()
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("explore")
        etask, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params=params,
                idempotency_key=self._bench_key(rows, operator_priority=operator_priority),
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if was_existing:
            return False
        log.info("auto_bench: enqueued explore task_id=%s (variants=%d)", etask.task_id, len(grid))
        record_config_proposal(
            self._coord, proposal_id, "explore", {"params": params, **kb_exposure_of(rows)}, outcome="auto_bench"
        )
        return True
