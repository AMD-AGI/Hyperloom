# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT proposals: candidates and integrate patches sent to Critic review, and untested custom proposals benched as a grid."""

from __future__ import annotations

import logging as _logging
import uuid
from collections.abc import Mapping
from typing import Any
from hyperloom_common.coerce import first_str, to_mapping
from . import machine_state as _phase_state
from ..state.task_registry import TaskSpec
from ..state.shared_state import inject_stack_base_params
from ..state.pending_proposal import PendingProposal
from ..loop.proposals import integrate_review_in_flight, record_proposal
from hyperloom.orchestrator.lever import LEVER_UPSTREAM_PR, patch_owner_phase
from .proposal_bench import CUSTOM_PROPOSAL_DOMAIN, kb_exposure_of, waiting_proposal_rows
from .framework_records import ProgressStamp, ProposalStep, _record_step

log = _logging.getLogger(__name__)


#: Progress-row status for a candidate the Critic rejected. The gate writes it
#: and the priors reader selects on it, so the ledger is the only record of a
#: denial and both sites agree by construction.
FRAMEWORK_CRITIC_DENIED_STATUS: str = "critic_denied"


def _forward_enablement_carriers(src: dict[str, Any], dst: dict[str, Any]) -> None:
    """Copy the round's enablement context from specialist params to the integrate task.

    The dispatched verdict and the pre-patch boot observation travel on every
    enablement round; the eval-origin trigger context only on an eval-origin one.
    """
    signature = src.get("enablement_failure_signature")
    if isinstance(signature, dict) and signature:
        dst["enablement_failure_signature"] = dict(signature)
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


def _integrate_review_params(
    sid: str,
    spec_params: dict[str, Any],
    done_payload: dict[str, Any],
    deliverable: dict[str, Any],
) -> dict[str, Any]:
    """Build the synthetic integrate_patch params that review a specialist's deliverable."""
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
    """Records the phase's proposals in the loop's proposal ledger and submits them for review or bench."""

    # Tail of outcomes from the priors ledger to evaluate.
    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5

    # Backstop: max Critic-review submissions for a single candidate before the pump force-stamps
    # ``repeated_review_abort`` and stops re-selecting it.
    _MAX_REPEATED_REVIEW_SUBMISSIONS: int = 3

    def _collect_framework_agent_candidate_priors(self) -> dict[str, Any]:
        """Return compact session-local priors for the Critic gate."""
        raw_progress = self.shared_state.framework_agent_phase_progress or []
        terminal = {
            "kept",
            "kept_inert",
            "reverted",
            "no_patch",
            "enqueue_failed",
            FRAMEWORK_CRITIC_DENIED_STATUS,
        }
        tail = [r for r in raw_progress if isinstance(r, dict) and str(r.get("status") or "") in terminal]
        outcomes: list[dict[str, Any]] = []
        for row in tail[-self._CRITIC_PRIORS_OUTCOME_TAIL :]:
            entry: dict[str, Any] = {
                "candidate_id": str(row.get("candidate_id") or ""),
                "status": str(row.get("status") or ""),
                "gain_pct": row.get("gain_pct"),
            }
            rationale = str(row.get("rationale") or "")[:200]
            if rationale:
                entry["rationale"] = rationale
            outcomes.append(entry)
        return {
            "recent_outcomes": outcomes,
        }

    def _review_already_pending(self, cand_id: str) -> bool:
        """True while ``cand_id`` is still awaiting the Critic's pre-screen verdict."""
        return bool(cand_id) and any(
            first_str(to_mapping(getattr(p, "payload", None)).get("framework_agent_candidate_id")) == cand_id
            for p in self.state.pending_proposals.values()
            if getattr(p, "action_name", "") == "integrate_patch"
        )

    def _charge_review_submission(self, cand_id: str, batch_id: str) -> bool:
        """Count one review submission against ``cand_id``; False once the phase budget's cap is passed."""
        if not cand_id:
            return True
        counts = to_mapping(self.shared_state.framework_agent_review_counts)
        count = int(counts.get(cand_id, 0) or 0) + 1
        counts[cand_id] = count
        self.shared_state.framework_agent_review_counts = counts
        if count <= self._MAX_REPEATED_REVIEW_SUBMISSIONS:
            return True
        log.warning(
            "FRAMEWORK: candidate=%s submitted for review %d times (> cap %d); aborting to protect the phase budget",
            cand_id,
            count,
            self._MAX_REPEATED_REVIEW_SUBMISSIONS,
        )
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="repeated_review_abort",
                rationale=f"submitted for review {count} times (> cap {self._MAX_REPEATED_REVIEW_SUBMISSIONS})",
                provenance="pump",
                extra={"review_submissions": count},
            )
        )
        return False

    async def _submit_framework_agent_candidate_for_review(
        self,
        candidate: dict[str, Any],
        *,
        audit: dict[str, Any] | None = None,
        audit_step: str = "",
    ) -> None:
        """Submit a FRAMEWORK candidate as a normal ``proposal`` for async Critic review."""
        cand_id = self._framework_candidate_key(candidate)
        batch_id = first_str(candidate.get("batch_id"))
        if self._review_already_pending(cand_id) or not self._charge_review_submission(cand_id, batch_id):
            return
        propose_payload: dict[str, Any] = {
            "action_name": "integrate_patch",
            "provenance": LEVER_UPSTREAM_PR,
            "predicted_gain_pct": 0.0,
            "candidate": dict(candidate),
            "batch_id": batch_id,
            "framework_agent_candidate_id": cand_id,
            "audit": to_mapping(audit),
            "audit_step": first_str(audit_step),
            "priors": self._collect_framework_agent_candidate_priors(),
        }
        pending = await record_proposal(
            self._coord,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload=dict(propose_payload),
        )
        # The review-bus message id is a field, not a second identity: the
        # candidate is keyed by its own id throughout the phase.
        _record_step(
            self,
            cand_id,
            ProposalStep(
                step="routed", outcome=str(audit_step or ""), reason=f"submitted_for_review:{pending.proposal_msg_id}"
            ),
        )
        log.info(
            "FRAMEWORK: candidate submitted for Critic review msg_id=%s candidate=%s batch=%s audit_step=%s",
            pending.proposal_msg_id,
            cand_id,
            batch_id,
            audit_step or "<unknown>",
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "framework_agent_candidate_submitted_for_review",
                "proposal_msg_id": pending.proposal_msg_id,
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "audit_step": str(audit_step or ""),
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception(
                "save after framework_agent candidate submit failed for candidate=%s",
                cand_id,
            )

    async def _submit_integrate_review(
        self,
        sid: str,
        spec_params: dict[str, Any],
        done_payload: dict[str, Any],
        deliverable: dict[str, Any],
        *,
        observation: dict[str, Any],
    ) -> PendingProposal | None:
        """Propose a specialist deliverable to the Critic as a synthetic integrate_patch; idempotent per specialist.

        Args:
            sid: The specialist task id the review is keyed on.
            spec_params: The specialist's dispatch params, whose provenance and
                enablement context travel to integrate_patch.
            done_payload: The specialist's ``specialist_done`` payload.
            deliverable: The integrate_patch params that say what to apply.
            observation: The bus observation recorded with the proposal.

        Returns:
            The recorded proposal, or ``None`` when the Critic already ruled on
            this specialist or its review is still pending.
        """
        if self.shared_state.get_specialist_patch_verdict(sid) or integrate_review_in_flight(
            self.state.pending_proposals, sid
        ):
            return None
        params = _integrate_review_params(sid, spec_params, done_payload, deliverable)
        pending = await record_proposal(
            self._coord,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload={
                "action_name": "integrate_patch",
                "provenance": "specialist",
                "predicted_gain_pct": 0.0,
                "params": params,
            },
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                **observation,
                "specialist_task_id": sid,
                "proposal_msg_id": pending.proposal_msg_id,
                "patch_name": params["patch_name"],
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("save after integrate-review autosubmit failed for task=%s", sid)
        return pending

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
        from ..loop.proposals import record_config_proposal

        record_config_proposal(
            self._coord, proposal_id, "explore", {"params": params, **kb_exposure_of(rows)}, outcome="auto_bench"
        )
        return True
