# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT re-authoring: apply-failure retries, critic-feedback rewrites and the authored lane's re-arm."""

from __future__ import annotations

import logging as _logging
from collections.abc import Mapping
from typing import Any, NamedTuple
from hyperloom_common.coerce import first_str, to_list, to_mapping
from ..state.task_registry import TaskNotFound, TaskSpec
from ..state.pending_proposal import PendingProposal
from hyperloom.orchestrator.lever import LEVER_SOURCE_PATCH
from .framework_records import ProgressStamp, _record_review_outcome
from .framework_proposals import FRAMEWORK_CRITIC_DENIED_STATUS

log = _logging.getLogger(__name__)


# Unified authored-lane max attempts (apply-failure retries + Critic reauthor).
_AUTHORED_LANE_MAX_ATTEMPTS: int = 3


class AuthoringRetry(NamedTuple):
    """An apply-failure retry queued for a fresh authoring specialist."""

    lane: str
    candidate: dict[str, Any] | None = None
    batch_id: str = ""
    specialist_task_id: str = ""
    attempt: int = 1
    retry_feedback: list[dict[str, Any]] | None = None
    vetting_drops: list[str] | None = None


def _apply_failure_feedback_lines(retry: AuthoringRetry) -> list[str]:
    """The prompt lines telling an authoring retry why the previous round's patch never landed."""
    lines: list[str] = []
    if retry.vetting_drops:
        lines.append("")
        lines.append(
            "PATCH GROUNDING FAILURE: the prior round's patches were dropped by "
            "the safety gate before reaching integration. The worktree contains "
            "the correct framework tree — edit files there and the diff is "
            "harvested automatically. Do not switch to the artifacts_written "
            "channel to avoid the gate."
        )
        lines.append("Dropped targets: " + "; ".join(retry.vetting_drops[:4]))
        lines.append("")
    if retry.retry_feedback:
        from ..actions.executors.patching._apply_feedback import ApplyFeedback

        lines.append("")
        lines.append(
            "APPLY FAILURE FEEDBACK (previous patch failed to apply; "
            "study the errors below and produce a corrected patch):"
        )
        # Capped for prompt brevity.
        for fb_dict in retry.retry_feedback[:5]:
            fb = ApplyFeedback.from_dict(fb_dict) if isinstance(fb_dict, dict) else None
            if fb is not None:
                lines.append("")
                lines.append(fb.format_for_mandate())
        lines.append("")
    return lines


class _ReauthorMixin:
    """Re-authors a framework candidate after an apply failure or a critic denial, within its budget."""

    async def _maybe_rearm_authored_lane(self, res: dict[str, Any] | None, params: Mapping[str, Any]) -> None:
        """Route an authored lane's result to the handler that rearms it: enablement settles, a perf apply-failure re-authors the candidate ``params`` name, under cap."""
        if not isinstance(res, dict):
            return
        lane = first_str(res.get("lane"))
        if lane == "enablement" or res.get("enablement"):
            await self._coord.enablement_lane.maybe_rearm_enablement(res)
            return
        # Every other perf-lane status goes through the writeback path.
        if first_str(res.get("status")) != "apply_failed" or lane not in ("perf_framework", "perf_explore"):
            return

        candidate = self._candidate_row(self._authored_candidate_id(params))
        cand_id = self._framework_candidate_key(candidate) or first_str(res.get("specialist_task_id")).strip()
        if not cand_id:
            return
        batch_id = first_str(candidate.get("batch_id"), res.get("batch_id"))

        state = self.shared_state
        apply_fail_attempts = to_mapping(state.apply_fail_reauthor_attempts)
        attempt = int(apply_fail_attempts.get(cand_id, 0) or 0) + 1
        apply_fail_attempts[cand_id] = attempt
        state.apply_fail_reauthor_attempts = apply_fail_attempts

        log.info(
            "AUTHORED_LANE rearm: lane=%s cand_id=%s apply_fail_attempt=%d cap=%d",
            lane,
            cand_id,
            attempt,
            _AUTHORED_LANE_MAX_ATTEMPTS,
        )

        if attempt > _AUTHORED_LANE_MAX_ATTEMPTS:
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="apply_fail_cap",
                    rationale=f"apply_failed {attempt} times (cap={_AUTHORED_LANE_MAX_ATTEMPTS})",
                    provenance="apply_fail_retry",
                )
            )
            try:
                state.save(self.session_dir)
            except OSError:
                log.debug("authored_lane: save after cap stamp failed", exc_info=True)
            return

        retry_ctx: dict[str, Any] = {
            "cand_id": cand_id,
            "batch_id": batch_id,
            "lane": lane,
            "attempt": attempt,
            "retry_feedback": to_list(res.get("retry_feedback")),
            "candidate": candidate,
            "specialist_task_id": first_str(res.get("specialist_task_id")),
        }
        vetting_drops = to_list(res.get("patches_ungrounded"))
        if vetting_drops:
            retry_ctx["vetting_drops"] = [str(d) for d in vetting_drops[:8]]
        pending = to_list(state.apply_fail_retry_pending)
        pending.append(retry_ctx)
        state.apply_fail_retry_pending = pending
        try:
            state.save(self.session_dir)
        except OSError:
            log.debug("authored_lane: save after retry-pending failed", exc_info=True)

    async def _authoring_specialist_params(self, task_id: str) -> dict[str, Any]:
        """The dispatch params of the authoring specialist ``task_id`` names; empty when the registry has no such row."""
        if not task_id:
            return {}
        try:
            return to_mapping((await self.tasks.get(task_id)).params)
        except TaskNotFound:
            return {}

    async def _enqueue_author_specialist(self, retry: AuthoringRetry) -> str:
        """Dispatch a fresh authoring specialist for an apply-failure retry."""
        candidate = to_mapping(retry.candidate)
        if retry.batch_id and not candidate.get("batch_id"):
            candidate["batch_id"] = retry.batch_id
        feedback = "\n".join(_apply_failure_feedback_lines(retry))
        if retry.lane == "perf_framework":
            return await self._reauthor_framework_candidate(retry, candidate, feedback)
        if retry.lane == "perf_explore":
            return await self._reauthor_explore_patch(retry, feedback)
        log.warning("_enqueue_author_specialist: unsupported lane=%s — skipping", retry.lane)
        return ""

    async def _reauthor_framework_candidate(
        self, retry: AuthoringRetry, candidate: dict[str, Any], feedback: str
    ) -> str:
        """Re-dispatch the upstream-PR authoring specialist, seeding it with the apply failure as Critic feedback."""
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            log.warning("_enqueue_author_specialist: perf_framework missing cand_id")
            return ""
        spec_params = await self._authoring_specialist_params(retry.specialist_task_id)
        new_task_id = await self._enqueue_framework_agent_authoring_specialist(
            candidate,
            audit=to_mapping(spec_params.get("framework_audit")),
            reauthor_attempt=retry.attempt,
            critic_feedback={"advice_text": feedback} if feedback else None,
        )
        log.info(
            "AUTHORED_LANE: dispatched perf_framework retry specialist cand=%s attempt=%d task=%s",
            cand_id,
            retry.attempt,
            new_task_id,
        )
        return new_task_id

    async def _reauthor_explore_patch(self, retry: AuthoringRetry, feedback: str) -> str:
        """Re-dispatch a serving specialist on the gap the apply-failed local-exploration patch came from."""
        spec_params = await self._authoring_specialist_params(retry.specialist_task_id)
        gap_cid = first_str(spec_params.get("gap_canonical_id")).strip()
        if not gap_cid:
            gap_cid = f"gap.explore.retry.{retry.specialist_task_id or 'unknown'}"
        gap_symptom = first_str(spec_params.get("gap_symptom")).strip()
        framework_name = first_str(spec_params.get("framework"), self.shared_state.framework).strip().lower()
        notes = feedback.strip()
        params: dict[str, Any] = {
            "domain": "serving_specialist",
            "source_phase": "FRAMEWORK_AGENT",
            "lever_kind": LEVER_SOURCE_PATCH,
            "gap_canonical_id": gap_cid,
            "gap_symptom": gap_symptom or f"Retry apply-failed patch for {gap_cid}",
            "gap_layer": "perf_explore",
            "framework": framework_name,
            "task_kind": "explore_apply_retry",
            "source": "coordinator_internal",
            "notes": notes,
            "apply_retry_attempt": retry.attempt,
        }
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        # Gap id and attempt both repeat across cycles.
        idem = f"perf_explore_authoring:{gap_cid}:retry:{retry.attempt}{self._coord.dispatcher.cycle_idem_suffix()}"
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _ = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        new_tid = str(getattr(spec_task, "task_id", "") or "")
        log.info(
            "AUTHORED_LANE: dispatched perf_explore retry specialist gap=%s attempt=%d task=%s",
            gap_cid,
            retry.attempt,
            new_tid,
        )
        return new_tid

    async def _drain_apply_fail_retry_pending(self) -> None:
        """Dispatch authoring specialists for any queued apply-failure retries."""
        state = self.shared_state
        pending: list[dict[str, Any]] = state.apply_fail_retry_pending or []
        if not isinstance(pending, list) or not pending:
            return
        # Consume the list atomically so a concurrent call doesn't double-fire.
        to_dispatch = list(pending)
        state.apply_fail_retry_pending = []
        for ctx in to_dispatch:
            if not isinstance(ctx, dict):
                continue
            lane = str(ctx.get("lane") or "")
            attempt = int(ctx.get("attempt") or 1)
            candidate = ctx.get("candidate") or {}
            specialist_task_id = str(ctx.get("specialist_task_id") or "")
            batch_id = str(ctx.get("batch_id") or "")
            retry_feedback = list(ctx.get("retry_feedback") or [])
            vetting_drops = list(ctx.get("vetting_drops") or [])
            await self._enqueue_author_specialist(
                AuthoringRetry(
                    lane=lane,
                    candidate=candidate,
                    batch_id=batch_id,
                    specialist_task_id=specialist_task_id,
                    attempt=attempt,
                    retry_feedback=retry_feedback,
                    vetting_drops=vetting_drops or None,
                )
            )
        try:
            state.save(self.session_dir)
        except OSError:
            log.debug("drain_apply_fail: save failed", exc_info=True)

    async def record_critic_denial(
        self,
        pending: "PendingProposal",
        reasoning: str,
    ) -> None:
        """Record a ``critic_denied`` FRAMEWORK row when the async gate rejects a candidate."""
        payload = pending.payload or {}
        cand_id = str(
            payload.get("framework_agent_candidate_id")
            or self._framework_candidate_key(
                payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
            )
        )
        batch_id = str(payload.get("batch_id") or "")
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status=FRAMEWORK_CRITIC_DENIED_STATUS,
                rationale=str(reasoning or ""),
                provenance="critic",
            )
        )
        _record_review_outcome(
            self,
            cand_id,
            verdict="reject",
            reason=str(reasoning or ""),
            denied=True,
        )
        log.info(
            "FRAMEWORK: critic rejected candidate=%s batch=%s rationale=%r",
            cand_id,
            batch_id,
            str(reasoning or "")[:200],
        )

    async def maybe_reauthor_from_critic_feedback(
        self,
        pending: "PendingProposal",
        advisory: dict[str, Any] | None,
    ) -> None:
        """Re-author a framework_agent deliverable once, seeding the next authoring round with the Critic's ``required_evidence``."""
        advisory = to_mapping(advisory)
        if first_str(getattr(pending, "action_name", "")) != "integrate_patch":
            return
        # Resolve the candidate identity FIRST so the two dead-end returns below (no required_evidence / reauthor cap)
        # can stamp a terminal progress row — a needs_review verdict that neither re-authors nor materializes would
        # otherwise leave the candidate row-less and re-selected forever.
        payload = to_mapping(getattr(pending, "payload", None))
        authored = to_mapping(payload.get("params")) if not payload.get("framework_agent_candidate_id") else {}
        old_sid = first_str(authored.get("specialist_task_id")).strip()
        candidate, audit = await self._reauthor_target(payload)
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            return
        batch_id = first_str(candidate.get("batch_id"), payload.get("batch_id"))
        required_evidence = [str(x).strip() for x in to_list(advisory.get("required_evidence")) if str(x).strip()]
        if not required_evidence:
            # needs_review with nothing to act on: no re-author is possible, so this is terminal for the candidate.
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="needs_review_no_evidence",
                    rationale=first_str(advisory.get("advice_text"))[:500],
                    provenance="critic",
                )
            )
            return
        if await self._candidate_is_materializing(cand_id):
            return
        attempts = to_mapping(self.shared_state.specialist_reauthor_attempts)
        prior = int(attempts.get(cand_id, 0) or 0)
        if prior >= _AUTHORED_LANE_MAX_ATTEMPTS:
            await self._close_out_spent_reauthor_budget(pending, cand_id=cand_id, batch_id=batch_id, attempts=prior)
            return
        attempt = prior + 1
        attempts[cand_id] = attempt
        self.shared_state.specialist_reauthor_attempts = attempts
        critic_feedback = {
            "required_evidence": required_evidence,
            "advice_text": first_str(advisory.get("advice_text")),
            "risks": [str(r).strip() for r in to_list(advisory.get("risks")) if str(r).strip()],
        }
        new_task_id = await self._enqueue_framework_agent_authoring_specialist(
            candidate, audit=audit, reauthor_attempt=attempt, critic_feedback=critic_feedback
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("save after re-author dispatch failed candidate=%s", cand_id)
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_reauthor_dispatched",
                "candidate_id": cand_id,
                "attempt": attempt,
                "proposal_msg_id": first_str(getattr(pending, "proposal_msg_id", "")),
                "old_specialist_task_id": old_sid,
                "new_specialist_task_id": new_task_id,
                "verdict": "needs_review",
                "required_evidence": required_evidence[:6],
            },
        )

    async def _reauthor_target(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The candidate row and the audit a needs_review integrate_patch proposal re-authors from."""
        if payload.get("framework_agent_candidate_id"):
            return to_mapping(payload.get("candidate")), to_mapping(payload.get("audit"))
        params = to_mapping(payload.get("params"))
        if not params.get("framework_agent_authoring"):
            return {}, {}
        spec_params = await self._authoring_specialist_params(first_str(params.get("specialist_task_id")).strip())
        candidate = {
            "candidate_id": first_str(
                params.get("framework_agent_candidate_id"), spec_params.get("framework_agent_candidate_id")
            ),
            "batch_id": first_str(params.get("framework_batch_id"), spec_params.get("framework_batch_id")),
            "title": first_str(spec_params.get("gap_symptom")),
            "framework": first_str(spec_params.get("framework")),
            "gap_canonical_id": first_str(spec_params.get("gap_canonical_id")),
        }
        return candidate, to_mapping(spec_params.get("framework_audit"))

    async def _candidate_is_materializing(self, cand_id: str) -> bool:
        """True while an integrate_patch task for ``cand_id`` is already queued or running."""
        return any(
            first_str(to_mapping(t.params).get("framework_agent_candidate_id")) == cand_id
            for t in (*await self.tasks.queued(), *await self.tasks.running())
            if t.kind == "integrate_patch"
        )

    async def _close_out_spent_reauthor_budget(
        self, pending: "PendingProposal", *, cand_id: str, batch_id: str, attempts: int
    ) -> None:
        """Settle a candidate whose re-author budget is spent: the cap is terminal for it."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "reauthor_cap_reached",
                "candidate_id": cand_id,
                "attempts": attempts,
                "proposal_msg_id": first_str(getattr(pending, "proposal_msg_id", "")),
                "verdict": "needs_review",
            },
        )
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="reauthor_cap",
                rationale=f"reauthor attempts >= cap ({_AUTHORED_LANE_MAX_ATTEMPTS})",
                provenance="pump",
            )
        )
