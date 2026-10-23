# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two-phase orchestration for the Critic decision pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.framework_arm import is_upstream_pr_prescreen
from hyperloom.inference_optimizer.protocol.intent import ALLOWED_VERDICTS


from .errors import (
    IntentEnvelopeValidationError,
    ReviewValidationError,
)
from .importance_mapping import CRITIC_IMPORTANCE_CEILING
from .inbox_parser import parse_inbox_prompt
from .intent_envelope import (
    Intent,
    ReviewVerdict,
    build_advice_intent,
    build_envelope,
    build_idle_intent,
    build_review_verdict_intent,
)
from .kb_writer import KBWriter, WriteContext
from .request_models import (
    CRITICAL_CONTEXT_KEYS,
    CriticRequest,
    Proposal,
    parse_request,
)
from .scope_builder import build_scope, scope_cache_key


# Review-constraints taxonomy: proposals split into four classes and the bundle-level ``approve_requires`` collapses a
# batch to its strictest class.
ACTION_CLASS_PATCH_LANDING = "patch_landing"
ACTION_CLASS_EVIDENCE_PRODUCER = "evidence_producer"
ACTION_CLASS_FRAMEWORK_OP = "framework_op"
# Pre-boot enablement patches (enablement=True): a patch-landing action whose sole purpose
# is *runnability* (make the model boot at all), not throughput.
ACTION_CLASS_ENABLEMENT_LANDING = "enablement_landing"

_PATCH_LANDING_ACTIONS: frozenset[str] = frozenset(
    {
        "integrate",
        "integrate_patch",
        "apply_patch",
    }
)

_FRAMEWORK_OP_ACTIONS: frozenset[str] = frozenset(
    {
        "baseline",
        "target_analysis",
        "report",
        "session_breakdown",
    }
)

_APPROVE_REQUIRES_PATCH_LANDING: tuple[str, ...] = (
    "comparable_before_after_benchmark",
    "accuracy_gate_or_waiver",
    "active_path_proof_when_relevant",
    "rollback_plan",
)

_APPROVE_REQUIRES_EVIDENCE_PRODUCER: tuple[str, ...] = (
    "specialist_or_default_grid_provenance",
    "in_phase_allowed_action",
    # Reject on a contradicting KB prior; absence of priors is NOT a blocker.
    "no_contradicting_kb_prior",
)

# Pre-boot enablement patch: keep only the review-time-checkable safety checks.
_APPROVE_REQUIRES_ENABLEMENT_LANDING: tuple[str, ...] = (
    "specialist_or_default_grid_provenance",
    "in_phase_allowed_action",
    "no_contradicting_kb_prior",
)

_APPROVE_REQUIRES_FRAMEWORK_OP: tuple[str, ...] = ()

# Class precedence for collapsing a batch — strictest class wins.
_CLASS_RANK: dict[str, int] = {
    ACTION_CLASS_FRAMEWORK_OP: 0,
    ACTION_CLASS_ENABLEMENT_LANDING: 1,
    ACTION_CLASS_EVIDENCE_PRODUCER: 1,
    ACTION_CLASS_PATCH_LANDING: 2,
}

_APPROVE_REQUIRES_BY_CLASS: dict[str, tuple[str, ...]] = {
    ACTION_CLASS_PATCH_LANDING: _APPROVE_REQUIRES_PATCH_LANDING,
    ACTION_CLASS_EVIDENCE_PRODUCER: _APPROVE_REQUIRES_EVIDENCE_PRODUCER,
    ACTION_CLASS_ENABLEMENT_LANDING: _APPROVE_REQUIRES_ENABLEMENT_LANDING,
    ACTION_CLASS_FRAMEWORK_OP: _APPROVE_REQUIRES_FRAMEWORK_OP,
}


def _is_enablement_patch(payload: dict[str, Any] | None) -> bool:
    """Whether a patch-landing proposal is a pre-boot enablement patch."""
    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    if not isinstance(params, dict):
        return False
    return bool(params.get("enablement"))


def classify_proposal_action(action_name: str | None, payload: dict[str, Any] | None = None) -> str:
    """Map an action name (and optional payload) to its review class."""
    if not isinstance(action_name, str):
        return ACTION_CLASS_EVIDENCE_PRODUCER
    name = action_name.strip()
    if not name:
        return ACTION_CLASS_EVIDENCE_PRODUCER
    if name in _PATCH_LANDING_ACTIONS:
        if is_upstream_pr_prescreen(name, payload):
            return ACTION_CLASS_FRAMEWORK_OP
        if _is_enablement_patch(payload):
            return ACTION_CLASS_ENABLEMENT_LANDING
        return ACTION_CLASS_PATCH_LANDING
    if name in _FRAMEWORK_OP_ACTIONS:
        return ACTION_CLASS_FRAMEWORK_OP
    return ACTION_CLASS_EVIDENCE_PRODUCER


# ---------------------------------------------------------------------------
@dataclass
class JudgeBundle:
    """Phase-1 output. The SKILL prompts read this to produce review JSON."""

    kind: str
    session_id: str
    decision_id: str | None
    phase: str = ""
    merged_context: dict[str, Any] = field(default_factory=dict)
    missing_context: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    kb_priors_by_proposal: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Audit trail for the historical KB prior reads; consumed by the Coordinator.
    kb_priors_trace: dict[str, Any] = field(default_factory=dict)
    kb_read_skipped_reason: str | None = None
    review_constraints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-serialisable copy of the judge bundle."""
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "phase": self.phase,
            "merged_context": dict(self.merged_context),
            "missing_context": list(self.missing_context),
            "required_context": list(self.required_context),
            "proposals": [dict(p) for p in self.proposals],
            "kb_priors_by_proposal": {k: [dict(p) for p in v] for k, v in self.kb_priors_by_proposal.items()},
            "kb_priors_trace": dict(self.kb_priors_trace),
            "kb_read_skipped_reason": self.kb_read_skipped_reason,
            "review_constraints": dict(self.review_constraints),
            "notes": list(self.notes),
        }


@dataclass
class CommitOutcome:
    """Phase-2 output."""

    kind: str
    session_id: str
    decision_id: str | None
    intent_envelope: dict[str, Any] | None = None
    kb_writes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the commit outcome."""
        out = {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "kb_writes": [dict(w) for w in self.kb_writes],
        }
        if self.intent_envelope is not None:
            out["intent_envelope"] = self.intent_envelope
        return out


# ---------------------------------------------------------------------------
def _review_verdict(item: dict[str, Any], target: str, verdict: str, advice_text: str) -> ReviewVerdict:
    """The verdict one validated ``review.review_verdicts`` item states, carrying its merged advice."""
    return ReviewVerdict(
        target_proposal_msg_id=target,
        verdict=verdict,
        reasoning=item.get("reasoning", ""),
        source=item.get("source", "critic"),
        confidence=item.get("confidence"),
        predicted_gain_pct=item.get("predicted_gain_pct"),
        kb_evidence=item.get("kb_evidence") or [],
        packet_evidence=item.get("packet_evidence") or [],
        risks=item.get("risks") or [],
        required_evidence=item.get("required_evidence") or [],
        alternative_action=item.get("alternative_action"),
        advice_text=advice_text,
        notes=item.get("notes") or [],
        failure_reason_code=str(item.get("failure_reason_code") or ""),
    )


class DecisionReviewer:
    """Orchestrator for the Critic prepare/commit lifecycle."""

    def __init__(self, *, kb_writer: KBWriter):
        """Wire up the reviewer with its KB writer and the session memory that writer shares."""
        self.kb_writer = kb_writer
        self.session_memory = kb_writer.session_memory

    # Phase 1: prepare-review
    def prepare_review(self, raw_request: dict[str, Any]) -> JudgeBundle:
        """Build the phase-1 judge bundle for a review request."""
        req = parse_request(raw_request)
        bundle = JudgeBundle(
            kind=req.kind,
            session_id=req.session_id,
            decision_id=req.decision_id,
            phase=str(req.context.get("phase") or "").strip().upper(),
        )
        self._populate_inbox(req)
        merge = self.session_memory.merge_context(req.session_id, req.context)
        bundle.merged_context = merge.merged
        bundle.missing_context = list(merge.missing_keys)
        bundle.proposals = [p.to_dict() for p in req.proposals]
        bundle.review_constraints = self._review_constraints(req.proposals)
        known = req.options.get("known_actions")
        if isinstance(known, list) and known:
            bundle.review_constraints["known_actions"] = sorted(str(a) for a in known if isinstance(a, str))

        # If model/framework unknown, skip KB reads and fall back to needs_review.
        critical_missing = [k for k in CRITICAL_CONTEXT_KEYS if merge.merged.get(k) in (None, "", "unknown")]
        if critical_missing:
            bundle.required_context = critical_missing
            bundle.kb_read_skipped_reason = "missing_critical_context"
            bundle.notes.append("model and/or framework unknown — KB priors not fetched")
            return bundle

        # KB reads disabled wholesale (operator switch) → reflect in bundle.
        if not self.kb_writer.read_enabled:
            bundle.kb_read_skipped_reason = "kb_read_disabled"
            bundle.notes.append("KB_READ_ENABLED=false — proceeding without priors")
            return bundle

        # Breaker open from an earlier failure → skip another timeout.
        if self.kb_writer.is_kb_unreachable():
            bundle.kb_read_skipped_reason = "kb_unreachable"
            bundle.notes.append("KB service unreachable (circuit breaker open); proceeding without priors")
            bundle.review_constraints["kb_breaker"] = self.kb_writer.kb_breaker_state()
            return bundle

        self._fetch_kb_priors(bundle, req, merge.merged)
        return bundle

    def _fetch_kb_priors(self, bundle: JudgeBundle, req: CriticRequest, merged: dict[str, Any]) -> None:
        """Read the KB priors for each proposal into ``bundle`` with their audit trail."""
        scope = build_scope(req.context, session_context=merged, require_critical=False)
        # Hide "unknown" so the service doesn't filter to literal "unknown" rows.
        scope_filter = {k: v for k, v in scope.items() if v != "unknown"}
        ctx = WriteContext(session_id=req.session_id, review_id=req.decision_id)
        prior_limit = int(os.environ.get("CRITIC_KB_PRIOR_LIMIT", "5"))
        priors_requests: list[dict[str, Any]] = []
        for proposal in req.proposals:
            topic = self._topic_for_proposal(proposal)
            priors = self.kb_writer.list_priors(scope=scope_filter, kind=None, topic=topic, limit=prior_limit, ctx=ctx)
            hits = priors.get("priors") or []
            bundle.kb_priors_by_proposal[proposal.msg_id] = hits
            priors_requests.append(
                {"msg_id": proposal.msg_id, "topic": topic, "cache": priors.get("cache"), "count": len(hits)}
            )

        bundle.kb_priors_trace = {
            "configured": True,
            "mode": "per_proposal",
            "client_mode": os.environ.get("CRITIC_KB_CLIENT_MODE", ""),
            "scope_filter": dict(scope_filter),
            "limit": prior_limit,
            "requests": priors_requests,
        }
        if any(r["cache"] == "kb_unreachable" for r in priors_requests):
            bundle.kb_read_skipped_reason = "kb_unreachable"
            bundle.notes.append("KB service unreachable for at least one lookup — priors may be incomplete")
            bundle.review_constraints["kb_breaker"] = self.kb_writer.kb_breaker_state()
        if scope.get("model") == "unknown" or scope.get("framework") == "unknown":
            bundle.notes.append("scope partially unknown — proceed with caution")
        bundle.notes.append(f"scope_cache_key={scope_cache_key(scope_filter)}")

    # Phase 2: commit-review
    def commit_review(
        self,
        raw_request: dict[str, Any],
        review: dict[str, Any],
    ) -> CommitOutcome:
        """Validate and commit a SKILL-produced review (phase 2)."""
        req = parse_request(raw_request)
        self._populate_inbox(req)
        if not isinstance(review, dict):
            raise ReviewValidationError(f"review must be a dict, got {type(review).__name__}")

        outcome = CommitOutcome(
            kind=req.kind,
            session_id=req.session_id,
            decision_id=req.decision_id,
        )
        session_ctx = self.session_memory.load_context(req.session_id)
        self._commit_coordinator_inbox(req, review, outcome, session_ctx)
        return outcome

    # Implementation helpers
    def _populate_inbox(self, req: CriticRequest) -> None:
        """Parse proposals from a coordinator-inbox prompt, in place."""
        if req.proposals:
            return
        parsed = parse_inbox_prompt(req.raw_prompt)
        if not req.context:
            req.context.update({k: parsed.shared_state[k] for k in parsed.shared_state})
        # Filter out proposals already handled in this session (idempotency).
        new_msg_ids = self.session_memory.filter_unreviewed(req.session_id, [p.msg_id for p in parsed.proposals])
        keep = set(new_msg_ids)
        req.proposals = [p for p in parsed.proposals if p.msg_id in keep]

    def _review_constraints(
        self,
        proposals: list[Proposal] | None = None,
    ) -> dict[str, Any]:
        """Return the per-bundle review constraints payload."""
        constraints: dict[str, Any] = {
            "allowed_verdicts": sorted(ALLOWED_VERDICTS),
            "ceiling_importance": CRITIC_IMPORTANCE_CEILING,
        }
        if not proposals:
            constraints["approve_requires"] = list(_APPROVE_REQUIRES_PATCH_LANDING)
            return constraints
        per_proposal: dict[str, str] = {}
        max_rank = -1
        max_class = ACTION_CLASS_EVIDENCE_PRODUCER
        for p in proposals:
            cls = classify_proposal_action(p.action_name, p.payload)
            per_proposal[p.msg_id] = cls
            rank = _CLASS_RANK[cls]
            if rank > max_rank:
                max_rank = rank
                max_class = cls
        constraints["approve_requires"] = list(_APPROVE_REQUIRES_BY_CLASS[max_class])
        constraints["bundle_action_class"] = max_class
        constraints["proposal_action_classes"] = per_proposal
        constraints["approve_requires_by_class"] = {cls: list(reqs) for cls, reqs in _APPROVE_REQUIRES_BY_CLASS.items()}
        return constraints

    def _topic_for_proposal(self, proposal: Proposal) -> str:
        """Derive the KB lookup topic for a proposal."""
        if proposal.action_name:
            return proposal.action_name
        body = proposal.payload.get("topic") or proposal.payload.get("summary")
        if isinstance(body, str) and body.strip():
            return body.strip()
        return f"proposal-{proposal.msg_id}"

    # ---- coordinator_inbox commit ------------------------------------
    def _commit_coordinator_inbox(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
        session_ctx: dict[str, Any],
    ) -> None:
        """Commit a coordinator-inbox review into an intent envelope."""
        verdicts_raw = review.get("review_verdicts")
        if not isinstance(verdicts_raw, list):
            raise ReviewValidationError("coordinator_inbox commit expects review.review_verdicts to be a list")

        advice = [a for a in review.get("advice") or [] if isinstance(a, dict) and a.get("body_md")]
        advice_by_target: dict[str, list[str]] = {}
        for advisory in advice:
            advice_target = advisory.get("target_proposal_msg_id")
            if isinstance(advice_target, str) and advice_target:
                advice_by_target.setdefault(advice_target, []).append(advisory["body_md"])

        # Every verdict is validated before any of them is committed.
        seen_targets: set[str] = set()
        intents: list[Intent] = []
        for i, item in enumerate(verdicts_raw):
            if not isinstance(item, dict):
                raise ReviewValidationError(f"review.review_verdicts[{i}] must be an object")
            target = item.get("target_proposal_msg_id")
            verdict = item.get("verdict")
            if verdict not in ALLOWED_VERDICTS:
                raise ReviewValidationError(f"review.review_verdicts[{i}].verdict {verdict!r} is not valid")
            if not isinstance(target, str) or not target:
                raise ReviewValidationError(f"review.review_verdicts[{i}].target_proposal_msg_id missing")
            if target in seen_targets:
                raise ReviewValidationError(
                    f"review.review_verdicts[{i}].target_proposal_msg_id {target!r} appears more than once"
                )
            seen_targets.add(target)
            advice_parts = [
                part
                for part in [item.get("advice_text", ""), *advice_by_target.get(target, [])]
                if isinstance(part, str) and part.strip()
            ]
            advice_text = "\n\n".join(advice_parts)
            try:
                intents.append(build_review_verdict_intent(_review_verdict(item, target, verdict, advice_text)))
            except IntentEnvelopeValidationError as exc:
                raise ReviewValidationError(str(exc)) from exc

        for item in verdicts_raw:
            target, verdict = item["target_proposal_msg_id"], item["verdict"]
            self.session_memory.mark_reviewed(req.session_id, target, verdict, decision_id=req.decision_id)
            self.session_memory.append_decision(
                req.session_id,
                {
                    "target_proposal_msg_id": target,
                    "verdict": verdict,
                    "reasoning": item.get("reasoning"),
                    "source": item.get("source", "critic"),
                    "failure_reason_code": str(item.get("failure_reason_code") or ""),
                    "kb_evidence": item.get("kb_evidence") or [],
                },
            )
            self._maybe_write_kb_for_verdict(item, req, session_ctx, outcome)

        # Every advisory also goes out as a standalone advice message, whatever the target's verdict (or if
        # untargeted); targeted bodies are already inlined into the verdict's ``advice_text`` above.
        for advisory in advice:
            intents.append(
                build_advice_intent(advisory["body_md"], target_proposal_msg_id=advisory.get("target_proposal_msg_id"))
            )

        # Nothing to review — still report in.
        if not intents:
            intents.append(build_idle_intent())

        outcome.intent_envelope = build_envelope(intents).to_dict()

    def _maybe_write_kb_for_verdict(
        self,
        verdict_item: dict[str, Any],
        req: CriticRequest,
        session_ctx: dict[str, Any],
        outcome: CommitOutcome,
    ) -> None:
        """Write a KB lesson for a verdict when the SKILL opted in."""
        verdict = verdict_item.get("verdict")
        if verdict not in {"reject", "redirect", "approve"}:
            return
        # Only persist when the SKILL opted in.
        if not verdict_item.get("persist_to_kb"):
            return
        res = self.kb_writer.write_verdict(
            verdict=verdict_item,
            packet_context=req.context,
            session_context=session_ctx,
            ctx=WriteContext(session_id=req.session_id, review_id=req.decision_id),
        )
        outcome.kb_writes.append(
            {
                "trigger": "review_verdict",
                "target_proposal_msg_id": verdict_item.get("target_proposal_msg_id"),
                "result": res.to_dict(),
            }
        )


__all__ = [
    "CommitOutcome",
    "DecisionReviewer",
    "JudgeBundle",
]
