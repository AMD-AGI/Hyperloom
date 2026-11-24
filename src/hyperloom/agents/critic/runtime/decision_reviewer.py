# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two-phase orchestration for the Critic decision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.framework_arm import is_upstream_pr_prescreen
from hyperloom.inference_optimizer.protocol.intent import ALLOWED_VERDICTS


from .errors import (
    IntentEnvelopeValidationError,
    ReviewValidationError,
)
from .inbox_parser import parse_inbox_prompt
from .intent_envelope import (
    Intent,
    ReviewVerdict,
    build_advice_intent,
    build_envelope,
    build_idle_intent,
    build_review_verdict_intent,
)
from .request_models import (
    CRITICAL_CONTEXT_KEYS,
    CriticRequest,
    Proposal,
    parse_request,
)
from .session_memory import SessionMemory


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

    session_id: str
    phase: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    required_context: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    review_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-serialisable copy of the judge bundle."""
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "context": dict(self.context),
            "required_context": list(self.required_context),
            "proposals": [dict(p) for p in self.proposals],
            "review_constraints": dict(self.review_constraints),
        }


@dataclass
class CommitOutcome:
    """Phase-2 output."""

    session_id: str
    intent_envelope: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the commit outcome."""
        out: dict[str, Any] = {"session_id": self.session_id}
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


def _advice_by_target(advice: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The ``body_md`` of each advisory, grouped by the proposal it targets; untargeted ones are left out."""
    by_target: dict[str, list[str]] = {}
    for advisory in advice:
        advice_target = advisory.get("target_proposal_msg_id")
        if isinstance(advice_target, str) and advice_target:
            by_target.setdefault(advice_target, []).append(advisory["body_md"])
    return by_target


class DecisionReviewer:
    """Orchestrator for the Critic prepare/commit lifecycle."""

    def __init__(self, *, session_memory: SessionMemory):
        """Wire up the reviewer with the session memory it reads and records into."""
        self.session_memory = session_memory

    # Phase 1: prepare-review
    def prepare_review(self, raw_request: dict[str, Any]) -> JudgeBundle:
        """Build the phase-1 judge bundle for a review request."""
        req = parse_request(raw_request)
        bundle = JudgeBundle(
            session_id=req.session_id,
            phase=str(req.context.get("phase") or "").strip().upper(),
        )
        proposals = self._unreviewed_proposals(req)
        bundle.context = dict(req.context)
        bundle.proposals = [p.to_dict() for p in proposals]
        bundle.review_constraints = self._review_constraints(proposals)

        # If model/framework unknown, fall back to needs_review.
        bundle.required_context = [k for k in CRITICAL_CONTEXT_KEYS if req.context.get(k) in (None, "", "unknown")]
        return bundle

    # Phase 2: commit-review
    def commit_review(
        self,
        raw_request: dict[str, Any],
        review: dict[str, Any],
    ) -> CommitOutcome:
        """Validate and commit a SKILL-produced review (phase 2)."""
        req = parse_request(raw_request)
        if not isinstance(review, dict):
            raise ReviewValidationError(f"review must be a dict, got {type(review).__name__}")

        outcome = CommitOutcome(session_id=req.session_id)
        self._commit_coordinator_inbox(req, review, outcome)
        return outcome

    # Implementation helpers
    def _unreviewed_proposals(self, req: CriticRequest) -> list[Proposal]:
        """The proposals in a coordinator-inbox prompt that this session has not reviewed yet."""
        proposals = parse_inbox_prompt(req.raw_prompt)
        keep = set(self.session_memory.filter_unreviewed(req.session_id, [p.msg_id for p in proposals]))
        return [p for p in proposals if p.msg_id in keep]

    def _review_constraints(
        self,
        proposals: list[Proposal] | None = None,
    ) -> dict[str, Any]:
        """Return the per-bundle review constraints payload."""
        constraints: dict[str, Any] = {
            "allowed_verdicts": sorted(ALLOWED_VERDICTS),
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

    # ---- coordinator_inbox commit ------------------------------------
    def _commit_coordinator_inbox(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
    ) -> None:
        """Commit a coordinator-inbox review into an intent envelope."""
        verdicts_raw = review.get("review_verdicts")
        if not isinstance(verdicts_raw, list):
            raise ReviewValidationError("coordinator_inbox commit expects review.review_verdicts to be a list")

        advice = [a for a in review.get("advice") or [] if isinstance(a, dict) and a.get("body_md")]
        advice_by_target = _advice_by_target(advice)

        # Every verdict is validated before any of them is committed.
        seen_targets: set[str] = set()
        intents: list[Intent] = []
        reviews: list[ReviewVerdict] = []
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
            verdict_review = _review_verdict(item, target, verdict, advice_text)
            reviews.append(verdict_review)
            try:
                intents.append(build_review_verdict_intent(verdict_review))
            except IntentEnvelopeValidationError as exc:
                raise ReviewValidationError(str(exc)) from exc

        for verdict_review in reviews:
            self.session_memory.mark_reviewed(
                req.session_id, verdict_review.target_proposal_msg_id, verdict_review.verdict
            )

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


__all__ = [
    "CommitOutcome",
    "DecisionReviewer",
    "JudgeBundle",
]
