"""Two-phase orchestration for the Critic decision pipeline.

The Critic SKILL is hosted by an A2A chat server (Codex). Each turn looks
roughly like:

1. Codex receives a prompt from the Coordinator (or a dialogue-style
   request). The SKILL extracts a request JSON and runs::

       python -m runtime.cli prepare-review --request request.json --out judge.json

   This produces a *judge bundle* with merged context, KB priors, and a
   list of proposals to review (when applicable).

2. The SKILL reasons over the judge bundle to produce a Critic-shaped
   review JSON (one of the schemas in ``references/``).

3. The SKILL runs::

       python -m runtime.cli commit-review --request request.json --review review.json --out emit.json

   to validate the review, persist it in session memory, optionally write
   to KB, and emit the Coordinator-compatible intent envelope.

This module hosts the deterministic logic for both phases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import (
    InboxParseError,
    IntentEnvelopeValidationError,
    RequestValidationError,
    ReviewValidationError,
    RuntimeAdapterError,
    ScopeError,
)
from .inbox_parser import parse_inbox_prompt
from .intent_envelope import (
    ALLOWED_VERDICTS,
    Intent,
    build_advice_intent,
    build_envelope,
    build_heartbeat_intent,
    build_review_verdict_intent,
)
from .kb_client import KBClient
from .kb_writer import KBWriter, WriteContext
from .metrics import CRITIC_REVIEW_VERDICT_TOTAL, get_registry
from .request_models import (
    COORDINATOR_INBOX,
    CRITICAL_CONTEXT_KEYS,
    CONTEXT_DIMENSIONS,
    CriticRequest,
    DECISION_REQUEST,
    KB_DRAFT_REQUEST,
    Proposal,
    parse_request,
)
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
@dataclass
class JudgeBundle:
    """Phase-1 output. The SKILL prompts read this to produce review JSON."""

    kind: str
    session_id: str
    decision_id: str | None
    merged_context: dict[str, Any] = field(default_factory=dict)
    missing_context: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    kb_priors_by_proposal: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    kb_priors_for_decision: list[dict[str, Any]] = field(default_factory=list)
    kb_read_skipped_reason: str | None = None
    review_constraints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "merged_context": dict(self.merged_context),
            "missing_context": list(self.missing_context),
            "required_context": list(self.required_context),
            "proposals": [dict(p) for p in self.proposals],
            "messages": [dict(m) for m in self.messages],
            "decision": dict(self.decision),
            "kb_priors_by_proposal": {
                k: [dict(p) for p in v] for k, v in self.kb_priors_by_proposal.items()
            },
            "kb_priors_for_decision": [dict(p) for p in self.kb_priors_for_decision],
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
    decision_review: dict[str, Any] | None = None
    kb_writes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "kb_writes": [dict(w) for w in self.kb_writes],
            "notes": list(self.notes),
        }
        if self.intent_envelope is not None:
            out["intent_envelope"] = self.intent_envelope
        if self.decision_review is not None:
            out["critic_decision_review"] = self.decision_review
        return out


# ---------------------------------------------------------------------------
class DecisionReviewer:
    """Orchestrator for the Critic prepare/commit lifecycle."""

    def __init__(
        self,
        *,
        session_memory: SessionMemory | None = None,
        kb_client: KBClient | None = None,
        kb_writer: KBWriter | None = None,
    ):
        self.session_memory = session_memory or SessionMemory()
        if kb_writer is None:
            if kb_client is None:
                # Lazily import to avoid creating a registry when not needed.
                from .in_memory_kb_client import InMemoryKBClient

                kb_client = InMemoryKBClient()
            kb_writer = KBWriter(kb_client, session_memory=self.session_memory)
        self.kb_writer = kb_writer

    # ------------------------------------------------------------------
    # Phase 0: init / close session
    # ------------------------------------------------------------------
    def init_session(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        req = parse_request(raw_request)
        merge = self.session_memory.merge_context(req.session_id, req.context)
        self.session_memory.append_event(req.session_id, {
            "kind": "init_session",
            "explicit_keys": merge.explicit_keys,
            "from_memory_keys": merge.from_memory_keys,
            "missing_keys": merge.missing_keys,
        })
        return {
            "session_id": req.session_id,
            "merged_context": merge.merged,
            "missing_context": merge.missing_keys,
        }

    def close_session(
        self,
        raw_request: dict[str, Any],
        kb_draft: dict[str, Any] | None = None,
    ) -> CommitOutcome:
        req = parse_request(raw_request)
        outcome = CommitOutcome(
            kind="session_close",
            session_id=req.session_id,
            decision_id=req.decision_id,
        )
        if kb_draft:
            drafts = list(kb_draft.get("kb_drafts") or [])
            ctx = WriteContext(
                session_id=req.session_id,
                review_id=req.decision_id,
                source_type="critic_kb_draft",
                topic=None,
            )
            session_ctx = self.session_memory.load_context(req.session_id)
            res = self.kb_writer.write_kb_drafts(
                kb_drafts=drafts,
                packet_context=req.context,
                session_context=session_ctx,
                ctx=ctx,
            )
            outcome.kb_writes.append({
                "trigger": "session_close",
                "result": res.to_dict(),
                "items": len(drafts),
            })
        self.session_memory.append_event(req.session_id, {
            "kind": "close_session",
            "kb_writes": [w["result"]["status"] for w in outcome.kb_writes],
        })
        return outcome

    # ------------------------------------------------------------------
    # Phase 1: prepare-review
    # ------------------------------------------------------------------
    def prepare_review(self, raw_request: dict[str, Any]) -> JudgeBundle:
        req = parse_request(raw_request)
        bundle = JudgeBundle(
            kind=req.kind,
            session_id=req.session_id,
            decision_id=req.decision_id,
        )
        self._populate_inbox(req)
        merge = self.session_memory.merge_context(req.session_id, req.context)
        bundle.merged_context = merge.merged
        bundle.missing_context = list(merge.missing_keys)
        bundle.proposals = [p.to_dict() for p in req.proposals]
        bundle.messages = list(req.messages)
        bundle.decision = dict(req.decision)
        bundle.review_constraints = self._review_constraints()

        # Hard requirement: if model/framework still unknown, KB reads must be
        # skipped and the Critic should fall back to needs_review.
        critical_missing = [
            k for k in CRITICAL_CONTEXT_KEYS if merge.merged.get(k) in (None, "", "unknown")
        ]
        if critical_missing:
            bundle.required_context = critical_missing
            bundle.kb_read_skipped_reason = "missing_critical_context"
            bundle.notes.append(
                "model and/or framework unknown — KB priors not fetched"
            )
            self.session_memory.append_event(req.session_id, {
                "kind": "prepare_review",
                "missing_critical": critical_missing,
                "kb_skipped": True,
            })
            return bundle

        # Skip KB reads only if explicitly disabled or inappropriate kind.
        if req.kind in (KB_DRAFT_REQUEST,):
            bundle.kb_read_skipped_reason = "kb_draft_does_not_need_priors"
            return bundle

        # KB reads disabled wholesale (operator switch) → reflect in bundle.
        if not self.kb_writer.read_enabled:
            bundle.kb_read_skipped_reason = "kb_read_disabled"
            bundle.notes.append("KB_READ_ENABLED=false — proceeding without priors")
            self.session_memory.append_event(req.session_id, {
                "kind": "prepare_review",
                "kb_skipped": True,
                "reason": "kb_read_disabled",
            })
            return bundle

        # Breaker already open from an earlier failure → short-circuit
        # before paying another timeout for this request.
        if self.kb_writer.is_kb_unreachable():
            bundle.kb_read_skipped_reason = "kb_unreachable"
            bundle.notes.append(
                "KB service unreachable (circuit breaker open); proceeding without priors"
            )
            bundle.review_constraints["kb_breaker"] = self.kb_writer.kb_breaker_state()
            self.session_memory.append_event(req.session_id, {
                "kind": "prepare_review",
                "kb_skipped": True,
                "reason": "kb_unreachable",
                "breaker": self.kb_writer.kb_breaker_state(),
            })
            return bundle

        scope = build_scope(req.context, session_context=merge.merged, require_critical=False)
        # Hide unknown values when listing priors so the service doesn't filter
        # to literal "unknown" rows.
        scope_filter = {k: v for k, v in scope.items() if v != "unknown"}

        topic_hits: dict[str, list[dict[str, Any]]] = {}
        ctx = WriteContext(session_id=req.session_id, review_id=req.decision_id)
        any_kb_unreachable = False
        if req.proposals:
            for p in req.proposals:
                topic = self._topic_for_proposal(p)
                priors = self.kb_writer.list_priors(
                    scope=scope_filter,
                    kind=None,
                    topic=topic,
                    metadata_filter=None,
                    limit=int(os.environ.get("CRITIC_KB_PRIOR_LIMIT", "5")),
                    ctx=ctx,
                )
                topic_hits[p.msg_id] = priors.get("priors") or []
                if priors.get("cache") == "kb_unreachable":
                    any_kb_unreachable = True
                self.session_memory.append_event(req.session_id, {
                    "kind": "kb_prior_lookup",
                    "msg_id": p.msg_id,
                    "topic": topic,
                    "cache": priors.get("cache"),
                    "count": len(topic_hits[p.msg_id]),
                })
            bundle.kb_priors_by_proposal = topic_hits
        else:
            topic = self._topic_for_decision(req)
            priors = self.kb_writer.list_priors(
                scope=scope_filter,
                kind=None,
                topic=topic,
                metadata_filter=None,
                limit=int(os.environ.get("CRITIC_KB_PRIOR_LIMIT", "5")),
                ctx=ctx,
            )
            bundle.kb_priors_for_decision = priors.get("priors") or []
            if priors.get("cache") == "kb_unreachable":
                any_kb_unreachable = True
            self.session_memory.append_event(req.session_id, {
                "kind": "kb_prior_lookup_decision",
                "topic": topic,
                "cache": priors.get("cache"),
                "count": len(bundle.kb_priors_for_decision),
            })

        if any_kb_unreachable:
            bundle.kb_read_skipped_reason = "kb_unreachable"
            bundle.notes.append(
                "KB service unreachable for at least one lookup — priors may be incomplete"
            )
            bundle.review_constraints["kb_breaker"] = self.kb_writer.kb_breaker_state()

        if scope.get("model") == "unknown" or scope.get("framework") == "unknown":
            bundle.notes.append("scope partially unknown — proceed with caution")
        bundle.notes.append(f"scope_cache_key={scope_cache_key(scope_filter)}")
        return bundle

    # ------------------------------------------------------------------
    # Phase 2: commit-review
    # ------------------------------------------------------------------
    def commit_review(
        self,
        raw_request: dict[str, Any],
        review: dict[str, Any],
    ) -> CommitOutcome:
        req = parse_request(raw_request)
        self._populate_inbox(req)
        if not isinstance(review, dict):
            raise ReviewValidationError(
                f"review must be a dict, got {type(review).__name__}"
            )

        outcome = CommitOutcome(
            kind=req.kind,
            session_id=req.session_id,
            decision_id=req.decision_id,
        )
        session_ctx = self.session_memory.load_context(req.session_id)

        if req.kind == COORDINATOR_INBOX:
            self._commit_coordinator_inbox(req, review, outcome, session_ctx)
        elif req.kind == DECISION_REQUEST:
            self._commit_decision_request(req, review, outcome, session_ctx)
        elif req.kind == KB_DRAFT_REQUEST:
            self._commit_kb_draft(req, review, outcome, session_ctx)
        else:
            raise ReviewValidationError(
                f"commit_review does not support kind={req.kind!r}"
            )
        return outcome

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    def _populate_inbox(self, req: CriticRequest) -> None:
        if req.kind != COORDINATOR_INBOX:
            return
        if req.proposals:
            return
        if not req.raw_prompt:
            return
        try:
            parsed = parse_inbox_prompt(req.raw_prompt)
        except InboxParseError:
            return
        if not req.context:
            req.context.update({k: parsed.shared_state[k] for k in parsed.shared_state})
        # Filter out proposals already handled in this session (idempotency).
        new_msg_ids = self.session_memory.filter_unreviewed(
            req.session_id, [p.msg_id for p in parsed.proposals]
        )
        keep = {mid for mid in new_msg_ids}
        req.proposals = [p for p in parsed.proposals if p.msg_id in keep]

    def _review_constraints(self) -> dict[str, Any]:
        return {
            "allowed_verdicts": sorted(ALLOWED_VERDICTS),
            "approve_requires": [
                "comparable_before_after_benchmark",
                "accuracy_gate_or_waiver",
                "active_path_proof_when_relevant",
                "rollback_plan",
            ],
            "ceiling_importance": 0.84,
        }

    def _topic_for_proposal(self, proposal: Proposal) -> str:
        if proposal.action_name:
            return proposal.action_name
        body = proposal.payload.get("topic") or proposal.payload.get("summary")
        if isinstance(body, str) and body.strip():
            return body.strip()
        return f"proposal-{proposal.msg_id}"

    def _topic_for_decision(self, req: CriticRequest) -> str:
        decision = req.decision
        if isinstance(decision, dict):
            for key in ("topic", "summary", "target"):
                value = decision.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return f"decision-{req.decision_id or req.session_id}"

    # ---- coordinator_inbox commit ------------------------------------
    def _commit_coordinator_inbox(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
        session_ctx: dict[str, Any],
    ) -> None:
        verdicts_raw = review.get("review_verdicts")
        if verdicts_raw is None:
            verdicts_raw = review.get("verdicts")
        if not isinstance(verdicts_raw, list):
            raise ReviewValidationError(
                "coordinator_inbox commit expects review.review_verdicts to be a list"
            )
        intents: list[Intent] = []
        for i, item in enumerate(verdicts_raw):
            if not isinstance(item, dict):
                raise ReviewValidationError(
                    f"review.review_verdicts[{i}] must be an object"
                )
            target = item.get("target_proposal_msg_id")
            verdict = item.get("verdict")
            if verdict not in ALLOWED_VERDICTS:
                raise ReviewValidationError(
                    f"review.review_verdicts[{i}].verdict {verdict!r} is not valid"
                )
            if not isinstance(target, str) or not target:
                raise ReviewValidationError(
                    f"review.review_verdicts[{i}].target_proposal_msg_id missing"
                )
            try:
                intent = build_review_verdict_intent(
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
                    advice_text=item.get("advice_text", ""),
                    notes=item.get("notes") or [],
                )
            except IntentEnvelopeValidationError as exc:
                raise ReviewValidationError(str(exc)) from exc
            intents.append(intent)
            self.session_memory.mark_reviewed(
                req.session_id, target, verdict, decision_id=req.decision_id
            )
            self.session_memory.append_decision(req.session_id, {
                "ts": _now_iso(),
                "target_proposal_msg_id": target,
                "verdict": verdict,
                "reasoning": item.get("reasoning"),
                "source": item.get("source", "critic"),
                "kb_evidence": item.get("kb_evidence") or [],
            })
            get_registry().counter(CRITIC_REVIEW_VERDICT_TOTAL).inc({"verdict": verdict})
            self._maybe_write_kb_for_verdict(item, req, session_ctx, outcome)

        # Optional advice for proposals approved with caveats.
        for advisory in review.get("advice") or []:
            if not isinstance(advisory, dict):
                continue
            body = advisory.get("body_md") or advisory.get("text")
            if not body:
                continue
            intents.append(build_advice_intent(
                body, target_proposal_msg_id=advisory.get("target_proposal_msg_id")
            ))

        # Heartbeat fallback if no proposals to review.
        if not intents:
            intents.append(build_heartbeat_intent())

        outcome.intent_envelope = build_envelope(intents).to_dict()

    # ---- decision_request commit -------------------------------------
    def _commit_decision_request(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
        session_ctx: dict[str, Any],
    ) -> None:
        for k in ("verdict", "reason"):
            if k not in review:
                raise ReviewValidationError(
                    f"decision_request review missing required key {k!r}"
                )
        if review["verdict"] not in {"adopt", "reject", "revise", "needs_info"}:
            raise ReviewValidationError(
                f"decision_request review.verdict {review['verdict']!r} not in "
                f"adopt|reject|revise|needs_info"
            )
        decision_review = {
            "kind": "critic_decision_review",
            "session_id": req.session_id,
            "decision_id": req.decision_id,
            "verdict": review["verdict"],
            "confidence": review.get("confidence", "medium"),
            "reason": review["reason"],
            "recommendation": review.get("recommendation", ""),
            "basis": review.get("basis", "mixed"),
            "kb_evidence": review.get("kb_evidence") or [],
            "session_evidence": review.get("session_evidence") or [],
            "required_context": review.get("required_context") or [],
            "notes": review.get("notes") or [],
        }
        self.session_memory.append_decision(req.session_id, {
            "ts": _now_iso(),
            "decision_review": decision_review,
        })
        # Translate adopt/reject into KB writes when there's reusable lesson.
        verdict_for_kb = {
            "adopt": "approve",
            "reject": "reject",
            "revise": "redirect",
            "needs_info": "needs_review",
        }[review["verdict"]]
        kb_payload = {
            "verdict": verdict_for_kb,
            "reasoning": review.get("reason", ""),
            "packet_evidence": review.get("session_evidence") or [],
            "kb_evidence": review.get("kb_evidence") or [],
            "risks": review.get("risks") or [],
            "confidence": review.get("confidence", "medium"),
            "predicted_gain_pct": review.get("predicted_gain_pct"),
        }
        ctx = WriteContext(
            session_id=req.session_id,
            review_id=req.decision_id,
            source_type=f"critic_decision_{review['verdict']}",
            topic=review.get("topic") or (req.decision.get("summary") if isinstance(req.decision, dict) else None),
        )
        write_res = self.kb_writer.write_verdict(
            verdict=kb_payload,
            packet_context=req.context,
            session_context=session_ctx,
            ctx=ctx,
        )
        outcome.kb_writes.append({
            "trigger": "decision_request",
            "result": write_res.to_dict(),
        })
        outcome.decision_review = decision_review

    # ---- kb_draft commit ----------------------------------------------
    def _commit_kb_draft(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
        session_ctx: dict[str, Any],
    ) -> None:
        drafts = review.get("kb_drafts") or []
        if not isinstance(drafts, list):
            raise ReviewValidationError("kb_draft commit expects review.kb_drafts list")
        ctx = WriteContext(
            session_id=req.session_id,
            review_id=req.decision_id,
            source_type="critic_kb_draft",
        )
        res = self.kb_writer.write_kb_drafts(
            kb_drafts=drafts,
            packet_context=req.context,
            session_context=session_ctx,
            ctx=ctx,
        )
        outcome.kb_writes.append({"trigger": "kb_draft", "result": res.to_dict()})
        outcome.decision_review = {
            "kind": "critic_kb_draft",
            "session_id": req.session_id,
            "kb_drafts_attempted": len(drafts),
            "kb_writes": [w["result"] for w in outcome.kb_writes],
        }

    def _maybe_write_kb_for_verdict(
        self,
        verdict_item: dict[str, Any],
        req: CriticRequest,
        session_ctx: dict[str, Any],
        outcome: CommitOutcome,
    ) -> None:
        verdict = verdict_item.get("verdict")
        if verdict not in {"reject", "redirect", "approve"}:
            return
        # Only write to KB when the SKILL flagged the verdict as
        # producing a reusable lesson — surface as ``persist_to_kb=True``.
        if not verdict_item.get("persist_to_kb"):
            return
        ctx = WriteContext(
            session_id=req.session_id,
            review_id=req.decision_id,
            source_type=f"critic_verdict_{verdict}",
            topic=verdict_item.get("topic"),
        )
        try:
            res = self.kb_writer.write_verdict(
                verdict=verdict_item,
                packet_context=req.context,
                session_context=session_ctx,
                ctx=ctx,
            )
            outcome.kb_writes.append({
                "trigger": "review_verdict",
                "target_proposal_msg_id": verdict_item.get("target_proposal_msg_id"),
                "result": res.to_dict(),
            })
        except RuntimeAdapterError as exc:
            outcome.notes.append(f"kb_write_skipped: {exc}")


__all__ = [
    "CommitOutcome",
    "DecisionReviewer",
    "JudgeBundle",
]
