# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two-phase orchestration for the Critic decision pipeline.

Phase 1 (``prepare-review``) builds a *judge bundle* with merged context,
KB priors, and proposals to review. Phase 2 (``commit-review``) validates
the SKILL's review JSON, persists it to session memory, optionally writes
to KB, and emits the Coordinator-compatible intent envelope. This module
hosts the deterministic logic for both phases.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.agents.robustness.role.findings import FINDINGS_SUBDIR
from hyperloom.common.timeutil import now_iso


log = logging.getLogger(__name__)

from .errors import (
    InboxParseError,
    IntentEnvelopeValidationError,
    ReviewValidationError,
    RuntimeAdapterError,
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
    CriticRequest,
    DECISION_REQUEST,
    KB_DRAFT_REQUEST,
    Proposal,
    parse_request,
)
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory


# Review-constraints taxonomy: proposals split into four classes and the
# bundle-level ``approve_requires`` collapses a batch to its strictest class.
ACTION_CLASS_PATCH_LANDING = "patch_landing"
ACTION_CLASS_EVIDENCE_PRODUCER = "evidence_producer"
ACTION_CLASS_FRAMEWORK_OP = "framework_op"
# Pre-boot enablement patches (framework-agent authoring / enablement=True):
# a patch-landing action whose sole purpose is *runnability* (make the model
# boot at all), not throughput. It is dispatched before any usable baseline
# exists, so the production ``patch_landing`` evidence — a comparable
# before/after benchmark and an accuracy gate — is impossible by construction.
# Rollback is provided structurally by the enablement integrate executor
# (``git apply`` + REVERT via ``git reset --hard`` + artifact backup) and by
# the downstream runnable-decision gate (boot-probe -> REVERT on failure), so
# it must not be judged under the full production bar.
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
        "recover",
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
# The production evidence (comparable_before_after_benchmark / accuracy_gate)
# cannot exist before the model boots, and rollback is guaranteed by the
# enablement integrate executor + runnable-decision gate, so both are dropped.
_APPROVE_REQUIRES_ENABLEMENT_LANDING: tuple[str, ...] = (
    "specialist_or_default_grid_provenance",
    "in_phase_allowed_action",
    "no_contradicting_kb_prior",
)

_APPROVE_REQUIRES_FRAMEWORK_OP: tuple[str, ...] = ()

# Class precedence for collapsing a batch — strictest class wins. Enablement
# landing sits below production patch-landing (a real promotion in the same
# batch still forces the strict bar) but above framework ops.
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


def _is_upstream_pr_prescreen(payload: dict[str, Any] | None) -> bool:
    """Whether this proposal only decides *whether to spend a bench* on a PR.

    A candidate pre-screen arrives as ``integrate_patch`` -- one action lands
    every patch source -- but nothing has been applied or measured yet, so the
    patch-landing evidence bar (before/after benchmark, accuracy gate, rollback
    plan) cannot be met by construction. Approval here means only "this
    candidate is worth a GPU bench"; the resulting measurement is what the
    executor's own gate then judges.

    Distinguished by a top-level ``framework_agent_candidate_id``, which the
    pre-screen carries and an actual patch application does not.

    Args:
        payload: The proposal payload.

    Returns:
        True when this is a candidate pre-screen rather than a patch landing.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("patches") or (payload.get("params") or {}).get("patches"):
        return False
    return bool(payload.get("framework_agent_candidate_id"))


def _is_enablement_patch(payload: dict[str, Any] | None) -> bool:
    """Whether a patch-landing proposal is a pre-boot enablement patch.

    Enablement patches are tagged by the framework-agent authoring path via
    ``payload.params.enablement`` / ``payload.params.framework_agent_authoring``
    (see ``orchestrator.phases.framework``). Their purpose is runnability, not
    throughput, so they are reviewed under the lighter ``enablement_landing``
    bar instead of the production ``patch_landing`` bar.

    Args:
        payload: The raw proposal payload, if any.

    Returns:
        ``True`` when the payload carries an enablement authoring marker.
    """
    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    if not isinstance(params, dict):
        return False
    return bool(params.get("enablement")) or bool(params.get("framework_agent_authoring"))


def classify_proposal_action(action_name: str | None, payload: dict[str, Any] | None = None) -> str:
    """Map an action name (and optional payload) to its review class.

    Unknown or missing actions fall back to the evidence-producer class,
    which is the cold-start-safe default. A patch-landing action carrying an
    enablement marker in ``payload`` is routed to the lighter
    ``enablement_landing`` class (see :func:`_is_enablement_patch`), and an
    upstream-PR candidate pre-screen to ``framework_op`` (see
    :func:`_is_upstream_pr_prescreen`).

    Args:
        action_name: The proposed action's name, if any.
        payload: The proposal payload, used to detect enablement patches.

    Returns:
        The review class constant for the action.
    """
    if not isinstance(action_name, str):
        return ACTION_CLASS_EVIDENCE_PRODUCER
    name = action_name.strip()
    if not name:
        return ACTION_CLASS_EVIDENCE_PRODUCER
    if name in _PATCH_LANDING_ACTIONS:
        if _is_upstream_pr_prescreen(payload):
            return ACTION_CLASS_FRAMEWORK_OP
        if _is_enablement_patch(payload):
            return ACTION_CLASS_ENABLEMENT_LANDING
        return ACTION_CLASS_PATCH_LANDING
    if name in _FRAMEWORK_OP_ACTIONS:
        return ACTION_CLASS_FRAMEWORK_OP
    return ACTION_CLASS_EVIDENCE_PRODUCER


# Robustness finding discovery / load helpers.

# Severity rank for the "min_severity" filter: high > medium > low.
_SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

# Findings-sink JSONL subdir; imported from the robustness role layer
# (``role.findings.FINDINGS_SUBDIR``, the on-disk source of truth) so this
# cross-package path cannot silently drift from where the FindingSink writes.
_ROBUSTNESS_FINDINGS_SUBDIR: str = FINDINGS_SUBDIR


def _discover_robustness_findings_path(session_id: str) -> Path | None:
    """Locate ``<session>.jsonl`` from the Robustness FindingSink.

    Resolution order: ``$CRITIC_ROBUSTNESS_FINDINGS_DIR`` (explicit override
    dir) then ``$ROBUSTNESS_AGENT_SESSION_DIR`` (auto-discovery).

    Args:
        session_id: Session identifier used to build the ``<session>.jsonl``
            filename (falls back to ``"default"`` when empty).

    Returns:
        Path to the findings file, or ``None`` when neither env var is set
        or the file does not exist.
    """
    explicit = os.environ.get("CRITIC_ROBUSTNESS_FINDINGS_DIR", "").strip()
    if explicit:
        candidate = Path(explicit) / f"{session_id or 'default'}.jsonl"
        return candidate if candidate.is_file() else None
    session_dir = os.environ.get("ROBUSTNESS_AGENT_SESSION_DIR", "").strip()
    if not session_dir:
        return None
    candidate = Path(session_dir) / _ROBUSTNESS_FINDINGS_SUBDIR / f"{session_id or 'default'}.jsonl"
    return candidate if candidate.is_file() else None


def _load_robustness_priors(
    path: Path,
    *,
    limit: int,
    min_severity: str,
) -> list[dict[str, Any]]:
    """Tail the JSONL and return priors that meet the severity floor.

    Args:
        path: Path to the robustness findings JSONL file.
        limit: Maximum number of priors to return.
        min_severity: Minimum severity to include (``"high"``,
            ``"medium"``, or ``"low"``).

    Returns:
        Up to ``limit`` prior records matching the severity filter; an
        empty list if the file cannot be read.
    """
    min_rank = _SEVERITY_RANK.get(min_severity, _SEVERITY_RANK["high"])
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning(
            "critic: cannot read robustness findings %s: %s",
            path,
            exc,
        )
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        sev = str(obj.get("severity") or "").lower()
        if _SEVERITY_RANK.get(sev, 0) < min_rank:
            continue
        rows.append(obj)
    if not rows:
        return []
    selected = rows[-limit:]
    # Narrow projection to what the SKILL needs.
    out: list[dict[str, Any]] = []
    for row in selected:
        out.append(
            {
                "symptom_name": row.get("symptom_name"),
                "severity": row.get("severity"),
                "tick_index": row.get("tick_index"),
                "timestamp_unix": row.get("timestamp_unix"),
                "summary": row.get("summary"),
                "rca_text": row.get("rca_text"),
                "intents": [
                    {
                        "intent_type": (i or {}).get("intent_type"),
                        "payload": (i or {}).get("payload"),
                    }
                    for i in (row.get("intents") or [])
                    if isinstance(i, dict)
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
@dataclass
class JudgeBundle:
    """Phase-1 output. The SKILL prompts read this to produce review JSON.

    Attributes:
        kind (str): The request kind being prepared.
        session_id (str): The owning A2A session id.
        decision_id (str | None): The decision id when reviewing a decision.
        phase (str): Coordinator pipeline phase this review belongs to, taken
            from ``request.context.phase``; ``""`` when the caller does not
            track phases.
        merged_context (dict[str, Any]): Context after session-memory merge.
        missing_context (list[str]): Mergeable context keys still missing.
        required_context (list[str]): Critical keys that block KB reads.
        proposals (list[dict[str, Any]]): Proposals to review, as dicts.
        messages (list[dict[str, Any]]): Inbox messages carried for review.
        decision (dict[str, Any]): The decision payload, when applicable.
        kb_priors_by_proposal (dict[str, list[dict[str, Any]]]): KB priors keyed
            by proposal ``msg_id``.
        kb_priors_for_decision (list[dict[str, Any]]): KB priors for a
            decision-level review.
        robustness_priors (list[dict[str, Any]]): Recent Robustness findings
            injected as priors.
        kb_read_skipped_reason (str | None): Why KB reads were skipped, if any.
        review_constraints (dict[str, Any]): Constraints/checklists for the SKILL.
        notes (list[str]): Free-form diagnostic notes.
    """

    kind: str
    session_id: str
    decision_id: str | None
    phase: str = ""
    merged_context: dict[str, Any] = field(default_factory=dict)
    missing_context: list[str] = field(default_factory=list)
    required_context: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] = field(default_factory=dict)
    kb_priors_by_proposal: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    kb_priors_for_decision: list[dict[str, Any]] = field(default_factory=list)
    # Audit trail for the historical KB prior reads; consumed by the Coordinator.
    kb_priors_trace: dict[str, Any] = field(default_factory=dict)
    # Recent Robustness findings; empty when absent or disabled.
    robustness_priors: list[dict[str, Any]] = field(default_factory=list)
    kb_read_skipped_reason: str | None = None
    review_constraints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-serialisable copy of the judge bundle.

        Returns:
            dict[str, Any]: All bundle fields with nested dicts/lists copied so
            the result can be serialised and emitted without aliasing internal
            state.
        """
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "decision_id": self.decision_id,
            "phase": self.phase,
            "merged_context": dict(self.merged_context),
            "missing_context": list(self.missing_context),
            "required_context": list(self.required_context),
            "proposals": [dict(p) for p in self.proposals],
            "messages": [dict(m) for m in self.messages],
            "decision": dict(self.decision),
            "kb_priors_by_proposal": {k: [dict(p) for p in v] for k, v in self.kb_priors_by_proposal.items()},
            "kb_priors_for_decision": [dict(p) for p in self.kb_priors_for_decision],
            "kb_priors_trace": dict(self.kb_priors_trace),
            "robustness_priors": [dict(p) for p in self.robustness_priors],
            "kb_read_skipped_reason": self.kb_read_skipped_reason,
            "review_constraints": dict(self.review_constraints),
            "notes": list(self.notes),
        }


@dataclass
class CommitOutcome:
    """Phase-2 output.

    Attributes:
        kind (str): The committed request kind.
        session_id (str): The owning A2A session id.
        decision_id (str | None): The decision id, when applicable.
        intent_envelope (dict[str, Any] | None): The Coordinator-compatible
            intent envelope, when one was produced.
        decision_review (dict[str, Any] | None): The persisted decision-review
            record, when one was produced.
        kb_writes (list[dict[str, Any]]): Summaries of KB write attempts.
        notes (list[str]): Free-form diagnostic notes.
    """

    kind: str
    session_id: str
    decision_id: str | None
    intent_envelope: dict[str, Any] | None = None
    decision_review: dict[str, Any] | None = None
    kb_writes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the commit outcome.

        Optional fields (``intent_envelope`` / ``decision_review``) are only
        included when set; the latter is emitted under the
        ``critic_decision_review`` key.

        Returns:
            dict[str, Any]: The serialisable outcome payload.
        """
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
        """Wire up the reviewer with its memory, KB client and writer.

        Args:
            session_memory (SessionMemory | None): Session store; a default is
                created when ``None``.
            kb_client (KBClient | None): KB client used to build a default
                ``KBWriter`` when none is supplied; an ``InMemoryKBClient`` is
                created if both ``kb_writer`` and ``kb_client`` are ``None``.
            kb_writer (KBWriter | None): KB write façade; built from
                ``kb_client`` and ``session_memory`` when ``None``.
        """
        self.session_memory = session_memory or SessionMemory()
        if kb_writer is None:
            if kb_client is None:
                from .in_memory_kb_client import InMemoryKBClient

                kb_client = InMemoryKBClient()
            kb_writer = KBWriter(kb_client, session_memory=self.session_memory)
        self.kb_writer = kb_writer

    # ------------------------------------------------------------------
    # Phase 0: init / close session
    # ------------------------------------------------------------------
    def init_session(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        """Initialise a session by merging its first context payload.

        Args:
            raw_request (dict[str, Any]): The raw Critic request envelope.

        Returns:
            dict[str, Any]: A dict with ``session_id``, ``merged_context`` and
            the list of still-``missing_context`` keys.
        """
        req = parse_request(raw_request)
        merge = self.session_memory.merge_context(req.session_id, req.context)
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
        """Close a session, optionally flushing KB drafts to the KB.

        Args:
            raw_request (dict[str, Any]): The raw Critic request envelope.
            kb_draft (dict[str, Any] | None): Optional payload whose
                ``kb_drafts`` list is written on close.

        Returns:
            CommitOutcome: Outcome of kind ``session_close`` recording any KB
            writes performed.
        """
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
            outcome.kb_writes.append(
                {
                    "trigger": "session_close",
                    "result": res.to_dict(),
                    "items": len(drafts),
                }
            )
        return outcome

    # ------------------------------------------------------------------
    # Phase 1: prepare-review
    # ------------------------------------------------------------------
    def prepare_review(self, raw_request: dict[str, Any]) -> JudgeBundle:
        """Build the phase-1 judge bundle for a review request.

        Merges context, classifies proposals, and (when critical context is
        present and KB reads are enabled and the breaker is closed) fetches KB
        priors per proposal or per decision plus recent Robustness priors.

        Args:
            raw_request (dict[str, Any]): The raw Critic request envelope.

        Returns:
            JudgeBundle: The assembled bundle the SKILL prompt reasons over to
            produce a review.
        """
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
        bundle.messages = list(req.messages)
        bundle.decision = dict(req.decision)
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

        # Skip KB reads only if explicitly disabled or inappropriate kind.
        if req.kind in (KB_DRAFT_REQUEST,):
            bundle.kb_read_skipped_reason = "kb_draft_does_not_need_priors"
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

        scope = build_scope(req.context, session_context=merge.merged, require_critical=False)
        # Hide "unknown" so the service doesn't filter to literal "unknown" rows.
        scope_filter = {k: v for k, v in scope.items() if v != "unknown"}

        topic_hits: dict[str, list[dict[str, Any]]] = {}
        ctx = WriteContext(session_id=req.session_id, review_id=req.decision_id)
        any_kb_unreachable = False
        prior_limit = int(os.environ.get("CRITIC_KB_PRIOR_LIMIT", "5"))
        priors_requests: list[dict[str, Any]] = []
        if req.proposals:
            for p in req.proposals:
                topic = self._topic_for_proposal(p)
                priors = self.kb_writer.list_priors(
                    scope=scope_filter,
                    kind=None,
                    topic=topic,
                    limit=prior_limit,
                    ctx=ctx,
                )
                topic_hits[p.msg_id] = priors.get("priors") or []
                if priors.get("cache") == "kb_unreachable":
                    any_kb_unreachable = True
                priors_requests.append(
                    {
                        "msg_id": p.msg_id,
                        "topic": topic,
                        "cache": priors.get("cache"),
                        "count": len(topic_hits[p.msg_id]),
                    }
                )
            bundle.kb_priors_by_proposal = topic_hits
        else:
            topic = self._topic_for_decision(req)
            priors = self.kb_writer.list_priors(
                scope=scope_filter,
                kind=None,
                topic=topic,
                limit=prior_limit,
                ctx=ctx,
            )
            bundle.kb_priors_for_decision = priors.get("priors") or []
            if priors.get("cache") == "kb_unreachable":
                any_kb_unreachable = True
            priors_requests.append(
                {
                    "msg_id": None,
                    "topic": topic,
                    "cache": priors.get("cache"),
                    "count": len(bundle.kb_priors_for_decision),
                }
            )

        # Audit trail for the historical KB prior reads (always injected).
        bundle.kb_priors_trace = {
            "configured": True,
            "mode": "per_proposal" if req.proposals else "per_decision",
            "client_mode": os.environ.get("CRITIC_KB_CLIENT_MODE", ""),
            "scope_filter": dict(scope_filter),
            "limit": prior_limit,
            "requests": priors_requests,
        }

        if any_kb_unreachable:
            bundle.kb_read_skipped_reason = "kb_unreachable"
            bundle.notes.append("KB service unreachable for at least one lookup — priors may be incomplete")
            bundle.review_constraints["kb_breaker"] = self.kb_writer.kb_breaker_state()

        if scope.get("model") == "unknown" or scope.get("framework") == "unknown":
            bundle.notes.append("scope partially unknown — proceed with caution")
        bundle.notes.append(f"scope_cache_key={scope_cache_key(scope_filter)}")

        # Best-effort recent Robustness findings; never blocks the review.
        self._inject_robustness_priors(bundle)
        return bundle

    def _inject_robustness_priors(self, bundle: JudgeBundle) -> None:
        """Populate ``bundle.robustness_priors`` from recent findings.

        Best-effort: disabled via ``CRITIC_ROBUSTNESS_PRIORS_DISABLED`` and
        silently skipped when no findings file exists or loading fails. On
        success, sets the priors and appends a diagnostic note to the bundle.

        Args:
            bundle (JudgeBundle): The bundle to enrich in place.
        """
        if os.environ.get("CRITIC_ROBUSTNESS_PRIORS_DISABLED", "").lower() in {"1", "true", "yes"}:
            return
        findings_path = _discover_robustness_findings_path(bundle.session_id)
        if findings_path is None:
            return
        limit = int(os.environ.get("CRITIC_ROBUSTNESS_PRIORS_LIMIT") or 5)
        # Severity floor: HIGH by default; drop to MEDIUM via the env knob.
        min_severity = os.environ.get("CRITIC_ROBUSTNESS_PRIORS_MIN_SEVERITY", "high").lower()
        try:
            priors = _load_robustness_priors(
                findings_path,
                limit=max(1, limit),
                min_severity=min_severity,
            )
        except Exception:  # noqa: BLE001 — best-effort injection
            log.exception(
                "critic: failed to load robustness priors from %s",
                findings_path,
            )
            return
        if priors:
            bundle.robustness_priors = priors
            bundle.notes.append(f"robustness_priors_injected count={len(priors)} path={findings_path}")

    # ------------------------------------------------------------------
    # Phase 2: commit-review
    # ------------------------------------------------------------------
    def commit_review(
        self,
        raw_request: dict[str, Any],
        review: dict[str, Any],
    ) -> CommitOutcome:
        """Validate and commit a SKILL-produced review (phase 2).

        Dispatches by request kind to persist verdicts, optionally write to
        KB, and build the Coordinator-compatible intent envelope.

        Args:
            raw_request (dict[str, Any]): The raw Critic request envelope.
            review (dict[str, Any]): The review JSON produced by the SKILL.

        Returns:
            CommitOutcome: The persisted outcome for the request kind.

        Raises:
            ReviewValidationError: If ``review`` is not a dict or the request
                kind is not supported by ``commit_review``.
        """
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

        if req.kind == COORDINATOR_INBOX:
            self._commit_coordinator_inbox(req, review, outcome, session_ctx)
        elif req.kind == DECISION_REQUEST:
            self._commit_decision_request(req, review, outcome, session_ctx)
        elif req.kind == KB_DRAFT_REQUEST:
            self._commit_kb_draft(req, review, outcome, session_ctx)
        else:
            raise ReviewValidationError(f"commit_review does not support kind={req.kind!r}")
        return outcome

    # ------------------------------------------------------------------
    # Implementation helpers
    # ------------------------------------------------------------------
    def _populate_inbox(self, req: CriticRequest) -> None:
        """Parse proposals from a coordinator-inbox prompt, in place.

        Only acts on ``COORDINATOR_INBOX`` requests that lack pre-parsed
        proposals but carry a raw prompt. Fills in missing context from the
        parsed shared state and drops proposals already reviewed this session
        for idempotency. Parse failures are swallowed.

        Args:
            req (CriticRequest): The request to mutate in place.
        """
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
        new_msg_ids = self.session_memory.filter_unreviewed(req.session_id, [p.msg_id for p in parsed.proposals])
        keep = set(new_msg_ids)
        req.proposals = [p for p in parsed.proposals if p.msg_id in keep]

    def _review_constraints(
        self,
        proposals: list[Proposal] | None = None,
    ) -> dict[str, Any]:
        """Return the per-bundle review constraints payload.

        With ``proposals``, ``approve_requires`` is the batch's strictest
        action class and ``proposal_action_classes`` carries the per-proposal
        bar; empty/None defaults to the strict ``patch_landing`` checklist.

        Args:
            proposals: Proposals in the bundle; when ``None`` or empty the
                strict patch-landing checklist is used.

        Returns:
            A constraints payload with allowed verdicts, the approval
            checklist, and (when proposals are given) per-proposal classes.
        """
        constraints: dict[str, Any] = {
            "allowed_verdicts": sorted(ALLOWED_VERDICTS),
            "ceiling_importance": 0.84,
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
        """Derive the KB lookup topic for a proposal.

        Prefers the action name, then a ``topic``/``summary`` in the payload,
        and finally a synthetic ``proposal-<msg_id>`` fallback.

        Args:
            proposal (Proposal): The proposal to derive a topic for.

        Returns:
            str: The topic string used for KB prior lookups.
        """
        if proposal.action_name:
            return proposal.action_name
        body = proposal.payload.get("topic") or proposal.payload.get("summary")
        if isinstance(body, str) and body.strip():
            return body.strip()
        return f"proposal-{proposal.msg_id}"

    def _topic_for_decision(self, req: CriticRequest) -> str:
        """Derive the KB lookup topic for a decision-level review.

        Prefers ``topic``/``summary``/``target`` from the decision payload and
        falls back to a synthetic ``decision-<decision_id|session_id>``.

        Args:
            req (CriticRequest): The request whose decision to inspect.

        Returns:
            str: The topic string used for KB prior lookups.
        """
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
        """Commit a coordinator-inbox review into an intent envelope.

        Every verdict is validated before any of them is persisted, so a
        malformed entry cannot leave earlier proposals marked reviewed with
        their intents undelivered. Valid batches then record session memory,
        increment metrics, optionally write a KB lesson, append any advice
        intents, and fall back to a heartbeat intent when nothing was reviewed.

        Args:
            req (CriticRequest): The parsed request.
            review (dict[str, Any]): The SKILL review with ``review_verdicts``.
            outcome (CommitOutcome): Outcome mutated in place with the envelope.
            session_ctx (dict[str, Any]): Stored session context for KB writes.

        Raises:
            ReviewValidationError: If verdicts are malformed, invalid, or
                target the same proposal twice.
        """
        verdicts_raw = review.get("review_verdicts")
        if not isinstance(verdicts_raw, list):
            raise ReviewValidationError("coordinator_inbox commit expects review.review_verdicts to be a list")

        advice_by_target: dict[str, list[str]] = {}
        for advisory in review.get("advice") or []:
            if not isinstance(advisory, dict):
                continue
            body = advisory.get("body_md")
            advice_target = advisory.get("target_proposal_msg_id")
            if not body or not isinstance(advice_target, str) or not advice_target:
                continue
            advice_by_target.setdefault(advice_target, []).append(body)

        seen_targets: set[str] = set()
        validated: list[tuple[dict[str, Any], str, str, Intent]] = []
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
                    advice_text=advice_text,
                    notes=item.get("notes") or [],
                    failure_reason_code=str(item.get("failure_reason_code") or ""),
                )
            except IntentEnvelopeValidationError as exc:
                raise ReviewValidationError(str(exc)) from exc
            validated.append((item, target, verdict, intent))

        intents: list[Intent] = []
        for item, target, verdict, intent in validated:
            intents.append(intent)
            self.session_memory.mark_reviewed(req.session_id, target, verdict, decision_id=req.decision_id)
            self.session_memory.append_decision(
                req.session_id,
                {
                    "ts": now_iso(timespec="microseconds"),
                    "target_proposal_msg_id": target,
                    "verdict": verdict,
                    "reasoning": item.get("reasoning"),
                    "source": item.get("source", "critic"),
                    "failure_reason_code": str(item.get("failure_reason_code") or ""),
                    "kb_evidence": item.get("kb_evidence") or [],
                },
            )
            get_registry().counter(CRITIC_REVIEW_VERDICT_TOTAL).inc({"verdict": verdict})
            self._maybe_write_kb_for_verdict(item, req, session_ctx, outcome)

        # Every advisory also goes out as a standalone advice message, whatever
        # the target's verdict (or if untargeted); targeted bodies are already
        # inlined into the verdict's ``advice_text`` above.
        for advisory in review.get("advice") or []:
            if not isinstance(advisory, dict):
                continue
            body = advisory.get("body_md")
            if not body:
                continue
            intents.append(build_advice_intent(body, target_proposal_msg_id=advisory.get("target_proposal_msg_id")))

        # Heartbeat fallback when nothing to review.
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
        """Commit a decision-request review and persist it.

        Validates the verdict, records the decision review in session memory,
        maps the verdict to a KB write, and stores the review on ``outcome``.

        Args:
            req (CriticRequest): The parsed request.
            review (dict[str, Any]): The SKILL review (``verdict``/``reason``).
            outcome (CommitOutcome): Outcome mutated in place with the review.
            session_ctx (dict[str, Any]): Stored session context for KB writes.

        Raises:
            ReviewValidationError: If required keys are missing or the verdict
                is not one of adopt/reject/revise/needs_info.
        """
        for k in ("verdict", "reason"):
            if k not in review:
                raise ReviewValidationError(f"decision_request review missing required key {k!r}")
        if review["verdict"] not in {"adopt", "reject", "revise", "needs_info"}:
            raise ReviewValidationError(
                f"decision_request review.verdict {review['verdict']!r} not in adopt|reject|revise|needs_info"
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
        self.session_memory.append_decision(
            req.session_id,
            {
                "ts": now_iso(timespec="microseconds"),
                "decision_review": decision_review,
            },
        )
        # Translate the verdict into a KB write verdict.
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
        outcome.kb_writes.append(
            {
                "trigger": "decision_request",
                "result": write_res.to_dict(),
            }
        )
        outcome.decision_review = decision_review

    # ---- kb_draft commit ----------------------------------------------
    def _commit_kb_draft(
        self,
        req: CriticRequest,
        review: dict[str, Any],
        outcome: CommitOutcome,
        session_ctx: dict[str, Any],
    ) -> None:
        """Commit a kb-draft review by batch-writing the drafts to KB.

        Args:
            req (CriticRequest): The parsed request.
            review (dict[str, Any]): The SKILL review with a ``kb_drafts`` list.
            outcome (CommitOutcome): Outcome mutated in place with write results.
            session_ctx (dict[str, Any]): Stored session context for KB writes.

        Raises:
            ReviewValidationError: If ``review.kb_drafts`` is not a list.
        """
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
        """Write a KB lesson for a verdict when the SKILL opted in.

        Only acts on approve/reject/redirect verdicts flagged with
        ``persist_to_kb``. KB write failures are recorded as a note rather
        than raised so the review pipeline never blocks.

        Args:
            verdict_item (dict[str, Any]): The single verdict payload.
            req (CriticRequest): The parsed request.
            session_ctx (dict[str, Any]): Stored session context for KB writes.
            outcome (CommitOutcome): Outcome mutated in place with the result.
        """
        verdict = verdict_item.get("verdict")
        if verdict not in {"reject", "redirect", "approve"}:
            return
        # Only persist when the SKILL opted in.
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
            outcome.kb_writes.append(
                {
                    "trigger": "review_verdict",
                    "target_proposal_msg_id": verdict_item.get("target_proposal_msg_id"),
                    "result": res.to_dict(),
                }
            )
        except RuntimeAdapterError as exc:
            outcome.notes.append(f"kb_write_skipped: {exc}")


__all__ = [
    "CommitOutcome",
    "DecisionReviewer",
    "JudgeBundle",
]
