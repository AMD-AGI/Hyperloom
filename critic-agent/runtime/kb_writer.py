"""High-level KB write façade used by the decision reviewer.

This module never raises to the caller for transport / 4xx errors — it
catches them, dead-letters, and returns a typed :class:`WriteResult` so
the Critic decision pipeline never blocks on KB issues
(contract §6 — "writes must not block review_verdict").

Three triggers map to three methods:

* :meth:`write_verdict` (Trigger A)  — gate decisions that produced a
  reusable lesson get an upsert; ``defer`` / ``inconclusive`` /
  ``advise`` are skipped.
* :meth:`write_kb_drafts` (Trigger B) — session-close kb_draft batch
  insert with ``on_conflict=upsert``.
* :meth:`add_contradiction` (Trigger C) — single-side contradicts edge
  (KB service mirrors).

Plus the read helper :meth:`list_priors` with TTL'd cache backed by
:class:`runtime.session_memory.SessionMemory`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .category_mapping import map_category_to_kind
from .dead_letter import DeadLetter
from .errors import KBError, KBValidationError, RuntimeAdapterError, ScopeError
from .importance_mapping import (
    cap_importance,
    importance_for_kb_draft,
    importance_for_verdict,
)
from .kb_client import KBClient
from .metrics import (
    CRITIC_KB_PRIOR_CACHE_HIT,
    CRITIC_KB_PRIOR_CACHE_MISS,
    CRITIC_KB_WRITE_TOTAL,
    CRITIC_REVIEW_VERDICT_TOTAL,
    get_registry,
)
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory
from .slugify import slugify, slugify_safe


# Verdicts that should produce a KB write (per contract §5.1 — defer /
# inconclusive / advise are pure dispatch decisions, no reusable lesson).
_KB_RELEVANT_VERDICTS: frozenset[str] = frozenset({
    "approve", "reject", "redirect", "needs_review",
})


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class WriteContext:
    """Per-write metadata threaded through every call."""

    session_id: str
    review_id: str | None = None
    source_role: str = "critic"
    source_type: str = "critic_decision_review"
    topic: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteResult:
    """Outcome of a KB write attempt."""

    status: str  # ok | dead_lettered | skipped | disabled
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": dict(self.detail)}


# ---------------------------------------------------------------------------
class KBWriter:
    """Façade with side-effect orchestration around a :class:`KBClient`."""

    def __init__(
        self,
        client: KBClient,
        *,
        session_memory: SessionMemory | None = None,
        dead_letter: DeadLetter | None = None,
    ):
        self.client = client
        self.session_memory = session_memory or SessionMemory()
        self.dead_letter = dead_letter or DeadLetter()
        self.write_enabled = _read_bool_env("KB_WRITE_ENABLED", True)
        self.read_enabled = _read_bool_env("KB_READ_ENABLED", True)

    # ------------------------------------------------------------------
    # list_priors (Trigger D — read; not gated by KB_WRITE_ENABLED)
    # ------------------------------------------------------------------
    def list_priors(
        self,
        *,
        scope: dict[str, Any],
        kind: str | None = None,
        topic: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        ctx: WriteContext | None = None,
    ) -> dict[str, Any]:
        """Look up KB priors with session-memory caching.

        Returns ``{priors, cache: 'hit'|'miss'|'disabled', cache_key}``.
        """
        if not self.read_enabled:
            return {"priors": [], "cache": "disabled", "cache_key": ""}

        cache_key = scope_cache_key(scope, topic=topic)
        if ctx is not None:
            cached = self.session_memory.get_cached_priors(ctx.session_id, cache_key)
            if cached is not None:
                get_registry().counter(CRITIC_KB_PRIOR_CACHE_HIT).inc()
                return {"priors": list(cached), "cache": "hit", "cache_key": cache_key}

        get_registry().counter(CRITIC_KB_PRIOR_CACHE_MISS).inc()
        try:
            response = self.client.list(
                scope_filter=scope,
                kind=kind,
                metadata_filter=metadata_filter,
                limit=limit,
            )
        except KBError as exc:
            return {
                "priors": [],
                "cache": "miss",
                "cache_key": cache_key,
                "error": str(exc),
            }
        priors = response.get("entries") or []
        if ctx is not None:
            self.session_memory.put_cached_priors(ctx.session_id, cache_key, priors)
        return {"priors": priors, "cache": "miss", "cache_key": cache_key}

    # ------------------------------------------------------------------
    # write_verdict (Trigger A)
    # ------------------------------------------------------------------
    def write_verdict(
        self,
        *,
        verdict: dict[str, Any],
        packet_context: dict[str, Any],
        session_context: dict[str, Any] | None = None,
        ctx: WriteContext,
    ) -> WriteResult:
        if not self.write_enabled:
            return WriteResult("disabled", {"reason": "KB_WRITE_ENABLED=false"})
        verdict_label = verdict.get("verdict")
        get_registry().counter(CRITIC_REVIEW_VERDICT_TOTAL).inc({
            "verdict": str(verdict_label),
        })
        if verdict_label not in _KB_RELEVANT_VERDICTS:
            return WriteResult("skipped", {"reason": f"verdict={verdict_label}"})

        try:
            scope = build_scope(packet_context, session_context=session_context)
        except ScopeError as exc:
            return WriteResult("skipped", {"reason": "scope_construction_failed", "error": str(exc)})

        topic = ctx.topic or verdict.get("topic") or _topic_from_reasoning(verdict)
        try:
            slug = slugify_safe(topic) if topic else None
        except Exception as exc:  # noqa: BLE001
            return WriteResult("skipped", {"reason": "slug_failed", "error": str(exc)})
        if not slug:
            return WriteResult("skipped", {"reason": "no_topic"})

        kind = "pitfall" if verdict_label in ("reject", "redirect", "needs_review") else "technique"
        has_measurement = bool(verdict.get("packet_evidence"))
        importance = cap_importance(importance_for_verdict(
            verdict=verdict_label,
            confidence=verdict.get("confidence"),
            has_measurement=has_measurement,
        ))

        payload = {
            "scope": scope,
            "kind": kind,
            "slug": slug,
            "importance": importance,
            "summary": (verdict.get("reasoning") or "")[:2000],
            "metadata": {
                "source_session": ctx.session_id,
                "source_review_id": ctx.review_id,
                "source_type": "critic_verdict_" + verdict_label,
                "source_role": ctx.source_role,
                "topic": topic or "",
                "evidence": {
                    "packet_evidence": verdict.get("packet_evidence", []),
                    "kb_evidence": verdict.get("kb_evidence", []),
                    "risks": verdict.get("risks", []),
                    "predicted_gain_pct": verdict.get("predicted_gain_pct"),
                },
                **ctx.extra_metadata,
            },
        }
        return self._upsert_with_dead_letter(payload, ctx)

    # ------------------------------------------------------------------
    # write_kb_drafts (Trigger B)
    # ------------------------------------------------------------------
    def write_kb_drafts(
        self,
        *,
        kb_drafts: list[dict[str, Any]],
        packet_context: dict[str, Any],
        session_context: dict[str, Any] | None = None,
        ctx: WriteContext,
    ) -> WriteResult:
        if not self.write_enabled:
            return WriteResult("disabled", {"reason": "KB_WRITE_ENABLED=false"})
        if not kb_drafts:
            return WriteResult("skipped", {"reason": "no_drafts"})
        try:
            scope = build_scope(packet_context, session_context=session_context)
        except ScopeError as exc:
            return WriteResult("skipped", {"reason": "scope_construction_failed", "error": str(exc)})

        items: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for d in kb_drafts:
            category = d.get("category")
            try:
                kind = map_category_to_kind(category)
            except RuntimeAdapterError as exc:
                rejected.append({"draft": d, "reason": str(exc)})
                continue
            topic = d.get("action") or d.get("lesson") or category
            try:
                slug = slug_for_kind(kind, topic, d)
            except Exception as exc:  # noqa: BLE001
                rejected.append({"draft": d, "reason": f"slug_failed: {exc}"})
                continue
            importance = cap_importance(importance_for_kb_draft(confidence=d.get("confidence")))
            items.append({
                "scope": scope,
                "kind": kind,
                "slug": slug,
                "importance": importance,
                "summary": (d.get("lesson") or d.get("action") or "")[:2000],
                "metadata": {
                    "source_session": ctx.session_id,
                    "source_review_id": ctx.review_id,
                    "source_type": "critic_kb_draft",
                    "source_role": ctx.source_role,
                    "topic": topic,
                    "tags": list(d.get("tags") or []),
                    "result": d.get("result") or {},
                    "context": d.get("context") or "",
                    "strategy_tested": d.get("strategy_tested") or [],
                    **ctx.extra_metadata,
                },
            })
        if not items:
            return WriteResult("skipped", {"reason": "all_rejected", "rejected": rejected})

        try:
            response = self.client.batch_insert(items, on_conflict="upsert")
            get_registry().counter(CRITIC_KB_WRITE_TOTAL).inc({
                "endpoint": "batch_insert",
                "status": "200",
            })
            return WriteResult(
                "ok",
                {"response": response, "rejected": rejected},
            )
        except KBValidationError as exc:
            self.dead_letter.append(
                "batch_insert",
                {"items": items, "on_conflict": "upsert"},
                attempts=1,
                last_error=str(exc),
                context={"session_id": ctx.session_id, "review_id": ctx.review_id},
            )
            return WriteResult(
                "dead_lettered",
                {"reason": "validation_error", "error": str(exc), "rejected": rejected},
            )
        except KBError as exc:
            self.dead_letter.append(
                "batch_insert",
                {"items": items, "on_conflict": "upsert"},
                attempts=1,
                last_error=str(exc),
                context={"session_id": ctx.session_id, "review_id": ctx.review_id},
            )
            return WriteResult(
                "dead_lettered",
                {"reason": "transport_error", "error": str(exc), "rejected": rejected},
            )

    # ------------------------------------------------------------------
    # add_contradiction (Trigger C)
    # ------------------------------------------------------------------
    def add_contradiction(
        self,
        *,
        new_id: str,
        old_ids: list[str],
        ctx: WriteContext,
    ) -> WriteResult:
        if not self.write_enabled:
            return WriteResult("disabled", {"reason": "KB_WRITE_ENABLED=false"})
        if not new_id or not old_ids:
            return WriteResult("skipped", {"reason": "missing_ids"})
        edges = [
            {"kind": "contradicts", "from_id": new_id, "to_id": old_id}
            for old_id in old_ids
        ]
        try:
            response = self.client.add_edges(edges)
            return WriteResult("ok", {"response": response})
        except KBError as exc:
            # Edge writes are supplemental — best-effort. Don't dead-letter.
            return WriteResult("skipped", {"reason": "edge_write_failed", "error": str(exc)})

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _upsert_with_dead_letter(
        self,
        payload: dict[str, Any],
        ctx: WriteContext,
    ) -> WriteResult:
        try:
            response = self.client.upsert(payload)
            get_registry().counter(CRITIC_KB_WRITE_TOTAL).inc({
                "endpoint": "upsert",
                "status": "200",
            })
            return WriteResult("ok", {"response": response})
        except KBValidationError as exc:
            self.dead_letter.append(
                "upsert",
                payload,
                attempts=1,
                last_error=str(exc),
                context={"session_id": ctx.session_id, "review_id": ctx.review_id},
            )
            return WriteResult("dead_lettered", {"reason": "validation_error", "error": str(exc)})
        except KBError as exc:
            self.dead_letter.append(
                "upsert",
                payload,
                attempts=1,
                last_error=str(exc),
                context={"session_id": ctx.session_id, "review_id": ctx.review_id},
            )
            return WriteResult("dead_lettered", {"reason": "transport_error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _topic_from_reasoning(verdict: dict[str, Any]) -> str | None:
    """Derive a slug-safe topic from verdict.reasoning when not provided."""
    reasoning = (verdict.get("reasoning") or "").strip()
    if not reasoning:
        return None
    # Take the first 8 ASCII words to keep within slugify length bounds.
    words = [w for w in reasoning.split() if w.isascii()][:8]
    if not words:
        return None
    return " ".join(words)


def slug_for_kind(kind: str, topic: str, draft: dict[str, Any] | None = None) -> str:
    """Build a slug for ``(kind, topic, draft)`` per contract §2.2 templates."""
    draft = draft or {}
    if kind == "params_catalog":
        # Param entries should slugify the param name itself.
        param_name = draft.get("action") or topic
        return slugify(param_name)
    if kind == "model_profile":
        model = draft.get("model") or draft.get("model_family") or topic
        return slugify(f"{model}-profile")
    if kind == "pitfall":
        return slugify_safe(topic)
    # technique
    return slugify_safe(topic)


__all__ = [
    "KBWriter",
    "WriteContext",
    "WriteResult",
    "slug_for_kind",
]
