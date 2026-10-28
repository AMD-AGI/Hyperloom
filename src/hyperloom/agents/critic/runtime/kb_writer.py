# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""High-level KB write façade used by the decision reviewer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.env import env_bool, env_float, env_int

from .dead_letter import DeadLetter
from .errors import (
    KBError,
    KBTransportError,
    KBValidationError,
    ScopeError,
    SlugifyError,
)
from .importance_mapping import cap_importance, importance_for_verdict
from .kb_client import KBClient, KBListQuery
from .scope_builder import build_scope, scope_cache_key
from .session_memory import SessionMemory
from .slugify import slugify_safe


# Verdicts that should produce a KB write.
_KB_RELEVANT_VERDICTS: frozenset[str] = frozenset(
    {
        "approve",
        "reject",
        "redirect",
        "needs_review",
    }
)


# Circuit-breaker defaults: after ``threshold`` consecutive transport failures, reads/writes short-circuit for
# ``cooldown`` seconds.
_DEFAULT_BREAKER_THRESHOLD = 1
_DEFAULT_BREAKER_COOLDOWN_SECONDS = 60.0


@dataclass
class WriteContext:
    """Per-write metadata threaded through every call."""

    session_id: str
    review_id: str | None = None


@dataclass
class WriteResult:
    """Outcome of a KB write attempt."""

    status: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the write result."""
        return {"status": self.status, "detail": dict(self.detail)}


# ---------------------------------------------------------------------------
class KBWriter:
    """Façade with side-effect orchestration around a :class:`KBClient`."""

    def __init__(
        self,
        client: KBClient,
        *,
        session_memory: SessionMemory,
        dead_letter: DeadLetter | None = None,
        time_fn=time.time,
    ):
        """Wire the writer to a KB client and read its env-driven config."""
        self.client = client
        self.session_memory = session_memory
        self.dead_letter = dead_letter or DeadLetter()
        self.write_enabled = env_bool("KB_WRITE_ENABLED", True)
        self.read_enabled = env_bool("KB_READ_ENABLED", True)
        self._time_fn = time_fn

        self._breaker_threshold = max(1, env_int("CRITIC_KB_BREAKER_THRESHOLD", _DEFAULT_BREAKER_THRESHOLD))
        self._breaker_cooldown = max(
            0.0,
            env_float("CRITIC_KB_BREAKER_COOLDOWN_SECONDS", _DEFAULT_BREAKER_COOLDOWN_SECONDS),
        )
        self._consecutive_failures = 0
        self._unreachable_until = 0.0

    # Circuit-breaker helpers
    def is_kb_unreachable(self) -> bool:
        """Return True iff the breaker is currently open."""
        return self._time_fn() < self._unreachable_until

    def kb_breaker_state(self) -> dict[str, Any]:
        """Snapshot for ``judge_bundle.notes``."""
        now = self._time_fn()
        return {
            "open": now < self._unreachable_until,
            "remaining_seconds": max(0.0, self._unreachable_until - now),
            "consecutive_failures": self._consecutive_failures,
            "threshold": self._breaker_threshold,
            "cooldown_seconds": self._breaker_cooldown,
        }

    def _record_kb_failure(self) -> None:
        """Account a transport error and possibly open the breaker."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._breaker_threshold:
            self._unreachable_until = self._time_fn() + self._breaker_cooldown

    def _record_kb_success(self) -> None:
        """Reset the breaker after any successful KB call."""
        self._consecutive_failures = 0
        self._unreachable_until = 0.0

    # list_priors (read; not gated by KB_WRITE_ENABLED)
    def list_priors(
        self,
        *,
        scope: dict[str, Any],
        kind: str | None = None,
        topic: str | None = None,
        limit: int = 10,
        ctx: WriteContext | None = None,
    ) -> dict[str, Any]:
        """Look up KB priors with caching + circuit-breaker short-circuit."""
        if not self.read_enabled:
            return {"priors": [], "cache": "disabled", "cache_key": ""}

        cache_key = scope_cache_key(scope, topic=topic, kind=kind, limit=limit)

        # Cache wins regardless of breaker state.
        if ctx is not None:
            cached = self.session_memory.get_cached_priors(ctx.session_id, cache_key)
            if cached is not None:
                return {"priors": list(cached), "cache": "hit", "cache_key": cache_key}

        # Breaker open → short-circuit.
        if self.is_kb_unreachable():
            return {
                "priors": [],
                "cache": "kb_unreachable",
                "cache_key": cache_key,
                "breaker": self.kb_breaker_state(),
            }

        try:
            response = self.client.list(KBListQuery(scope_filter=scope, kind=kind, limit=limit))
        except KBTransportError as exc:
            self._record_kb_failure()
            return {
                "priors": [],
                "cache": "kb_unreachable",
                "cache_key": cache_key,
                "error": str(exc),
                "breaker": self.kb_breaker_state(),
            }
        except KBError as exc:
            # 4xx / validation errors — KB is up, request was bad.
            return {
                "priors": [],
                "cache": "miss",
                "cache_key": cache_key,
                "error": str(exc),
            }

        self._record_kb_success()
        priors = response.get("entries") or []
        if ctx is not None:
            self.session_memory.put_cached_priors(ctx.session_id, cache_key, priors)
        return {"priors": priors, "cache": "miss", "cache_key": cache_key}

    # write_verdict
    def write_verdict(
        self,
        *,
        verdict: dict[str, Any],
        packet_context: dict[str, Any],
        session_context: dict[str, Any] | None = None,
        ctx: WriteContext,
    ) -> WriteResult:
        """Write a single verdict lesson to KB."""
        if not self.write_enabled:
            return WriteResult("disabled", {"reason": "KB_WRITE_ENABLED=false"})
        if self.is_kb_unreachable():
            return WriteResult(
                "disabled",
                {
                    "reason": "kb_unreachable",
                    "breaker": self.kb_breaker_state(),
                },
            )
        verdict_label = verdict.get("verdict")
        if verdict_label not in _KB_RELEVANT_VERDICTS:
            return WriteResult("skipped", {"reason": f"verdict={verdict_label}"})

        try:
            scope = build_scope(packet_context, session_context=session_context)
        except ScopeError as exc:
            return WriteResult("skipped", {"reason": "scope_construction_failed", "error": str(exc)})

        topic = verdict.get("topic") or _topic_from_reasoning(verdict)
        try:
            slug = slugify_safe(topic) if topic else None
        except (SlugifyError, UnicodeEncodeError) as exc:
            return WriteResult("skipped", {"reason": "slug_failed", "error": str(exc)})
        if not slug:
            return WriteResult("skipped", {"reason": "no_topic"})

        kind = "pitfall" if verdict_label in ("reject", "redirect", "needs_review") else "technique"
        has_measurement = bool(verdict.get("packet_evidence"))
        importance = cap_importance(
            importance_for_verdict(
                verdict=verdict_label,
                confidence=verdict.get("confidence"),
                has_measurement=has_measurement,
            )
        )

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
                "source_role": "critic",
                "topic": topic or "",
                "evidence": {
                    "packet_evidence": verdict.get("packet_evidence", []),
                    "kb_evidence": verdict.get("kb_evidence", []),
                    "risks": verdict.get("risks", []),
                    "predicted_gain_pct": verdict.get("predicted_gain_pct"),
                },
            },
        }
        return self._upsert_with_dead_letter(payload, ctx)

    def _upsert_with_dead_letter(
        self,
        payload: dict[str, Any],
        ctx: WriteContext,
    ) -> WriteResult:
        """Upsert a payload, dead-lettering any transport/4xx failure."""
        try:
            response = self.client.upsert(payload)
        except KBError as exc:
            # Only a transport failure counts toward the breaker; 404/409 report as transport errors without it.
            if isinstance(exc, KBTransportError):
                self._record_kb_failure()
            self.dead_letter.append(
                "upsert",
                payload,
                attempts=1,
                last_error=str(exc),
                context={"session_id": ctx.session_id, "review_id": ctx.review_id},
            )
            reason = "validation_error" if isinstance(exc, KBValidationError) else "transport_error"
            return WriteResult("dead_lettered", {"reason": reason, "error": str(exc)})
        self._record_kb_success()
        return WriteResult("ok", {"response": response})


# Helpers
def _topic_from_reasoning(verdict: dict[str, Any]) -> str | None:
    """Derive a slug-safe topic from verdict.reasoning when not provided."""
    reasoning = (verdict.get("reasoning") or "").strip()
    if not reasoning:
        return None
    words = [w for w in reasoning.split() if w.isascii()][:8]
    if not words:
        return None
    return " ".join(words)


__all__ = [
    "KBWriter",
    "WriteContext",
    "WriteResult",
]
