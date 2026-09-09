# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and validate Coordinator-compatible intent envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
)
from hyperloom.inference_optimizer.protocol.intent import (
    validate_envelope as _validate_protocol_envelope,
)

from .errors import IntentEnvelopeValidationError


ENVELOPE_SCHEMA_VERSION = "v0.6"


# Intent types the Critic role is allowed to emit.
ALLOWED_CRITIC_INTENTS: frozenset[str] = frozenset(
    {
        IntentType.REVIEW_VERDICT.value,
        IntentType.SEND_MESSAGE.value,
        IntentType.ALERT.value,
    }
)


# Verdict vocabulary.
ALLOWED_VERDICTS: frozenset[str] = frozenset(
    {
        "approve",
        "reject",
        "redirect",
        "advise",
        "needs_review",
    }
)


# Verdict source vocabulary.
ALLOWED_VERDICT_SOURCES: frozenset[str] = frozenset(
    {
        "critic",
        "mock",
        "timeout",
        "critic_unavailable",
    }
)


# Default content for the heartbeat fallback.
DEFAULT_HEARTBEAT_TOPIC = "heartbeat"
DEFAULT_HEARTBEAT_BODY = "ok (critic)"
DEFAULT_ADVICE_TOPIC = "advice"


# ---------------------------------------------------------------------------
@dataclass
class Intent:
    """One envelope item — exactly what the Coordinator parses."""

    intent_type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the intent as a plain JSON-serialisable dict."""
        return {"intent_type": self.intent_type, "payload": dict(self.payload)}


@dataclass
class IntentEnvelope:
    """A list of :class:`Intent` items; serialises to ``{"intents": [...]}``."""

    intents: list[Intent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as the Coordinator-compatible dict form."""
        return {"intents": [i.to_dict() for i in self.intents]}

    def append(self, intent: Intent) -> None:
        """Append one intent to the envelope."""
        self.intents.append(intent)


def validate_envelope(envelope: dict[str, Any]) -> IntentEnvelope:
    """Validate the dict form and return a typed :class:`IntentEnvelope`."""
    if isinstance(envelope, dict) and envelope.get("intents") == []:
        raise IntentEnvelopeValidationError("envelope.intents must be a non-empty list")
    try:
        validated = _validate_protocol_envelope(envelope)
    except IntentValidationError as exc:
        raise IntentEnvelopeValidationError(str(exc)) from exc
    out = IntentEnvelope()
    for i, item in enumerate(validated):
        intent_type = item.type.value
        if intent_type not in ALLOWED_CRITIC_INTENTS:
            raise IntentEnvelopeValidationError(
                f"envelope.intents[{i}].intent_type {intent_type!r} is not "
                f"a Critic-permitted intent (allowed: "
                f"{sorted(ALLOWED_CRITIC_INTENTS)!r})"
            )
        out.append(Intent(intent_type=intent_type, payload=dict(item.payload)))
    return out


# Builders — convenience constructors used by decision_reviewer.
def build_review_verdict_intent(
    *,
    target_proposal_msg_id: str,
    verdict: str,
    reasoning: str = "",
    source: str = "critic",
    confidence: str | None = None,
    predicted_gain_pct: float | None = None,
    kb_evidence: Iterable[str] | None = None,
    packet_evidence: Iterable[str] | None = None,
    risks: list[dict[str, Any]] | None = None,
    required_evidence: Iterable[str] | None = None,
    alternative_action: str | None = None,
    advice_text: str = "",
    notes: Iterable[str] | None = None,
    failure_reason_code: str = "",
) -> Intent:
    """Build a validated ``review_verdict`` intent."""
    if verdict not in ALLOWED_VERDICTS:
        raise IntentEnvelopeValidationError(f"verdict {verdict!r} not in {sorted(ALLOWED_VERDICTS)!r}")
    if source not in ALLOWED_VERDICT_SOURCES:
        raise IntentEnvelopeValidationError(f"source {source!r} not in {sorted(ALLOWED_VERDICT_SOURCES)!r}")
    if not target_proposal_msg_id:
        raise IntentEnvelopeValidationError("target_proposal_msg_id is required")
    payload: dict[str, Any] = {
        "target_proposal_msg_id": target_proposal_msg_id,
        "verdict": verdict,
        "source": source,
        "reasoning": reasoning,
        "predicted_gain_pct": predicted_gain_pct,
        "kb_evidence": list(kb_evidence or []),
        "packet_evidence": list(packet_evidence or []),
        "risks": list(risks or []),
        "required_evidence": list(required_evidence or []),
        "alternative_action": alternative_action,
        "advice_text": advice_text,
        "notes": list(notes or []),
        "failure_reason_code": failure_reason_code,
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return Intent(intent_type="review_verdict", payload=payload)


def build_heartbeat_intent(body_md: str = DEFAULT_HEARTBEAT_BODY) -> Intent:
    """Build the heartbeat ``send_message`` intent."""
    return Intent(
        intent_type="send_message",
        payload={"topic": DEFAULT_HEARTBEAT_TOPIC, "body_md": body_md},
    )


def build_advice_intent(body_md: str, *, target_proposal_msg_id: str | None = None) -> Intent:
    """Build a devil's-advocate ``advice`` ``send_message`` intent."""
    payload: dict[str, Any] = {"topic": DEFAULT_ADVICE_TOPIC, "body_md": body_md}
    if target_proposal_msg_id:
        payload["about_proposal_msg_id"] = target_proposal_msg_id
    return Intent(intent_type="send_message", payload=payload)


def build_envelope(intents: Iterable[Intent]) -> IntentEnvelope:
    """Wrap a non-empty iterable of intents into an envelope."""
    materialised = list(intents)
    if not materialised:
        materialised = [build_heartbeat_intent()]
    env = IntentEnvelope()
    for intent in materialised:
        env.append(intent)
    # Self-check at build time.
    validate_envelope(env.to_dict())
    return env


__all__ = [
    "ALLOWED_CRITIC_INTENTS",
    "ALLOWED_VERDICTS",
    "ALLOWED_VERDICT_SOURCES",
    "DEFAULT_ADVICE_TOPIC",
    "DEFAULT_HEARTBEAT_BODY",
    "DEFAULT_HEARTBEAT_TOPIC",
    "ENVELOPE_SCHEMA_VERSION",
    "Intent",
    "IntentEnvelope",
    "build_advice_intent",
    "build_envelope",
    "build_heartbeat_intent",
    "build_review_verdict_intent",
    "validate_envelope",
]
