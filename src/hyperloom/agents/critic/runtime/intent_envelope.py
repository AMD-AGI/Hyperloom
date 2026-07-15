# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Build and validate Coordinator-compatible intent envelopes.

The Coordinator accepts intent envelopes of the form

```
{"intents": [{"intent_type": "<type>", "payload": {...}}, ...]}
```

The Critic normally produces two intent types: ``review_verdict``
(per proposal) and ``send_message`` (heartbeat / ``advice``). Raw
envelope validation delegates to the inference-optimizer protocol
validator so the Coordinator and standalone critic runtime share one
payload schema.
"""

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
        """Return the intent as a plain JSON-serialisable dict.

        Returns:
            dict[str, Any]: ``{"intent_type": ..., "payload": ...}`` with the
            payload copied.
        """
        return {"intent_type": self.intent_type, "payload": dict(self.payload)}


@dataclass
class IntentEnvelope:
    """A list of :class:`Intent` items + envelope metadata."""

    intents: list[Intent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as the Coordinator-compatible dict form.

        Returns:
            dict[str, Any]: ``{"intents": [...]}`` with each intent converted
            via :meth:`Intent.to_dict`.
        """
        return {"intents": [i.to_dict() for i in self.intents]}

    def append(self, intent: Intent) -> None:
        """Append one intent to the envelope.

        Args:
            intent (Intent): The intent to add.
        """
        self.intents.append(intent)


def validate_envelope(envelope: dict[str, Any]) -> IntentEnvelope:
    """Validate the dict form and return a typed :class:`IntentEnvelope`.

    Args:
        envelope (dict[str, Any]): The raw envelope dict to validate.

    Returns:
        IntentEnvelope: The validated, typed envelope.

    Raises:
        IntentEnvelopeValidationError: If the envelope shape is invalid, an
            intent type is not Critic-permitted, or a payload fails
            validation.
    """
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
) -> Intent:
    """Build a validated ``review_verdict`` intent.

    Args:
        target_proposal_msg_id (str): The proposal ``msg_id`` being reviewed.
        verdict (str): One of :data:`ALLOWED_VERDICTS`.
        reasoning (str): Free-text justification for the verdict.
        source (str): Verdict source; one of :data:`ALLOWED_VERDICT_SOURCES`.
        confidence (str | None): Optional confidence label; omitted when
            ``None``.
        predicted_gain_pct (float | None): Optional predicted gain percent.
        kb_evidence (Iterable[str] | None): KB evidence references.
        packet_evidence (Iterable[str] | None): Packet evidence references.
        risks (list[dict[str, Any]] | None): Structured risk entries.
        required_evidence (Iterable[str] | None): Evidence still required.
        alternative_action (str | None): Suggested alternative action.
        advice_text (str): Devil's-advocate advice text.
        notes (Iterable[str] | None): Additional free-text notes.

    Returns:
        Intent: The constructed ``review_verdict`` intent.

    Raises:
        IntentEnvelopeValidationError: If ``verdict`` or ``source`` is not in
            the allowed set, or ``target_proposal_msg_id`` is empty.
    """
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
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return Intent(intent_type="review_verdict", payload=payload)


def build_heartbeat_intent(body_md: str = DEFAULT_HEARTBEAT_BODY) -> Intent:
    """Build the heartbeat ``send_message`` intent.

    Args:
        body_md (str): Message body; defaults to :data:`DEFAULT_HEARTBEAT_BODY`.

    Returns:
        Intent: A ``send_message`` intent on the heartbeat topic.
    """
    return Intent(
        intent_type="send_message",
        payload={"topic": DEFAULT_HEARTBEAT_TOPIC, "body_md": body_md},
    )


def build_advice_intent(body_md: str, *, target_proposal_msg_id: str | None = None) -> Intent:
    """Build a devil's-advocate ``advice`` ``send_message`` intent.

    Args:
        body_md (str): The advice message body.
        target_proposal_msg_id (str | None): Optional proposal the advice is
            about; added as ``about_proposal_msg_id`` when provided.

    Returns:
        Intent: A ``send_message`` intent on the advice topic.
    """
    payload: dict[str, Any] = {"topic": DEFAULT_ADVICE_TOPIC, "body_md": body_md}
    if target_proposal_msg_id:
        payload["about_proposal_msg_id"] = target_proposal_msg_id
    return Intent(intent_type="send_message", payload=payload)


def build_envelope(intents: Iterable[Intent]) -> IntentEnvelope:
    """Wrap a non-empty iterable of intents into an envelope.

    Empty input falls back to a single heartbeat so the Coordinator never
    times out on an empty envelope; this matches MockCriticBackend.

    Args:
        intents (Iterable[Intent]): The intents to wrap.

    Returns:
        IntentEnvelope: The validated envelope (heartbeat-only if ``intents``
        was empty).

    Raises:
        IntentEnvelopeValidationError: If the resulting envelope fails the
            self-check in :func:`validate_envelope`.
    """
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
