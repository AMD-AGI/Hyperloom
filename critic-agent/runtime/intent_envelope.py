"""Build and validate Coordinator-compatible intent envelopes.

The Coordinator () accepts intent envelopes of the form

```
{"intents": [{"intent_type": "<type>", "payload": {...}}, ...]}
```

The Critic agent only ever produces three intent types in normal operation:

* ``review_verdict`` — primary output for any ``proposal`` it sees.
* ``send_message`` — heartbeat (no proposal in inbox) or devil's advocate
  ``advice`` topic.
* ``update_persona`` — append-only persona update (Critic role permission).

We deliberately mirror the schema and verdict vocabulary from
``inference_optimizer/orchestrator/intent_parser.py`` and
``inference_optimizer/orchestrator/policy.py`` rather than importing them, so
the Critic skill stays usable as a standalone package (no Hyperloom-wide
runtime dependency at install time).

If the Coordinator ever extends the envelope schema, update this file and
the ``ENVELOPE_SCHEMA_VERSION`` constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import IntentEnvelopeValidationError


ENVELOPE_SCHEMA_VERSION = "v0.6"


# Intent types the Critic role is allowed to emit ( +
# ``_CRITIC_INTENTS`` in agent_role.py).
ALLOWED_CRITIC_INTENTS: frozenset[str] = frozenset({
    "review_verdict",
    "send_message",
    "ask_question",
    "answer",
    "alert",
    "update_persona",
})


# Verdict vocabulary from policy.REVIEW_VERDICTS.
ALLOWED_VERDICTS: frozenset[str] = frozenset({
    "approve",
    "reject",
    "redirect",
    "advise",
    "needs_review",
})


# Verdict source vocabulary from references/verdict_schema.md.
ALLOWED_VERDICT_SOURCES: frozenset[str] = frozenset({
    "critic",
    "mock",
    "timeout",
    "critic_unavailable",
})


# Required payload fields per intent type — same set the Coordinator's
# PolicyGate enforces. See ``_PAYLOAD_REQUIRED`` in intent_parser.py.
_PAYLOAD_REQUIRED: dict[str, tuple[str, ...]] = {
    "send_message": ("topic",),
    "ask_question": ("topic", "question"),
    "answer": ("in_reply_to", "answer"),
    "alert": ("severity", "summary"),
    "update_persona": ("body_md",),
    "review_verdict": ("target_proposal_msg_id", "verdict"),
}


# Default content for the heartbeat fallback. Matches MockCriticBackend
# behaviour so callers can unit-test parity.
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


# ---------------------------------------------------------------------------
def _validate_payload(intent_type: str, payload: dict[str, Any]) -> None:
    """Validate a single intent payload against the per-type requirements.

    Enforces required keys for each intent type and, for ``review_verdict``,
    the allowed verdict / source vocabulary and a non-empty target msg id.

    Args:
        intent_type (str): The intent type whose rules apply.
        payload (dict[str, Any]): The payload to validate.

    Raises:
        IntentEnvelopeValidationError: If the payload is not a dict, is
            missing a required key, or violates the ``review_verdict`` rules.
    """
    if not isinstance(payload, dict):
        raise IntentEnvelopeValidationError(
            f"intent {intent_type!r}: payload must be an object, "
            f"got {type(payload).__name__}"
        )
    required = _PAYLOAD_REQUIRED.get(intent_type, ())
    for key in required:
        if key not in payload:
            raise IntentEnvelopeValidationError(
                f"intent {intent_type!r}: missing required payload key {key!r}"
            )
    if intent_type == "review_verdict":
        verdict = payload.get("verdict")
        if verdict not in ALLOWED_VERDICTS:
            raise IntentEnvelopeValidationError(
                f"review_verdict.verdict {verdict!r} not in "
                f"{sorted(ALLOWED_VERDICTS)!r}"
            )
        target = payload.get("target_proposal_msg_id")
        if not isinstance(target, str) or not target:
            raise IntentEnvelopeValidationError(
                "review_verdict.target_proposal_msg_id must be a non-empty string"
            )
        source = payload.get("source")
        if source is not None and source not in ALLOWED_VERDICT_SOURCES:
            raise IntentEnvelopeValidationError(
                f"review_verdict.source {source!r} not in "
                f"{sorted(ALLOWED_VERDICT_SOURCES)!r}"
            )


def validate_envelope(envelope: dict[str, Any]) -> IntentEnvelope:
    """Validate the dict form and return a typed :class:`IntentEnvelope`.

    The function is the inverse of :meth:`IntentEnvelope.to_dict` — it's the
    only place we accept envelopes constructed by hand (e.g. when reading
    a Skill-produced ``review.json``).

    Args:
        envelope (dict[str, Any]): The raw envelope dict to validate.

    Returns:
        IntentEnvelope: The validated, typed envelope.

    Raises:
        IntentEnvelopeValidationError: If the envelope shape is invalid, an
            intent type is not Critic-permitted, or a payload fails
            validation.
    """
    if not isinstance(envelope, dict):
        raise IntentEnvelopeValidationError(
            f"envelope must be an object, got {type(envelope).__name__}"
        )
    if "intents" not in envelope:
        raise IntentEnvelopeValidationError("envelope missing 'intents' key")
    items = envelope["intents"]
    if not isinstance(items, list) or not items:
        raise IntentEnvelopeValidationError(
            "envelope.intents must be a non-empty list"
        )
    out = IntentEnvelope()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise IntentEnvelopeValidationError(
                f"envelope.intents[{i}] must be an object"
            )
        if "intent_type" not in item or "payload" not in item:
            raise IntentEnvelopeValidationError(
                f"envelope.intents[{i}] missing intent_type or payload"
            )
        intent_type = item["intent_type"]
        if intent_type not in ALLOWED_CRITIC_INTENTS:
            raise IntentEnvelopeValidationError(
                f"envelope.intents[{i}].intent_type {intent_type!r} is not "
                f"a Critic-permitted intent (allowed: "
                f"{sorted(ALLOWED_CRITIC_INTENTS)!r})"
            )
        payload = item["payload"]
        _validate_payload(intent_type, payload)
        out.append(Intent(intent_type=intent_type, payload=dict(payload)))
    return out


# ---------------------------------------------------------------------------
# Builders — convenience constructors used by decision_reviewer.
# ---------------------------------------------------------------------------
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
        raise IntentEnvelopeValidationError(
            f"verdict {verdict!r} not in {sorted(ALLOWED_VERDICTS)!r}"
        )
    if source not in ALLOWED_VERDICT_SOURCES:
        raise IntentEnvelopeValidationError(
            f"source {source!r} not in {sorted(ALLOWED_VERDICT_SOURCES)!r}"
        )
    if not target_proposal_msg_id:
        raise IntentEnvelopeValidationError(
            "target_proposal_msg_id is required"
        )
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
    # Self-check by routing through validate_envelope so we surface bugs at
    # build time rather than after the LLM has already produced a response.
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
