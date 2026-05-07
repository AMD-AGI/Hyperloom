"""Structured-intent transport (DESIGN v0.6 §14).

Two transports share the same envelope schema:

* **Claude transport** (§14.2) — ``emit_intent`` MCP tool_call
* **Codex transport** (§14.3) — ``validated_json_output`` (single JSON object)

Both validate via :func:`validate_envelope`; downstream consumers see a
uniform list of :class:`Intent` objects.

v0.6 changes vs v0.5:

* ``OBJECTION`` / ``VOTE`` removed (parliament gone — DESIGN ADR-38).
* ``REVIEW_VERDICT`` added (Critic Review Protocol — DESIGN §18.2).
* Triage-only intents kept but renamed: source allowlist now
  ``{"robustness"}`` (see ``policy.py``); intent literal strings unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
class IntentType(str, Enum):
    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    PROPOSE_ACTION = "propose_action"
    UPDATE_STATE = "update_state"
    UPDATE_PERSONA = "update_persona"
    ASK_QUESTION = "ask_question"
    ANSWER = "answer"
    ALERT = "alert"
    # Bidirectional agent-to-agent RPC. PolicyGate restricts (source,
    # target_agent) pairs and `kind` per pair. v0.6: only Orchestration→Kernel.
    REQUEST = "request"
    RESPONSE = "response"
    # Critic Review Protocol (DESIGN §18.2). Critic-only.
    REVIEW_VERDICT = "review_verdict"
    # Robustness-only — task cancellation. Payload.scope must be "task".
    KILL_TASK = "kill_task"
    # Robustness-only — scheduling police (DESIGN §19.3).
    FORCE_DISPATCH = "force_dispatch"
    PRUNE_BRANCH = "prune_branch"
    ESCALATE_STRATEGY_CHANGE = "escalate_strategy_change"


_ALL_INTENT_VALUES: tuple[str, ...] = tuple(t.value for t in IntentType)


@dataclass
class Intent:
    """One validated intent from any transport."""

    type: IntentType
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_envelope_item(cls, item: dict[str, Any]) -> "Intent":
        intent_type = IntentType(item["intent_type"])
        return cls(type=intent_type, payload=dict(item.get("payload") or {}))


# ---------------------------------------------------------------------------
# Envelope schema (DESIGN §14.1) — kept inline so we don't need jsonschema
# ---------------------------------------------------------------------------
INTENT_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent_type": {
                        "type": "string",
                        "enum": list(_ALL_INTENT_VALUES),
                    },
                    "payload": {
                        "type": "object",
                        "description": "Schema depends on intent_type.",
                    },
                },
                "required": ["intent_type", "payload"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["intents"],
    "additionalProperties": False,
}


# Per-intent payload required-field map (DESIGN §14.1 / §15 dispatch / §18.2)
_PAYLOAD_REQUIRED: dict[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE:    ("topic",),
    IntentType.DELEGATE:        ("action_name",),
    IntentType.PROPOSE_ACTION:  ("action_name", "predicted_gain_pct"),
    IntentType.UPDATE_STATE:    ("changes",),
    IntentType.UPDATE_PERSONA:  ("body_md",),
    IntentType.ASK_QUESTION:    ("topic", "question"),
    IntentType.ANSWER:          ("in_reply_to", "answer"),
    IntentType.ALERT:           ("severity", "summary"),
    IntentType.REQUEST:         ("target_agent", "kind"),
    IntentType.RESPONSE:        ("in_reply_to", "kind"),
    IntentType.REVIEW_VERDICT:  ("target_proposal_msg_id", "verdict"),
    IntentType.KILL_TASK:       ("task_id", "reason"),
    IntentType.FORCE_DISPATCH:  ("task_id", "reason"),
    IntentType.PRUNE_BRANCH:    ("family", "reason"),
    IntentType.ESCALATE_STRATEGY_CHANGE: ("reason", "next_action_hint"),
}


# ---------------------------------------------------------------------------
# Tool schema for the Claude SDK (DESIGN §14.2)
# ---------------------------------------------------------------------------
EMIT_INTENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_intent",
    "description": (
        "The ONLY way to communicate decisions, messages, or actions to "
        "the system. Free-text responses are ignored. Call this tool one "
        "or more times per turn; each call carries exactly one intent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": list(_ALL_INTENT_VALUES),
            },
            "payload": {
                "type": "object",
                "description": (
                    "Per-intent payload. send_message: {topic, body_md, to?}; "
                    "propose_action: {action_name, predicted_gain_pct, "
                    "reason?}; delegate: {action_name, params?, "
                    "idempotency_key?}; alert: {severity, summary, detail?}; "
                    "update_state: {changes}; update_persona: {body_md}; "
                    "ask_question: {topic, question}; answer: {in_reply_to, "
                    "answer}; request: {target_agent, kind, params?, "
                    "reason?}; response: {in_reply_to, kind, status?, "
                    "result?}; review_verdict: {target_proposal_msg_id, "
                    "verdict ∈ approve/reject/redirect/advise/needs_review, "
                    "reasoning, kb_evidence?} — Critic-only; "
                    "kill_task / force_dispatch / prune_branch / "
                    "escalate_strategy_change — Robustness-only (PolicyGate)."
                ),
            },
        },
        "required": ["intent_type", "payload"],
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class NoIntentEmitted(RuntimeError):
    """Backend produced no parseable envelope and no tool_use blocks."""


class IntentValidationError(RuntimeError):
    """Envelope present but schema invalid (raw + reason captured)."""

    def __init__(self, reason: str, raw: str | None = None):
        super().__init__(reason)
        self.raw = raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_envelope(envelope: dict[str, Any]) -> list[Intent]:
    """Validate the top-level envelope shape + per-intent payloads.

    Returns the validated :class:`Intent` list. Raises
    :class:`IntentValidationError` on any structural issue so the caller
    can surface a single repair-prompt path (DESIGN §14.4).
    """
    if not isinstance(envelope, dict):
        raise IntentValidationError(f"envelope must be object, got {type(envelope).__name__}")
    if "intents" not in envelope:
        raise IntentValidationError("envelope missing required 'intents' key")
    items = envelope["intents"]
    if not isinstance(items, list):
        raise IntentValidationError("envelope.intents must be a list")

    validated: list[Intent] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise IntentValidationError(f"intents[{i}] must be object, got {type(item).__name__}")
        if "intent_type" not in item or "payload" not in item:
            raise IntentValidationError(
                f"intents[{i}] missing intent_type or payload"
            )
        try:
            it = IntentType(item["intent_type"])
        except ValueError:
            raise IntentValidationError(
                f"intents[{i}].intent_type {item['intent_type']!r} not in allowed set"
            )
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise IntentValidationError(
                f"intents[{i}].payload must be object, got {type(payload).__name__}"
            )
        for required in _PAYLOAD_REQUIRED[it]:
            if required not in payload:
                raise IntentValidationError(
                    f"intents[{i}] (type={it.value}) missing required "
                    f"payload field: {required!r}"
                )
        validated.append(Intent(type=it, payload=dict(payload)))
    return validated


def parse_codex_validated_json(raw: str) -> list[Intent]:
    """Codex transport — single JSON object containing one envelope."""
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IntentValidationError(f"codex json parse error: {exc}", raw=raw) from exc
    return validate_envelope(envelope)


def parse_claude_tool_calls(tool_uses: list[dict[str, Any]]) -> list[Intent]:
    """Claude transport — gather every ``emit_intent`` tool_use into intents."""
    items: list[dict[str, Any]] = []
    for use in tool_uses:
        if use.get("name") != EMIT_INTENT_TOOL_SCHEMA["name"]:
            continue
        inp = use.get("input") or {}
        items.append({
            "intent_type": inp.get("intent_type"),
            "payload": inp.get("payload") or {},
        })
    if not items:
        raise NoIntentEmitted("no emit_intent tool_use blocks in Claude reply")
    return validate_envelope({"intents": items})


__all__ = [
    "EMIT_INTENT_TOOL_SCHEMA",
    "INTENT_ENVELOPE_SCHEMA",
    "Intent",
    "IntentType",
    "IntentValidationError",
    "NoIntentEmitted",
    "parse_claude_tool_calls",
    "parse_codex_validated_json",
    "validate_envelope",
]
