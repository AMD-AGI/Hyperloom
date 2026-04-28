"""Structured-intent transport — DESIGN §10.5.

Two parser entry points:

- :func:`parse_claude_trajectory` walks the message stream produced by the
  ``claude-agent-sdk`` ``query()`` iterator and pulls out intents from
  ``emit_intent`` tool_use blocks (the Claude transport, §10.5.4) **and**,
  for v0.5 ClaudeBackend simplicity, falls back to a JSON envelope parsed
  out of the assistant's plain-text reply.
- :func:`parse_codex_validated_json` consumes the single JSON object that
  Codex roles emit (§10.5.5) and validates it against the envelope schema.

Both eventually route through :func:`validate_envelope` which is the only
place the envelope shape is enforced. ``ClaudeBackend`` and the (planned)
``CodexBackend`` therefore share the same downstream contract.

Envelope schema is mirrored from DESIGN §10.5.3.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
class IntentType(str, Enum):
    SEND_MESSAGE = "send_message"
    DELEGATE = "delegate"
    PROPOSE_ACTION = "propose_action"
    OBJECTION = "objection"
    VOTE = "vote"
    UPDATE_STATE = "update_state"
    UPDATE_PERSONA = "update_persona"
    ASK_QUESTION = "ask_question"
    ANSWER = "answer"
    ALERT = "alert"


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
# Envelope schema (DESIGN §10.5.3) — kept inline as a Python dict so we don't
# need jsonschema as a dependency.
# ---------------------------------------------------------------------------
_ALL_INTENT_VALUES: tuple[str, ...] = tuple(t.value for t in IntentType)

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

# Per-intent payload required-field map (DESIGN §10.5.3 / §15 dispatch).
_PAYLOAD_REQUIRED: dict[IntentType, tuple[str, ...]] = {
    IntentType.SEND_MESSAGE:    ("topic",),
    IntentType.DELEGATE:        ("action_name",),
    IntentType.PROPOSE_ACTION:  ("action_name", "predicted_gain_pct"),
    IntentType.OBJECTION:       ("target_msg_id", "reason"),
    IntentType.VOTE:             ("target_msg_id", "vote"),
    IntentType.UPDATE_STATE:    ("changes",),
    IntentType.UPDATE_PERSONA:  ("body_md",),
    IntentType.ASK_QUESTION:    ("topic", "question"),
    IntentType.ANSWER:           ("in_reply_to", "answer"),
    IntentType.ALERT:           ("severity", "summary"),
}


# ---------------------------------------------------------------------------
# Tool schema for the Claude SDK (DESIGN §10.5.4)
# ---------------------------------------------------------------------------
EMIT_INTENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_intent",
    "description": (
        "The ONLY way to communicate decisions, messages, or actions to the "
        "system. Free-text responses are ignored. Call this tool one or more "
        "times per turn; each call carries exactly one intent."
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
                    "Per-intent payload. send_message: {topic, body_md, "
                    "to?}; propose_action: {action_name, predicted_gain_pct, "
                    "reason?}; delegate: {action_name, params?, "
                    "idempotency_key?}; vote: {target_msg_id, vote}; "
                    "objection: {target_msg_id, reason}; alert: {severity, "
                    "summary, detail?}; update_state: {changes}; "
                    "update_persona: {body_md}; ask_question: {topic, "
                    "question}; answer: {in_reply_to, answer}."
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


class ProtocolError(RuntimeError):
    """Critic/Sage repair-prompt fallback path also failed."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_payload_for_type(intent_type: IntentType, payload: dict[str, Any]) -> None:
    required = _PAYLOAD_REQUIRED.get(intent_type, ())
    missing = [k for k in required if k not in payload]
    if missing:
        raise IntentValidationError(
            f"intent_type={intent_type.value} missing required payload "
            f"fields: {missing!r}",
        )


def validate_envelope(envelope: dict[str, Any]) -> list[Intent]:
    """Validate against ``INTENT_ENVELOPE_SCHEMA`` and return Intent list.

    Implements just enough of JSON-schema to enforce the envelope shape
    without taking on a runtime jsonschema dep.
    """
    if not isinstance(envelope, dict):
        raise IntentValidationError(
            f"envelope must be an object, got {type(envelope).__name__}",
        )
    extra = set(envelope.keys()) - {"intents"}
    if extra:
        raise IntentValidationError(
            f"envelope has unexpected top-level keys: {sorted(extra)!r}",
        )
    if "intents" not in envelope:
        raise IntentValidationError("envelope missing required key 'intents'")
    intents_raw = envelope["intents"]
    if not isinstance(intents_raw, list):
        raise IntentValidationError(
            f"'intents' must be an array, got {type(intents_raw).__name__}",
        )
    if not intents_raw:
        raise IntentValidationError("'intents' array is empty")

    out: list[Intent] = []
    for i, item in enumerate(intents_raw):
        if not isinstance(item, dict):
            raise IntentValidationError(
                f"intents[{i}] must be an object, got {type(item).__name__}",
            )
        item_extra = set(item.keys()) - {"intent_type", "payload"}
        if item_extra:
            raise IntentValidationError(
                f"intents[{i}] has unexpected keys: {sorted(item_extra)!r}",
            )
        if "intent_type" not in item:
            raise IntentValidationError(
                f"intents[{i}] missing 'intent_type'"
            )
        if "payload" not in item:
            raise IntentValidationError(
                f"intents[{i}] missing 'payload'"
            )
        try:
            intent_type = IntentType(item["intent_type"])
        except ValueError as exc:
            raise IntentValidationError(
                f"intents[{i}] has unknown intent_type "
                f"{item['intent_type']!r}",
            ) from exc
        payload = item["payload"]
        if not isinstance(payload, dict):
            raise IntentValidationError(
                f"intents[{i}].payload must be an object, got "
                f"{type(payload).__name__}",
            )
        _validate_payload_for_type(intent_type, payload)
        out.append(Intent(type=intent_type, payload=dict(payload)))
    return out


# ---------------------------------------------------------------------------
# JSON object extraction helpers
# ---------------------------------------------------------------------------
# Accept any language tag after ``` (or no tag). Codex roles emit
# ```validated_json_output blocks (DESIGN §10.5.5), Claude tends to
# emit ```json. Trailing whitespace before the closing fence is allowed.
_JSON_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\s*\n(?P<body>\{.*?\})\s*\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Find a JSON object inside arbitrary text.

    Tries (in order):
        1. A fenced code block ``\\`\\`\\`json ... \\`\\`\\``` or ``\\`\\`\\`\\n{...}\\n\\`\\`\\``.
        2. The whole string parsed directly.
        3. The largest balanced ``{...}`` substring.
    """
    if not isinstance(text, str):
        raise IntentValidationError(
            f"expected text, got {type(text).__name__}", raw=str(text)[:500],
        )
    text = text.strip()
    if not text:
        raise IntentValidationError("empty assistant response", raw=text)

    m = _JSON_FENCE_RE.search(text)
    if m:
        body = m.group("body")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise IntentValidationError(
                f"fenced JSON not parseable: {exc}", raw=body,
            ) from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise IntentValidationError(
            "no JSON object found in response", raw=text[:500],
        )
    fragment = text[start:end + 1]
    try:
        return json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise IntentValidationError(
            f"could not parse JSON fragment: {exc}", raw=fragment[:500],
        ) from exc


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------
def parse_codex_validated_json(text: str) -> list[Intent]:
    """Parse a Codex-style ``validated_json_output`` (DESIGN §10.5.5).

    Used by ``CodexBackend`` once it lands; also used by ``ClaudeBackend``
    as a *fallback* when the SDK trajectory contains no tool_use blocks
    (some prompts/policies steer Claude to plain-text JSON output).
    """
    envelope = _extract_first_json_object(text)
    return validate_envelope(envelope)


def _is_emit_intent_name(name: Any) -> bool:
    """True iff *name* refers to the ``emit_intent`` tool, qualified or not.

    The Claude SDK rewrites in-process MCP tools as
    ``mcp__<server_name>__<tool_name>`` before exposing them to the model
    (see :mod:`backends.mcp_emit_intent`), so the tool_use blocks we get
    back carry the qualified name (e.g.
    ``mcp__inference_optimizer__emit_intent``). We accept both the bare
    name (used in unit tests, JSONL replay, and any future non-MCP
    transport) and any ``mcp__<server>__emit_intent`` variant so the
    parser stays stable when the server name changes.
    """
    if not isinstance(name, str):
        return False
    if name == "emit_intent":
        return True
    return name.startswith("mcp__") and name.endswith("__emit_intent")


def _walk_tool_use_blocks(messages: Iterable[Any]) -> Iterable[dict[str, Any]]:
    """Yield ``input`` dicts from every ``ToolUseBlock`` whose name resolves
    to ``emit_intent`` (qualified or unqualified — see
    :func:`_is_emit_intent_name`).

    The shape we accept is duck-typed because the SDK objects evolve; we
    only ever read ``.name`` and ``.input``. If a message has ``.content``
    iterable (assistant message), we walk it; if it's a dict with
    ``"type": "tool_use"``, we read the dict directly (handy for tests).
    """
    for msg in messages:
        # Claude SDK assistant message
        content = getattr(msg, "content", None)
        if content is not None:
            for block in content:
                if _is_emit_intent_name(getattr(block, "name", None)):
                    yield dict(getattr(block, "input", {}) or {})
            continue

        # Plain dict (tests / replay)
        if isinstance(msg, dict):
            if msg.get("type") == "tool_use" and _is_emit_intent_name(msg.get("name")):
                yield dict(msg.get("input", {}) or {})
            elif "content" in msg and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and _is_emit_intent_name(block.get("name"))
                    ):
                        yield dict(block.get("input", {}) or {})


def parse_claude_trajectory(
    trajectory: Iterable[Any],
    *,
    fallback_text: str | None = None,
) -> list[Intent]:
    """Parse a Claude SDK trajectory (DESIGN §10.5.4 / §10.5.6).

    Each ``emit_intent`` tool_use block carries exactly one envelope item
    in its ``input``. We aggregate them into one envelope and reuse the
    same :func:`validate_envelope` that Codex flows through.

    The block name may be the bare ``"emit_intent"`` (legacy /
    JSON-in-text fallback / unit tests) or the SDK-qualified
    ``"mcp__<server>__emit_intent"`` shape produced by the in-process MCP
    server (DESIGN §10.5.4). Both forms are accepted — see
    :func:`_is_emit_intent_name`.

    If there are no matching tool_use blocks but ``fallback_text`` is
    provided, we fall back to JSON-in-text mode (used by ClaudeBackend in
    v0.5 before MCP custom tools land).
    """
    items = []
    for input_obj in _walk_tool_use_blocks(trajectory):
        # Allow either {intent_type, payload} (single intent) or the full
        # {intents: [...]} envelope when a model dumps a batch in one call.
        if "intents" in input_obj and isinstance(input_obj["intents"], list):
            items.extend(input_obj["intents"])
        else:
            items.append(input_obj)

    if items:
        return validate_envelope({"intents": items})

    if fallback_text:
        return parse_codex_validated_json(fallback_text)

    raise NoIntentEmitted("no emit_intent tool_use blocks and no fallback text")


# ---------------------------------------------------------------------------
# Repair-prompt helper (DESIGN §10.5.5, IMPL-CHECKLIST §2.14b)
# ---------------------------------------------------------------------------
def build_repair_prompt(
    original_prompt: str,
    error: Exception | None,
    *,
    fenced_label: str = "json",
) -> str:
    """Compose a follow-up prompt that asks the model to retry, citing the
    exact validation error.

    Used by both ``ClaudeBackend`` and ``CodexBackend`` for their single
    repair attempt. Codex roles want ``validated_json_output`` for the
    fence label; Claude is happy with ``json``. The error string is
    truncated to keep the repaired prompt short.
    """
    reason = str(error) if error else "unknown parse failure"
    if len(reason) > 500:
        reason = reason[:497] + "..."
    return (
        f"{original_prompt.rstrip()}\n\n"
        f"Your previous reply did not validate ({reason}). "
        f"Please retry — return ONLY the JSON envelope inside a "
        f"```{fenced_label} fenced block. Do not add commentary."
    )


__all__ = [
    "EMIT_INTENT_TOOL_SCHEMA",
    "INTENT_ENVELOPE_SCHEMA",
    "Intent",
    "IntentType",
    "IntentValidationError",
    "NoIntentEmitted",
    "ProtocolError",
    "build_repair_prompt",
    "parse_claude_trajectory",
    "parse_codex_validated_json",
    "validate_envelope",
]
