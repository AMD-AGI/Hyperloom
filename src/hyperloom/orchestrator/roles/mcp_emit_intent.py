# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The intent contract, and the in-process MCP server exposing ``emit_intent``."""

from __future__ import annotations

import json
import logging
from typing import Any

from hyperloom.inference_optimizer.protocol.intent import (
    EXTEND_LEASE_MAX_SEC,
    PAYLOAD_REQUIRED,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from hyperloom.orchestrator.mcp_tool_names import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
)

from .mcp_sdk import build_sdk_mcp_server


log = logging.getLogger(__name__)


# Vocabulary a required-field name cannot carry on its own: closed value sets, numeric bounds, and the optional dials
# each type honours.
_PAYLOAD_FIELD_NOTES: dict[IntentType, tuple[str, ...]] = {
    IntentType.EXTEND_LEASE: (
        f"extra_sec ≤ {EXTEND_LEASE_MAX_SEC} per call, repeatable while the task runs",
        "reason is optional",
    ),
    IntentType.PRUNE_BRANCH: ("scope ∈ family|queued",),
    IntentType.ALERT: ("severity ∈ low|medium|high",),
}


def payload_contract() -> str:
    """Describe the required payload keys of each intent type."""
    parts: list[str] = []
    for intent_type in IntentType:
        fields = PAYLOAD_REQUIRED.get(intent_type, ())
        parts.append(f"{intent_type.value}:{{{','.join(fields)}}}")
    return ", ".join(parts)


def payload_constraints() -> str:
    """Describe the value sets and optional dials the required-key list cannot."""
    parts: list[str] = []
    for intent_type in IntentType:
        for note in _PAYLOAD_FIELD_NOTES.get(intent_type, ()):
            parts.append(f"{intent_type.value}.{note}")
    return "; ".join(parts)


def constraints_sentence() -> str:
    """Render the constraints as a trailing sentence."""
    return f" Constraints: {payload_constraints()}."


def build_intent_envelope_schema() -> dict[str, Any]:
    """Render the intent contract as an OpenAI-strict JSON schema."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intents": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "intent_type": {
                            "type": "string",
                            "enum": [t.value for t in IntentType],
                        },
                        "payload": {
                            "type": "string",
                            "description": (
                                "The intent payload as a JSON object serialized into a string. "
                                f"Required keys per intent_type: {payload_contract()}."
                                f"{constraints_sentence()}"
                            ),
                        },
                    },
                    "required": ["intent_type", "payload"],
                },
            }
        },
        "required": ["intents"],
    }


# No top-level required or anyOf/oneOf/allOf: the MCP layer validates against this schema before the handler runs,
# and Claude Code's __unparsedToolInput wrapper carries neither key; the Messages API rejects top-level combinators,
# and Claude Code can drop such a tool without an error. validate_emit_intent_input enforces both keys instead.
def build_emit_intent_tool_input_schema() -> dict[str, Any]:
    """Render the intent contract as the ``emit_intent`` tool input schema."""
    return {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": [t.value for t in IntentType],
            },
            "payload": {
                "type": "object",
                "description": (
                    f"Per-intent payload. Required keys per intent_type: {payload_contract()}.{constraints_sentence()}"
                ),
            },
            "__unparsedToolInput": {
                "type": "object",
                "description": (
                    "Internal fallback produced by the Claude Code streaming parser. Never emit this deliberately."
                ),
            },
        },
        "additionalProperties": False,
    }


EMIT_INTENT_TOOL_DESCRIPTION = (
    "Emit ONE structured intent into the inference_optimizer system. This "
    "is the only way to communicate decisions, messages, or actions; "
    "free-text replies are ignored. Call once per intent — to emit several "
    "intents in a single turn, call this tool multiple times. Always pass "
    "both intent_type and payload."
)

# Exception-only: Claude Code stores a tool JSON string the streaming parser did not promote to an object.
_UNPARSED_TOOL_INPUT_KEY = "__unparsedToolInput"
_EMIT_INTENT_TOP_LEVEL_KEYS = {"intent_type", "payload", _UNPARSED_TOOL_INPUT_KEY}


def decode_emit_intent_input(raw_input: Any) -> tuple[dict[str, Any], str | None]:
    """Decode a parser fallback and report why decoding failed.

    Accepts the three shapes ``emit_intent`` tool input arrives in:

    1. a native object (direct Anthropic API);
    2. Claude Code's ``__unparsedToolInput.raw`` wrapper;
    3. a JSON *string* envelope, as emitted by OpenAI-compatible / litellm
       style proxies, optionally with ``payload`` itself stringified.

    A caller that can see shape 3 hands it through unchanged: a non-dict input replaced
    with ``{}`` would decode "successfully" to an empty intent and stall the
    orchestrator with no error.
    """
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except (TypeError, ValueError):
            return {}, "emit_intent tool input string is not valid JSON"
        if not isinstance(raw_input, dict):
            return {}, "emit_intent tool input string did not decode to a JSON object"
    if not isinstance(raw_input, dict):
        return {}, f"emit_intent tool input must be an object, got {type(raw_input).__name__}"
    payload = raw_input.get("payload")
    if isinstance(payload, str):
        try:
            decoded_payload = json.loads(payload)
        except (TypeError, ValueError):
            decoded_payload = None
        if isinstance(decoded_payload, dict):
            raw_input = {**raw_input, "payload": decoded_payload}
    if "intent_type" in raw_input or _UNPARSED_TOOL_INPUT_KEY not in raw_input:
        return raw_input, None
    wrapped = raw_input.get(_UNPARSED_TOOL_INPUT_KEY)
    if not isinstance(wrapped, dict):
        return raw_input, f"{_UNPARSED_TOOL_INPUT_KEY} must be an object"
    raw = wrapped.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return raw_input, f"{_UNPARSED_TOOL_INPUT_KEY}.raw must be a non-empty JSON object string"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw_input, f"{_UNPARSED_TOOL_INPUT_KEY}.raw is not valid JSON"
    if not isinstance(parsed, dict):
        return raw_input, f"{_UNPARSED_TOOL_INPUT_KEY}.raw must decode to a JSON object"
    return parsed, None


def validate_emit_intent_input(payload: dict[str, Any]) -> None:
    """Eager single-intent validation, by the same rule the harvest will apply.

    The eager path decides only the tool ack; :func:`validate_envelope` decides
    whether the intent survives. A rule written twice here drifts — a null
    required field was acked ``ok`` and then dropped by the harvest, leaving the
    agent told it succeeded with nothing to retry on — so the eager path builds
    the one-intent envelope and defers to that function.
    """
    if not isinstance(payload, dict):
        raise IntentValidationError(f"emit_intent input must be an object, got {type(payload).__name__}")
    original_extra = set(payload) - _EMIT_INTENT_TOP_LEVEL_KEYS
    if original_extra:
        raise IntentValidationError(f"emit_intent input has unexpected keys: {sorted(original_extra)!r}")
    coerced, decode_error = decode_emit_intent_input(payload)
    if decode_error is not None:
        raise IntentValidationError(f"emit_intent: {decode_error}")
    extra = set(coerced) - _EMIT_INTENT_TOP_LEVEL_KEYS
    if extra:
        raise IntentValidationError(f"emit_intent input has unexpected keys: {sorted(extra)!r}")
    if "intent_type" not in coerced or "payload" not in coerced:
        raise IntentValidationError("emit_intent input requires both 'intent_type' and 'payload'")
    try:
        IntentType(coerced["intent_type"])
    except ValueError as exc:
        raise IntentValidationError(f"emit_intent: unknown intent_type {coerced['intent_type']!r}") from exc
    if not isinstance(coerced["payload"], dict):
        raise IntentValidationError(
            f"emit_intent: 'payload' must be an object, got {type(coerced['payload']).__name__}"
        )
    validate_envelope({"intents": [{"intent_type": coerced["intent_type"], "payload": coerced["payload"]}]})


async def _emit_intent_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Default handler — validate then ack; errors return is_error=True."""
    try:
        validate_emit_intent_input(args)
    except IntentValidationError as exc:
        log.info("emit_intent rejected: %s", exc)
        return {
            "content": [{"type": "text", "text": f"validation_error: {exc}"}],
            "is_error": True,
        }
    return {"content": [{"type": "text", "text": "ok"}]}


def build_emit_intent_server(sdk: Any) -> Any:
    """Build the in-process MCP server config exposing ``emit_intent`` with the imported ``claude_agent_sdk``."""
    tool = (
        EMIT_INTENT_TOOL_NAME,
        EMIT_INTENT_TOOL_DESCRIPTION,
        build_emit_intent_tool_input_schema(),
        _emit_intent_handler,
    )
    return build_sdk_mcp_server(sdk, MCP_SERVER_NAME, [tool])


__all__ = [
    "EMIT_INTENT_TOOL_DESCRIPTION",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "build_emit_intent_server",
    "build_emit_intent_tool_input_schema",
    "build_intent_envelope_schema",
    "decode_emit_intent_input",
    "payload_contract",
    "validate_emit_intent_input",
]
