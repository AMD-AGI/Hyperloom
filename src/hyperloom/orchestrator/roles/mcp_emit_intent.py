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
    Intent,
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


def _unwrap_unparsed_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return the input Claude Code's ``__unparsedToolInput`` wrapper holds; one that names its intent type is returned as it is.

    Raises:
        IntentValidationError: The wrapper does not hold one JSON object.
    """
    if "intent_type" in tool_input or _UNPARSED_TOOL_INPUT_KEY not in tool_input:
        return tool_input
    wrapped = tool_input[_UNPARSED_TOOL_INPUT_KEY]
    if not isinstance(wrapped, dict):
        raise IntentValidationError(f"emit_intent: {_UNPARSED_TOOL_INPUT_KEY} must be an object")
    raw = wrapped.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        raise IntentValidationError(
            f"emit_intent: {_UNPARSED_TOOL_INPUT_KEY}.raw must be a non-empty JSON object string"
        )
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise IntentValidationError(f"emit_intent: {_UNPARSED_TOOL_INPUT_KEY}.raw is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise IntentValidationError(f"emit_intent: {_UNPARSED_TOOL_INPUT_KEY}.raw must decode to a JSON object")
    return parsed


def validate_emit_intent_input(payload: dict[str, Any]) -> Intent:
    """Validate one ``emit_intent`` tool input and return the intent it asks for.

    The input is a native object or Claude Code's ``__unparsedToolInput`` wrapper around one, the two shapes the
    tool schema admits. The intent is built by :func:`validate_envelope`, the rule every transport's intents pass,
    from the one-intent envelope of this call.

    Raises:
        IntentValidationError: The input is not one well-formed intent.
    """
    if not isinstance(payload, dict):
        raise IntentValidationError(f"emit_intent input must be an object, got {type(payload).__name__}")
    original_extra = set(payload) - _EMIT_INTENT_TOP_LEVEL_KEYS
    if original_extra:
        raise IntentValidationError(f"emit_intent input has unexpected keys: {sorted(original_extra)!r}")
    coerced = _unwrap_unparsed_tool_input(payload)
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
    return validate_envelope({"intents": [{"intent_type": coerced["intent_type"], "payload": coerced["payload"]}]})[0]


def build_emit_intent_server(sdk: Any, accepted: list[Intent]) -> Any:
    """Build the in-process MCP server exposing ``emit_intent`` for one attempt of a turn.

    The tool's answer and the intent's fate are one decision: a call answered ``ok`` has appended its intent to
    ``accepted``, and a call answered with an error has appended nothing. The Coordinator routes ``accepted`` once
    the turn has ended.

    Args:
        sdk: The imported ``claude_agent_sdk`` module.
        accepted: The attempt's intents, in the order the calls were answered.

    Returns:
        The SDK's server config, for ``ClaudeAgentOptions.mcp_servers``.
    """

    async def _emit_intent(args: dict[str, Any]) -> dict[str, Any]:
        try:
            intent = validate_emit_intent_input(args)
        except IntentValidationError as exc:
            log.info("emit_intent rejected: %s", exc)
            return {
                "content": [{"type": "text", "text": f"validation_error: {exc}"}],
                "is_error": True,
            }
        accepted.append(intent)
        return {"content": [{"type": "text", "text": "ok"}]}

    tool = (
        EMIT_INTENT_TOOL_NAME,
        EMIT_INTENT_TOOL_DESCRIPTION,
        build_emit_intent_tool_input_schema(),
        _emit_intent,
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
    "payload_contract",
    "validate_emit_intent_input",
]
