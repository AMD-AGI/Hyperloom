# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The intent contract, and the in-process MCP server exposing ``emit_intent``.

This module owns what an agent is allowed to say, for every provider. The
Claude path takes it as a real tool: :func:`build_emit_intent_server` wires
:data:`EMIT_INTENT_TOOL_NAME` into the SDK and each tool_use block becomes one
validated :class:`Intent`. The Codex path takes it as an enforced structured
output: :func:`build_intent_envelope_schema` renders the same contract as a
JSON schema for one role's intent set.

Both are generated from :class:`IntentType` and ``_PAYLOAD_REQUIRED``, never
from a hand-written literal — a hand-written Codex list once offered 5 of the
orchestration role's 12 intents, which silently removed the Coordinator's only
route to the kernel agent.

In-process MCP avoids extra processes; :func:`build_emit_intent_server` accepts
factory overrides for tests. The SDK rewrites the tool name to
:data:`EMIT_INTENT_TOOL_QUALIFIED` when forwarding to Claude.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable

from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    _PAYLOAD_REQUIRED,  # type: ignore[attr-defined]
)
from .mcp_context_tools import _resolve_sdk


log = logging.getLogger(__name__)


MCP_SERVER_NAME = "inference_optimizer"
EMIT_INTENT_TOOL_NAME = "emit_intent"
EMIT_INTENT_TOOL_QUALIFIED = f"mcp__{MCP_SERVER_NAME}__{EMIT_INTENT_TOOL_NAME}"


# Vocabulary a required-field name cannot carry on its own: closed value sets
# and the one optional dial each type honours. Rendered as its own clause by
# :func:`payload_constraints`, never mixed into the required-field list --
# appended there, they read as required keys, and a note naming a field that is
# already required printed it twice.
_PAYLOAD_FIELD_NOTES: dict[IntentType, tuple[str, ...]] = {
    IntentType.REVIEW_VERDICT: ("verdict ∈ approve|reject|redirect|advise|needs_review",),
    IntentType.EXTEND_LEASE: ("reason is optional",),
    IntentType.PRUNE_BRANCH: ("scope ∈ family|queued",),
    IntentType.ALERT: ("severity ∈ low|medium|high",),
}


def _ordered_intents(allowed_intents: Iterable[IntentType]) -> list[IntentType]:
    """Return the allowed types in :class:`IntentType` declaration order.

    A role's intent set is a frozenset, whose iteration order varies per
    process. The wire contract must not.

    Args:
        allowed_intents: The intent types a role may emit.

    Returns:
        The allowed types, deduplicated and in enum declaration order.
    """
    allowed = set(allowed_intents)
    return [t for t in IntentType if t in allowed]


def payload_contract(allowed_intents: Iterable[IntentType]) -> str:
    """Describe the required payload keys of each allowed intent type.

    Generated from ``_PAYLOAD_REQUIRED`` so the description cannot drift from
    what :func:`validate_envelope` actually enforces. Only the required keys:
    what a value may be, and which optional dial a type honours, are different
    claims and are rendered by :func:`payload_constraints`.

    Args:
        allowed_intents: The intent types a role may emit.

    Returns:
        A single-line ``"<type>:{<field>,...}"`` listing, comma separated.
    """
    parts: list[str] = []
    for intent_type in _ordered_intents(allowed_intents):
        fields = _PAYLOAD_REQUIRED.get(intent_type, ())
        parts.append(f"{intent_type.value}:{{{','.join(fields)}}}")
    return ", ".join(parts)


def payload_constraints(allowed_intents: Iterable[IntentType]) -> str:
    """Describe the value sets and optional dials the required-key list cannot.

    Kept apart from :func:`payload_contract` so neither claim is stated as the
    other: a constraint listed among required keys reads as a key the model must
    send, and one naming an already-required field printed that field twice.

    Args:
        allowed_intents: The intent types a role may emit.

    Returns:
        A single-line ``"<type>.<clause>"`` listing, semicolon separated, or an
        empty string when no allowed type carries a note.
    """
    parts: list[str] = []
    for intent_type in _ordered_intents(allowed_intents):
        for note in _PAYLOAD_FIELD_NOTES.get(intent_type, ()):
            parts.append(f"{intent_type.value}.{note}")
    return "; ".join(parts)


def constraints_sentence(allowed_intents: Iterable[IntentType]) -> str:
    """Render the constraints as a trailing sentence, or nothing at all.

    A role whose types carry no note has no constraints to state, and embedding
    the empty string unconditionally told it ``Constraints: .``

    Args:
        allowed_intents: The intent types a role may emit.

    Returns:
        ``" Constraints: <clauses>."`` with a leading space, or ``""``.
    """
    constraints = payload_constraints(allowed_intents)
    return f" Constraints: {constraints}." if constraints else ""


def build_intent_envelope_schema(allowed_intents: Iterable[IntentType]) -> dict[str, Any]:
    """Render one role's intent contract as an OpenAI-strict JSON schema.

    Strict structured outputs (which Azure enforces) require every object to
    declare ``additionalProperties: false`` and list every property in
    ``required``, and cannot express a free-form object at all — so the payload
    travels as a JSON string that the caller decodes and hands to the shared
    :func:`validate_envelope`.

    Args:
        allowed_intents: The intent types the role may emit; becomes the
            ``intent_type`` enum verbatim.

    Returns:
        The schema for the ``{"intents": [...]}`` envelope.

    Raises:
        ValueError: If ``allowed_intents`` is empty, which would produce a
            schema no reply could ever satisfy.
    """
    ordered = _ordered_intents(allowed_intents)
    if not ordered:
        raise ValueError("build_intent_envelope_schema requires at least one allowed IntentType")
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
                            "enum": [t.value for t in ordered],
                        },
                        "payload": {
                            "type": "string",
                            "description": (
                                "The intent payload as a JSON object serialized into a string. "
                                f"Required keys per intent_type: {payload_contract(ordered)}."
                                f"{constraints_sentence(ordered)}"
                            ),
                        },
                    },
                    "required": ["intent_type", "payload"],
                },
            }
        },
        "required": ["intents"],
    }


EMIT_INTENT_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent_type": {
            "type": "string",
            "enum": [t.value for t in IntentType],
        },
        "payload": {
            "type": "object",
            "description": (
                "Per-intent payload. Required keys per intent_type: "
                f"{payload_contract(IntentType)}."
                f"{constraints_sentence(IntentType)}"
            ),
        },
        "__unparsedToolInput": {
            "type": "object",
            "description": (
                "Internal fallback produced by the Claude Code streaming parser. Never emit this deliberately."
            ),
        },
    },
    "anyOf": [
        {"required": ["intent_type", "payload"]},
        {"required": ["__unparsedToolInput"]},
    ],
    "additionalProperties": False,
}

# The fallback branch lets the MCP handler acknowledge parser-generated
# wrappers instead of returning an error the model cannot repair. Native
# ``intent_type`` + ``payload`` remains the canonical and preferred contract.

EMIT_INTENT_TOOL_DESCRIPTION = (
    "Emit ONE structured intent into the inference_optimizer system. This "
    "is the only way to communicate decisions, messages, or actions; "
    "free-text replies are ignored. Call once per intent — to emit several "
    "intents in a single turn, call this tool multiple times."
)

# Exception-only: Claude Code stores a tool JSON string the streaming parser
# did not promote to an object. Canonical input remains ``intent_type`` +
# ``payload`` and is never rewritten when that shape is already present.
_UNPARSED_TOOL_INPUT_KEY = "__unparsedToolInput"
_EMIT_INTENT_TOP_LEVEL_KEYS = {"intent_type", "payload", _UNPARSED_TOOL_INPUT_KEY}


def _decode_emit_intent_input(raw_input: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Decode a parser fallback and report why decoding failed.

    Canonical input always wins when ``intent_type`` is present. The fallback
    is inspected only when canonical input is absent.
    """
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


def is_unparsed_tool_wrapper(raw_input: Any) -> bool:
    """Whether ``input`` is Claude Code's wrapped unparsed tool JSON object."""
    if not isinstance(raw_input, dict):
        return False
    if "intent_type" in raw_input:
        return False
    wrapped = raw_input.get(_UNPARSED_TOOL_INPUT_KEY)
    return isinstance(wrapped, dict) and isinstance(wrapped.get("raw"), str)


def coerce_emit_intent_input(raw_input: Any) -> dict[str, Any]:
    """Return native emit_intent input, decoding the wrapper only as fallback.

    Canonical ``{intent_type, payload}`` is returned unchanged. The
    ``__unparsedToolInput.raw`` object is decoded only when that native shape
    is absent. Malformed JSON leaves the original dict so validation still
    fails.
    """
    if not isinstance(raw_input, dict):
        return {}
    coerced, _ = _decode_emit_intent_input(raw_input)
    return coerced


def validate_emit_intent_input(payload: dict[str, Any]) -> None:
    """Eager single-intent validation (mirrors :func:`validate_envelope`).

    Canonical input is ``intent_type`` + ``payload``. A Claude Code
    ``__unparsedToolInput`` wrapper is decoded first only when that native
    shape is missing, then the same checks apply: only those two top-level
    keys, a known :class:`IntentType`, and every required payload field.

    Args:
        payload (dict[str, Any]): The raw ``emit_intent`` tool input to
            validate.

    Raises:
        IntentValidationError: If the input is not a dict, has unexpected or
            missing top-level keys, names an unknown intent type, or omits a
            required payload field.
    """
    if not isinstance(payload, dict):
        raise IntentValidationError(f"emit_intent input must be an object, got {type(payload).__name__}")
    original_extra = set(payload) - _EMIT_INTENT_TOP_LEVEL_KEYS
    if original_extra:
        raise IntentValidationError(f"emit_intent input has unexpected keys: {sorted(original_extra)!r}")
    coerced, decode_error = _decode_emit_intent_input(payload)
    if decode_error is not None:
        raise IntentValidationError(f"emit_intent: {decode_error}")
    extra = set(coerced) - _EMIT_INTENT_TOP_LEVEL_KEYS
    if extra:
        raise IntentValidationError(f"emit_intent input has unexpected keys: {sorted(extra)!r}")
    if "intent_type" not in coerced or "payload" not in coerced:
        raise IntentValidationError("emit_intent input requires both 'intent_type' and 'payload'")
    try:
        intent_type = IntentType(coerced["intent_type"])
    except ValueError as exc:
        raise IntentValidationError(f"emit_intent: unknown intent_type {coerced['intent_type']!r}") from exc
    inner = coerced["payload"]
    if not isinstance(inner, dict):
        raise IntentValidationError(f"emit_intent: 'payload' must be an object, got {type(inner).__name__}")
    required = _PAYLOAD_REQUIRED.get(intent_type, ())
    missing = [k for k in required if k not in inner]
    if missing:
        raise IntentValidationError(
            f"emit_intent: intent_type={intent_type.value} missing required fields: {missing!r}"
        )


async def _emit_intent_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Default handler — validate then ack; errors return is_error=True.

    Args:
        args: The raw ``emit_intent`` tool input to validate.

    Returns:
        An MCP tool result dict acknowledging success, or carrying the
        validation error with ``is_error=True``.
    """
    try:
        validate_emit_intent_input(args)
    except IntentValidationError as exc:
        log.info("emit_intent rejected: %s", exc)
        return {
            "content": [{"type": "text", "text": f"validation_error: {exc}"}],
            "is_error": True,
        }
    return {"content": [{"type": "text", "text": "ok"}]}


def build_emit_intent_server(
    *,
    sdk_module: Any | None = None,
    tool_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] | None = None,
    handler: Callable[[dict[str, Any]], Any] | None = None,
) -> Any | None:
    """Build the in-process MCP server config exposing ``emit_intent``.

    Returns the SDK ``McpSdkServerConfig`` for
    :class:`ClaudeAgentOptions.mcp_servers`, or ``None`` if the SDK lacks
    in-process MCP helpers. ``tool_factory`` / ``server_factory`` /
    ``handler`` are test seams.

    Args:
        sdk_module: Explicit SDK module to use, or ``None`` to import the real
            one.
        tool_factory: Override for the SDK ``tool`` decorator factory (tests).
        server_factory: Override for the SDK ``create_sdk_mcp_server`` factory
            (tests).
        handler: Override for the tool handler; defaults to
            :func:`_emit_intent_handler`.

    Returns:
        The constructed in-process MCP server config, or ``None`` when the SDK
        lacks the required in-process MCP helpers.
    """
    sdk = _resolve_sdk(sdk_module)
    handler = handler or _emit_intent_handler

    if tool_factory is None:
        tool_factory = getattr(sdk, "tool", None) if sdk is not None else None
    if server_factory is None:
        server_factory = getattr(sdk, "create_sdk_mcp_server", None) if sdk is not None else None
    if tool_factory is None or server_factory is None:
        log.info(
            "emit_intent MCP server unavailable (sdk=%s).",
            getattr(sdk, "__name__", "<none>"),
        )
        return None

    decorator = tool_factory(
        EMIT_INTENT_TOOL_NAME,
        EMIT_INTENT_TOOL_DESCRIPTION,
        EMIT_INTENT_TOOL_INPUT_SCHEMA,
    )
    decorated = decorator(handler)
    return server_factory(MCP_SERVER_NAME, "1.0.0", [decorated])


__all__ = [
    "EMIT_INTENT_TOOL_DESCRIPTION",
    "EMIT_INTENT_TOOL_INPUT_SCHEMA",
    "EMIT_INTENT_TOOL_NAME",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "MCP_SERVER_NAME",
    "build_emit_intent_server",
    "build_intent_envelope_schema",
    "coerce_emit_intent_input",
    "is_unparsed_tool_wrapper",
    "payload_contract",
    "validate_emit_intent_input",
]
