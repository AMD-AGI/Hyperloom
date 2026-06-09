# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""In-process MCP server exposing the ``emit_intent`` tool.

Wires :data:`EMIT_INTENT_TOOL_NAME` into the Claude SDK as a real tool; each
tool_use block becomes one validated :class:`Intent`. In-process avoids extra
processes; :func:`build_emit_intent_server` accepts factory overrides for
tests. The SDK rewrites the name to :data:`EMIT_INTENT_TOOL_QUALIFIED` when
forwarding to Claude.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from ...protocol.intent import (
    IntentType,
    IntentValidationError,
    _PAYLOAD_REQUIRED,  # type: ignore[attr-defined]
)


log = logging.getLogger(__name__)


MCP_SERVER_NAME = "inference_optimizer"
EMIT_INTENT_TOOL_NAME = "emit_intent"
EMIT_INTENT_TOOL_QUALIFIED = f"mcp__{MCP_SERVER_NAME}__{EMIT_INTENT_TOOL_NAME}"


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
                "send_message:{topic}, delegate:{action_name}, "
                "propose_action:{action_name,predicted_gain_pct}, "
                "request:{target_agent,kind}, response:{in_reply_to,kind}, "
                "review_verdict:{target_proposal_msg_id,verdict ∈ "
                "approve|reject|redirect|advise|needs_review}, "
                "kill_task:{task_id,reason}, force_dispatch:{task_id,reason}, "
                "prune_branch:{family,reason}, escalate_strategy_change:"
                "{reason,next_action_hint}, update_state:{changes}, "
                "update_persona:{body_md}, ask_question:{topic,question}, "
                "answer:{in_reply_to,answer}, alert:{severity,summary}."
            ),
        },
    },
    "required": ["intent_type", "payload"],
    "additionalProperties": False,
}


EMIT_INTENT_TOOL_DESCRIPTION = (
    "Emit ONE structured intent into the inference_optimizer system. This "
    "is the only way to communicate decisions, messages, or actions; "
    "free-text replies are ignored. Call once per intent — to emit several "
    "intents in a single turn, call this tool multiple times."
)


def validate_emit_intent_input(payload: dict[str, Any]) -> None:
    """Eager single-intent validation (mirrors :func:`validate_envelope`)."""
    if not isinstance(payload, dict):
        raise IntentValidationError(
            f"emit_intent input must be an object, got {type(payload).__name__}"
        )
    extra = set(payload.keys()) - {"intent_type", "payload"}
    if extra:
        raise IntentValidationError(
            f"emit_intent input has unexpected keys: {sorted(extra)!r}"
        )
    if "intent_type" not in payload or "payload" not in payload:
        raise IntentValidationError(
            "emit_intent input requires both 'intent_type' and 'payload'"
        )
    try:
        intent_type = IntentType(payload["intent_type"])
    except ValueError as exc:
        raise IntentValidationError(
            f"emit_intent: unknown intent_type {payload['intent_type']!r}"
        ) from exc
    inner = payload["payload"]
    if not isinstance(inner, dict):
        raise IntentValidationError(
            f"emit_intent: 'payload' must be an object, got {type(inner).__name__}"
        )
    required = _PAYLOAD_REQUIRED.get(intent_type, ())
    missing = [k for k in required if k not in inner]
    if missing:
        raise IntentValidationError(
            f"emit_intent: intent_type={intent_type.value} missing required "
            f"fields: {missing!r}"
        )


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


def _resolve_sdk(sdk_module: Any | None) -> Any | None:
    if sdk_module is not None:
        return sdk_module
    try:
        return importlib.import_module("claude_agent_sdk")
    except ImportError:
        return None


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
    """
    sdk = _resolve_sdk(sdk_module)
    handler = handler or _emit_intent_handler

    if tool_factory is None:
        tool_factory = getattr(sdk, "tool", None) if sdk is not None else None
    if server_factory is None:
        server_factory = (
            getattr(sdk, "create_sdk_mcp_server", None) if sdk is not None else None
        )
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
    "validate_emit_intent_input",
]
