"""In-process MCP server exposing the ``emit_intent`` tool — DESIGN §10.5.4.

This is the F4 wiring that makes ``emit_intent`` a *real* Claude tool instead
of just a JSON-in-text convention. The Claude SDK ships an in-process MCP
helper (``create_sdk_mcp_server`` + the ``@tool`` decorator) that lets us
register Python callables which the SDK then advertises to the model exactly
like any other tool.

Why an in-process server (and not a stdio/SSE one)?
---------------------------------------------------
- Zero extra processes. Marathon mode already has a Conductor + reactors +
  sub-agents, and the watchdog cares about pid hygiene. Embedding the server
  in the Conductor process keeps the topology trim.
- Synchronous lookups. The handler does nothing asynchronous beyond returning
  a confirmation; intent capture happens via the SDK trajectory parser
  (``parse_claude_trajectory``), so the handler can stay tiny.
- Test seam. We can construct the server with a fake "tool" / "factory"
  callable, so unit tests don't need to import ``claude_agent_sdk``.

Output contract
---------------
The handler validates the input against ``EMIT_INTENT_TOOL_SCHEMA``-shaped
payloads (single-intent form, since each tool call carries one intent).
On success it returns ``{"content": [{"type": "text", "text": "ok"}]}``,
which is the MCP-tool-result shape the SDK expects. On validation error it
returns the same shape but with an error string, so the model can see the
mistake and retry without crashing the run.

The tool name used inside the server is ``emit_intent``; the SDK
automatically rewrites it to ``mcp__<server_name>__emit_intent`` when
forwarding to Claude. Use :data:`EMIT_INTENT_TOOL_QUALIFIED` to allow-list
it in :class:`ClaudeAgentOptions`.

References
----------
- DESIGN §10.5.4   Claude transport (tool_use blocks)
- DESIGN §10.5.6   Trajectory parser (consumes the tool input again
                   downstream — handler does not need to publish)
- IMPLEMENTATION-CHECKLIST.md F4
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from ..intent_parser import (
    IntentType,
    IntentValidationError,
    _PAYLOAD_REQUIRED,  # type: ignore[attr-defined]
)


log = logging.getLogger(__name__)


MCP_SERVER_NAME = "inference_optimizer"
EMIT_INTENT_TOOL_NAME = "emit_intent"
EMIT_INTENT_TOOL_QUALIFIED = f"mcp__{MCP_SERVER_NAME}__{EMIT_INTENT_TOOL_NAME}"


# ---------------------------------------------------------------------------
# Schema (a stripped-down JSON-schema fragment the SDK will pass to Claude).
# Single-intent form — each tool call carries exactly one intent. The model is
# expected to make multiple tool calls when it wants to emit several intents
# in the same turn.
# ---------------------------------------------------------------------------
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
                "objection:{target_msg_id,reason}, vote:{target_msg_id,vote}, "
                "update_state:{changes}, update_persona:{body_md}, "
                "ask_question:{topic,question}, answer:{in_reply_to,answer}, "
                "alert:{severity,summary}."
            ),
        },
    },
    "required": ["intent_type", "payload"],
    "additionalProperties": False,
}

EMIT_INTENT_TOOL_DESCRIPTION = (
    "Emit ONE structured intent into the inference-optimizer system. This is "
    "the only way to communicate decisions, messages, or actions; free-text "
    "replies are ignored. Call once per intent — to emit multiple intents in "
    "a single turn, call this tool multiple times."
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_emit_intent_input(payload: dict[str, Any]) -> None:
    """Raise :class:`IntentValidationError` if ``payload`` is malformed.

    Mirrors the validation done by :func:`validate_envelope` for a single
    item. Keeping the rule in two places is fine: the tool handler validates
    eagerly (so the model sees a useful error), and the parser revalidates
    when stitching the trajectory together.
    """
    if not isinstance(payload, dict):
        raise IntentValidationError(
            f"emit_intent input must be an object, got {type(payload).__name__}",
        )
    extra = set(payload.keys()) - {"intent_type", "payload"}
    if extra:
        raise IntentValidationError(
            f"emit_intent input has unexpected keys: {sorted(extra)!r}",
        )
    if "intent_type" not in payload or "payload" not in payload:
        raise IntentValidationError(
            "emit_intent input requires both 'intent_type' and 'payload'",
        )
    try:
        intent_type = IntentType(payload["intent_type"])
    except ValueError as exc:
        raise IntentValidationError(
            f"emit_intent: unknown intent_type {payload['intent_type']!r}",
        ) from exc
    inner = payload["payload"]
    if not isinstance(inner, dict):
        raise IntentValidationError(
            f"emit_intent: 'payload' must be an object, got {type(inner).__name__}",
        )
    required = _PAYLOAD_REQUIRED.get(intent_type, ())
    missing = [k for k in required if k not in inner]
    if missing:
        raise IntentValidationError(
            f"emit_intent: intent_type={intent_type.value} missing required "
            f"fields: {missing!r}",
        )


# ---------------------------------------------------------------------------
# Handler — pure-Python, no SDK import required.
# ---------------------------------------------------------------------------
async def _emit_intent_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Handler invoked by the in-process MCP server on every tool call.

    The trajectory parser is the source-of-truth for collecting intents, so
    this handler only has to:

    - Validate the args (so the model gets a fast, structured error).
    - Return an MCP tool-result envelope acknowledging the call.

    Errors are returned as ``is_error=True`` MCP results rather than raised,
    so a single bad call doesn't tear down the whole turn.
    """
    try:
        validate_emit_intent_input(args)
    except IntentValidationError as exc:
        log.info("emit_intent rejected: %s", exc)
        return {
            "content": [{"type": "text", "text": f"validation_error: {exc}"}],
            "is_error": True,
        }
    return {
        "content": [{"type": "text", "text": "ok"}],
    }


# ---------------------------------------------------------------------------
# SDK wiring
# ---------------------------------------------------------------------------
def _resolve_sdk(sdk_module: Any | None) -> Any | None:
    """Return ``claude_agent_sdk`` (or ``claude_code_sdk``) if importable.

    Returns ``None`` if neither is present so callers can degrade gracefully
    to JSON-in-text mode without raising.
    """
    if sdk_module is not None:
        return sdk_module
    for name in ("claude_agent_sdk", "claude_code_sdk"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def build_emit_intent_server(
    *,
    sdk_module: Any | None = None,
    tool_factory: Callable[..., Any] | None = None,
    server_factory: Callable[..., Any] | None = None,
    handler: Callable[[dict[str, Any]], Any] | None = None,
) -> Any | None:
    """Build the in-process MCP server config exposing ``emit_intent``.

    Returns the SDK-specific ``McpSdkServerConfig`` value to plug into
    :class:`ClaudeAgentOptions.mcp_servers`, or ``None`` if the SDK lacks
    the in-process MCP helpers (in which case :class:`ClaudeBackend` falls
    back to JSON-in-text mode silently).

    Test seams:
        ``tool_factory``: replacement for ``sdk.tool``. Called with the same
            kwargs as the real decorator and is expected to return a
            decorator that accepts an async callable.
        ``server_factory``: replacement for ``sdk.create_sdk_mcp_server``.
        ``handler``: replacement for the default validation handler. Useful
            for tests that want to record every call.
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
            "emit_intent MCP server unavailable (sdk=%s); falling back to "
            "JSON-in-text mode.",
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
