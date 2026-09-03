# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for Claude Code ``__unparsedToolInput`` emit_intent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.roles.claude import ClaudeBackend, _intent_fingerprint
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    build_emit_intent_server,
    coerce_emit_intent_input,
    is_unparsed_tool_wrapper,
)


@dataclass
class _FakeToolUse:
    name: str
    input: dict[str, Any]


class ToolUseBlock(_FakeToolUse):
    pass


def _backend() -> ClaudeBackend:
    async def _q(*, prompt, options):  # pragma: no cover
        if False:
            yield None

    class _Opts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    return ClaudeBackend(
        sdk_query_factory=_q,
        sdk_options_cls=_Opts,
        api_key_env="UNSET_KEY_ENV_FOR_TEST",
        enable_mcp_emit_intent=False,
    )


def test_is_unparsed_wrapper_false_for_native_and_junk() -> None:
    assert is_unparsed_tool_wrapper(None) is False
    assert is_unparsed_tool_wrapper({"intent_type": "send_message"}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": "raw"}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": {"raw": 1}}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": {"raw": '{"intent_type": "send_message"}'}}) is True


def test_coerce_returns_empty_for_non_dict() -> None:
    assert coerce_emit_intent_input(None) == {}
    assert coerce_emit_intent_input("x") == {}


def test_coerce_keeps_native_when_intent_type_present() -> None:
    native = {
        "intent_type": "send_message",
        "payload": {"topic": "heartbeat"},
        "__unparsedToolInput": {"raw": '{"intent_type": "alert"}'},
    }
    assert coerce_emit_intent_input(native) is native


def test_coerce_decodes_wrapper_object() -> None:
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
    out = coerce_emit_intent_input({"__unparsedToolInput": {"raw": raw, "len": len(raw)}})
    assert out["intent_type"] == "send_message"
    assert out["payload"]["topic"] == "heartbeat"


def test_coerce_leaves_wrapper_when_raw_is_not_object_json() -> None:
    wrapped = {"__unparsedToolInput": {"raw": "[1, 2]", "len": 6}}
    assert coerce_emit_intent_input(wrapped) is wrapped
    empty = {"__unparsedToolInput": {"raw": "   "}}
    assert coerce_emit_intent_input(empty) is empty
    missing = {"__unparsedToolInput": {}}
    assert coerce_emit_intent_input(missing) is missing


def test_coerce_leaves_wrapper_when_raw_is_malformed() -> None:
    wrapped = {"__unparsedToolInput": {"raw": "{not-json"}}
    assert coerce_emit_intent_input(wrapped) is wrapped


def test_parse_unparsed_wrapper_missing_required_payload_returns_none() -> None:
    b = _backend()
    raw = '{"intent_type": "propose_action", "payload": {"action_name": "baseline"}}'
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": raw, "len": len(raw)}},
    )
    assert b._parse_tool_use_block(block) is None


def test_diagnostic_records_unwrapped_intent_type() -> None:
    b = _backend()
    b._active_turn_diagnostic = {"tool_blocks": []}
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": raw, "len": len(raw)}},
    )
    b._record_tool_block_diagnostic(block)
    summary = b._active_turn_diagnostic["tool_blocks"][0]
    assert summary["input_keys"] == ["__unparsedToolInput"]
    assert summary["intent_type"] == "send_message"


def test_intent_fingerprint_is_stable() -> None:
    a = Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    b = Intent(type=IntentType.SEND_MESSAGE, payload={"body_md": "ok", "topic": "heartbeat"})
    assert _intent_fingerprint(a) == _intent_fingerprint(b)


_NATIVE_EMIT = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
_WRAPPER_RAW = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
_WRAPPER_EMIT = {"__unparsedToolInput": {"raw": _WRAPPER_RAW, "len": len(_WRAPPER_RAW)}}


def _emit_intent_server() -> Any:
    """The in-process MCP server, or a skip when the SDK is not installed."""
    pytest.importorskip("claude_agent_sdk")
    pytest.importorskip("mcp")
    cfg = build_emit_intent_server()
    if cfg is None:
        pytest.skip("in-process MCP helpers unavailable")
    return cfg["instance"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,expect_error,text_substr",
    [
        (_NATIVE_EMIT, False, "ok"),
        (_WRAPPER_EMIT, False, "ok"),
        ({"nonsense": 1}, True, "Additional properties"),
    ],
    ids=["native", "wrapper", "junk"],
)
async def test_sdk_tools_call_schema_accepts_native_and_wrapper(
    arguments: dict[str, Any],
    expect_error: bool,
    text_substr: str,
) -> None:
    """The tool schema is validated before the handler runs, so a shape the
    handler accepts but the schema rejects never reaches it. Driven through a
    real client session rather than the handler directly, which is the only
    way that ordering is exercised."""
    memory = pytest.importorskip("mcp.shared.memory")
    server = _emit_intent_server()
    async with memory.create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(EMIT_INTENT_TOOL_NAME, arguments)
    assert result.isError is expect_error
    assert text_substr in result.content[0].text


@pytest.mark.asyncio
async def test_broadcast_schema_offers_both_shapes() -> None:
    """tools/list is what the model reads, so the wrapper alternative has to
    survive into the advertised schema and say it is internal."""
    memory = pytest.importorskip("mcp.shared.memory")
    server = _emit_intent_server()
    async with memory.create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
    schema = next(t.inputSchema for t in tools.tools if t.name == EMIT_INTENT_TOOL_NAME)
    assert {"required": ["intent_type", "payload"]} in schema["anyOf"]
    assert {"required": ["__unparsedToolInput"]} in schema["anyOf"]
    assert "Never emit this deliberately" in schema["properties"]["__unparsedToolInput"]["description"]
