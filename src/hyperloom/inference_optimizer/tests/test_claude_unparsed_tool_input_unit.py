# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for Claude Code ``__unparsedToolInput`` emit_intent fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import IntentType
from hyperloom.orchestrator.roles import claude as claude_module
from hyperloom.orchestrator.roles.claude import ClaudeBackend
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    build_emit_intent_server,
)


@dataclass
class _FakeToolUse:
    name: str
    input: dict[str, Any]


class ToolUseBlock(_FakeToolUse):
    pass


@pytest.mark.usefixtures("claude_sdk")
def test_parse_unparsed_wrapper_missing_required_payload_returns_none() -> None:
    b = ClaudeBackend()
    raw = '{"intent_type": "delegate", "payload": {"params": {}}}'
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": raw, "len": len(raw)}},
    )
    assert b._parse_tool_use_block(block) is None


@pytest.mark.usefixtures("claude_sdk")
def test_parse_tool_use_block_ignores_extra_native_keys() -> None:
    """Stream ingest stays lenient: extra native keys are not a reason to drop."""
    b = ClaudeBackend()
    block = ToolUseBlock(
        name="emit_intent",
        input={
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat"},
            "reasoning": "why this emit exists",
        },
    )
    intent = b._parse_tool_use_block(block)
    assert intent is not None
    assert intent.type is IntentType.SEND_MESSAGE
    assert intent.payload["topic"] == "heartbeat"


@pytest.mark.usefixtures("claude_sdk")
def test_parse_tool_use_block_logs_malformed_wrapper_decode_error(caplog: pytest.LogCaptureFixture) -> None:
    b = ClaudeBackend()
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": "{not-json"}},
    )
    with caplog.at_level(logging.INFO, logger=claude_module.log.name):
        assert b._parse_tool_use_block(block) is None
    assert any("__unparsedToolInput.raw is not valid JSON" in message for message in caplog.messages)


@pytest.mark.usefixtures("claude_sdk")
def test_parse_tool_use_block_prefers_native_intent_type() -> None:
    """Canonical input remains authoritative when a fallback is also present."""
    b = ClaudeBackend()
    block = ToolUseBlock(
        name="emit_intent",
        input={
            "intent_type": "send_message",
            "payload": {"topic": "native"},
            "__unparsedToolInput": {"raw": '{"intent_type": "send_message", "payload": {"topic": "fallback"}}'},
        },
    )
    intent = b._parse_tool_use_block(block)
    assert intent is not None
    assert intent.payload["topic"] == "native"


_NATIVE_EMIT = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
_WRAPPER_RAW = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
_WRAPPER_EMIT = {"__unparsedToolInput": {"raw": _WRAPPER_RAW, "len": len(_WRAPPER_RAW)}}
_MIXED_EMIT = {**_NATIVE_EMIT, **_WRAPPER_EMIT}
_WRAPPER_WITH_JUNK = {**_WRAPPER_EMIT, "junk": 1}

_DELEGATE_NATIVE = {"intent_type": "delegate", "payload": {"action_name": "baseline"}}
_DELEGATE_RAW = json.dumps(_DELEGATE_NATIVE)
_DELEGATE_WRAPPER = {"__unparsedToolInput": {"raw": _DELEGATE_RAW, "len": len(_DELEGATE_RAW)}}
_UNKNOWN_TYPE_NATIVE = {"intent_type": "review_verdict", "payload": {"verdict": "approve"}}
_UNKNOWN_TYPE_RAW = json.dumps(_UNKNOWN_TYPE_NATIVE)
_UNKNOWN_TYPE_WRAPPER = {"__unparsedToolInput": {"raw": _UNKNOWN_TYPE_RAW, "len": len(_UNKNOWN_TYPE_RAW)}}


def _registered_emit_intent_tool() -> Any:
    """The emit_intent tool as the real SDK decorator registers it."""
    sdk = pytest.importorskip("claude_agent_sdk")
    captured: list[Any] = []

    def capture(name: str, version: str, tools: list[Any]) -> None:
        captured.extend(tools)

    build_emit_intent_server(SimpleNamespace(tool=sdk.tool, create_sdk_mcp_server=capture))
    tool = next((t for t in captured if getattr(t, "name", None) == EMIT_INTENT_TOOL_NAME), None)
    if tool is None:
        pytest.skip("SDK did not register the emit_intent tool")
    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,accepted",
    [
        (_NATIVE_EMIT, True),
        (_WRAPPER_EMIT, True),
        (_MIXED_EMIT, True),
        ({"nonsense": 1}, False),
        (_WRAPPER_WITH_JUNK, False),
        (_DELEGATE_NATIVE, True),
        (_DELEGATE_WRAPPER, True),
        (_UNKNOWN_TYPE_NATIVE, False),
    ],
    ids=[
        "native",
        "wrapper",
        "native-with-wrapper",
        "junk",
        "wrapper-with-junk",
        "native-delegate",
        "wrapper-delegate",
        "native-outside-the-enum",
    ],
)
async def test_registered_schema_and_handler_agree(arguments: dict[str, Any], accepted: bool) -> None:
    """The declared schema gates the call before the handler ever runs, so a shape the handler accepts but the schema rejects can never land."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool()

    if accepted:
        jsonschema.validate(instance=arguments, schema=tool.input_schema)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=arguments, schema=tool.input_schema)

    result = await tool.handler(arguments)
    assert ("is_error" not in result) is accepted


def test_registered_schema_refuses_a_native_intent_type_outside_the_enum() -> None:
    """The SDK returns the schema validation message to the model before the handler runs, so the enum is what refuses a native call, naming the types that exist."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool()

    with pytest.raises(jsonschema.ValidationError) as exc:
        jsonschema.validate(instance=_UNKNOWN_TYPE_NATIVE, schema=tool.input_schema)

    assert "'review_verdict'" in exc.value.message
    assert [t.value for t in IntentType if t.value not in exc.value.message] == []


@pytest.mark.asyncio
async def test_registered_handler_refuses_a_wrapped_intent_type_outside_the_enum() -> None:
    """The enum cannot see inside the parser wrapper, so the handler refuses that form in the same turn instead."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool()
    jsonschema.validate(instance=_UNKNOWN_TYPE_WRAPPER, schema=tool.input_schema)

    result = await tool.handler(_UNKNOWN_TYPE_WRAPPER)

    assert result["is_error"] is True
    assert "unknown intent_type 'review_verdict'" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_handler_rejects_extra_key_inside_wrapped_raw() -> None:
    """Decoded fallback JSON observes the canonical top-level key contract."""
    tool = _registered_emit_intent_tool()
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}, "junk": 1}'
    result = await tool.handler({"__unparsedToolInput": {"raw": raw, "len": len(raw)}})
    assert result["is_error"] is True
    assert "unexpected keys: ['junk']" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_handler_reports_malformed_wrapped_raw() -> None:
    """A parser fallback reports its malformed JSON instead of blaming its key."""
    tool = _registered_emit_intent_tool()
    result = await tool.handler({"__unparsedToolInput": {"raw": "{not-json"}})
    assert result["is_error"] is True
    assert "__unparsedToolInput.raw is not valid JSON" in result["content"][0]["text"]


def test_registered_schema_offers_both_shapes() -> None:
    """The declared schema is what the model reads, so the wrapper alternative has to survive into it and say it is internal."""
    schema = _registered_emit_intent_tool().input_schema
    assert set(schema["properties"]) == {"intent_type", "payload", "__unparsedToolInput"}
    assert "Never emit this deliberately" in schema["properties"]["__unparsedToolInput"]["description"]
