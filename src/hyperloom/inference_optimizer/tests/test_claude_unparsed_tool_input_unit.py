# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for Claude Code ``__unparsedToolInput`` emit_intent fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    build_emit_intent_server,
)


_NATIVE_EMIT = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
_WRAPPER_RAW = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
_WRAPPER_EMIT = {"__unparsedToolInput": {"raw": _WRAPPER_RAW, "len": len(_WRAPPER_RAW)}}
_MIXED_EMIT = {**_NATIVE_EMIT, **_WRAPPER_EMIT}
_STRING_PAYLOAD_EMIT = {"intent_type": "send_message", "payload": json.dumps({"topic": "heartbeat"})}
_WRAPPER_WITH_JUNK = {**_WRAPPER_EMIT, "junk": 1}

_DELEGATE_NATIVE = {"intent_type": "delegate", "payload": {"action_name": "baseline"}}
_DELEGATE_RAW = json.dumps(_DELEGATE_NATIVE)
_DELEGATE_WRAPPER = {"__unparsedToolInput": {"raw": _DELEGATE_RAW, "len": len(_DELEGATE_RAW)}}
_UNKNOWN_TYPE_NATIVE = {"intent_type": "review_verdict", "payload": {"verdict": "approve"}}
_UNKNOWN_TYPE_RAW = json.dumps(_UNKNOWN_TYPE_NATIVE)
_UNKNOWN_TYPE_WRAPPER = {"__unparsedToolInput": {"raw": _UNKNOWN_TYPE_RAW, "len": len(_UNKNOWN_TYPE_RAW)}}


def _registered_emit_intent_tool(accepted: list[Intent] | None = None) -> Any:
    """The emit_intent tool as the real SDK decorator registers it; its handler fills ``accepted``."""
    sdk = pytest.importorskip("claude_agent_sdk")
    captured: list[Any] = []

    def capture(name: str, version: str, tools: list[Any]) -> None:
        captured.extend(tools)

    build_emit_intent_server(
        SimpleNamespace(tool=sdk.tool, create_sdk_mcp_server=capture), [] if accepted is None else accepted
    )
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
        (_STRING_PAYLOAD_EMIT, False),
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
        "payload-as-a-json-string",
    ],
)
async def test_registered_schema_and_handler_agree(arguments: dict[str, Any], accepted: bool) -> None:
    """The declared schema gates the call before the handler ever runs, so a shape the handler accepts but the schema rejects can never land."""
    jsonschema = pytest.importorskip("jsonschema")
    kept: list[Intent] = []
    tool = _registered_emit_intent_tool(kept)

    if accepted:
        jsonschema.validate(instance=arguments, schema=tool.input_schema)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=arguments, schema=tool.input_schema)

    result = await tool.handler(arguments)
    assert ("is_error" not in result) is accepted
    assert len(kept) == (1 if accepted else 0)


@pytest.mark.asyncio
async def test_handler_keeps_the_native_intent_when_a_wrapper_rides_along() -> None:
    """Canonical input remains authoritative when a fallback is also present."""
    kept: list[Intent] = []
    tool = _registered_emit_intent_tool(kept)
    fallback = '{"intent_type": "send_message", "payload": {"topic": "fallback"}}'

    result = await tool.handler(
        {
            "intent_type": "send_message",
            "payload": {"topic": "native"},
            "__unparsedToolInput": {"raw": fallback},
        }
    )

    assert "is_error" not in result
    assert [intent.payload["topic"] for intent in kept] == ["native"]


@pytest.mark.asyncio
async def test_handler_refuses_a_wrapped_intent_missing_a_required_payload_field() -> None:
    kept: list[Intent] = []
    tool = _registered_emit_intent_tool(kept)
    raw = '{"intent_type": "delegate", "payload": {"params": {}}}'

    result = await tool.handler({"__unparsedToolInput": {"raw": raw, "len": len(raw)}})

    assert result["is_error"] is True
    assert "action_name" in result["content"][0]["text"]
    assert kept == []


def test_registered_schema_refuses_a_native_intent_type_outside_the_enum() -> None:
    """The SDK returns the schema validation message to the model before the handler runs, so the enum is what refuses a native call, naming the types that exist."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool()

    with pytest.raises(jsonschema.ValidationError) as exc:
        jsonschema.validate(instance=_UNKNOWN_TYPE_NATIVE, schema=tool.input_schema)

    assert "'review_verdict'" in exc.value.message
    for intent_type in IntentType:
        assert intent_type.value in exc.value.message


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
