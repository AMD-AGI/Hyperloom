# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for Claude Code ``__unparsedToolInput`` emit_intent fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.roles import claude as claude_module
from hyperloom.orchestrator.roles.claude import ClaudeBackend, _intent_fingerprint
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    build_emit_intent_server,
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
        mcp_server_factory=lambda *a, **k: None,
    )


def test_is_unparsed_wrapper_false_for_native_and_junk() -> None:
    assert is_unparsed_tool_wrapper(None) is False
    assert is_unparsed_tool_wrapper({"intent_type": "send_message"}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": "raw"}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": {"raw": 1}}) is False
    assert is_unparsed_tool_wrapper({"__unparsedToolInput": {"raw": '{"intent_type": "send_message"}'}}) is True


def test_parse_unparsed_wrapper_missing_required_payload_returns_none() -> None:
    b = _backend()
    raw = '{"intent_type": "propose_action", "payload": {"action_name": "baseline"}}'
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": raw, "len": len(raw)}},
    )
    assert b._parse_tool_use_block(block) is None


def test_parse_tool_use_block_ignores_extra_native_keys() -> None:
    """Stream ingest stays lenient: extra native keys are not a reason to drop."""
    b = _backend()
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


def test_parse_tool_use_block_logs_malformed_wrapper_decode_error(caplog: pytest.LogCaptureFixture) -> None:
    b = _backend()
    block = ToolUseBlock(
        name="emit_intent",
        input={"__unparsedToolInput": {"raw": "{not-json"}},
    )
    with caplog.at_level(logging.INFO, logger=claude_module.log.name):
        assert b._parse_tool_use_block(block) is None
    assert any("__unparsedToolInput.raw is not valid JSON" in message for message in caplog.messages)


def test_parse_tool_use_block_prefers_native_intent_type() -> None:
    """Canonical input remains authoritative when a fallback is also present."""
    b = _backend()
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


def test_intent_fingerprint_is_stable() -> None:
    a = Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})
    b = Intent(type=IntentType.SEND_MESSAGE, payload={"body_md": "ok", "topic": "heartbeat"})
    assert _intent_fingerprint(a) == _intent_fingerprint(b)


_NATIVE_EMIT = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
_WRAPPER_RAW = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
_WRAPPER_EMIT = {"__unparsedToolInput": {"raw": _WRAPPER_RAW, "len": len(_WRAPPER_RAW)}}
_MIXED_EMIT = {**_NATIVE_EMIT, **_WRAPPER_EMIT}
_WRAPPER_WITH_JUNK = {**_WRAPPER_EMIT, "junk": 1}

_ORCHESTRATION_INTENTS = default_role_registry()["orchestration"].allowed_intents
_CRITIC_INTENTS = default_role_registry()["critic"].allowed_intents
_REVIEW_VERDICT_NATIVE = {
    "intent_type": "review_verdict",
    "payload": {"target_proposal_msg_id": "p1", "verdict": "approve"},
}
_REVIEW_VERDICT_RAW = json.dumps(_REVIEW_VERDICT_NATIVE)
_REVIEW_VERDICT_WRAPPER = {"__unparsedToolInput": {"raw": _REVIEW_VERDICT_RAW, "len": len(_REVIEW_VERDICT_RAW)}}


def _registered_emit_intent_tool(allowed_intents: frozenset[IntentType] | None = None) -> Any:
    """The emit_intent tool as the real SDK decorator registers it."""
    sdk = pytest.importorskip("claude_agent_sdk")
    captured: list[Any] = []

    def capture(name: str, version: str, tools: list[Any]) -> None:
        captured.extend(tools)

    build_emit_intent_server(sdk_module=sdk, server_factory=capture, allowed_intents=allowed_intents)
    tool = next((t for t in captured if getattr(t, "name", None) == EMIT_INTENT_TOOL_NAME), None)
    if tool is None:
        pytest.skip("SDK did not register the emit_intent tool")
    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_intents,arguments,accepted",
    [
        (None, _NATIVE_EMIT, True),
        (None, _WRAPPER_EMIT, True),
        (None, _MIXED_EMIT, True),
        (None, {"nonsense": 1}, False),
        (None, _WRAPPER_WITH_JUNK, False),
        (_CRITIC_INTENTS, _REVIEW_VERDICT_NATIVE, True),
        (_CRITIC_INTENTS, _REVIEW_VERDICT_WRAPPER, True),
        (_ORCHESTRATION_INTENTS, _REVIEW_VERDICT_NATIVE, False),
    ],
    ids=[
        "native",
        "wrapper",
        "native-with-wrapper",
        "junk",
        "wrapper-with-junk",
        "critic-native-in-role",
        "critic-wrapper-in-role",
        "orchestration-native-out-of-role",
    ],
)
async def test_registered_schema_and_handler_agree(
    allowed_intents: frozenset[IntentType] | None, arguments: dict[str, Any], accepted: bool
) -> None:
    """The declared schema gates the call before the handler ever runs, so a shape the handler accepts but the schema rejects can never land."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool(allowed_intents)

    if accepted:
        jsonschema.validate(instance=arguments, schema=tool.input_schema)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=arguments, schema=tool.input_schema)

    result = await tool.handler(arguments)
    assert ("is_error" not in result) is accepted


def _assert_names_the_orchestration_intents(refusal: str) -> None:
    assert "'review_verdict'" in refusal
    for intent_type in _ORCHESTRATION_INTENTS:
        assert intent_type.value in refusal


def test_registered_schema_refuses_a_native_out_of_role_intent() -> None:
    """The SDK returns the schema validation message to the model before the handler runs, so the role's enum is what refuses a native call, naming the types the role may emit."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool(_ORCHESTRATION_INTENTS)

    with pytest.raises(jsonschema.ValidationError) as exc:
        jsonschema.validate(instance=_REVIEW_VERDICT_NATIVE, schema=tool.input_schema)

    _assert_names_the_orchestration_intents(exc.value.message)


@pytest.mark.asyncio
async def test_registered_handler_refuses_a_wrapped_out_of_role_intent() -> None:
    """The enum cannot see inside the parser wrapper, so the handler refuses that form in the same turn instead."""
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool(_ORCHESTRATION_INTENTS)
    jsonschema.validate(instance=_REVIEW_VERDICT_WRAPPER, schema=tool.input_schema)

    result = await tool.handler(_REVIEW_VERDICT_WRAPPER)

    assert result["is_error"] is True
    _assert_names_the_orchestration_intents(result["content"][0]["text"])


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
