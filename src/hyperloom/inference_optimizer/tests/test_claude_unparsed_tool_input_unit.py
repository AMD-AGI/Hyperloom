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


def _registered_emit_intent_tool() -> Any:
    """The emit_intent tool as the real SDK decorator registers it.

    Captured off the ``server_factory`` seam so the ``tool`` decorator, the
    schema and the handler are all the production ones, without depending on
    where a given ``mcp`` release keeps its server internals.
    """
    sdk = pytest.importorskip("claude_agent_sdk")
    captured: list[Any] = []

    def capture(name: str, version: str, tools: list[Any]) -> None:
        captured.extend(tools)

    build_emit_intent_server(sdk_module=sdk, server_factory=capture)
    tool = next((t for t in captured if getattr(t, "name", None) == EMIT_INTENT_TOOL_NAME), None)
    if tool is None:
        pytest.skip("SDK did not register the emit_intent tool")
    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,schema_ok",
    [(_NATIVE_EMIT, True), (_WRAPPER_EMIT, True), ({"nonsense": 1}, False)],
    ids=["native", "wrapper", "junk"],
)
async def test_registered_schema_and_handler_agree(arguments: dict[str, Any], schema_ok: bool) -> None:
    """The declared schema gates the call before the handler ever runs, so a
    shape the handler accepts but the schema rejects can never land. Both are
    checked here against the same tool the SDK registers, because they used to
    disagree: the handler decoded the wrapper the schema had already refused.
    """
    jsonschema = pytest.importorskip("jsonschema")
    tool = _registered_emit_intent_tool()

    if not schema_ok:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        return

    jsonschema.validate(instance=arguments, schema=tool.input_schema)
    result = await tool.handler(arguments)
    assert result["content"][0]["text"] == "ok"
    assert "is_error" not in result


def test_registered_schema_offers_both_shapes() -> None:
    """The declared schema is what the model reads, so the wrapper alternative
    has to survive into it and say it is internal."""
    schema = _registered_emit_intent_tool().input_schema
    assert {"required": ["intent_type", "payload"]} in schema["anyOf"]
    assert {"required": ["__unparsedToolInput"]} in schema["anyOf"]
    assert "Never emit this deliberately" in schema["properties"]["__unparsedToolInput"]["description"]
