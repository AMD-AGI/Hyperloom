# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ClaudeBackend + emit_intent MCP server tests."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.roles import ClaudeBackend
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed, RetryPolicy
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    build_emit_intent_server,
    build_emit_intent_tool_input_schema,
    validate_emit_intent_input,
)
from hyperloom.orchestrator.roles import claude as claude_module
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
)


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# Subclass so type(block).__name__ == "ToolUseBlock" (matching ClaudeBackend).
class ToolUseBlock(FakeToolUseBlock):  # type: ignore[misc, valid-type]
    pass


class TextBlock:
    def __init__(self, text: str):
        self.text = text


@dataclass
class FakeAssistantMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class FakeResultMessage:
    """Stand-in for the SDK's terminal ResultMessage (carries .result)."""

    result: str = ""
    content: list[Any] = field(default_factory=list)


@dataclass
class FakeSystemMessage:
    """Stand-in for the SDK's SystemMessage; the CLI's init frame carries its tool list in ``data``."""

    subtype: str
    data: dict[str, Any] = field(default_factory=dict)


def _make_query_factory(messages: list[Any]):
    async def _q(*, prompt, options):
        for m in messages:
            yield m

    return _q


def _make_raising_query_factory(exc: BaseException):
    async def _q(*, prompt, options):
        raise exc
        yield  # pragma: no cover — keeps this an async generator

    return _q


@pytest.mark.asyncio
async def test_stream_api_error_is_marked_as_llm_call_failed(claude_sdk):
    """A non-timeout gateway error must be countable, not just a timeout."""
    claude_sdk.query = _make_raising_query_factory(RuntimeError("litellm.BadRequestError: AnthropicException"))
    backend = ClaudeBackend(
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(LLMCallFailed, match="Claude backend call failed"):
        await backend.run("hello")


@pytest.mark.asyncio
async def test_cancellation_is_not_marked_as_an_llm_failure(claude_sdk):
    """Cancellation is a BaseException and must pass through untouched."""
    claude_sdk.query = _make_raising_query_factory(asyncio.CancelledError())
    backend = ClaudeBackend(
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(asyncio.CancelledError):
        await backend.run("hello")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
            id="minimal_send_message",
        ),
        pytest.param(
            {
                "intent_type": "alert",
                "payload": {"severity": "high", "summary": "server down"},
            },
            id="alert_full",
        ),
        pytest.param(
            {
                "__unparsedToolInput": {
                    "raw": '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}',
                    "len": 62,
                }
            },
            id="unparsed_wrapper_fallback",
        ),
    ],
)
def test_validate_emit_intent_input_accepts_well_formed(payload):
    validate_emit_intent_input(payload)


@pytest.mark.parametrize(
    "payload, error_match",
    [
        pytest.param(
            {"intent_type": "objection", "payload": {}},
            "unknown intent_type",
            id="unknown_intent_type",
        ),
        pytest.param(
            {"payload": {"topic": "x"}},
            "requires both",
            id="missing_intent_type",
        ),
        pytest.param(
            {"intent_type": "send_message", "payload": "not_object"},
            "must be an object",
            id="payload_must_be_object",
        ),
        pytest.param(
            {
                "intent_type": "send_message",
                "payload": {"topic": "heartbeat"},
                "extra": "nope",
            },
            "unexpected keys",
            id="unexpected_top_keys",
        ),
        pytest.param(
            {"intent_type": "alert", "payload": {"severity": "high"}},
            "summary",
            id="alert_missing_summary",
        ),
    ],
)
def test_validate_emit_intent_input_rejects_malformed(payload, error_match):
    with pytest.raises(IntentValidationError, match=error_match):
        validate_emit_intent_input(payload)


@pytest.mark.parametrize("absent", ["query", "ClaudeAgentOptions", "tool", "create_sdk_mcp_server"])
def test_claude_backend_refuses_an_sdk_that_lacks_a_name_it_calls(claude_sdk, absent):
    """An SDK that cannot serve ``emit_intent`` fails the backend at construction, naming what is missing."""
    delattr(claude_sdk, absent)

    with pytest.raises(BackendError, match=f"claude_agent_sdk loaded but missing {absent}"):
        ClaudeBackend()


_DELEGATE_EMIT = {"intent_type": "delegate", "payload": {"action_name": "baseline"}}


def _capturing_tool_factory(captured: dict[str, Any]):
    def fake_tool(name, desc, schema):
        captured["schema"] = schema

        def deco(handler):
            captured["handler"] = handler
            return handler

        return deco

    return fake_tool


def _registered_emit_intent() -> dict[str, Any]:
    """The schema and handler ``build_emit_intent_server`` registers."""
    captured: dict[str, Any] = {}
    build_emit_intent_server(
        SimpleNamespace(
            tool=_capturing_tool_factory(captured),
            create_sdk_mcp_server=lambda name, version, tools: {"server": name},
        )
    )
    return captured


def _offered_intent_types(schema: dict[str, Any]) -> set[str]:
    return set(schema["properties"]["intent_type"]["enum"])


def test_emit_intent_schema_offers_every_intent():
    schema = _registered_emit_intent()["schema"]

    assert schema == build_emit_intent_tool_input_schema()
    assert _offered_intent_types(schema) == {t.value for t in IntentType}
    assert "delegate" in schema["properties"]["payload"]["description"]
    assert "__unparsedToolInput" in schema["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True], ids=["native-if-schema-bypassed", "unparsed-wrapper"])
async def test_the_emit_intent_handler_refuses_an_intent_type_outside_the_enum(wrapped):
    """The handler refuses by the decoded type.

    Under the real SDK only the wrapped form reaches the handler: the schema's enum refuses a native call first
    (pinned in test_claude_unparsed_tool_input_unit). The native case pins the handler's own check.
    """
    handler = _registered_emit_intent()["handler"]
    emitted = {"intent_type": "review_verdict", "payload": {"verdict": "approve"}}
    arguments: dict[str, Any] = emitted
    if wrapped:
        raw = json.dumps(emitted)
        arguments = {"__unparsedToolInput": {"raw": raw, "len": len(raw)}}

    result = await handler(arguments)

    assert result["is_error"] is True
    assert "unknown intent_type 'review_verdict'" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "unparsed-wrapper"])
async def test_the_emit_intent_tool_acks_a_valid_intent(wrapped):
    handler = _registered_emit_intent()["handler"]
    arguments: dict[str, Any] = _DELEGATE_EMIT
    if wrapped:
        raw = json.dumps(_DELEGATE_EMIT)
        arguments = {"__unparsedToolInput": {"raw": raw, "len": len(raw)}}

    result = await handler(arguments)

    assert "is_error" not in result
    assert result["content"][0]["text"] == "ok"


def test_claude_backend_registers_emit_intent_with_every_intent_type(claude_sdk):
    captured: dict[str, Any] = {}
    claude_sdk.tool = _capturing_tool_factory(captured)
    ClaudeBackend()

    assert _offered_intent_types(captured["schema"]) == {t.value for t in IntentType}


def test_claude_backend_raises_without_the_sdk(monkeypatch):
    """A missing SDK is a BackendError that names the package to install."""
    import importlib

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(BackendError, match="claude-agent-sdk"):
        ClaudeBackend()


def _clear_effort_env(monkeypatch):
    for k in (
        "INFERENCE_OPTIMIZER_CLAUDE_EFFORT",
        "INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT",
        "INFERENCE_OPTIMIZER_CLAUDE_THINKING",
    ):
        monkeypatch.delenv(k, raising=False)


def test_build_options_effort_defaults_to_medium(claude_sdk, monkeypatch):
    _clear_effort_env(monkeypatch)
    backend = ClaudeBackend(model="m")
    o = backend._build_options(tools=[], system_prompt="sp")
    assert o.kwargs["effort"] == "medium"
    assert o.kwargs["thinking"] == {"type": "adaptive"}


def test_build_options_runs_the_cli_the_subprocess_specialists_spawn(claude_sdk, monkeypatch, tmp_path):
    """The in-process backend hands the SDK the same CLI ``resolve_claude_executable`` gives subprocesses."""
    recorded = tmp_path / "claude"
    recorded.write_text("#!/bin/sh\n", encoding="utf-8")
    recorded.chmod(0o755)
    monkeypatch.delenv("HYPERLOOM_CLAUDE_CLI_PATH", raising=False)
    monkeypatch.setenv("GEAK_CLAUDE_BIN", str(recorded))
    backend = ClaudeBackend(model="m")
    assert backend._build_options(tools=[], system_prompt=None).kwargs["cli_path"] == str(recorded)


def test_build_options_effort_env_override_and_thinking_off(claude_sdk, monkeypatch):
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT", "high")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_THINKING", "off")
    b = ClaudeBackend(model="m")
    o = b._build_options(tools=[], system_prompt="sp")
    assert o.kwargs["effort"] == "high"
    assert "thinking" not in o.kwargs


def test_real_sdk_options_accept_hyperloom_kwargs(monkeypatch):
    """SDK compat: the pinned SDK must accept the kwargs _build_options sends."""
    _clear_effort_env(monkeypatch)
    pytest.importorskip("claude_agent_sdk")
    b = ClaudeBackend(model="m")
    # Must not raise: effort, thinking and the emit_intent server are all accepted by ClaudeAgentOptions.
    b._build_options(tools=[], system_prompt="sp")


@pytest.mark.asyncio
async def test_run_extracts_single_emit_intent_tool_use(claude_sdk):
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat", "body_md": "ok"},
                },
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("hello", system_prompt="sys", tools=["Read"])
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_run_extracts_multiple_emit_intent_tool_uses(claude_sdk):
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "delegate",
                    "payload": {"action_name": "baseline"},
                },
            ),
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "request",
                    "payload": {"target_agent": "kernel_agent", "kind": "trace_analyze"},
                },
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert len(res.intents) == 2
    assert res.intents[0].type == IntentType.DELEGATE
    assert res.intents[1].type == IntentType.REQUEST


@pytest.mark.asyncio
async def test_run_accepts_short_tool_name_too(claude_sdk):
    """SDK in-process tools sometimes show as plain `emit_intent` (no mcp__ prefix)."""
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_NAME,
                input={
                    "intent_type": "alert",
                    "payload": {"severity": "low", "summary": "x"},
                },
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert res.intents and res.intents[0].type == IntentType.ALERT


@pytest.mark.asyncio
async def test_run_ignores_other_tool_uses(claude_sdk):
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name="Read", input={"path": "/tmp/x"}),
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat"},
                },
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert len(res.intents) == 1


@pytest.mark.asyncio
async def test_run_text_blocks_collected_into_raw_text(claude_sdk):
    msg = FakeAssistantMessage(
        content=[
            TextBlock(text="thinking..."),
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat"},
                },
            ),
            TextBlock(text=" done."),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert "thinking..." in res.raw_text
    assert " done." in res.raw_text


@pytest.mark.asyncio
async def test_run_without_an_emit_intent_call_returns_no_intents(claude_sdk):
    msg = FakeAssistantMessage(
        content=[
            TextBlock(text="just thinking, no tool call."),
            ToolUseBlock(name="Read", input={"path": "/tmp/x"}),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    result = await backend.run("p")
    assert result.intents == []
    assert "just thinking" in result.raw_text


@pytest.mark.parametrize(
    ("offered_tools", "offered"),
    [
        pytest.param([EMIT_INTENT_TOOL_QUALIFIED, "mcp__other__tool"], True, id="offered"),
        pytest.param(["mcp__other__tool"], False, id="dropped"),
    ],
)
@pytest.mark.asyncio
async def test_mcp_setup_diagnostic_reports_what_the_cli_offered(claude_sdk, offered_tools, offered):
    """Registering emit_intent does not mean the CLI offered it, so the setup snapshot carries the CLI's own account."""
    init = FakeSystemMessage(
        subtype="init",
        data={
            "claude_code_version": "2.1.294",
            "tools": ["Read", *offered_tools],
            "mcp_servers": [{"name": "inference_optimizer", "status": "connected"}],
        },
    )
    claude_sdk.query = _make_query_factory([init, FakeAssistantMessage(content=[TextBlock(text="hi")])])
    backend = ClaudeBackend()
    assert backend.get_mcp_setup_diagnostic()["emit_intent"]["offered"] is None
    await backend.run("p")
    setup = backend.get_mcp_setup_diagnostic()
    assert setup["cli_version"] == "2.1.294"
    assert setup["mcp_tools"] == sorted(offered_tools)
    assert setup["mcp_server_status"] == {"inference_optimizer": "connected"}
    assert setup["emit_intent"]["offered"] is offered


def test_mcp_setup_diagnostic_hashes_the_registered_schema(claude_sdk):
    """The recorded schema hash identifies the schema the backend registers."""
    import hashlib

    backend = ClaudeBackend()
    expected = json.dumps(build_emit_intent_tool_input_schema(), sort_keys=True, separators=(",", ":"))

    recorded = backend.get_mcp_setup_diagnostic()["emit_intent"]["schema_sha256"]

    assert recorded == hashlib.sha256(expected.encode("utf-8")).hexdigest()


@pytest.mark.asyncio
async def test_run_invalid_tool_use_input_drops_block_silently(claude_sdk):
    """Bad tool_use input shouldn't crash the run — just drop the bad block."""
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "send_message",
                    "payload": {},  # missing required `topic`
                },
            ),
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={"intent_type": "send_message"},  # no `payload` key at all
            ),
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat"},
                },
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert len(res.intents) == 1


@pytest.mark.parametrize("as_tool_input", [dict, json.dumps], ids=["object", "json-string"])
@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_every_unparsed_tool_wrapper_becomes_an_intent(claude_sdk, tool_name, as_tool_input):
    """Claude Code's wrapped JSON decodes to its intent, also inside the JSON string an OpenAI-compatible proxy
    sends, and every call is kept: a repeated one is the model's call to make, as it is for a native input."""
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "ok"}}'
    other = '{"intent_type": "send_message", "payload": {"topic": "status", "body_md": "next"}}'
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=tool_name, input=as_tool_input({"__unparsedToolInput": {"raw": wrapped, "len": len(wrapped)}})
            )
            for wrapped in (raw, raw, other)
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert [intent.payload["topic"] for intent in res.intents] == ["heartbeat", "heartbeat", "status"]


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_empty_string_tool_input_reports_decode_error(claude_sdk, tool_name, caplog):
    """An empty-string input reports a decode error, not a missing intent_type.

    ``input or {}`` used to swallow ``""`` into ``{}``, which decoded cleanly
    and surfaced later as ``intent_type None not in allowed set`` — pointing a
    reader at the model rather than at the undecodable envelope.
    """
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=""),
            ToolUseBlock(
                name=tool_name,
                input={"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
            ),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    with caplog.at_level(logging.INFO, logger=claude_module.log.name):
        res = await backend.run("p")
    # The empty-string block is dropped; the rest of the turn is unaffected.
    assert len(res.intents) == 1
    assert "claude tool_use decode failed: emit_intent tool input string is not valid JSON" in caplog.messages


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_identical_native_tool_inputs_are_not_deduped(claude_sdk, tool_name):
    """Native Claude objects keep every emit_intent call, even duplicates."""
    native = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=dict(native)),
            ToolUseBlock(name=tool_name, input=dict(native)),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert len(res.intents) == 2


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_wrapper_then_native_same_intent_keeps_both(claude_sdk, tool_name):
    """A wrapper block and a native block are two tool_use events."""
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
    wrapped = {"__unparsedToolInput": {"raw": raw, "len": len(raw)}}
    native = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=dict(wrapped)),
            ToolUseBlock(name=tool_name, input=dict(native)),
        ]
    )
    claude_sdk.query = _make_query_factory([msg])
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert len(res.intents) == 2


@pytest.mark.asyncio
async def test_raw_text_prefers_result_no_duplication(claude_sdk):
    """The streamed TextBlock and ResultMessage.result share content; raw_text must not concatenate both."""
    text = '{"tool": "emit_proposal", "args": {}}'
    msgs = [
        FakeAssistantMessage(content=[TextBlock(text=text)]),
        FakeResultMessage(result=text),
    ]
    claude_sdk.query = _make_query_factory(msgs)
    backend = ClaudeBackend()
    res = await backend.run("p")
    assert res.raw_text == text
    assert res.raw_text.count('"tool"') == 1


@pytest.mark.asyncio
async def test_options_include_the_system_prompt_and_the_backend_turn_cap(claude_sdk):
    captured: dict[str, Any] = {}

    async def q(*, prompt, options):
        captured["options_kwargs"] = options.kwargs
        captured["prompt"] = prompt
        captured["system_prompt_text"] = Path(options.kwargs["system_prompt"]["path"]).read_text(encoding="utf-8")
        return
        yield  # make it a generator

    claude_sdk.query = q
    backend = ClaudeBackend()
    await backend.run("the prompt", system_prompt="sys", tools=["Read"])
    # Claude Code counts the model's own messages as turns, so the cap leaves room for a tool loop.
    assert captured["options_kwargs"]["max_turns"] == backend.max_turns_default == 32
    assert captured["options_kwargs"]["system_prompt"]["type"] == "file"
    assert captured["system_prompt_text"] == "sys"
    assert "Read" in captured["options_kwargs"]["allowed_tools"]
    assert captured["options_kwargs"]["tools"] == ["Read"]
    # Output instructions appended to prompt
    assert "OUTPUT FORMAT" in captured["prompt"]


@pytest.mark.asyncio
async def test_options_includes_mcp_server_when_emit_intent_enabled(claude_sdk):
    captured: dict[str, Any] = {}

    async def q(*, prompt, options):
        captured["options_kwargs"] = options.kwargs
        return
        yield

    def fake_tool(name, desc, schema):
        def deco(handler):
            return ("decorated", handler)

        return deco

    def fake_server(name, version, tools):
        return {"name": name}

    claude_sdk.query = q
    claude_sdk.tool = fake_tool
    claude_sdk.create_sdk_mcp_server = fake_server
    backend = ClaudeBackend()
    assert backend.mcp_server_config is not None
    assert backend.get_mcp_setup_diagnostic()["allowed_tools"] == []
    assert backend.get_mcp_setup_diagnostic()["mcp_servers"] == []
    await backend.run("p", tools=["Read", "emit_intent", "get_failure"])
    kw = captured["options_kwargs"]
    setup = backend.get_mcp_setup_diagnostic()
    assert setup["allowed_tools"] == kw["allowed_tools"]
    assert setup["mcp_servers"] == sorted(kw["mcp_servers"])
    assert "mcp_servers" in kw
    assert "inference_optimizer" in kw["mcp_servers"]
    # emit_intent qualified tool name auto-added to allowed_tools
    assert EMIT_INTENT_TOOL_QUALIFIED in kw["allowed_tools"]
    assert kw["tools"] == ["Read"]
