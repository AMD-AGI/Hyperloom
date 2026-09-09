# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ClaudeBackend + emit_intent MCP server tests.

All tests use SDK test seams so no real Claude API calls are made and no
``ANTHROPIC_API_KEY`` is required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    build_emit_intent_server,
    validate_emit_intent_input,
)
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed, RetryPolicy
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
)


@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]

    @classmethod
    def __init_subclass__(cls, **kwargs):  # noqa: D401
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
class FakeOptions:
    """Stand-in for ClaudeAgentOptions — captures kwargs for assertions."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.kwargs = kwargs


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
async def test_stream_api_error_is_marked_as_llm_call_failed():
    """A non-timeout gateway error must be countable, not just a timeout.

    The failure that motivated this telemetry is a gateway 400
    (``litellm.BadRequestError: AnthropicException``) surfacing out of the SDK
    stream. Left unmarked it reaches the Coordinator's "unexpected crash" path
    and no error row is written, so the LLM error rate silently misses exactly
    the case it was added for.
    """
    backend = ClaudeBackend(
        sdk_query_factory=_make_raising_query_factory(RuntimeError("litellm.BadRequestError: AnthropicException")),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(LLMCallFailed, match="Claude backend call failed"):
        await backend.run("hello")


@pytest.mark.asyncio
async def test_cancellation_is_not_marked_as_an_llm_failure():
    """Cancellation is a BaseException and must pass through untouched."""
    backend = ClaudeBackend(
        sdk_query_factory=_make_raising_query_factory(asyncio.CancelledError()),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
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
                "intent_type": "review_verdict",
                "payload": {"target_proposal_msg_id": "abc", "verdict": "approve"},
            },
            id="review_verdict_full",
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
            {"intent_type": "review_verdict", "payload": {"verdict": "approve"}},
            "target_proposal_msg_id",
            id="review_verdict_missing_target",
        ),
    ],
)
def test_validate_emit_intent_input_rejects_malformed(payload, error_match):
    with pytest.raises(IntentValidationError, match=error_match):
        validate_emit_intent_input(payload)


def test_build_emit_intent_server_returns_none_without_factories():
    """Empty SDK shim with no `tool` or `create_sdk_mcp_server` → None."""

    class EmptySdk:
        __name__ = "empty"

    cfg = build_emit_intent_server(
        sdk_module=EmptySdk(),
        tool_factory=None,
        server_factory=None,
    )
    assert cfg is None


def test_build_emit_intent_server_uses_factories():
    captured: dict[str, Any] = {}

    def fake_tool(name, desc, schema):
        captured["tool_name"] = name

        def deco(handler):
            captured["handler"] = handler
            return ("decorated", handler)

        return deco

    def fake_server(name, version, tools):
        captured["server_name"] = name
        captured["version"] = version
        captured["tools"] = tools
        return {"server": name}

    cfg = build_emit_intent_server(
        tool_factory=fake_tool,
        server_factory=fake_server,
    )
    assert cfg == {"server": "inference_optimizer"}
    assert captured["tool_name"] == EMIT_INTENT_TOOL_NAME
    assert captured["server_name"] == "inference_optimizer"
    assert captured["version"] == "1.0.0"
    assert len(captured["tools"]) == 1


def test_claude_backend_raises_without_sdk_or_seams(monkeypatch):
    """If neither real SDK nor test seams provided, BackendError fires."""
    import importlib

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(BackendError, match="claude-agent-sdk"):
        ClaudeBackend(
            sdk_query_factory=None,
            sdk_options_cls=None,
            sdk_module=None,
            enable_mcp_emit_intent=False,
        )


def test_claude_backend_with_seams_constructs(monkeypatch):
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    assert backend.sdk_query_factory is not None


def test_claude_backend_warns_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    assert any("ANTHROPIC_API_KEY" in str(c) for c in backend.calls)


def _clear_effort_env(monkeypatch):
    for k in (
        "INFERENCE_OPTIMIZER_CLAUDE_EFFORT",
        "INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT",
        "INFERENCE_OPTIMIZER_CLAUDE_KERNEL_EFFORT",
        "INFERENCE_OPTIMIZER_CLAUDE_THINKING",
    ):
        monkeypatch.delenv(k, raising=False)


def test_build_options_effort_defaults_by_role(monkeypatch):
    _clear_effort_env(monkeypatch)
    orch = ClaudeBackend(
        model="m", conversational=True, sdk_query_factory=_make_query_factory([]), sdk_options_cls=FakeOptions
    )
    o = orch._build_options(tools=[], max_turns=4, system_prompt="sp")
    assert o.kwargs["effort"] == "medium"
    assert o.kwargs["thinking"] == {"type": "adaptive"}

    kernel = ClaudeBackend(
        model="m", conversational=False, sdk_query_factory=_make_query_factory([]), sdk_options_cls=FakeOptions
    )
    k = kernel._build_options(tools=[], max_turns=4, system_prompt="sp")
    assert k.kwargs["effort"] == "low"


def test_build_options_effort_env_override_and_thinking_off(monkeypatch):
    _clear_effort_env(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT", "high")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_THINKING", "off")
    b = ClaudeBackend(
        model="m", conversational=True, sdk_query_factory=_make_query_factory([]), sdk_options_cls=FakeOptions
    )
    o = b._build_options(tools=[], max_turns=4, system_prompt="sp")
    assert o.kwargs["effort"] == "high"
    assert "thinking" not in o.kwargs


def test_real_sdk_options_accept_hyperloom_kwargs(monkeypatch):
    """SDK compat: the pinned SDK must accept the kwargs _build_options sends."""
    _clear_effort_env(monkeypatch)
    sdk = pytest.importorskip("claude_agent_sdk")
    b = ClaudeBackend(
        model="m",
        conversational=True,
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=sdk.ClaudeAgentOptions,
        enable_mcp_emit_intent=False,
    )
    # Must not raise: effort + thinking + resume all accepted by ClaudeAgentOptions.
    b._build_options(tools=[], max_turns=4, system_prompt="sp", resume_session_id="sess-1")


@pytest.mark.asyncio
async def test_run_extracts_single_emit_intent_tool_use():
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
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("hello", system_prompt="sys", tools=["Read"])
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"
    assert res.metadata["tool_blocks"] == 1


@pytest.mark.asyncio
async def test_run_extracts_multiple_emit_intent_tool_uses():
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(
                name=EMIT_INTENT_TOOL_QUALIFIED,
                input={
                    "intent_type": "propose_action",
                    "payload": {"action_name": "baseline", "predicted_gain_pct": 0.0},
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
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert len(res.intents) == 2
    assert res.intents[0].type == IntentType.PROPOSE_ACTION
    assert res.intents[1].type == IntentType.REQUEST


@pytest.mark.asyncio
async def test_run_accepts_short_tool_name_too():
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
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert res.intents and res.intents[0].type == IntentType.ALERT


@pytest.mark.asyncio
async def test_run_ignores_other_tool_uses():
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
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert len(res.intents) == 1


@pytest.mark.asyncio
async def test_run_text_blocks_collected_into_raw_text():
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
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert "thinking..." in res.raw_text
    assert " done." in res.raw_text


@pytest.mark.asyncio
async def test_run_no_emit_intent_raises_no_intent_emitted():
    msg = FakeAssistantMessage(
        content=[
            TextBlock(text="just thinking, no tool call."),
            ToolUseBlock(name="Read", input={"path": "/tmp/x"}),
        ]
    )
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    with pytest.raises(NoIntentEmitted):
        await backend.run("p")


@pytest.mark.asyncio
async def test_run_invalid_tool_use_input_drops_block_silently():
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
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat"},
                },
            ),
        ]
    )
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    # 2 tool_blocks counted, but only 1 valid intent
    assert res.metadata["tool_blocks"] == 2
    assert len(res.intents) == 1


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_unparsed_tool_wrapper_retries_dedupe_to_one_intent(tool_name):
    """Claude Code retries the same wrapped JSON; keep one validated intent."""
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "ok"}}'
    wrapped = {"__unparsedToolInput": {"raw": raw, "len": len(raw)}}
    other = '{"intent_type": "send_message", "payload": {"topic": "status", "body_md": "next"}}'
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=dict(wrapped)),
            ToolUseBlock(name=tool_name, input=dict(wrapped)),
            ToolUseBlock(
                name=tool_name,
                input={"__unparsedToolInput": {"raw": other, "len": len(other)}},
            ),
        ]
    )
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
        capture_turn_diagnostics=True,
    )
    res = await backend.run("p")
    assert res.metadata["tool_blocks"] == 3
    assert len(res.intents) == 2
    assert [intent.payload["topic"] for intent in res.intents] == ["heartbeat", "status"]
    assert backend.get_turn_diagnostic()["deduped_fallback_intents"] == 1


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_identical_native_tool_inputs_are_not_deduped(tool_name):
    """Native Claude objects keep every emit_intent call, even duplicates."""
    native = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=dict(native)),
            ToolUseBlock(name=tool_name, input=dict(native)),
        ]
    )
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert res.metadata["tool_blocks"] == 2
    assert len(res.intents) == 2


@pytest.mark.parametrize("tool_name", [EMIT_INTENT_TOOL_QUALIFIED, EMIT_INTENT_TOOL_NAME])
@pytest.mark.asyncio
async def test_wrapper_then_native_same_intent_keeps_both(tool_name):
    """A wrapper block and a native block are two tool_use events.

    The coordinator executes every intent with no merge. After the MCP
    handler acks the wrapper, the model should not retry; if both still
    appear in one query they are kept rather than dropping a real second
    emit.
    """
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}'
    wrapped = {"__unparsedToolInput": {"raw": raw, "len": len(raw)}}
    native = {"intent_type": "send_message", "payload": {"topic": "heartbeat"}}
    msg = FakeAssistantMessage(
        content=[
            ToolUseBlock(name=tool_name, input=dict(wrapped)),
            ToolUseBlock(name=tool_name, input=dict(native)),
        ]
    )
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert res.metadata["tool_blocks"] == 2
    assert len(res.intents) == 2


@pytest.mark.asyncio
async def test_raw_completion_returns_raw_text_without_intent():
    """raw_completion mode: a text-only reply yields raw_text and does NOT raise NoIntentEmitted."""
    action = '{"tool": "read_source", "args": {"path": "/x"}}'
    msgs = [
        FakeAssistantMessage(content=[TextBlock(text=action)]),
        FakeResultMessage(result=action),
    ]
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory(msgs),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    res = await backend.run("p")
    assert res.intents == []
    assert res.raw_text == action  # not duplicated


@pytest.mark.asyncio
async def test_raw_completion_prefers_result_no_duplication():
    """The streamed TextBlock and ResultMessage.result share content; raw_text must not concatenate both."""
    text = '{"tool": "emit_proposal", "args": {}}'
    msgs = [
        FakeAssistantMessage(content=[TextBlock(text=text)]),
        FakeResultMessage(result=text),
    ]
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory(msgs),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    res = await backend.run("p")
    assert res.raw_text == text
    assert res.raw_text.count('"tool"') == 1


@pytest.mark.asyncio
async def test_raw_completion_falls_back_to_text_blocks_without_result():
    """When no ResultMessage is emitted, raw_text is the joined TextBlocks."""
    msgs = [
        FakeAssistantMessage(
            content=[
                TextBlock(text="part-a "),
                TextBlock(text="part-b"),
            ]
        )
    ]
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory(msgs),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    res = await backend.run("p")
    assert res.raw_text == "part-a part-b"


@pytest.mark.asyncio
async def test_raw_completion_options_disable_tools_and_skip_suffix():
    """raw_completion options: tools disallowed, no MCP server, no OUTPUT FORMAT suffix, max_turns bumped."""
    captured: dict[str, Any] = {}

    async def q(*, prompt, options):
        captured["kwargs"] = options.kwargs
        captured["prompt"] = prompt
        yield FakeResultMessage(result="ok")

    backend = ClaudeBackend(
        sdk_query_factory=q,
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    await backend.run("the prompt", system_prompt="sys", max_turns=1)
    kw = captured["kwargs"]
    assert kw["allowed_tools"] == []
    assert kw["disallowed_tools"]  # built-ins blocked
    assert "mcp_servers" not in kw
    assert kw["max_turns"] >= 8
    assert "OUTPUT FORMAT" not in captured["prompt"]
    assert captured["prompt"] == "the prompt"


def test_raw_completion_disables_emit_intent_mcp():
    """Constructing with raw_completion=True must not register the emit_intent MCP server."""
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    assert backend.mcp_server_config is None
    assert backend.mcp_tool_name is None


@pytest.mark.asyncio
async def test_options_includes_system_prompt_and_max_turns():
    captured: dict[str, Any] = {}

    async def q(*, prompt, options):
        captured["options_kwargs"] = options.kwargs
        captured["prompt"] = prompt
        return
        yield  # make it a generator

    backend = ClaudeBackend(
        sdk_query_factory=q,
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
        max_turns_default=7,
    )
    with pytest.raises(NoIntentEmitted):
        await backend.run("the prompt", system_prompt="sys", tools=["Read"], max_turns=3)
    # max_turns is floored to _RAW_COMPLETION_MIN_MAX_TURNS (8) for every mode:
    # Claude Code counts its own messages as turns, so a literal max_turns=3
    # would trip before the model can emit an intent.
    assert captured["options_kwargs"]["max_turns"] == 8
    assert captured["options_kwargs"]["system_prompt"] == "sys"
    assert "Read" in captured["options_kwargs"]["allowed_tools"]
    # Output instructions appended to prompt
    assert "OUTPUT FORMAT" in captured["prompt"]


@pytest.mark.asyncio
async def test_options_includes_mcp_server_when_emit_intent_enabled():
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

    backend = ClaudeBackend(
        sdk_query_factory=q,
        sdk_options_cls=FakeOptions,
        mcp_tool_factory=fake_tool,
        mcp_server_factory=fake_server,
        enable_mcp_emit_intent=True,
    )
    assert backend.mcp_server_config is not None
    assert backend.mcp_tool_name == EMIT_INTENT_TOOL_QUALIFIED
    with pytest.raises(NoIntentEmitted):
        await backend.run("p", tools=["Read"])
    kw = captured["options_kwargs"]
    assert "mcp_servers" in kw
    assert "inference_optimizer" in kw["mcp_servers"]
    # emit_intent qualified tool name auto-added to allowed_tools
    assert EMIT_INTENT_TOOL_QUALIFIED in kw["allowed_tools"]
