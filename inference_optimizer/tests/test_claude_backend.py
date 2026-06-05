"""P1-5 ClaudeBackend + emit_intent MCP server tests.

All tests use SDK test seams (``sdk_query_factory`` / ``sdk_options_cls``
/ ``mcp_*_factory``) so no real Claude API calls are made and no
``ANTHROPIC_API_KEY`` is required.

Covers:

* ``validate_emit_intent_input`` accepts well-formed payloads, rejects
  unknown intent_type, missing required fields, extra top-level keys
* ``build_emit_intent_server`` returns None when SDK helpers absent;
  builds a server when factories are provided
* ``ClaudeBackend.__post_init__`` raises BackendError when SDK + seams
  both absent
* ``ClaudeBackend.run`` collects intents from ToolUseBlock instances
* Multiple ToolUseBlocks in one stream → multiple Intents
* ToolUseBlock for a different tool → ignored
* Non-emit_intent text content → returned in raw_text
* Stream with zero matching tool_use blocks → NoIntentEmitted
* Validation error in tool_use input → block dropped (no exception)
* mcp_servers + allowed_tools propagated through ClaudeAgentOptions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import (
    ClaudeBackend,
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    build_emit_intent_server,
    validate_emit_intent_input,
)
from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.protocol.intent import (
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
)


# ===========================================================================
# Fakes — minimal stand-ins for SDK classes
# ===========================================================================
@dataclass
class FakeToolUseBlock:
    name: str
    input: dict[str, Any]

    @classmethod
    def __init_subclass__(cls, **kwargs):  # noqa: D401
        super().__init_subclass__(**kwargs)


# Subclass so type(block).__name__ == "ToolUseBlock" (matching ClaudeBackend)
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
    """Stand-in for the SDK's terminal ResultMessage (carries .result —
    the consolidated final assistant text)."""
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


# ===========================================================================
# validate_emit_intent_input
# ===========================================================================
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


# ===========================================================================
# build_emit_intent_server
# ===========================================================================
def test_build_emit_intent_server_returns_none_without_factories():
    """Empty SDK shim with no `tool` or `create_sdk_mcp_server` → None."""
    class EmptySdk:
        __name__ = "empty"
    cfg = build_emit_intent_server(
        sdk_module=EmptySdk(), tool_factory=None, server_factory=None,
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
        tool_factory=fake_tool, server_factory=fake_server,
    )
    assert cfg == {"server": "inference_optimizer"}
    assert captured["tool_name"] == EMIT_INTENT_TOOL_NAME
    assert captured["server_name"] == "inference_optimizer"
    assert captured["version"] == "1.0.0"
    assert len(captured["tools"]) == 1


# ===========================================================================
# ClaudeBackend construction
# ===========================================================================
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
            sdk_query_factory=None, sdk_options_cls=None, sdk_module=None,
            enable_mcp_emit_intent=False,
        )


def test_claude_backend_with_seams_constructs(monkeypatch):
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    assert backend.has_emit_intent_tool is False
    assert backend.sdk_query_factory is not None


def test_claude_backend_warns_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    assert any("ANTHROPIC_API_KEY" in str(c) for c in backend.calls)


# ===========================================================================
# ClaudeBackend.run — intent extraction
# ===========================================================================
@pytest.mark.asyncio
async def test_run_extracts_single_emit_intent_tool_use():
    msg = FakeAssistantMessage(content=[
        ToolUseBlock(
            name=EMIT_INTENT_TOOL_QUALIFIED,
            input={
                "intent_type": "send_message",
                "payload": {"topic": "heartbeat", "body_md": "ok"},
            },
        ),
    ])
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
    msg = FakeAssistantMessage(content=[
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "propose_action",
            "payload": {"action_name": "baseline", "predicted_gain_pct": 0.0},
        }),
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "request",
            "payload": {"target_agent": "kernel", "kind": "trace_analyze"},
        }),
    ])
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
    msg = FakeAssistantMessage(content=[
        ToolUseBlock(name=EMIT_INTENT_TOOL_NAME, input={
            "intent_type": "alert",
            "payload": {"severity": "low", "summary": "x"},
        }),
    ])
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert res.intents and res.intents[0].type == IntentType.ALERT


@pytest.mark.asyncio
async def test_run_ignores_other_tool_uses():
    msg = FakeAssistantMessage(content=[
        ToolUseBlock(name="Read", input={"path": "/tmp/x"}),
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat"},
        }),
    ])
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    assert len(res.intents) == 1


@pytest.mark.asyncio
async def test_run_text_blocks_collected_into_raw_text():
    msg = FakeAssistantMessage(content=[
        TextBlock(text="thinking..."),
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "send_message", "payload": {"topic": "heartbeat"},
        }),
        TextBlock(text=" done."),
    ])
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
    msg = FakeAssistantMessage(content=[
        TextBlock(text="just thinking, no tool call."),
        ToolUseBlock(name="Read", input={"path": "/tmp/x"}),
    ])
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
    msg = FakeAssistantMessage(content=[
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "send_message",
            "payload": {},  # missing required `topic`
        }),
        ToolUseBlock(name=EMIT_INTENT_TOOL_QUALIFIED, input={
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat"},
        }),
    ])
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([msg]),
        sdk_options_cls=FakeOptions,
        enable_mcp_emit_intent=False,
    )
    res = await backend.run("p")
    # 2 tool_blocks counted, but only 1 valid intent
    assert res.metadata["tool_blocks"] == 2
    assert len(res.intents) == 1


# ===========================================================================
# raw_completion mode (dynamic_action ReAct runner backend)
# ===========================================================================
@pytest.mark.asyncio
async def test_raw_completion_returns_raw_text_without_intent():
    """raw_completion mode: a text-only reply yields raw_text and does
    NOT raise NoIntentEmitted."""
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
    """The streamed TextBlock and the terminal ResultMessage.result
    carry the same content; raw_text must not concatenate both."""
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
    """When no ResultMessage is emitted, raw_text is the joined
    TextBlocks."""
    msgs = [FakeAssistantMessage(content=[
        TextBlock(text="part-a "), TextBlock(text="part-b"),
    ])]
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory(msgs),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    res = await backend.run("p")
    assert res.raw_text == "part-a part-b"


@pytest.mark.asyncio
async def test_raw_completion_options_disable_tools_and_skip_suffix():
    """raw_completion options: all tools disallowed, no MCP server, no
    OUTPUT FORMAT suffix, and max_turns bumped above the SDK's strict
    single-turn cap."""
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
    assert kw["max_turns"] >= 2
    assert "OUTPUT FORMAT" not in captured["prompt"]
    assert captured["prompt"] == "the prompt"


def test_raw_completion_disables_emit_intent_mcp():
    """Constructing with raw_completion=True must not register the
    emit_intent MCP server."""
    backend = ClaudeBackend(
        sdk_query_factory=_make_query_factory([]),
        sdk_options_cls=FakeOptions,
        raw_completion=True,
    )
    assert backend.has_emit_intent_tool is False


# ===========================================================================
# Options propagation
# ===========================================================================
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
    assert captured["options_kwargs"]["max_turns"] == 3
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
    assert backend.has_emit_intent_tool
    with pytest.raises(NoIntentEmitted):
        await backend.run("p", tools=["Read"])
    kw = captured["options_kwargs"]
    assert "mcp_servers" in kw
    assert "inference_optimizer" in kw["mcp_servers"]
    # emit_intent qualified tool name auto-added to allowed_tools
    assert EMIT_INTENT_TOOL_QUALIFIED in kw["allowed_tools"]
