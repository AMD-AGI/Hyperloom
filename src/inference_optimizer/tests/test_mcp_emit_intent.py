"""Tests for the in-process ``emit_intent`` MCP server (F4)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.claude import ClaudeBackend
from inference_optimizer.orchestrator.backends.mcp_emit_intent import (
    EMIT_INTENT_TOOL_DESCRIPTION,
    EMIT_INTENT_TOOL_INPUT_SCHEMA,
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    _emit_intent_handler,
    build_emit_intent_server,
    validate_emit_intent_input,
)
from inference_optimizer.orchestrator.intent_parser import (
    IntentType,
    IntentValidationError,
)


# ---------------------------------------------------------------------------
# validate_emit_intent_input
# ---------------------------------------------------------------------------
def test_validate_emit_intent_input_accepts_valid():
    validate_emit_intent_input(
        {
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat", "body_md": "ok"},
        }
    )


def test_validate_emit_intent_input_rejects_non_object():
    with pytest.raises(IntentValidationError):
        validate_emit_intent_input("not a dict")  # type: ignore[arg-type]


def test_validate_emit_intent_input_rejects_unknown_intent_type():
    with pytest.raises(IntentValidationError):
        validate_emit_intent_input(
            {"intent_type": "telepathy", "payload": {}}
        )


def test_validate_emit_intent_input_rejects_extra_top_level_keys():
    with pytest.raises(IntentValidationError):
        validate_emit_intent_input(
            {
                "intent_type": "send_message",
                "payload": {"topic": "x"},
                "extra": True,
            }
        )


def test_validate_emit_intent_input_rejects_missing_required_payload_field():
    with pytest.raises(IntentValidationError):
        validate_emit_intent_input(
            {
                "intent_type": "alert",
                "payload": {"summary": "no severity"},
            }
        )


def test_validate_emit_intent_input_rejects_payload_non_object():
    with pytest.raises(IntentValidationError):
        validate_emit_intent_input(
            {"intent_type": "alert", "payload": ["bad"]}
        )


# ---------------------------------------------------------------------------
# _emit_intent_handler
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handler_returns_ok_for_valid_input():
    res = await _emit_intent_handler(
        {
            "intent_type": "alert",
            "payload": {"severity": "info", "summary": "ok"},
        }
    )
    assert "content" in res
    assert res["content"][0]["type"] == "text"
    assert res["content"][0]["text"] == "ok"
    assert "is_error" not in res or not res.get("is_error")


@pytest.mark.asyncio
async def test_handler_returns_error_envelope_for_bad_input():
    res = await _emit_intent_handler({"intent_type": "send_message"})
    assert res.get("is_error") is True
    assert "validation_error" in res["content"][0]["text"]


# ---------------------------------------------------------------------------
# build_emit_intent_server: degrades to None when SDK helpers absent
# ---------------------------------------------------------------------------
class _StubSdkNoMcp:
    """Pretend SDK module that lacks ``tool``/``create_sdk_mcp_server``."""

    __name__ = "stub_sdk_no_mcp"


def test_build_returns_none_when_sdk_lacks_mcp_helpers():
    cfg = build_emit_intent_server(sdk_module=_StubSdkNoMcp())
    assert cfg is None


def test_build_uses_provided_factories():
    """If we inject ``tool_factory`` + ``server_factory`` directly we don't
    need an SDK at all — useful for unit tests and for environments where
    ``claude_agent_sdk`` is not yet installed but we still want to dry-run
    the wiring."""

    captured: dict[str, Any] = {}

    def fake_tool(name: str, description: str, input_schema: dict[str, Any]):
        captured["tool_args"] = (name, description, input_schema)

        def decorator(handler):
            captured["handler"] = handler
            return {"name": name, "handler": handler, "schema": input_schema}

        return decorator

    def fake_server(name: str, version: str, tools: list[Any]):
        captured["server_args"] = (name, version, list(tools))
        return {"server": name, "tools": tools}

    cfg = build_emit_intent_server(
        sdk_module=_StubSdkNoMcp(),
        tool_factory=fake_tool,
        server_factory=fake_server,
    )

    assert cfg == {"server": MCP_SERVER_NAME, "tools": cfg["tools"]}
    name, description, schema = captured["tool_args"]
    assert name == EMIT_INTENT_TOOL_NAME
    assert description == EMIT_INTENT_TOOL_DESCRIPTION
    assert schema == EMIT_INTENT_TOOL_INPUT_SCHEMA
    server_name, version, tools = captured["server_args"]
    assert server_name == MCP_SERVER_NAME
    assert version == "1.0.0"
    assert len(tools) == 1


def test_build_with_real_sdk_returns_mcp_sdk_server_config():
    """End-to-end check using the real ``claude-agent-sdk`` (already a
    project dependency).

    ``McpSdkServerConfig`` is a ``TypedDict`` so we can't use ``isinstance``
    on it; instead we duck-type the result against the SDK's documented
    shape: ``{type: "sdk", name, instance}``.
    """

    pytest.importorskip("claude_agent_sdk")

    cfg = build_emit_intent_server()
    assert cfg is not None
    assert isinstance(cfg, dict)
    assert cfg.get("type") == "sdk"
    assert cfg.get("name") == MCP_SERVER_NAME
    assert cfg.get("instance") is not None


# ---------------------------------------------------------------------------
# ClaudeBackend integration: tool injection + prompt switch
# ---------------------------------------------------------------------------
@dataclass
class _FlexOptions:
    """Permissive options class — accepts any kwargs the backend hands us.

    The real ``ClaudeAgentOptions`` accepts ``mcp_servers``, ``allowed_tools``,
    ``max_turns``, ``model`` and many others. For tests we just need a
    placeholder that records everything.
    """

    allowed_tools: list = field(default_factory=list)
    mcp_servers: dict = field(default_factory=dict)
    max_turns: int = 10
    model: str | None = None


def _fake_query_yielding(messages):
    async def _gen():
        for m in messages:
            yield m

    def fake(*, prompt, options):
        fake.last_prompt = prompt  # type: ignore[attr-defined]
        fake.last_options = options  # type: ignore[attr-defined]
        return _gen()

    fake.last_prompt = None  # type: ignore[attr-defined]
    fake.last_options = None  # type: ignore[attr-defined]
    return fake


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _AssistantMessage:
    content: list


@pytest.mark.asyncio
async def test_claude_backend_registers_emit_intent_when_factories_provided():
    """The backend builds an MCP server when given test-only factories.

    This avoids depending on the real SDK in CI: we feed in a fake tool
    factory + server factory, and assert the backend then injects
    ``mcp_servers={inference_optimizer: <stub>}`` plus the qualified tool
    name into ``allowed_tools``.
    """

    sentinel_server = {"server": MCP_SERVER_NAME, "fake": True}

    def fake_tool(name, description, input_schema):
        def decorator(handler):
            return {"name": name, "handler": handler}

        return decorator

    def fake_server(name, version, tools):
        return sentinel_server

    msg = _AssistantMessage(
        content=[
            _ToolUseBlock(
                name="emit_intent",
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat", "body_md": "hi"},
                },
            )
        ]
    )
    fake_query = _fake_query_yielding([msg])

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FlexOptions,
        sdk_extract_text=lambda m: "",
        sdk_module=_StubSdkNoMcp(),
        mcp_tool_factory=fake_tool,
        mcp_server_factory=fake_server,
    )

    assert backend.has_emit_intent_tool is True
    assert backend.mcp_tool_name == EMIT_INTENT_TOOL_QUALIFIED
    assert backend.mcp_server_config is sentinel_server

    intents = await backend.run("system prompt", agent_name="executor")
    assert intents and intents[0].type == IntentType.SEND_MESSAGE

    opts = fake_query.last_options
    assert EMIT_INTENT_TOOL_QUALIFIED in opts.allowed_tools
    assert MCP_SERVER_NAME in opts.mcp_servers
    assert opts.mcp_servers[MCP_SERVER_NAME] is sentinel_server
    # Prompt should reference the tool path explicitly.
    assert EMIT_INTENT_TOOL_NAME in fake_query.last_prompt


@pytest.mark.asyncio
async def test_claude_backend_disabled_mcp_does_not_inject_anything():
    msg = _AssistantMessage(
        content=[
            _ToolUseBlock(
                name="emit_intent",
                input={
                    "intent_type": "send_message",
                    "payload": {"topic": "heartbeat", "body_md": "hi"},
                },
            )
        ]
    )
    fake_query = _fake_query_yielding([msg])

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FlexOptions,
        sdk_extract_text=lambda m: "",
        enable_mcp_emit_intent=False,
    )

    assert backend.has_emit_intent_tool is False
    await backend.run("system prompt", agent_name="executor")

    opts = fake_query.last_options
    assert EMIT_INTENT_TOOL_QUALIFIED not in opts.allowed_tools
    assert opts.mcp_servers == {}
