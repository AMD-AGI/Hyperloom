# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ClaudeBackend pure helpers (no real SDK): prompt composition,
usage coercion, block iteration/classification, tool-use parsing, text
extraction, and conversation-session accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from hyperloom.orchestrator.roles import (
    ClaudeBackend,
    EMIT_INTENT_TOOL_NAME,
)
from hyperloom.orchestrator.roles.base import safe_int


@dataclass
class _FakeToolUse:
    name: str
    input: dict[str, Any]


class ToolUseBlock(_FakeToolUse):  # type(block).__name__ == "ToolUseBlock"
    pass


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


@dataclass
class _Msg:
    content: list[Any] = field(default_factory=list)


class _FakeOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _make_query_factory():
    async def _q(*, prompt, options):  # pragma: no cover - not invoked here
        if False:
            yield None

    return _q


def _backend(**over: Any) -> ClaudeBackend:
    kwargs: dict[str, Any] = dict(
        sdk_query_factory=_make_query_factory(),
        sdk_options_cls=_FakeOptions,
        api_key_env="UNSET_KEY_ENV_FOR_TEST",
    )
    kwargs.update(over)
    return ClaudeBackend(**kwargs)


def test_safe_int() -> None:
    assert safe_int(None) == 0
    assert safe_int("bad") == 0
    assert safe_int(7) == 7
    assert safe_int("9") == 9


def test_compose_prompt_raw_vs_normal() -> None:
    raw = _backend(raw_completion=True)
    assert raw._compose_prompt("hello") == "hello"
    normal = _backend(raw_completion=False)
    out = normal._compose_prompt("hello")
    assert out.startswith("hello")
    assert len(out) > len("hello")  # suffix appended


def test_set_context_provider_none_clears() -> None:
    b = _backend()
    b.set_context_provider(None)
    assert b._context_server_config is None


def test_reset_conversation_clears_session_id() -> None:
    b = _backend(conversational=True)
    b._session_id = "sess-1"
    b.reset_conversation()
    assert b._session_id is None


def test_build_options_pins_gateway_env_and_ignores_global_settings(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_SMALL_FAST_MODEL", raising=False)
    b = _backend(model="claude-opus-4-6")

    opts = b._build_options(tools=[], max_turns=4, system_prompt=None)

    assert opts.kwargs["setting_sources"] == []
    child_env = opts.kwargs["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://llm.example.invalid/anthropic"
    assert child_env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: sub-key"
    assert child_env["ANTHROPIC_MODEL"] == "claude-opus-4-6"
    assert child_env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-opus-4-6"


def test_build_options_leaves_settings_sources_unset_without_gateway_env(monkeypatch) -> None:
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    b = _backend(model="claude-opus-4-6")

    opts = b._build_options(tools=[], max_turns=4, system_prompt=None)

    assert "env" not in opts.kwargs
    assert "setting_sources" not in opts.kwargs


def test_build_options_never_maps_openai_key_onto_the_anthropic_side(monkeypatch) -> None:
    """An OpenAI-only environment produces no Anthropic credentials for the Claude
    child process."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: openai-key")
    b = _backend(model="claude-opus-4-6")

    opts = b._build_options(tools=[], max_turns=4, system_prompt=None)

    child_env = opts.kwargs["env"]
    assert opts.kwargs["setting_sources"] == []
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "ANTHROPIC_AUTH_TOKEN" not in child_env
    # OPENAI_CUSTOM_HEADERS is not copied to the Anthropic side.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in child_env


def test_iter_blocks() -> None:
    assert ClaudeBackend._iter_blocks(_Msg(content=[1, 2])) == [1, 2]
    assert ClaudeBackend._iter_blocks(_Msg(content=None)) == []
    assert ClaudeBackend._iter_blocks(object()) == []


def test_is_tool_use_for_emit_intent() -> None:
    b = _backend()
    assert (
        b._is_tool_use_for_emit_intent(
            ToolUseBlock(name=EMIT_INTENT_TOOL_NAME, input={}),
        )
        is True
    )
    # wrong tool name
    assert (
        b._is_tool_use_for_emit_intent(
            ToolUseBlock(name="other_tool", input={}),
        )
        is False
    )
    # wrong block class
    assert b._is_tool_use_for_emit_intent(TextBlock("hi")) is False


def test_parse_tool_use_block_valid_and_invalid() -> None:
    b = _backend()
    valid = ToolUseBlock(
        name=EMIT_INTENT_TOOL_NAME,
        input={"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "ok"}},
    )
    intent = b._parse_tool_use_block(valid)
    assert intent is not None
    # invalid envelope -> None (validation failure swallowed)
    bad = ToolUseBlock(name=EMIT_INTENT_TOOL_NAME, input={"intent_type": "not_a_real_type"})
    assert b._parse_tool_use_block(bad) is None


def test_parse_tool_use_block_prefers_native_intent_type() -> None:
    b = _backend()
    block = ToolUseBlock(
        name=EMIT_INTENT_TOOL_NAME,
        input={
            "intent_type": "send_message",
            "payload": {"topic": "heartbeat", "body_md": "native"},
            "__unparsedToolInput": {
                "raw": '{"intent_type": "alert", "payload": {"severity": "high", "summary": "x"}}',
                "len": 1,
            },
        },
    )
    intent = b._parse_tool_use_block(block)
    assert intent is not None
    assert intent.type.value == "send_message"
    assert intent.payload["body_md"] == "native"


def test_parse_tool_use_block_unwraps_claude_code_wrapper() -> None:
    b = _backend()
    raw = '{"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "ok"}}'
    wrapped = ToolUseBlock(
        name=EMIT_INTENT_TOOL_NAME,
        input={"__unparsedToolInput": {"raw": raw, "len": len(raw)}},
    )
    intent = b._parse_tool_use_block(wrapped)
    assert intent is not None
    assert intent.type.value == "send_message"
    assert intent.payload["topic"] == "heartbeat"


def test_parse_tool_use_block_unparsed_malformed_json_returns_none() -> None:
    b = _backend()
    bad = ToolUseBlock(
        name=EMIT_INTENT_TOOL_NAME,
        input={"__unparsedToolInput": {"raw": "{not-json", "len": 9}},
    )
    assert b._parse_tool_use_block(bad) is None


def test_extract_text_shapes() -> None:
    assert ClaudeBackend._extract_text(TextBlock("hello")) == "hello"
    assert ClaudeBackend._extract_text({"type": "text", "text": "d"}) == "d"
    assert ClaudeBackend._extract_text({"type": "image"}) == ""

    class _HasText:
        text = "attr-text"

    assert ClaudeBackend._extract_text(_HasText()) == "attr-text"

    class _NoText:
        text = 123  # non-string

    assert ClaudeBackend._extract_text(_NoText()) == ""
