# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for ClaudeBackend pure helpers (no real SDK): prompt composition,
usage coercion, block iteration/classification, tool-use parsing, text
extraction, and conversation-session accessors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import (
    ClaudeBackend,
    EMIT_INTENT_TOOL_NAME,
)


# -- SDK fakes -------------------------------------------------------------
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


# -- _safe_int -------------------------------------------------------------
def test_safe_int() -> None:
    assert ClaudeBackend._safe_int(None) == 0
    assert ClaudeBackend._safe_int("bad") == 0
    assert ClaudeBackend._safe_int(7) == 7
    assert ClaudeBackend._safe_int("9") == 9


# -- _compose_prompt -------------------------------------------------------
def test_compose_prompt_raw_vs_normal() -> None:
    raw = _backend(raw_completion=True)
    assert raw._compose_prompt("hello") == "hello"
    normal = _backend(raw_completion=False)
    out = normal._compose_prompt("hello")
    assert out.startswith("hello")
    assert len(out) > len("hello")  # suffix appended


# -- context provider + emit_intent wiring --------------------------------
def test_set_context_provider_none_clears() -> None:
    b = _backend()
    b.set_context_provider(None)
    assert b.has_context_tools is False


def test_has_emit_intent_tool_in_raw_mode_is_false() -> None:
    # raw_completion disables MCP emit_intent wiring
    b = _backend(raw_completion=True)
    assert b.has_emit_intent_tool is False


def test_conversation_session_accessors() -> None:
    b = _backend(conversational=True)
    assert b.conversation_session_id is None  # nothing captured yet
    b._session_id = "sess-1"
    assert b.conversation_session_id == "sess-1"
    b.reset_conversation()
    assert b._session_id is None
    # non-conversational backend never exposes a session id
    b2 = _backend(conversational=False)
    b2._session_id = "x"
    assert b2.conversation_session_id is None


# -- block helpers ---------------------------------------------------------
def test_iter_blocks() -> None:
    assert ClaudeBackend._iter_blocks(_Msg(content=[1, 2])) == [1, 2]
    assert ClaudeBackend._iter_blocks(_Msg(content=None)) == []
    assert ClaudeBackend._iter_blocks(object()) == []


def test_is_tool_use_for_emit_intent() -> None:
    b = _backend()
    assert b._is_tool_use_for_emit_intent(
        ToolUseBlock(name=EMIT_INTENT_TOOL_NAME, input={}),
    ) is True
    # wrong tool name
    assert b._is_tool_use_for_emit_intent(
        ToolUseBlock(name="other_tool", input={}),
    ) is False
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
