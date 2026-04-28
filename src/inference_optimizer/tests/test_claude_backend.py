"""Tests for ``ClaudeBackend`` — SDK is fully mocked."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.orchestrator.backends.claude import ClaudeBackend
from inference_optimizer.orchestrator.intent_parser import (
    EMIT_INTENT_TOOL_SCHEMA,
    Intent,
    IntentType,
)


# ---------------------------------------------------------------------------
# Tiny "SDK" surface — duck-types claude-agent-sdk shapes the parser cares about.
# ---------------------------------------------------------------------------
@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _AssistantMessage:
    content: list


@dataclass
class _FakeOptions:
    allowed_tools: list = field(default_factory=list)
    max_turns: int = 10
    model: str | None = None
    extra: dict = field(default_factory=dict)


def _make_query_factory(messages: list[Any]):
    """Return a fake ``query`` callable that yields ``messages`` once."""

    async def _generator():
        for m in messages:
            yield m

    def fake_query(*, prompt, options):
        fake_query.last_prompt = prompt  # type: ignore[attr-defined]
        fake_query.last_options = options  # type: ignore[attr-defined]
        return _generator()

    fake_query.last_prompt = None  # type: ignore[attr-defined]
    fake_query.last_options = None  # type: ignore[attr-defined]
    return fake_query


def _make_extract():
    def extract(message):
        chunks: list[str] = []
        for block in getattr(message, "content", []) or []:
            if isinstance(block, _TextBlock):
                chunks.append(block.text)
        return "".join(chunks)

    return extract


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_emit_intent_tool_schema_is_valid_for_sdk():
    assert EMIT_INTENT_TOOL_SCHEMA["name"] == "emit_intent"
    assert "intent_type" in EMIT_INTENT_TOOL_SCHEMA["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_claude_backend_uses_tool_use_blocks(monkeypatch):
    # SDK trajectory contains an emit_intent tool_use block.
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
    fake_query = _make_query_factory([msg])
    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        enable_mcp_emit_intent=False,
    )

    intents = await backend.run("system prompt body", agent_name="executor")
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE
    assert "OUTPUT FORMAT" in fake_query.last_prompt


@pytest.mark.asyncio
async def test_claude_backend_falls_back_to_text_envelope(monkeypatch):
    """No tool_use blocks; assistant returned a fenced JSON envelope."""
    text = (
        "thinking...\n"
        "```json\n"
        + json.dumps(
            {
                "intents": [
                    {
                        "intent_type": "alert",
                        "payload": {"severity": "info", "summary": "ok"},
                    }
                ]
            }
        )
        + "\n```"
    )
    msg = _AssistantMessage(content=[_TextBlock(text=text)])
    fake_query = _make_query_factory([msg])
    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        enable_mcp_emit_intent=False,
    )

    intents = await backend.run("hello", agent_name="executor")
    assert len(intents) == 1
    assert intents[0].type == IntentType.ALERT


@pytest.mark.asyncio
async def test_claude_backend_repairs_on_first_failure(monkeypatch):
    """First reply is junk; repair prompt yields valid envelope."""
    bad = _AssistantMessage(content=[_TextBlock(text="lol no JSON here")])
    good_text = json.dumps(
        {
            "intents": [
                {
                    "intent_type": "alert",
                    "payload": {"severity": "info", "summary": "after repair"},
                }
            ]
        }
    )
    good = _AssistantMessage(content=[_TextBlock(text=good_text)])

    iters = iter([bad, good])

    async def gen_one(msg):
        yield msg

    captured: list[str] = []

    def fake_query(*, prompt, options):
        captured.append(prompt)
        return gen_one(next(iters))

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        repair_attempts=1,
        enable_mcp_emit_intent=False,
    )
    intents = await backend.run("hi", agent_name="executor")
    assert len(intents) == 1
    assert intents[0].payload["summary"] == "after repair"
    assert "did not validate" in captured[1]


@pytest.mark.asyncio
async def test_claude_backend_raises_after_exhausting_retries():
    bad = _AssistantMessage(content=[_TextBlock(text="no json at all")])

    async def gen():
        yield bad

    def fake_query(*, prompt, options):
        return gen()

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        repair_attempts=0,
        enable_mcp_emit_intent=False,
    )
    with pytest.raises(BackendError):
        await backend.run("hi", agent_name="executor")


@pytest.mark.asyncio
async def test_claude_backend_passes_model_and_tools_to_options():
    msg = _AssistantMessage(
        content=[
            _ToolUseBlock(
                name="emit_intent",
                input={
                    "intent_type": "alert",
                    "payload": {"severity": "info", "summary": "x"},
                },
            )
        ]
    )
    fake_query = _make_query_factory([msg])
    backend = ClaudeBackend(
        model="claude-opus-4-7",
        allowed_tools_default=("Read", "Bash"),
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        enable_mcp_emit_intent=False,
    )
    await backend.run("hi", agent_name="executor")
    opts = fake_query.last_options
    assert opts.model == "claude-opus-4-7"
    assert opts.allowed_tools == ["Read", "Bash"]


@pytest.mark.asyncio
async def test_claude_backend_sdk_exception_wrapped():
    def fake_query(*, prompt, options):
        raise RuntimeError("network down")

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        sdk_extract_text=_make_extract(),
        enable_mcp_emit_intent=False,
    )
    with pytest.raises(BackendError) as exc:
        await backend.run("hi", agent_name="executor")
    assert "SDK call failed" in str(exc.value)
