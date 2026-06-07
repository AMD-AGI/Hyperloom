# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline-v2 N6: ClaudeBackend cache hit metric extraction.

Pins the contract N7 (verify + audit scripts) builds on top of:
ClaudeBackend exposes `cache_creation_input_tokens` /
`cache_read_input_tokens` / `input_tokens` / `output_tokens` from
the underlying SDK's `ResultMessage.usage` on every `run()`
invocation, in two places:

* `backend.calls[-1]` (existing per-call audit log)
* `BackendTurnResult.metadata` (returned to Coordinator)

The metric is extracted exclusively from `ResultMessage.usage` —
Anthropic Messages API response shape that Claude Code propagates
on its terminal SDK messages. See task_manager.py in Primus-Claw/OOB
(lines 152-153) for the same pattern in production.

We do NOT inject `cache_control` ourselves (claude-agent-sdk 0.2.82
doesn't expose that param), and we don't reorder prompt sections —
automatic caching already operates on the `system_prompt + tools +
messages` prefix per Anthropic docs. N6's contribution is purely
**measuring** how effective that automatic caching is.
"""

from __future__ import annotations


import pytest

from inference_optimizer.protocol.intent import IntentType
from inference_optimizer.orchestrator.backends.claude import ClaudeBackend


# ---------------------------------------------------------------------------
# Fake SDK plumbing
# ---------------------------------------------------------------------------
class _FakeBlock:
    """Mimics claude_agent_sdk's ToolUseBlock with name + input."""

    def __init__(self, name: str, input_dict: dict):
        self.__class__.__name__ = "ToolUseBlock"
        self.name = name
        self.input = input_dict


# noqa - class name mimic by inheriting
class _FakeToolUseBlock(_FakeBlock):
    pass


_FakeToolUseBlock.__name__ = "ToolUseBlock"


class _FakeMessage:
    """SDK message with content[] (blocks) + optional usage dict."""

    def __init__(self, *, content: list = None, usage: dict | None = None,
                  result: str | None = None):
        self.content = content or []
        if usage is not None:
            self.usage = usage
        if result is not None:
            self.result = result


def _emit_intent_block() -> _FakeToolUseBlock:
    return _FakeToolUseBlock(
        name="mcp__inference_optimizer__emit_intent",
        input_dict={
            "intent_type": IntentType.SEND_MESSAGE.value,
            "payload": {"topic": "heartbeat", "body_md": "ok"},
        },
    )


def _make_backend(messages: list[_FakeMessage]) -> ClaudeBackend:
    """Construct ClaudeBackend with a fake SDK that plays back the given
    message sequence."""

    async def fake_query(*, prompt, options):
        for m in messages:
            yield m

    class _FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    backend = ClaudeBackend(
        sdk_query_factory=fake_query,
        sdk_options_cls=_FakeOptions,
        # Disable MCP wiring so we don't trigger real Claude Code setup
        enable_mcp_emit_intent=False,
    )
    return backend


# ---------------------------------------------------------------------------
# Happy path — usage propagates to calls + metadata
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cache_metrics_extracted_from_result_message_usage():
    """Happy path: ResultMessage carries `usage` dict including the 4
    Anthropic token counters; they land on both `backend.calls[-1]`
    and `BackendTurnResult.metadata`."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 12000,
            },
            result="ok",
        ),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")

    # backend.calls log
    call = backend.calls[-1]
    assert call["cache_creation_input_tokens"] == 5000
    assert call["cache_read_input_tokens"] == 12000
    assert call["input_tokens"] == 100
    assert call["output_tokens"] == 50
    # BackendTurnResult metadata
    assert result.metadata["cache_creation_input_tokens"] == 5000
    assert result.metadata["cache_read_input_tokens"] == 12000
    assert result.metadata["input_tokens"] == 100
    assert result.metadata["output_tokens"] == 50


@pytest.mark.asyncio
async def test_cache_read_only_means_cache_hit():
    """When `cache_read > 0` and `cache_creation == 0`, the prompt
    prefix was fully served from cache — this is the success
    signal §10.3 looks for."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(
            usage={
                "input_tokens": 50,
                "output_tokens": 30,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 15000,
            },
        ),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    assert result.metadata["cache_read_input_tokens"] == 15000
    assert result.metadata["cache_creation_input_tokens"] == 0


@pytest.mark.asyncio
async def test_cache_creation_only_means_first_miss():
    """First request in a session: cache_creation > 0, cache_read = 0.
    The next request with the same prefix should hit."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 18000,
                "cache_read_input_tokens": 0,
            },
        ),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    assert result.metadata["cache_creation_input_tokens"] == 18000
    assert result.metadata["cache_read_input_tokens"] == 0


# ---------------------------------------------------------------------------
# Degraded paths — usage missing / non-dict / SDK silent on cache
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_usage_field_zero_metrics():
    """SDK silent on usage (older Claude Code versions / unusual model
    fallback paths) → metrics default to 0, no crash."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        # No usage attribute at all
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    assert result.metadata["cache_creation_input_tokens"] == 0
    assert result.metadata["cache_read_input_tokens"] == 0
    assert result.metadata["input_tokens"] == 0


@pytest.mark.asyncio
async def test_usage_partial_only_some_keys_present():
    """SDK reports usage but only `input_tokens` (legacy gateway path).
    Missing cache keys default to 0."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(usage={"input_tokens": 200}),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    assert result.metadata["input_tokens"] == 200
    assert result.metadata["cache_creation_input_tokens"] == 0
    assert result.metadata["cache_read_input_tokens"] == 0
    assert result.metadata["output_tokens"] == 0


@pytest.mark.asyncio
async def test_usage_non_dict_treated_as_missing():
    """`usage` field present but is e.g. a string (SDK schema drift) →
    metrics default to 0, no crash."""
    msg = _FakeMessage(content=[_emit_intent_block()])
    msg.usage = "garbage"  # type: ignore[assignment]
    backend = _make_backend([msg])
    result = await backend.run(prompt="hello")
    assert result.metadata["cache_creation_input_tokens"] == 0


@pytest.mark.asyncio
async def test_usage_non_numeric_value_coerced_to_zero():
    """`cache_creation_input_tokens` present but non-numeric (e.g.
    None / "n/a") → coerce to 0 (defensive against gateway proxies
    that filter values)."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(usage={
            "input_tokens": "n/a",
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": "garbage",
            "output_tokens": 50,
        }),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    assert result.metadata["input_tokens"] == 0
    assert result.metadata["cache_creation_input_tokens"] == 0
    assert result.metadata["cache_read_input_tokens"] == 0
    assert result.metadata["output_tokens"] == 50


# ---------------------------------------------------------------------------
# Multiple ResultMessages — last usage wins
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_last_usage_wins_when_multiple_result_messages():
    """In multi-turn streaming, only the terminal ResultMessage carries
    the cumulative session usage. Earlier ResultMessages (turn-level)
    must NOT clobber the final one."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(usage={"input_tokens": 50, "cache_read_input_tokens": 100}),
        _FakeMessage(usage={
            "input_tokens": 200,
            "cache_creation_input_tokens": 5000,
            "cache_read_input_tokens": 15000,
            "output_tokens": 100,
        }),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    # Last usage wins (cumulative)
    assert result.metadata["input_tokens"] == 200
    assert result.metadata["cache_read_input_tokens"] == 15000
    assert result.metadata["cache_creation_input_tokens"] == 5000
    assert result.metadata["output_tokens"] == 100


# ---------------------------------------------------------------------------
# safe_int helper
# ---------------------------------------------------------------------------
def test_safe_int_coerces_ints_strings_and_falsy():
    assert ClaudeBackend._safe_int(42) == 42
    assert ClaudeBackend._safe_int("100") == 100
    assert ClaudeBackend._safe_int(None) == 0
    assert ClaudeBackend._safe_int("") == 0
    assert ClaudeBackend._safe_int("garbage") == 0
    assert ClaudeBackend._safe_int(0) == 0
    assert ClaudeBackend._safe_int(False) == 0  # bool → int(False) == 0


# ---------------------------------------------------------------------------
# Pre-existing fields preserved
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_existing_metadata_fields_still_present():
    """N6 adds cache fields without removing tool_blocks / model."""
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(usage={"input_tokens": 10}),
    ]
    backend = _make_backend(messages)
    result = await backend.run(prompt="hello")
    # Old fields
    assert "tool_blocks" in result.metadata
    assert "model" in result.metadata
    # New fields
    assert "cache_creation_input_tokens" in result.metadata
    assert "cache_read_input_tokens" in result.metadata


@pytest.mark.asyncio
async def test_existing_backend_calls_fields_preserved():
    messages = [
        _FakeMessage(content=[_emit_intent_block()]),
        _FakeMessage(usage={"cache_read_input_tokens": 5000}),
    ]
    backend = _make_backend(messages)
    await backend.run(prompt="hello")
    call = backend.calls[-1]
    # Pre-existing keys
    assert "prompt_chars" in call
    assert "tool_blocks" in call
    assert "intents" in call
    assert "max_turns" in call
    # N6 keys
    assert call["cache_read_input_tokens"] == 5000
