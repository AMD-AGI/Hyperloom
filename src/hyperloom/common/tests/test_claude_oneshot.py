# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the single-shot Claude completion transport.

The SDK is stubbed at :func:`hyperloom.common.claude_oneshot._load_sdk`, so
these tests pin the request the module builds and the result it flattens
without spawning the Claude CLI.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.common import claude_oneshot


class _FakeOptions:
    """Stand-in for ``ClaudeAgentOptions`` recording the kwargs it received."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Stub the SDK; return a dict capturing the prompt and options used."""
    captured: dict[str, Any] = {}

    async def _query(*, prompt: str, options: Any):
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages or []:
            yield message

    monkeypatch.setattr(
        claude_oneshot,
        "_load_sdk",
        lambda: SimpleNamespace(query=_query, ClaudeAgentOptions=_FakeOptions),
    )
    return captured


def test_ensure_available_raises_when_the_sdk_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> Any:
        raise RuntimeError("claude_agent_sdk is not installed")

    monkeypatch.setattr(claude_oneshot, "_load_sdk", _missing)

    with pytest.raises(RuntimeError, match="not installed"):
        claude_oneshot.ensure_available()


@pytest.mark.asyncio
async def test_amessages_builds_a_tool_free_single_turn_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_sdk(monkeypatch, messages=[SimpleNamespace(result="verdict")])

    result = await claude_oneshot.ClaudeOneShotClient().amessages(
        model="claude-opus-5",
        system="be terse",
        messages=[{"role": "user", "content": "why did it crash"}],
    )

    assert result.text == "verdict"
    assert captured["prompt"] == "why did it crash"
    kwargs = captured["options"].kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["system_prompt"] == "be terse"
    assert kwargs["max_turns"] == 1
    assert kwargs["allowed_tools"] == []
    assert kwargs["tools"] == []
    assert kwargs["disallowed_tools"] == list(claude_oneshot.DISALLOWED_TOOLS)


@pytest.mark.asyncio
async def test_max_tokens_reaches_the_cli_through_its_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """ClaudeAgentOptions has no max_tokens, so the cap rides on the child env."""
    captured = _install_sdk(monkeypatch, messages=[SimpleNamespace(result="ok")])

    await claude_oneshot.ClaudeOneShotClient().amessages(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=600,
    )

    assert captured["options"].kwargs["env"][claude_oneshot.OUTPUT_TOKEN_CAP_ENV] == "600"


@pytest.mark.asyncio
async def test_no_cap_is_injected_when_max_tokens_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_sdk(monkeypatch, messages=[SimpleNamespace(result="ok")])

    await claude_oneshot.ClaudeOneShotClient().amessages(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert claude_oneshot.OUTPUT_TOKEN_CAP_ENV not in (captured["options"].kwargs.get("env") or {})


@pytest.mark.asyncio
async def test_usage_and_stop_reason_come_from_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(
        monkeypatch,
        messages=[
            SimpleNamespace(content=[{"type": "text", "text": "partial"}], stop_reason=None, usage=None),
            SimpleNamespace(
                result="final",
                stop_reason="max_tokens",
                usage={"input_tokens": 11, "output_tokens": 7},
            ),
        ],
    )

    result = await claude_oneshot.ClaudeOneShotClient().amessages(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.text == "final"
    assert result.stop_reason == "max_tokens"
    assert result.usage == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.asyncio
async def test_streamed_text_is_used_when_no_result_message_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(
        monkeypatch,
        messages=[
            SimpleNamespace(content=[{"type": "text", "text": "one "}]),
            SimpleNamespace(text="two"),
        ],
    )

    result = await claude_oneshot.ClaudeOneShotClient().amessages(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.text == "one two"


@pytest.mark.asyncio
async def test_multiple_turns_are_flattened_into_one_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_sdk(monkeypatch, messages=[SimpleNamespace(result="ok")])

    await claude_oneshot.ClaudeOneShotClient().amessages(
        model="m",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
        ],
    )

    assert captured["prompt"] == "first\n\nsecond"


@pytest.mark.asyncio
async def test_an_empty_message_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(monkeypatch)

    with pytest.raises(ValueError, match="at least one non-empty message"):
        await claude_oneshot.ClaudeOneShotClient().amessages(model="m", messages=[])


@pytest.mark.asyncio
async def test_the_sync_entry_point_refuses_to_run_inside_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(monkeypatch, messages=[SimpleNamespace(result="ok")])

    with pytest.raises(RuntimeError, match="active event loop"):
        claude_oneshot.ClaudeOneShotClient().messages(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_the_sync_entry_point_drives_the_call_outside_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_sdk(monkeypatch, messages=[SimpleNamespace(result="narrative")])

    result = claude_oneshot.ClaudeOneShotClient().messages(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.text == "narrative"


@pytest.mark.asyncio
async def test_a_slow_completion_hits_the_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _query(*, prompt: str, options: Any):
        await asyncio.sleep(5)
        yield SimpleNamespace(result="too late")

    monkeypatch.setattr(
        claude_oneshot,
        "_load_sdk",
        lambda: SimpleNamespace(query=_query, ClaudeAgentOptions=_FakeOptions),
    )

    with pytest.raises(asyncio.TimeoutError):
        await claude_oneshot.ClaudeOneShotClient(timeout_s=0.1).amessages(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
