# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P1-7 CodexBackend tests (mock OpenAI client — no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import CodexBackend
from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.orchestrator.backends.codex import _extract_envelope
from inference_optimizer.protocol.intent import (
    IntentType,
    NoIntentEmitted,
)


# Fakes
@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeResp:
    choices: list[FakeChoice] = field(default_factory=list)


class FakeChatCompletions:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self._replies:
            return FakeResp(choices=[FakeChoice(message=FakeMessage(content=""))])
        text = self._replies.pop(0)
        return FakeResp(choices=[FakeChoice(message=FakeMessage(content=text))])


class FakeChat:
    def __init__(self, completions: FakeChatCompletions):
        self.completions = completions


class FakeOpenAIClient:
    def __init__(self, replies: list[str]):
        self.completions = FakeChatCompletions(replies)
        self.chat = FakeChat(self.completions)


def _make_backend(replies: list[str], model: str = "gpt-5.4") -> CodexBackend:
    client = FakeOpenAIClient(replies)
    return CodexBackend(model=model, client_factory=lambda: client)


# _extract_envelope
def test_extract_envelope_fenced_json():
    text = """Reasoning here.
```json
{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}
```
trailing prose."""
    env = _extract_envelope(text)
    assert env is not None
    assert env["intents"][0]["intent_type"] == "send_message"


def test_extract_envelope_bare_json():
    text = """{"intents": [{"intent_type": "alert", "payload": {"severity": "low", "summary": "x"}}]}"""
    env = _extract_envelope(text)
    assert env is not None
    assert env["intents"][0]["intent_type"] == "alert"


def test_extract_envelope_bare_json_with_trailing_prose():
    text = """{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]} and some prose"""
    env = _extract_envelope(text)
    assert env is not None
    assert env["intents"][0]["payload"]["topic"] == "heartbeat"


def test_extract_envelope_returns_none_for_no_json():
    assert _extract_envelope("just plain text, no json here") is None


def test_extract_envelope_returns_none_for_empty():
    assert _extract_envelope("") is None


def test_extract_envelope_returns_none_for_json_without_intents_key():
    text = '```json\n{"some_other_key": []}\n```'
    assert _extract_envelope(text) is None


# CodexBackend.run
@pytest.mark.asyncio
async def test_run_extracts_review_verdict_intent():
    reply = """```json
{"intents": [{"intent_type": "review_verdict", "payload": {
  "target_proposal_msg_id": "abc123",
  "verdict": "approve",
  "reasoning": "Looks safe; baseline_tput is zero so this is the canonical first step."
}}]}
```"""
    b = _make_backend([reply])
    res = await b.run("review this proposal", system_prompt="You are critic.")
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.REVIEW_VERDICT
    assert res.intents[0].payload["verdict"] == "approve"
    assert res.intents[0].payload["target_proposal_msg_id"] == "abc123"


@pytest.mark.asyncio
async def test_run_handles_multiple_intents_in_one_envelope():
    reply = """```json
{"intents": [
  {"intent_type": "review_verdict", "payload": {"target_proposal_msg_id": "a", "verdict": "approve", "reasoning": "ok"}},
  {"intent_type": "send_message",   "payload": {"topic": "advice", "body_md": "consider profiling next"}}
]}
```"""
    b = _make_backend([reply])
    res = await b.run("p")
    assert len(res.intents) == 2
    assert {i.type for i in res.intents} == {IntentType.REVIEW_VERDICT, IntentType.SEND_MESSAGE}


@pytest.mark.asyncio
async def test_run_no_envelope_raises_no_intent_emitted():
    b = _make_backend(["I cannot decide right now."])
    with pytest.raises(NoIntentEmitted):
        await b.run("p")


@pytest.mark.asyncio
async def test_run_invalid_envelope_raises_no_intent_emitted():
    """Envelope present but invalid (unknown intent_type) → NoIntentEmitted."""
    reply = '```json\n{"intents": [{"intent_type": "objection", "payload": {}}]}\n```'
    b = _make_backend([reply])
    with pytest.raises(NoIntentEmitted, match="invalid"):
        await b.run("p")


@pytest.mark.asyncio
async def test_run_includes_system_and_user_messages():
    reply = '```json\n{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}\n```'
    b = _make_backend([reply])
    await b.run("user prompt", system_prompt="system role here")
    call = b._client.completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "system role here"}
    assert call["messages"][1]["role"] == "user"
    assert "user prompt" in call["messages"][1]["content"]
    assert "OUTPUT FORMAT" in call["messages"][1]["content"]
    assert call["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_run_records_call_metadata():
    reply = '{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}'
    b = _make_backend([reply])
    res = await b.run("p")
    assert res.metadata["model"] == "gpt-5.4"
    assert res.metadata["finish_reason"] == "stop"
    assert b.calls[0]["prompt_chars"] > 0
    assert b.calls[0]["reply_chars"] == len(reply)


# Construction
def test_construct_without_creds_raises_backend_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="not set"):
        CodexBackend(client_factory=None)
