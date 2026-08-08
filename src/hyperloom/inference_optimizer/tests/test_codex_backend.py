# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CodexBackend tests (mock OpenAI client — no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from hyperloom.common.llm_config import ResponsesResult
from hyperloom.orchestrator.roles import CodexBackend
from hyperloom.orchestrator.roles import codex as codex_module
from hyperloom.orchestrator.roles.base import BackendError, LLMCallFailed
from hyperloom.orchestrator.roles.codex import _extract_envelope
from hyperloom.inference_optimizer.protocol.intent import (
    IntentType,
    NoIntentEmitted,
)


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


@dataclass
class FakeUsageResp:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class FakeResponsesResult:
    output: list[Any] = field(default_factory=list)
    usage: FakeUsageResp = field(default_factory=FakeUsageResp)
    status: str = "completed"


class FakeResponses:
    """Fake OpenAI Responses API endpoint (``client.responses.create``)."""

    def __init__(self, replies: list[str], citations: list[str] | None = None):
        self._replies = list(replies)
        self._citations = list(citations or [])
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model, input, tools, **kwargs):  # noqa: A002 — mirror SDK kwarg name
        self.calls.append({"model": model, "input": input, "tools": tools, "kwargs": kwargs})
        text = self._replies.pop(0) if self._replies else ""
        annotations = [{"type": "url_citation", "url": u} for u in self._citations]
        message = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": annotations}],
        }
        return FakeResponsesResult(
            output=[{"type": "reasoning", "content": []}, message],
            usage=FakeUsageResp(input_tokens=11, output_tokens=7),
        )


class FakeOpenAIClient:
    def __init__(
        self,
        replies: list[str],
        responses_replies: list[str] | None = None,
        citations: list[str] | None = None,
    ):
        self.completions = FakeChatCompletions(replies)
        self.chat = FakeChat(self.completions)
        self.responses = FakeResponses(responses_replies or [], citations=citations)


def _make_backend(replies: list[str], model: str = "gpt-5.4") -> CodexBackend:
    client = FakeOpenAIClient(replies)
    return CodexBackend(model=model, client_factory=lambda: client)


def _make_ws_backend(
    responses_replies: list[str],
    model: str = "gpt-5.5",
    citations: list[str] | None = None,
) -> CodexBackend:
    client = FakeOpenAIClient([], responses_replies=responses_replies, citations=citations)
    return CodexBackend(
        model=model,
        web_search=True,
        web_search_context_size="medium",
        client_factory=lambda: client,
    )


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


@pytest.mark.asyncio
async def test_api_failure_raises_llm_call_failed():
    """A provider call that blows up must be distinguishable from a local fault.

    Only ``LLMCallFailed`` is counted in the LLM error rate; a plain
    ``BackendError`` here would let the reactor record a deterministic local
    fault as a provider failure.
    """

    class _BoomCompletions(FakeChatCompletions):
        async def create(self, **kwargs):
            raise RuntimeError("gateway 400")

    client = FakeOpenAIClient([])
    client.completions = _BoomCompletions([])
    client.chat = FakeChat(client.completions)
    backend = CodexBackend(model="gpt-5.4", client_factory=lambda: client)

    with pytest.raises(LLMCallFailed, match="Codex API call failed"):
        await backend.run("p")


@pytest.mark.asyncio
async def test_responses_api_failure_raises_llm_call_failed():
    class _BoomResponses(FakeResponses):
        async def create(self, **kwargs):
            raise RuntimeError("gateway 500")

    client = FakeOpenAIClient([], responses_replies=[])
    client.responses = _BoomResponses([])
    backend = CodexBackend(model="gpt-5.5", web_search=True, client_factory=lambda: client)

    with pytest.raises(LLMCallFailed, match="Codex Responses API call failed"):
        await backend.run("p")


def test_construct_without_creds_raises_backend_error(monkeypatch):
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(BackendError, match="not set"):
        CodexBackend(client_factory=None)


def _construct_real_codex_capturing_kwargs(monkeypatch):
    """Build a CodexBackend through the real SDK path, capturing AsyncOpenAI kwargs."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    CodexBackend(client_factory=None)
    return captured


def test_codex_client_comes_from_the_shared_llm_gateway(monkeypatch):
    """Codex owns no credentials: it forwards its env-var contract to llm_config."""
    captured: dict[str, Any] = {}
    sentinel = object()
    monkeypatch.setattr(
        codex_module,
        "get_async_openai_client",
        lambda **kwargs: captured.update(kwargs) or sentinel,
    )
    backend = CodexBackend(api_key_env="CODEX_KEY", base_url_env="CODEX_URL", client_factory=None)
    assert backend._client is sentinel
    assert captured == {"api_key_env": "CODEX_KEY", "base_url_env": "CODEX_URL"}


def test_codex_missing_openai_sdk_raises_backend_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "tok")
    with pytest.raises(BackendError, match="openai SDK not installed"):
        CodexBackend(client_factory=None)


def test_codex_prefers_explicit_openai_key_over_safe_filled_anthropic(monkeypatch):
    """Plan B: a user-set OPENAI_API_KEY wins over SAFE-filled ANTHROPIC_AUTH_TOKEN."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "safe-key-from-safe")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-user-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    captured = _construct_real_codex_capturing_kwargs(monkeypatch)
    assert captured["api_key"] == "openai-user-key"
    assert captured["base_url"] == "https://api.openai.com/v1"


def test_codex_authenticates_with_the_anthropic_gateway_token(monkeypatch):
    """Anthropic-only deployment: the gateway token authenticates the OpenAI protocol."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    captured = _construct_real_codex_capturing_kwargs(monkeypatch)
    assert captured["api_key"] == "anthropic-token"
    assert captured["base_url"] == "https://gateway.example/v1"


def test_codex_derives_its_base_url_in_an_anthropic_only_deployment(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-api.amd.com/Anthropic")
    captured = _construct_real_codex_capturing_kwargs(monkeypatch)
    assert captured["base_url"] == "https://llm-api.amd.com/Unified/v1"


# ---------------------------------------------------------------------------
# Web search (Responses API) path


@pytest.mark.asyncio
async def test_web_search_routes_through_the_shared_responses_entry_point(monkeypatch):
    """The Responses call is llm_config's; codex only supplies params and maps the result."""
    captured: dict[str, Any] = {}
    result = ResponsesResult(
        text='{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}',
        citations=["https://cited.example"],
        status="completed",
        input_tokens=13,
        output_tokens=5,
    )

    async def _fake_aresponse(client, **params):
        captured["client"] = client
        captured["params"] = params
        return result

    monkeypatch.setattr(codex_module, "aresponse", _fake_aresponse)
    backend = _make_ws_backend([])
    res = await backend.run("p", system_prompt="sys")

    assert captured["client"] is backend._client
    assert captured["params"]["tools"][0]["type"] == "web_search"
    assert captured["params"]["instructions"] == "sys"
    # No bare SDK call was made: the fake stood in for the whole endpoint.
    assert backend._client.responses.calls == []
    assert res.metadata["finish_reason"] == "completed"
    assert res.metadata["input_tokens"] == 13
    assert res.metadata["output_tokens"] == 5
    assert res.metadata["web_search_citations"] == ["https://cited.example"]


@pytest.mark.asyncio
async def test_web_search_omits_citations_metadata_when_there_are_none():
    reply = '```json\n{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}\n```'
    b = _make_ws_backend([reply])
    res = await b.run("p")
    assert "web_search_citations" not in res.metadata


@pytest.mark.asyncio
async def test_web_search_wires_reasoning_effort(monkeypatch):
    monkeypatch.setenv("HYPERLOOM_REASONING_EFFORT", "high")
    reply = '```json\n{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}\n```'
    b = _make_ws_backend([reply])
    await b.run("p")
    call = b._client.responses.calls[0]
    assert call["kwargs"].get("reasoning") == {"effort": "high"}


@pytest.mark.asyncio
async def test_web_search_uses_responses_api_and_parses_envelope():
    reply = '```json\n{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}\n```'
    b = _make_ws_backend([reply], citations=["https://github.com/vllm-project/vllm/releases"])
    res = await b.run("find the latest vLLM version", system_prompt="you are a researcher")

    # Went through the Responses API, NOT chat.completions.
    assert len(b._client.responses.calls) == 1
    assert len(b._client.completions.calls) == 0

    # web_search tool wired with the configured context size; system prompt as instructions.
    call = b._client.responses.calls[0]
    assert call["tools"][0]["type"] == "web_search"
    assert call["tools"][0]["search_context_size"] == "medium"
    assert call["kwargs"]["instructions"] == "you are a researcher"
    assert "find the latest vLLM version" in call["input"]
    assert "OUTPUT FORMAT" in call["input"]

    # Envelope parsed from output_text; usage mapped; citations recorded.
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.metadata["input_tokens"] == 11
    assert res.metadata["output_tokens"] == 7
    assert res.metadata["web_search_citations"] == ["https://github.com/vllm-project/vllm/releases"]


@pytest.mark.asyncio
async def test_web_search_disabled_uses_chat_completions():
    """Default (web_search off) still uses chat.completions — no Responses call."""
    reply = '{"intents": [{"intent_type": "send_message", "payload": {"topic": "heartbeat"}}]}'
    b = _make_backend([reply])
    assert b.web_search is False
    await b.run("p")
    assert len(b._client.completions.calls) == 1
    assert len(b._client.responses.calls) == 0


@pytest.mark.asyncio
async def test_web_search_no_envelope_raises_no_intent_emitted():
    b = _make_ws_backend(["I searched but have nothing structured to say."])
    with pytest.raises(NoIntentEmitted):
        await b.run("p")


# --- API error translation to BackendError ---------------------------------


class _RaisingEndpoint:
    """Fake ``create`` that raises the given exception."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def create(self, **kwargs):
        raise self._exc


class _RaisingChatClient:
    def __init__(self, exc: BaseException):
        self.completions = _RaisingEndpoint(exc)
        self.chat = FakeChat(self.completions)  # type: ignore[arg-type]
        self.responses = _RaisingEndpoint(exc)


def _chat_backend_raising(exc: BaseException) -> CodexBackend:
    return CodexBackend(model="gpt-5.4", client_factory=lambda: _RaisingChatClient(exc))


def _ws_backend_raising(exc: BaseException) -> CodexBackend:
    return CodexBackend(model="gpt-5.5", web_search=True, client_factory=lambda: _RaisingChatClient(exc))


@pytest.mark.asyncio
async def test_chat_timeout_translates_to_backend_error():
    import asyncio

    b = _chat_backend_raising(asyncio.TimeoutError())
    with pytest.raises(BackendError, match="timed out"):
        await b.run("p")


@pytest.mark.asyncio
async def test_chat_generic_error_translates_to_backend_error():
    b = _chat_backend_raising(RuntimeError("boom"))
    with pytest.raises(BackendError, match="Codex API call failed"):
        await b.run("p")


@pytest.mark.asyncio
async def test_responses_timeout_translates_to_backend_error():
    import asyncio

    b = _ws_backend_raising(asyncio.TimeoutError())
    with pytest.raises(BackendError, match="Responses API call timed out"):
        await b.run("p")


@pytest.mark.asyncio
async def test_responses_generic_error_translates_to_backend_error():
    b = _ws_backend_raising(RuntimeError("boom"))
    with pytest.raises(BackendError, match="Responses API call failed"):
        await b.run("p")
