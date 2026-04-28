"""Tests for ``CodexBackend`` — the OpenAI client is fully faked."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.base import BackendError
from inference_optimizer.orchestrator.backends.codex import (
    CodexBackend,
    _OUTPUT_INSTRUCTIONS,
    _resolve_verify_ssl,
)
from inference_optimizer.orchestrator.intent_parser import IntentType


# ---------------------------------------------------------------------------
# Tiny "OpenAI" surface — duck-types just enough of openai.AsyncOpenAI for
# CodexBackend.run() to be happy.
# ---------------------------------------------------------------------------
@dataclass
class _Message:
    content: str | None
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list[_Choice]
    id: str = "resp-fake"


class _FakeChatCompletions:
    def __init__(self, replies: list[str], *, raise_on_call: int | None = None):
        self._replies = list(replies)
        self.calls: list[dict] = []
        self._raise_on_call = raise_on_call

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if self._raise_on_call is not None and len(self.calls) - 1 == self._raise_on_call:
            raise RuntimeError("fake network glitch")
        if not self._replies:
            raise AssertionError("FakeChatCompletions exhausted; test set up too few replies")
        text = self._replies.pop(0)
        return _Response(choices=[_Choice(message=_Message(content=text))])


@dataclass
class _FakeChatNamespace:
    completions: _FakeChatCompletions


@dataclass
class _FakeClient:
    chat: _FakeChatNamespace
    closed: bool = False

    async def aclose(self) -> None:  # used by CodexBackend.aclose
        self.closed = True


def _fake_client(replies: list[str], *, raise_on_call: int | None = None) -> _FakeClient:
    fcc = _FakeChatCompletions(replies, raise_on_call=raise_on_call)
    return _FakeClient(chat=_FakeChatNamespace(completions=fcc))


def _envelope(intents: list[dict]) -> str:
    body = json.dumps({"intents": intents})
    return f"some preamble noise\n```validated_json_output\n{body}\n```\ntrailing junk"


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_codex_backend_parses_validated_json_block():
    text = _envelope([
        {"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": "hi"}},
    ])
    client = _fake_client([text])
    backend = CodexBackend(model="gpt-5.4", client=client)

    intents = await backend.run("static role prompt", agent_name="critic")

    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE
    assert intents[0].payload["topic"] == "heartbeat"
    # Compose-prompt should append the OUTPUT FORMAT block.
    sent = client.chat.completions.calls[0]
    assert sent["model"] == "gpt-5.4"
    assert sent["messages"][0]["role"] == "user"
    assert "OUTPUT FORMAT" in sent["messages"][0]["content"]
    assert _OUTPUT_INSTRUCTIONS.split("\n", 1)[0] in sent["messages"][0]["content"]


@pytest.mark.asyncio
async def test_codex_backend_supports_multiple_intents_in_envelope():
    text = _envelope([
        {"intent_type": "objection",
         "payload": {"target_msg_id": "m1", "reason": "stale evidence"}},
        {"intent_type": "send_message",
         "payload": {"topic": "observation", "body_md": "watch out"}},
    ])
    backend = CodexBackend(client=_fake_client([text]))

    intents = await backend.run("hi", agent_name="critic")

    assert len(intents) == 2
    assert intents[0].type == IntentType.OBJECTION
    assert intents[1].type == IntentType.SEND_MESSAGE


@pytest.mark.asyncio
async def test_codex_backend_repairs_on_first_failure():
    bad = "lol no JSON whatsoever"
    good = _envelope([
        {"intent_type": "alert",
         "payload": {"severity": "info", "summary": "after repair"}},
    ])
    client = _fake_client([bad, good])
    backend = CodexBackend(client=client, repair_attempts=1)

    intents = await backend.run("hi", agent_name="critic")
    assert len(intents) == 1
    assert intents[0].payload["summary"] == "after repair"

    # Repair prompt should reference the validation failure.
    repair_prompt = client.chat.completions.calls[1]["messages"][0]["content"]
    assert "did not validate" in repair_prompt
    # And must request the same fenced label.
    assert "validated_json_output" in repair_prompt


@pytest.mark.asyncio
async def test_codex_backend_raises_after_exhausting_retries():
    backend = CodexBackend(client=_fake_client(["no json", "still no json"]),
                            repair_attempts=1)
    with pytest.raises(BackendError) as exc:
        await backend.run("hi", agent_name="critic")
    assert "failed to parse intents after 2 attempt" in str(exc.value)


@pytest.mark.asyncio
async def test_codex_backend_zero_repair_raises_on_first_failure():
    backend = CodexBackend(client=_fake_client(["nope"]), repair_attempts=0)
    with pytest.raises(BackendError):
        await backend.run("hi", agent_name="critic")


@pytest.mark.asyncio
async def test_codex_backend_wraps_sdk_exceptions():
    client = _fake_client([_envelope([
        {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
    ])], raise_on_call=0)
    backend = CodexBackend(client=client, repair_attempts=0)
    with pytest.raises(BackendError) as exc:
        await backend.run("hi", agent_name="critic")
    assert "SDK call failed" in str(exc.value)


@pytest.mark.asyncio
async def test_codex_backend_handles_list_content_parts():
    """OpenAI vision-style responses sometimes return content as a list."""
    text_envelope = _envelope([
        {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
    ])

    class _ListContentClient(_FakeClient):
        pass

    client = _fake_client([""])
    # Override the response with list-shaped content.
    async def _create(**kwargs: Any) -> _Response:
        client.chat.completions.calls.append(kwargs)
        msg = _Message(content=None)
        msg.content = [{"type": "text", "text": text_envelope}]  # type: ignore[assignment]
        return _Response(choices=[_Choice(message=msg)])

    client.chat.completions.create = _create  # type: ignore[assignment]
    backend = CodexBackend(client=client)
    intents = await backend.run("hi", agent_name="sage")
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE


@pytest.mark.asyncio
async def test_codex_backend_passes_extra_options_to_sdk():
    text = _envelope([
        {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
    ])
    client = _fake_client([text])
    backend = CodexBackend(
        model="gpt-5.4",
        client=client,
        max_completion_tokens=128,
        temperature=0.2,
        request_timeout_s=42.0,
    )
    await backend.run("hi", agent_name="critic", extra={"role": "critic"})
    call = client.chat.completions.calls[0]
    assert call["max_completion_tokens"] == 128
    assert call["temperature"] == 0.2
    assert call["timeout"] == 42.0
    # `extra` is metadata only — it must NOT be forwarded as kwargs.
    assert "role" not in call


@pytest.mark.asyncio
async def test_codex_backend_records_calls_for_telemetry():
    text = _envelope([
        {"intent_type": "send_message", "payload": {"topic": "heartbeat"}},
    ])
    client = _fake_client([text])
    backend = CodexBackend(client=client)
    await backend.run("hi", agent_name="critic", extra={"task_id": "abc"})
    assert len(backend.calls) == 1
    assert backend.calls[0]["agent"] == "critic"
    assert backend.calls[0]["attempt"] == 0
    assert backend.calls[0]["text_chars"] == len(text)
    assert backend.calls[0]["extra"] == {"task_id": "abc"}


def test_resolve_verify_ssl_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL", "0")
    assert _resolve_verify_ssl(True) is True
    assert _resolve_verify_ssl(False) is False
    assert _resolve_verify_ssl(None) is False


def test_resolve_verify_ssl_default_true(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL", raising=False)
    assert _resolve_verify_ssl(None) is True


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("FALSE", False),
    ("no", False), ("off", False),
    ("1", True), ("true", True), ("yes", True), ("anything-else", True),
])
def test_resolve_verify_ssl_env_truthiness(monkeypatch, value, expected):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_OPENAI_VERIFY_SSL", value)
    assert _resolve_verify_ssl(None) is expected


@pytest.mark.asyncio
async def test_codex_backend_aclose_is_safe_with_no_owned_http_client():
    backend = CodexBackend(client=_fake_client(["x"]))
    # No exception even though _http_client is None.
    await backend.aclose()


@pytest.mark.asyncio
async def test_codex_backend_aclose_closes_owned_http_client():
    backend = CodexBackend(client=_fake_client(["x"]))
    sentinel = _fake_client(["unused"])
    backend._http_client = sentinel  # type: ignore[assignment]
    await backend.aclose()
    assert sentinel.closed is True
