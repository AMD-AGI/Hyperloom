# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Anthropic Messages transport reads every reply as server-sent events and returns what the whole body says."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from hyperloom_common.llm import llm_config, llm_request_hooks
from hyperloom_common.llm.llm_config import AnthropicCompletionRequest, aanthropic_messages, anthropic_messages
from hyperloom_common.llm.llm_request_hooks import LLMRequestRecord, add_llm_request_observer

#: A reply as the Messages API streams it: a thinking block, a ping, a text block in two deltas, then the stop.
REPLY: tuple[dict[str, Any], ...] = (
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-x",
            "content": [],
            "usage": {"input_tokens": 21, "cache_read_input_tokens": 3, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hello "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "world"}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
)
USAGE = {"input_tokens": 21, "cache_read_input_tokens": 3, "output_tokens": 7}
MESSAGES = [{"role": "user", "content": "hi"}]
OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


def _stream(*events: dict[str, Any]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
    ).encode()


def _with_text(text: str) -> tuple[dict[str, Any], ...]:
    """REPLY with its two text deltas replaced by one carrying ``text``."""

    delta = {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}}
    return (*REPLY[:6], delta, *REPLY[8:])


def _transport(content: Any, seen: list[httpx.Request] | None = None, status: int = 200) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=content, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handle)


def _client(content: Any, seen: list[httpx.Request] | None = None, status: int = 200) -> httpx.Client:
    return httpx.Client(base_url="https://gateway.test", transport=_transport(content, seen, status))


def _async_client(content: Any, seen: list[httpx.Request] | None = None, status: int = 200) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="https://gateway.test", transport=_transport(content, seen, status))


def _pings(seconds: float) -> Iterator[bytes]:
    """A reply whose text never comes: pings for ``seconds``, then the stream ends."""

    yield _stream(REPLY[0], REPLY[4])
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(0.05)
        yield _stream({"type": "ping"})


async def _apings(seconds: float) -> AsyncIterator[bytes]:
    yield _stream(REPLY[0], REPLY[4])
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        await asyncio.sleep(0.05)
        yield _stream({"type": "ping"})


def test_a_reply_reads_as_the_whole_body_would_and_asks_for_the_stream() -> None:
    seen: list[httpx.Request] = []

    with _client(_stream(*REPLY), seen) as client:
        result = anthropic_messages(client, model="claude", messages=MESSAGES, max_tokens=8)

    assert (result.text, result.stop_reason, result.usage) == ("hello world", "end_turn", USAGE)
    [request] = seen
    assert request.url.path == "/v1/messages"
    assert json.loads(request.content) == {"model": "claude", "messages": MESSAGES, "max_tokens": 8, "stream": True}


@pytest.mark.asyncio
async def test_the_async_twin_reads_the_same_reply() -> None:
    seen: list[httpx.Request] = []

    async with _async_client(_stream(*REPLY), seen) as client:
        result = await aanthropic_messages(client, model="claude", system="sys")

    assert (result.text, result.stop_reason, result.usage) == ("hello world", "end_turn", USAGE)
    assert json.loads(seen[0].content) == {"model": "claude", "system": "sys", "stream": True}


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ({"input_tokens": 21, "output_tokens": 1}, {"output_tokens": 7}),
        ({"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 21, "output_tokens": 7}),
    ],
    ids=["counts-split-across-the-stream", "every-count-at-the-end"],
)
def test_the_usage_is_message_starts_with_message_deltas_counts_over_it(
    start: dict[str, int], end: dict[str, int]
) -> None:
    """The Messages API sends the input count up front; the SaFE llm-proxy sends zeros there and every count last."""

    opening = {**REPLY[0], "message": {**REPLY[0]["message"], "usage": start}}
    with _client(_stream(opening, *REPLY[1:-2], {**REPLY[-2], "usage": end}, REPLY[-1])) as client:
        result = anthropic_messages(client, model="claude")

    assert result.usage == {"input_tokens": 21, "output_tokens": 7}


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"], ids=["u2028", "u2029", "u0085"])
def test_text_holding_a_unicode_line_separator_reads_as_the_whole_body_would(separator: str) -> None:
    text = f"line one{separator}line two"

    with _client(_stream(*_with_text(text))) as client:
        result = anthropic_messages(client, model="claude")

    assert result.text == text


def test_a_reply_that_arrives_a_byte_at_a_time_with_crlf_line_ends_reads_the_same() -> None:
    body = _stream(*_with_text("line one\u2028line two")).replace(b"\n", b"\r\n")

    with _client([body[i : i + 1] for i in range(len(body))]) as client:
        result = anthropic_messages(client, model="claude")

    assert (result.text, result.stop_reason, result.usage) == ("line one\u2028line two", "end_turn", USAGE)


@pytest.mark.parametrize(
    ("body", "found"),
    [
        (_stream(*REPLY[:7], OVERLOADED), "stream error: overloaded_error: Overloaded"),
        (_stream(*REPLY[:-1]), "ended before message_stop"),
        (b"event: message_start\ndata: {not json\n\n", "non-JSON event"),
        (b"data: [1, 2]\n\n", "malformed event"),
        (_stream(REPLY[0], REPLY[6], *REPLY[-2:]), "malformed event"),
    ],
    ids=["error-event", "cut-before-stop", "not-json", "not-an-object", "delta-without-its-block"],
)
def test_an_error_event_a_cut_stream_or_an_unreadable_event_raises(body: bytes, found: str) -> None:
    with _client(body) as client, pytest.raises(RuntimeError, match=found):
        anthropic_messages(client, model="claude")


def test_a_failed_status_raises_with_its_body() -> None:
    with (
        _client(b"unauthorized", status=401) as client,
        pytest.raises(RuntimeError, match="status=401 body=unauthorized"),
    ):
        anthropic_messages(client, model="claude")


@pytest.mark.asyncio
async def test_a_failed_status_raises_with_its_body_on_the_async_twin_too() -> None:
    async with _async_client(b"upstream request timeout", status=504) as client:
        with pytest.raises(RuntimeError, match="status=504 body=upstream request timeout"):
            await aanthropic_messages(client, model="claude")


def test_a_transport_error_propagates() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down", request=request)

    with httpx.Client(base_url="https://gateway.test", transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(httpx.ConnectError, match="gateway down"):
            anthropic_messages(client, model="claude")


def test_the_request_is_observed_as_streamed_with_its_first_token_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[LLMRequestRecord] = []
    monkeypatch.setattr(llm_request_hooks, "_OBSERVERS", [])
    add_llm_request_observer(records.append)

    with _client(_stream(*REPLY)) as client:
        anthropic_messages(client, model="claude", component="critic")

    [record] = records
    assert (record.streamed, record.component, record.response_id, record.model) == (
        True,
        "critic",
        "msg_1",
        "claude-x",
    )
    assert record.first_token is not None and record.stop_reason == "end_turn"
    assert record.usage == {**USAGE, "cache_creation_input_tokens": 0}


def test_timeout_s_bounds_the_whole_reply_on_the_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Events that keep coming do not outlast the budget, as the Claude CLI transport's deadline does not."""

    monkeypatch.setattr(llm_config, "get_anthropic_client", lambda **_: _client(_pings(3.0)))
    request = AnthropicCompletionRequest(model="claude", messages=MESSAGES, max_tokens=8, timeout_s=0.2)

    with pytest.raises(TimeoutError):
        llm_config.anthropic_completion(request=request, env={"ANTHROPIC_API_KEY": "k"})


@pytest.mark.asyncio
async def test_timeout_s_bounds_the_whole_reply_on_the_async_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_config, "get_async_anthropic_client", lambda **_: _async_client(_apings(3.0)))
    request = AnthropicCompletionRequest(model="claude", messages=MESSAGES, max_tokens=8, timeout_s=0.2)

    with pytest.raises(asyncio.TimeoutError):
        await llm_config.aanthropic_completion(request=request, env={"ANTHROPIC_API_KEY": "k"})
