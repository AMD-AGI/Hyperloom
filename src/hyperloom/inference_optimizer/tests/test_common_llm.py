# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json

import pytest

from hyperloom.common import llm


class _Response:
    def __init__(self, payload: object, *, fail_status: bool = False) -> None:
        self.payload = payload
        self.fail_status = fail_status

    def raise_for_status(self) -> None:
        if self.fail_status:
            raise RuntimeError("bad status")

    def json(self) -> object:
        return self.payload


def test_openai_chat_completions_posts_and_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _post(url: str, *, headers: dict[str, str], content: str, timeout: float) -> _Response:
        captured.update({"url": url, "headers": headers, "body": json.loads(content), "timeout": timeout})
        return _Response({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr("httpx.post", _post)

    result = llm.call_openai_chat_completions(
        base_url="https://openai.example/v1/",
        api_key="sk-test",
        model="m",
        system="sys",
        user="user",
        max_output_tokens=32,
        timeout_sec=3.0,
        temperature=0.0,
    )

    assert result == "hello"
    assert captured["url"] == "https://openai.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["messages"][0]["content"] == "sys"
    assert captured["body"]["temperature"] == 0.0
    assert captured["timeout"] == 3.0


def test_openai_chat_completions_wraps_http_and_shape_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: _Response({}, fail_status=True))
    with pytest.raises(llm.LLMClientError, match="openai chat-completions failed"):
        llm.call_openai_chat_completions(
            base_url="https://openai.example/v1",
            api_key="sk",
            model="m",
            system="s",
            user="u",
            max_output_tokens=1,
        )

    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: _Response({"choices": []}))
    with pytest.raises(llm.LLMClientError, match="unexpected openai response shape"):
        llm.call_openai_chat_completions(
            base_url="https://openai.example/v1",
            api_key="sk",
            model="m",
            system="s",
            user="u",
            max_output_tokens=1,
        )


def test_anthropic_messages_posts_and_joins_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _post(url: str, *, headers: dict[str, str], content: str, timeout: float) -> _Response:
        captured.update({"url": url, "headers": headers, "body": json.loads(content), "timeout": timeout})
        return _Response({"content": [{"type": "text", "text": "a"}, {"type": "image", "text": "skip"}, {"type": "text", "text": "b"}]})

    monkeypatch.setattr("httpx.post", _post)

    result = llm.call_anthropic_messages(
        base_url="https://anthropic.example/",
        api_key="ak-test",
        model="m",
        system="sys",
        user="user",
        max_output_tokens=8,
        timeout_sec=4.0,
        anthropic_version="2024-01-01",
    )

    assert result == "ab"
    assert captured["url"] == "https://anthropic.example/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"
    assert captured["headers"]["anthropic-version"] == "2024-01-01"
    assert captured["body"]["messages"][0]["content"] == "user"
    assert captured["timeout"] == 4.0


def test_anthropic_messages_wraps_http_and_shape_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: _Response({}, fail_status=True))
    with pytest.raises(llm.LLMClientError, match="anthropic messages failed"):
        llm.call_anthropic_messages(
            base_url="https://anthropic.example",
            api_key="ak",
            model="m",
            system="s",
            user="u",
            max_output_tokens=1,
        )

    monkeypatch.setattr("httpx.post", lambda *_args, **_kwargs: _Response({"content": [None]}))
    with pytest.raises(llm.LLMClientError, match="unexpected anthropic response shape"):
        llm.call_anthropic_messages(
            base_url="https://anthropic.example",
            api_key="ak",
            model="m",
            system="s",
            user="u",
            max_output_tokens=1,
        )
