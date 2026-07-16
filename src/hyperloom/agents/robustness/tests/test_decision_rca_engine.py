# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :class:`LlmRcaEngine` and :class:`RcaThrottle`."""

from __future__ import annotations

import json

import httpx
import pytest

from hyperloom.agents.robustness.decision.rca_engine import (
    AnthropicRcaEngine,
    LlmRcaEngine,
    NoopRcaEngine,
    RcaThrottle,
    RcaThrottleConfig,
)
from hyperloom.agents.robustness.signals import Symptom, SymptomSeverity


def _sym(
    name: str = "crash_count_high",
    severity: SymptomSeverity = SymptomSeverity.HIGH,
    *,
    summary: str = "crash_count=5",
    subject: dict | None = None,
    evidence: dict | None = None,
) -> Symptom:
    return Symptom(
        name=name,
        severity=severity,
        summary=summary,
        evidence=evidence or {"crash_count": 5},
        subject=subject or {"agent": "session"},
        source="test",
    )


def _engine(handler, *, throttle: RcaThrottle | None = None, **overrides) -> LlmRcaEngine:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        base_url="http://chat.test",
        transport=transport,
        timeout=httpx.Timeout(2.0),
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
    )
    return LlmRcaEngine(
        base_url="http://chat.test",
        api_key="secret",
        client=client,
        throttle=throttle,
        **overrides,
    )


# Noop engine


@pytest.mark.asyncio
async def test_noop_engine_returns_empty():
    engine = NoopRcaEngine()
    assert await engine.summarize(_sym()) == ""


# Happy path


@pytest.mark.asyncio
async def test_llm_engine_calls_chat_server_and_returns_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Likely OOM. Reduce batch_size."}}]},
        )

    engine = _engine(handler)
    try:
        engine.set_tick(1)
        text = await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert "Likely OOM" in text
    assert "/chat/completions" in captured["url"]
    assert captured["auth"] == "Bearer secret"
    assert "crash_count=5" in captured["body"]
    assert "claude-opus-4-8" in captured["body"]


@pytest.mark.asyncio
async def test_llm_engine_uses_max_completion_tokens_for_gpt5_models():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "root cause"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    engine = LlmRcaEngine(
        base_url="https://api.openai.com/v1",
        api_key="openai-token",
        model="gpt-5.5",
        client=client,
        throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=0.0)),
    )
    try:
        engine.set_tick(1)
        text = await engine.summarize(_sym())
    finally:
        await client.aclose()

    assert text == "root cause"
    assert captured["body"]["max_completion_tokens"] == 600
    assert "max_tokens" not in captured["body"]
    assert "temperature" not in captured["body"]


def test_llm_engine_injected_client_uses_openai_bearer_headers():
    client = httpx.AsyncClient(base_url="https://gateway.example/v1")

    try:
        engine = LlmRcaEngine(
            base_url="https://gateway.example/v1",
            api_key="openai-token",
            model="gpt-5.5",
            client=client,
            throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=0.0)),
        )

        assert engine.client is client
        assert client.headers["Authorization"] == "Bearer openai-token"
        assert "x-api-key" not in client.headers
        assert "anthropic-version" not in client.headers
    finally:
        import anyio

        anyio.run(client.aclose)


@pytest.mark.asyncio
async def test_anthropic_rca_engine_calls_messages_endpoint_and_returns_text():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = {k.lower(): v for k, v in request.headers.items()}
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "anthropic root cause"}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com/anthropic")
    engine = AnthropicRcaEngine(
        base_url="https://api.deepseek.com/anthropic",
        api_key="deepseek-token",
        model="deepseek-v4-pro",
        client=client,
        throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=0.0)),
    )
    try:
        engine.set_tick(1)
        text = await engine.summarize(
            _sym(
                name="gateway_auth_outage",
                summary="gateway returned 401",
                evidence={"status_code": 401},
            )
        )
    finally:
        await client.aclose()
    usage = engine.drain_usage()

    assert text == "anthropic root cause"
    assert seen["path"] == "/anthropic/v1/messages"
    assert seen["headers"]["x-api-key"] == "deepseek-token"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["json"]["model"] == "deepseek-v4-pro"
    assert usage is not None
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert usage["calls"] == 1
    assert usage["model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_drain_usage_accumulates_and_resets():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    engine = _engine(
        handler,
        throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=0.0)),
    )
    try:
        engine.set_tick(1)
        await engine.summarize(_sym())
        await engine.summarize(_sym(name="other_symptom"))
    finally:
        await engine.aclose()
    usage = engine.drain_usage()
    assert usage is not None
    assert usage["calls"] == 2
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 8
    assert usage["model"] == "claude-opus-4-8"
    assert usage["latency_ms"] >= 0
    assert engine.drain_usage() is None


@pytest.mark.asyncio
async def test_drain_usage_none_without_calls():
    engine = _engine(lambda r: httpx.Response(200, json={"choices": []}))
    try:
        assert engine.drain_usage() is None
    finally:
        await engine.aclose()


def test_noop_engine_drain_usage_is_none():
    assert NoopRcaEngine().drain_usage() is None


@pytest.mark.asyncio
async def test_drain_usage_counts_call_even_without_usage_block():
    # Provider omits a usage block: the call is still counted so the trace reflects it happened.
    engine = _engine(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    )
    try:
        engine.set_tick(1)
        await engine.summarize(_sym())
    finally:
        await engine.aclose()
    usage = engine.drain_usage()
    assert usage is not None
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 0 and usage["output_tokens"] == 0


@pytest.mark.asyncio
async def test_llm_engine_truncates_to_max_chars():
    long_text = "abcd" * 1000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": long_text}}]},
        )

    engine = _engine(handler, max_chars=50)
    try:
        engine.set_tick(1)
        text = await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert len(text) == 50
    assert text.endswith("...")


@pytest.mark.asyncio
async def test_llm_engine_handles_list_content_parts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello "},
                                {"type": "text", "text": "World"},
                            ]
                        }
                    }
                ]
            },
        )

    engine = _engine(handler)
    try:
        engine.set_tick(1)
        text = await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert text == "Hello World"


# Throttle


@pytest.mark.asyncio
async def test_throttle_skips_low_severity_symptoms():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    engine = _engine(handler, throttle=RcaThrottle(RcaThrottleConfig(severity_min=SymptomSeverity.HIGH)))
    try:
        engine.set_tick(1)
        result = await engine.summarize(_sym(severity=SymptomSeverity.MEDIUM))
    finally:
        await engine.aclose()
    assert result == ""
    assert calls == 0


@pytest.mark.asyncio
async def test_throttle_caps_calls_per_tick():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    engine = _engine(handler, throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=1)))
    try:
        engine.set_tick(1)
        first = await engine.summarize(_sym(name="a", subject={"k": "1"}))
        second = await engine.summarize(_sym(name="b", subject={"k": "2"}))
    finally:
        await engine.aclose()
    assert first
    assert second == ""
    assert calls == 1


@pytest.mark.asyncio
async def test_throttle_resets_per_tick():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    engine = _engine(handler, throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=1, cooldown_seconds=0.0)))
    try:
        engine.set_tick(1)
        await engine.summarize(_sym(name="a", subject={"k": "1"}))
        engine.set_tick(2)
        await engine.summarize(_sym(name="b", subject={"k": "2"}))
    finally:
        await engine.aclose()
    assert calls == 2


@pytest.mark.asyncio
async def test_throttle_enforces_per_key_cooldown():
    """Same dedup_key within cooldown window should not call again."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    throttle = RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=600.0))
    engine = _engine(handler, throttle=throttle)
    try:
        engine.set_tick(1)
        first = await engine.summarize(_sym(name="x", subject={"k": "1"}))
        engine.set_tick(2)
        second = await engine.summarize(_sym(name="x", subject={"k": "1"}))
    finally:
        await engine.aclose()
    assert first
    assert second == ""
    assert calls == 1


# Error paths


@pytest.mark.asyncio
async def test_llm_engine_returns_empty_on_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "boom"})

    engine = _engine(handler)
    try:
        engine.set_tick(1)
        result = await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert result == ""


@pytest.mark.asyncio
async def test_llm_engine_returns_empty_on_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    engine = _engine(handler)
    try:
        engine.set_tick(1)
        result = await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert result == ""


@pytest.mark.asyncio
async def test_llm_engine_skips_when_credentials_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://chat.test")
    engine = LlmRcaEngine(base_url="", api_key="", client=client)
    try:
        engine.set_tick(1)
        text = await engine.summarize(_sym())
    finally:
        await client.aclose()
    assert text == ""
