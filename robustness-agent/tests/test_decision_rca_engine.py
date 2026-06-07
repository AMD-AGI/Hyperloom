# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :class:`LlmRcaEngine` and :class:`RcaThrottle`."""

from __future__ import annotations

import httpx
import pytest

from robustness_agent.decision.rca_engine import (
    LlmRcaEngine,
    NoopRcaEngine,
    RcaThrottle,
    RcaThrottleConfig,
)
from robustness_agent.signals import Symptom, SymptomSeverity


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


# ---------------------------------------------------------------------------
# Noop engine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_noop_engine_returns_empty():
    engine = NoopRcaEngine()
    assert await engine.summarize(_sym()) == ""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_engine_calls_chat_server_and_returns_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read().decode()
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "Likely OOM. Reduce batch_size."}}
                ]
            },
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
    assert "claude-opus-4-7" in captured["body"]


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


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# extra_evidence_provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extra_evidence_appears_in_prompt():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    def evidence(_sym):
        return ["log line 1", "log line 2"]

    engine = _engine(handler, extra_evidence_provider=evidence)
    try:
        engine.set_tick(1)
        await engine.summarize(_sym())
    finally:
        await engine.aclose()
    assert "log line 1" in captured["body"]
    assert "log line 2" in captured["body"]
