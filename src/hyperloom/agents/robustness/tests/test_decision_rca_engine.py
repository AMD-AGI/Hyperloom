# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for :class:`LlmRcaEngine` and :class:`RcaThrottle`.

The engines call through ``hyperloom.common.llm_config``. Those entry points
are patched with ``raising=False`` so this suite is independent of whether the
provider contract has landed in the module yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hyperloom.agents.robustness.decision.rca_engine import (
    AnthropicRcaEngine,
    LlmRcaEngine,
    NoopRcaEngine,
    RcaThrottle,
    RcaThrottleConfig,
    load_rca_system_prompt,
)
from hyperloom.agents.robustness.signals import Symptom, SymptomSeverity

_LLM_CONFIG = "hyperloom.common.llm_config"


@dataclass
class _Reply:
    """Stand-in for the ``llm_config`` result objects the engines consume."""

    text: str
    usage: Any = None


class _OpenAIUsage:
    """OpenAI reports usage as SDK object attributes, not mapping keys."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _StubClient:
    """Injected provider client; the patched contract never talks to it."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class _StubAnthropicCompletion:
    """Stands in for ``llm_config.aanthropic_completion`` and records its params.

    The Anthropic engine holds no client of its own now: llm_config owns
    transport selection, so the seam is the entry point rather than an object.
    """

    def __init__(self, *, reply: _Reply | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._reply = reply
        self._error = error

    async def __call__(self, **params: Any) -> _Reply:
        self.calls.append(params)
        if self._error is not None:
            raise self._error
        return self._reply if self._reply is not None else _Reply("ok")


def _install_anthropic_completion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: _Reply | None = None,
    error: Exception | None = None,
) -> _StubAnthropicCompletion:
    """Route the Anthropic engine's single-shot entry point to a recorder."""
    stub = _StubAnthropicCompletion(reply=reply, error=error)
    monkeypatch.setattr(f"{_LLM_CONFIG}.aanthropic_completion", stub, raising=False)
    return stub


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


def _open_throttle() -> RcaThrottle:
    """Throttle that never blocks, so a test can focus on the transport."""
    return RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=0.0))


def _install_chat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply: _Reply | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Patch the async chat-completion entry point; return the recorded calls."""
    calls: list[dict[str, Any]] = []

    async def _fake(client: Any, **params: Any) -> _Reply:
        calls.append({"client": client, **params})
        if error is not None:
            raise error
        return reply if reply is not None else _Reply("ok")

    monkeypatch.setattr(f"{_LLM_CONFIG}.achat_completion", _fake, raising=False)
    return calls


def _install_client_factories(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Patch the OpenAI client factory; return its construction kwargs.

    Only the OpenAI engine builds a client. The Anthropic engine calls
    llm_config's single-shot entry point, which owns its own transport.
    """
    built: dict[str, list[dict[str, Any]]] = {"openai": []}

    def _openai(**kwargs: Any) -> _StubClient:
        built["openai"].append(kwargs)
        return _StubClient()

    monkeypatch.setattr(f"{_LLM_CONFIG}.get_async_openai_client", _openai, raising=False)
    monkeypatch.setattr(f"{_LLM_CONFIG}.build_http_timeout", lambda **kwargs: kwargs, raising=False)
    return built


def _engine(*, throttle: RcaThrottle | None = None, **overrides: Any) -> LlmRcaEngine:
    return LlmRcaEngine(
        base_url="http://chat.test/v1",
        api_key="secret",
        client=_StubClient(),
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
async def test_llm_engine_calls_chat_completion_and_returns_text(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch, reply=_Reply("Likely OOM. Reduce batch_size."))

    engine = _engine()
    engine.set_tick(1)
    text = await engine.summarize(_sym())

    assert "Likely OOM" in text
    assert len(calls) == 1
    assert calls[0]["model"] == "claude-opus-5"
    assert calls[0]["messages"][0]["content"] == load_rca_system_prompt()
    assert "crash_count=5" in calls[0]["messages"][1]["content"]
    assert calls[0]["max_tokens"] == 600
    assert calls[0]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_llm_engine_uses_max_completion_tokens_for_gpt5_models(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch, reply=_Reply("root cause"))

    engine = _engine(model="gpt-5.5", throttle=_open_throttle())
    engine.set_tick(1)
    text = await engine.summarize(_sym())

    assert text == "root cause"
    assert calls[0]["max_completion_tokens"] == 600
    assert "max_tokens" not in calls[0]
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_anthropic_rca_engine_calls_messages_contract_and_returns_text(monkeypatch: pytest.MonkeyPatch):
    stub = _install_anthropic_completion(
        monkeypatch,
        reply=_Reply("anthropic root cause", usage={"input_tokens": 11, "output_tokens": 7}),
    )
    calls = stub.calls

    engine = AnthropicRcaEngine(
        base_url="https://api.deepseek.com/anthropic",
        api_key="deepseek-token",
        model="deepseek-v4-pro",
        throttle=_open_throttle(),
    )
    engine.set_tick(1)
    text = await engine.summarize(
        _sym(
            name="gateway_auth_outage",
            summary="gateway returned 401",
            evidence={"status_code": 401},
        )
    )
    usage = engine.drain_usage()

    assert text == "anthropic root cause"
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["system"] == load_rca_system_prompt()
    assert "gateway returned 401" in calls[0]["messages"][0]["content"]
    assert calls[0]["max_tokens"] == 600
    assert usage is not None
    assert usage["input_tokens"] == 11
    assert usage["output_tokens"] == 7
    assert usage["calls"] == 1
    assert usage["model"] == "deepseek-v4-pro"


# Client construction


@pytest.mark.asyncio
async def test_llm_engine_builds_its_client_from_the_openai_factory(monkeypatch: pytest.MonkeyPatch):
    built = _install_client_factories(monkeypatch)
    calls = _install_chat(monkeypatch, reply=_Reply("ok"))
    anthropic = _install_anthropic_completion(monkeypatch)

    engine = LlmRcaEngine(
        base_url="https://gateway.example/v1",
        api_key="openai-token",
        timeout_s=3.0,
        throttle=_open_throttle(),
    )
    engine.set_tick(1)
    await engine.summarize(_sym(name="a", subject={"k": "1"}))
    await engine.summarize(_sym(name="b", subject={"k": "2"}))

    # Built lazily, exactly once, and never through the Anthropic entry point.
    assert len(built["openai"]) == 1
    assert anthropic.calls == []
    assert built["openai"][0]["env"]["OPENAI_API_KEY"] == "openai-token"
    assert built["openai"][0]["env"]["OPENAI_BASE_URL"] == "https://gateway.example/v1"
    assert built["openai"][0]["timeout"] == {"connect": 3.0, "read": 3.0, "write": 3.0, "pool": 3.0}
    assert calls[0]["client"] is calls[1]["client"] is engine.client


@pytest.mark.asyncio
async def test_anthropic_engine_calls_the_entry_point_without_in_process_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    """llm_config resolves the credential, so only the budgets cross over.

    Passing no base_url/api_key is the normal shape for a subscription-token
    host, and the call must still be issued.
    """
    built = _install_client_factories(monkeypatch)
    stub = _install_anthropic_completion(monkeypatch)

    engine = AnthropicRcaEngine(base_url="", api_key="", timeout_s=7.0, throttle=_open_throttle())
    engine.set_tick(1)
    await engine.summarize(_sym())

    assert len(stub.calls) == 1
    assert built["openai"] == [], "the Anthropic engine must not build an OpenAI client"
    assert stub.calls[0]["timeout_s"] == 7.0
    assert engine.client is None, "the Anthropic engine owns no client to leak"


@pytest.mark.asyncio
async def test_no_client_is_built_before_the_first_call(monkeypatch: pytest.MonkeyPatch):
    built = _install_client_factories(monkeypatch)

    engine = LlmRcaEngine(base_url="https://gateway.example/v1", api_key="openai-token")

    assert engine.client is None
    assert built["openai"] == []
    await engine.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_only_engine_owned_clients(monkeypatch: pytest.MonkeyPatch):
    _install_client_factories(monkeypatch)
    _install_chat(monkeypatch, reply=_Reply("ok"))

    owned = LlmRcaEngine(base_url="https://gateway.example/v1", api_key="token", throttle=_open_throttle())
    owned.set_tick(1)
    await owned.summarize(_sym())
    await owned.aclose()
    assert owned.client.closed is True

    injected = _StubClient()
    borrowed = LlmRcaEngine(base_url="https://gateway.example/v1", api_key="token", client=injected)
    await borrowed.aclose()
    assert injected.closed is False


@pytest.mark.asyncio
async def test_anthropic_engine_aclose_is_a_noop_without_a_client(monkeypatch: pytest.MonkeyPatch):
    """Nothing to close: the entry point owns any transport a call needs."""
    _install_client_factories(monkeypatch)
    _install_anthropic_completion(monkeypatch)

    engine = AnthropicRcaEngine(base_url="", api_key="", throttle=_open_throttle())
    engine.set_tick(1)
    await engine.summarize(_sym())
    await engine.aclose()

    assert engine.client is None


# Usage ledger


@pytest.mark.asyncio
async def test_drain_usage_accumulates_and_resets(monkeypatch: pytest.MonkeyPatch):
    _install_chat(monkeypatch, reply=_Reply("ok", usage=_OpenAIUsage(prompt_tokens=11, completion_tokens=4)))

    engine = _engine(throttle=_open_throttle())
    engine.set_tick(1)
    await engine.summarize(_sym())
    await engine.summarize(_sym(name="other_symptom"))

    usage = engine.drain_usage()
    assert usage is not None
    assert usage["calls"] == 2
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 8
    assert usage["model"] == "claude-opus-5"
    assert usage["latency_ms"] >= 0
    assert engine.drain_usage() is None


@pytest.mark.asyncio
async def test_drain_usage_none_without_calls():
    assert _engine().drain_usage() is None


def test_noop_engine_drain_usage_is_none():
    assert NoopRcaEngine().drain_usage() is None


@pytest.mark.asyncio
async def test_drain_usage_counts_call_even_without_usage_block(monkeypatch: pytest.MonkeyPatch):
    # Provider omits a usage block: the call is still counted so the trace reflects it happened.
    _install_chat(monkeypatch, reply=_Reply("ok"))

    engine = _engine()
    engine.set_tick(1)
    await engine.summarize(_sym())

    usage = engine.drain_usage()
    assert usage is not None
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 0 and usage["output_tokens"] == 0


@pytest.mark.asyncio
async def test_failed_call_is_not_counted_in_the_usage_ledger(monkeypatch: pytest.MonkeyPatch):
    _install_chat(monkeypatch, error=RuntimeError("gateway down"))

    engine = _engine()
    engine.set_tick(1)
    await engine.summarize(_sym())

    assert engine.drain_usage() is None


@pytest.mark.asyncio
async def test_llm_engine_truncates_to_max_chars(monkeypatch: pytest.MonkeyPatch):
    _install_chat(monkeypatch, reply=_Reply("abcd" * 1000))

    engine = _engine(max_chars=50)
    engine.set_tick(1)
    text = await engine.summarize(_sym())

    assert len(text) == 50
    assert text.endswith("...")


# Throttle


@pytest.mark.asyncio
async def test_throttle_skips_low_severity_symptoms(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch)

    engine = _engine(throttle=RcaThrottle(RcaThrottleConfig(severity_min=SymptomSeverity.HIGH)))
    engine.set_tick(1)
    result = await engine.summarize(_sym(severity=SymptomSeverity.MEDIUM))

    assert result == ""
    assert calls == []


@pytest.mark.asyncio
async def test_throttle_caps_calls_per_tick(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch)

    engine = _engine(throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=1)))
    engine.set_tick(1)
    first = await engine.summarize(_sym(name="a", subject={"k": "1"}))
    second = await engine.summarize(_sym(name="b", subject={"k": "2"}))

    assert first
    assert second == ""
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_throttle_resets_per_tick(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch)

    engine = _engine(throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=1, cooldown_seconds=0.0)))
    engine.set_tick(1)
    await engine.summarize(_sym(name="a", subject={"k": "1"}))
    engine.set_tick(2)
    await engine.summarize(_sym(name="b", subject={"k": "2"}))

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_throttle_enforces_per_key_cooldown(monkeypatch: pytest.MonkeyPatch):
    """Same dedup_key within cooldown window should not call again."""
    calls = _install_chat(monkeypatch)

    engine = _engine(throttle=RcaThrottle(RcaThrottleConfig(max_calls_per_tick=10, cooldown_seconds=600.0)))
    engine.set_tick(1)
    first = await engine.summarize(_sym(name="x", subject={"k": "1"}))
    engine.set_tick(2)
    second = await engine.summarize(_sym(name="x", subject={"k": "1"}))

    assert first
    assert second == ""
    assert len(calls) == 1


# Error paths


@pytest.mark.asyncio
async def test_llm_engine_returns_empty_when_the_provider_call_fails(monkeypatch: pytest.MonkeyPatch):
    _install_chat(monkeypatch, error=RuntimeError("503 Service Unavailable"))

    engine = _engine()
    engine.set_tick(1)

    assert await engine.summarize(_sym()) == ""


@pytest.mark.asyncio
async def test_anthropic_engine_returns_empty_when_the_provider_call_fails(monkeypatch: pytest.MonkeyPatch):
    stub = _install_anthropic_completion(monkeypatch, error=TimeoutError("slow"))
    engine = AnthropicRcaEngine(base_url="https://api.anthropic.com", api_key="key")
    engine.set_tick(1)

    assert await engine.summarize(_sym()) == ""
    assert len(stub.calls) == 1, "the call must be attempted, not skipped"


@pytest.mark.asyncio
async def test_llm_engine_skips_when_credentials_missing(monkeypatch: pytest.MonkeyPatch):
    calls = _install_chat(monkeypatch)

    engine = LlmRcaEngine(base_url="", api_key="", client=_StubClient())
    engine.set_tick(1)

    assert await engine.summarize(_sym()) == ""
    assert calls == []


# System prompt asset


def test_load_rca_system_prompt_reads_package_asset():
    prompt = load_rca_system_prompt()
    assert "insufficient evidence" in prompt
    assert "symptom" in prompt.lower()


@pytest.mark.asyncio
async def test_anthropic_engine_sends_asset_as_system_field(monkeypatch: pytest.MonkeyPatch):
    stub = _install_anthropic_completion(monkeypatch)

    engine = AnthropicRcaEngine(
        base_url="https://api.anthropic.com",
        api_key="key",
        throttle=_open_throttle(),
    )
    engine.set_tick(1)
    await engine.summarize(_sym())

    assert stub.calls[0]["system"] == load_rca_system_prompt()
