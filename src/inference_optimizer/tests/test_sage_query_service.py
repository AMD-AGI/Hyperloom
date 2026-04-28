"""Tests for ``orchestrator.sage_query_service`` — IMPL-CHECKLIST §5.20‒5.25."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inference_optimizer.orchestrator.sage_query_service import (
    SAGE_TIMEOUT_S,
    SageQueryService,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _StubKB:
    def __init__(self, value: str = "- lesson", *, raises: bool = False) -> None:
        self._value = value
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    def recall_for_model(self, model: str, action: str) -> str:
        self.calls.append((model, action))
        if self._raises:
            raise RuntimeError("simulated KB failure")
        return self._value


class _StubBackend:
    def __init__(self, body_md: str = "") -> None:
        self.body_md = body_md
        self.calls: list[str] = []

    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: tuple = (),
        extra: dict | None = None,
    ):
        from inference_optimizer.orchestrator.intent_parser import (
            Intent,
            IntentType,
        )
        self.calls.append(prompt)
        if not self.body_md:
            return []
        return [Intent(IntentType.SEND_MESSAGE, {"body_md": self.body_md})]


class _SlowBackend(_StubBackend):
    async def run(self, prompt: str, **kw):
        await asyncio.sleep(60.0)
        return []


# ---------------------------------------------------------------------------
# recall — KB only
# ---------------------------------------------------------------------------
def test_recall_returns_kb_snippet_when_no_backend():
    kb = _StubKB("- lesson body")
    svc = SageQueryService(codex_backend=None, kb=kb)
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert "lesson body" in out
    assert kb.calls == [("llama-3", "backends")]


def test_recall_returns_empty_when_kb_returns_empty():
    kb = _StubKB("")
    svc = SageQueryService(codex_backend=None, kb=kb)
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert out == ""


def test_recall_silent_on_kb_exception():
    kb = _StubKB(raises=True)
    svc = SageQueryService(codex_backend=None, kb=kb)
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert out == ""


# ---------------------------------------------------------------------------
# recall — backend annotation
# ---------------------------------------------------------------------------
def test_recall_uses_backend_body_when_provided():
    kb = _StubKB("- raw kb extract")
    backend = _StubBackend(body_md="- compressed bullet")
    svc = SageQueryService(codex_backend=backend, kb=kb)
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert out == "- compressed bullet"
    assert backend.calls  # backend was actually invoked


def test_recall_falls_back_to_raw_when_backend_returns_nothing():
    kb = _StubKB("- raw")
    backend = _StubBackend(body_md="")
    svc = SageQueryService(codex_backend=backend, kb=kb)
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert "raw" in out


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def test_recall_cache_short_circuits_second_call():
    kb = _StubKB("- one")
    svc = SageQueryService(codex_backend=None, kb=kb)
    asyncio.run(svc.recall("llama-3", "backends"))
    asyncio.run(svc.recall("llama-3", "backends"))
    assert len(kb.calls) == 1


def test_recall_cache_distinct_keys():
    kb = _StubKB("- snippet")
    svc = SageQueryService(codex_backend=None, kb=kb)
    asyncio.run(svc.recall("a", "x"))
    asyncio.run(svc.recall("a", "y"))
    assert len(kb.calls) == 2


def test_recall_cache_expiry():
    kb = _StubKB("- snippet")
    svc = SageQueryService(
        codex_backend=None, kb=kb, cache_ttl_s=0.01
    )
    asyncio.run(svc.recall("a", "x"))
    import time
    time.sleep(0.05)
    asyncio.run(svc.recall("a", "x"))
    assert len(kb.calls) == 2


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------
def test_recall_timeout_returns_empty():
    kb = _StubKB("- raw")
    backend = _SlowBackend()
    svc = SageQueryService(
        codex_backend=backend, kb=kb, timeout_s=0.05
    )
    out = asyncio.run(svc.recall("llama-3", "backends"))
    assert out == ""


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------
def test_prefetch_warms_cache():
    kb = _StubKB("- snippet")
    svc = SageQueryService(codex_backend=None, kb=kb)
    asyncio.run(svc.prefetch("llama-3", ["a", "b", "c"]))
    # a follow-up recall should hit the cache, not the kb
    asyncio.run(svc.recall("llama-3", "a"))
    assert len([c for c in kb.calls if c == ("llama-3", "a")]) == 1
