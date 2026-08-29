# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The completion budget has to leave room for a reasoning model to answer.

Measured against the gateway with claude-opus-5 on a real discovery prompt: at
2400 tokens every one of five attempts returned an empty completion, because the
model spends the budget thinking before it writes. At 16000 the same prompt
produced 5102 characters of closed JSON with four proposals.
"""

from __future__ import annotations

import sys
import types

import pytest

from kernelforge.fusion.discover import DEFAULT_LLM_MAX_TOKENS, default_llm_fn


def _capture_create_kwargs(monkeypatch) -> dict:
    """Fake openai/httpx that records the kwargs of chat.completions.create."""
    seen: dict = {}

    class _Resp:
        class _C:
            message = type("M", (), {"content": "[]"})()

        choices = [_C()]

    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _Resp()

    class _FakeClient:
        def __init__(self, **_kwargs):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    fake_openai.DefaultHttpxClient = lambda **_k: object()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gw")
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    return seen


def test_default_budget_leaves_room_for_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_FUSION_LLM_MAX_TOKENS", raising=False)
    seen = _capture_create_kwargs(monkeypatch)

    default_llm_fn()("prompt")

    assert seen["max_tokens"] == DEFAULT_LLM_MAX_TOKENS
    assert DEFAULT_LLM_MAX_TOKENS >= 8000, "a reasoning model needs headroom past its own thinking"


def test_budget_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FUSION_LLM_MAX_TOKENS", "4096")
    seen = _capture_create_kwargs(monkeypatch)

    default_llm_fn()("prompt")

    assert seen["max_tokens"] == 4096


def test_explicit_argument_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that pin a budget keep it, env or not."""
    monkeypatch.setenv("FORGE_FUSION_LLM_MAX_TOKENS", "4096")
    seen = _capture_create_kwargs(monkeypatch)

    default_llm_fn(max_tokens=1234)("prompt")

    assert seen["max_tokens"] == 1234
