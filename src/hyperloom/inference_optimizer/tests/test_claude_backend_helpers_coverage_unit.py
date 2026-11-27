# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ClaudeBackend helpers (on a stand-in SDK): gateway env pinning in the SDK options, and text collection
from the streamed content blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from hyperloom_common.llm.llm_config import CLAUDE_GATEWAY_SIGNAL_KEYS, LEGACY_DEEPSEEK_ENV_KEYS
from hyperloom.orchestrator.roles import ClaudeBackend


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


@dataclass
class _Msg:
    content: list[Any] = field(default_factory=list)


pytestmark = pytest.mark.usefixtures("claude_sdk")


def test_build_options_pins_gateway_env_and_ignores_global_settings(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_SMALL_FAST_MODEL", raising=False)
    b = ClaudeBackend(model="claude-opus-4-6")

    opts = b._build_options(b._option_kwargs(tools=[], system_prompt=None), [])

    assert opts.kwargs["setting_sources"] == []
    child_env = opts.kwargs["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "https://llm.example.invalid/anthropic"
    assert child_env["ANTHROPIC_CUSTOM_HEADERS"] == "Ocp-Apim-Subscription-Key: sub-key"
    assert child_env["ANTHROPIC_MODEL"] == "claude-opus-4-6"
    assert child_env["ANTHROPIC_SMALL_FAST_MODEL"] == "claude-opus-4-6"


def test_build_options_leaves_settings_sources_unset_without_gateway_env(monkeypatch) -> None:
    # Taken from the product's signal set so a newly recognised gateway variable cannot leak in from the shell.
    for key in (*CLAUDE_GATEWAY_SIGNAL_KEYS, *LEGACY_DEEPSEEK_ENV_KEYS, "LLM_GATEWAY_KEY"):
        monkeypatch.delenv(key, raising=False)
    b = ClaudeBackend(model="claude-opus-4-6")

    opts = b._build_options(b._option_kwargs(tools=[], system_prompt=None), [])

    assert "env" not in opts.kwargs
    assert "setting_sources" not in opts.kwargs


def test_build_options_never_maps_openai_key_onto_the_anthropic_side(monkeypatch) -> None:
    """An OpenAI-only environment produces no Anthropic credentials for the Claude child process."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: openai-key")
    b = ClaudeBackend(model="claude-opus-4-6")

    opts = b._build_options(b._option_kwargs(tools=[], system_prompt=None), [])

    child_env = opts.kwargs["env"]
    assert opts.kwargs["setting_sources"] == []
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "ANTHROPIC_AUTH_TOKEN" not in child_env
    # OPENAI_CUSTOM_HEADERS is not copied to the Anthropic side.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in child_env


async def test_raw_text_collects_every_text_shaped_block(claude_sdk) -> None:
    class _HasText:
        text = "attr-text"

    class _NoText:
        text = 123  # non-string

    async def _q(*, prompt, options):
        yield _Msg(
            content=[TextBlock("hello"), {"type": "text", "text": "d"}, {"type": "image"}, _HasText(), _NoText()]
        )

    claude_sdk.query = _q
    res = await ClaudeBackend().run("p")
    assert res.raw_text == "hellodattr-text"
