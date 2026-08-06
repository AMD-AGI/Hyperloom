# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``backends/ray_runtime.py`` ``safe_runtime_env`` key/URL derivation.

Locks the per-side alias derivation: each side's aliases come from that side's
own credentials, and the GEAK aliases are never derived at all (GEAK runs on the
Anthropic side via GEAK_CLAUDE_MODEL + ANTHROPIC_*).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR / "backends"))
sys.path.insert(0, str(_TOOLS_DIR))

import ray_runtime  # noqa: E402

# Every key alias derived by safe_runtime_env, split by provider protocol.
_OPENAI_KEYS = ("OPENAI_API_KEY", "LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY")
_ANTHROPIC_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_URL_ALIASES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "LLM_API_BASE")
_ALL_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEAK_API_KEY",
    "LLM_API_KEY",
    "AMD_LLM_API_KEY",
    "LLM_GATEWAY_KEY",
    "AMD_API_KEY",
)
_ALL_URL_VARS = ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE")


def _clear(monkeypatch):
    for name in (*_ALL_KEY_VARS, *_ALL_URL_VARS):
        monkeypatch.delenv(name, raising=False)


def test_openai_only_fills_openai_aliases_and_leaves_anthropic_unset(monkeypatch):
    """OpenAI side only: its own aliases are filled, nothing on the Anthropic side."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "ak-gateway")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    for alias in _OPENAI_KEYS:
        assert env[alias] == "ak-gateway", alias
    for alias in ("OPENAI_BASE_URL", "LLM_API_BASE"):
        assert env[alias] == "https://gateway.example/v1", alias
    for alias in (*_ANTHROPIC_KEYS, "ANTHROPIC_BASE_URL"):
        assert alias not in env, alias
    # GEAK is Anthropic-only, so an OpenAI-side value is never handed to it.
    for alias in ("GEAK_API_KEY", "GEAK_BASE_URL"):
        assert alias not in env, alias


def test_explicit_anthropic_key_stays_on_anthropic_side(monkeypatch):
    """An explicit Anthropic key stays on the Anthropic side; OpenAI aliases derive from OPENAI_API_KEY."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    # The explicit Anthropic key is preserved and drives the Anthropic aliases.
    assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "anthropic-key"
    # OpenAI-side aliases derive from OPENAI_API_KEY.
    for alias in ("LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY"):
        assert env[alias] == "openai-key", alias


def test_split_gateway_leaves_geak_aliases_to_the_operator(monkeypatch):
    """Split deploy: the generic OpenAI-protocol aliases derive from the OpenAI
    key, while the GEAK aliases stay unset for either side to claim."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    for alias in ("LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY"):
        assert env[alias] == "openai-test-key", alias
    # Explicit provider keys are preserved as-is.
    assert env["OPENAI_API_KEY"] == "openai-test-key"
    assert env["ANTHROPIC_API_KEY"] == "anthropic-test-key"
    for alias in ("GEAK_API_KEY", "GEAK_BASE_URL"):
        assert alias not in env, alias


def test_explicit_geak_aliases_are_forwarded_verbatim(monkeypatch):
    """An operator-set GEAK alias is forwarded unchanged, never recomputed."""
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com")
    monkeypatch.setenv("GEAK_API_KEY", "operator-geak-key")
    monkeypatch.setenv("GEAK_BASE_URL", "https://geak.example.com")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    assert env["GEAK_API_KEY"] == "operator-geak-key"
    assert env["GEAK_BASE_URL"] == "https://geak.example.com"


def test_anthropic_only_leaves_openai_side_unset(monkeypatch):
    """Anthropic-only entry: the OpenAI-protocol aliases stay unconfigured. GEAK
    itself runs from ANTHROPIC_* + GEAK_CLAUDE_MODEL, not from these aliases."""
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    assert env["ANTHROPIC_API_KEY"] == "anthropic-test-key"
    for alias in ("GEAK_API_KEY", "GEAK_BASE_URL", "LLM_API_KEY", "LLM_API_BASE", "OPENAI_API_KEY"):
        assert alias not in env, alias


def test_no_credentials_leaves_aliases_unset(monkeypatch):
    """No key/URL configured: no alias is invented."""
    _clear(monkeypatch)
    env = ray_runtime.safe_runtime_env()["env_vars"]
    for alias in (*_OPENAI_KEYS, *_ANTHROPIC_KEYS, *_URL_ALIASES):
        assert alias not in env, alias
