# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``backends/ray_runtime.py`` ``safe_runtime_env`` key/URL derivation.

Locks the single-gateway behaviour (every alias derives from OPENAI_API_KEY)
and verifies the split-gateway fallback (GEAK takes the OpenAI-side key/URL,
while an explicit Anthropic key stays on the Anthropic side).
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR / "backends"))
sys.path.insert(0, str(_TOOLS_DIR))

import ray_runtime  # noqa: E402

# Every key alias derived by safe_runtime_env, split by provider protocol.
_OPENAI_KEYS = ("OPENAI_API_KEY", "GEAK_API_KEY", "LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY")
_ANTHROPIC_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_URL_ALIASES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GEAK_BASE_URL", "LLM_API_BASE")
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


def test_single_gateway_all_aliases_from_openai_key(monkeypatch):
    """Single gateway: every key alias is OPENAI_API_KEY, every URL alias is OPENAI_BASE_URL."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "ak-gateway")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    for alias in (*_OPENAI_KEYS, *_ANTHROPIC_KEYS):
        assert env[alias] == "ak-gateway", alias
    for alias in _URL_ALIASES:
        assert env[alias] == "https://gateway.example/v1", alias


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
    for alias in ("GEAK_API_KEY", "LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY"):
        assert env[alias] == "openai-key", alias


def test_split_gateway_geak_takes_openai_key(monkeypatch):
    """Split deploy: GEAK derives from the OpenAI-side key."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    for alias in ("GEAK_API_KEY", "LLM_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY"):
        assert env[alias] == "openai-test-key", alias
    # Explicit provider keys are preserved as-is.
    assert env["OPENAI_API_KEY"] == "openai-test-key"
    assert env["ANTHROPIC_API_KEY"] == "anthropic-test-key"
    assert env["GEAK_BASE_URL"] == "https://api.openai.com/v1"


def test_split_anthropic_only_reuses_url_and_key_for_openai_side(monkeypatch):
    """Anthropic-only entry: the OpenAI side reuses the Anthropic URL/key so GEAK still resolves."""
    _clear(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    env = ray_runtime.safe_runtime_env()["env_vars"]

    # The Anthropic key backfills the OpenAI side.
    assert env["GEAK_API_KEY"] == "anthropic-test-key"
    assert env["GEAK_BASE_URL"] == "https://api.anthropic.com"


def test_no_credentials_leaves_aliases_unset(monkeypatch):
    """No key/URL configured: no alias is invented."""
    _clear(monkeypatch)
    env = ray_runtime.safe_runtime_env()["env_vars"]
    for alias in (*_OPENAI_KEYS, *_ANTHROPIC_KEYS, *_URL_ALIASES):
        assert alias not in env, alias
