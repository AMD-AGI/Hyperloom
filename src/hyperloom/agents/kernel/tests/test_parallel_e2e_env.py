# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``parallel_e2e_runner.load_env_file`` key/URL derivation.

Locks the single-gateway behaviour (all active aliases derive from SAFE_API_KEY)
and verifies the split-gateway fallback (GEAK / generic LLM aliases take the
OpenAI-side key/URL when SAFE_API_KEY is absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import parallel_e2e_runner as per  # noqa: E402


def _write_env(tmp_path: Path, **vars: str) -> Path:
    p = tmp_path / ".env"
    p.write_text("\n".join(f"{k}={v}" for k, v in vars.items()) + "\n", encoding="utf-8")
    return p


def test_single_gateway_all_aliases_from_safe_key(tmp_path):
    """Single gateway: every key alias is SAFE_API_KEY."""
    env = per.load_env_file(_write_env(tmp_path, SAFE_API_KEY="ak-safe", OPENAI_BASE_URL="https://gw/v1"))
    for alias in ("OPENAI_API_KEY", "GEAK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        assert env[alias] == "ak-safe", alias
    for alias in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "LLM_API_BASE"):
        assert env[alias] == "https://gw/v1", alias
    assert "_".join(("legacy backend", "API", "KEY")) not in env
    assert "_".join(("legacy backend", "BASE", "URL")) not in env


def test_split_gateway_geak_and_llm_aliases_take_openai_key(tmp_path):
    """Split deploy (no SAFE_API_KEY): GEAK/generic aliases derive from OpenAI."""
    env = per.load_env_file(
        _write_env(
            tmp_path,
            OPENAI_API_KEY="sk-openai",
            ANTHROPIC_API_KEY="sk-ant",
            OPENAI_BASE_URL="https://api.openai.com/v1",
            ANTHROPIC_BASE_URL="https://api.anthropic.com",
        )
    )
    for alias in ("GEAK_API_KEY", "LLM_API_KEY", "AMD_LLM_API_KEY"):
        assert env[alias] == "sk-openai", alias
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"
    assert env["GEAK_BASE_URL"] == "https://api.openai.com/v1"
    assert env["LLM_API_BASE"] == "https://api.openai.com/v1"
    assert "_".join(("legacy backend", "API", "KEY")) not in env
    assert "_".join(("legacy backend", "BASE", "URL")) not in env


def test_missing_file_returns_empty(tmp_path):
    assert per.load_env_file(tmp_path / "nope.env") == {}
