# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Codex sandbox gate takes the backend the orchestration role will run on."""

from __future__ import annotations

import argparse
import asyncio
import os

import pytest

from hyperloom_common.llm.llm_config import (
    AGENT_BACKEND_CLAUDE,
    AGENT_BACKEND_CODEX,
    AGENT_BACKEND_ENV,
    ANTHROPIC_CREDENTIAL_ENV_ORDER,
)
from hyperloom.inference_optimizer import cli

_PROVIDER_ENV = (
    AGENT_BACKEND_ENV,
    "ANTHROPIC_BASE_URL",
    *ANTHROPIC_CREDENTIAL_ENV_ORDER,
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


def _gated_backend(monkeypatch, env: dict[str, str], *, preflight_sets: dict[str, str] | None = None) -> str:
    """Run ``_prepare_run`` under ``env`` and return the backend its sandbox gate received."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def preflight(_args):
        os.environ.update(preflight_sets or {})
        return ("", "")

    gated: list[str] = []
    monkeypatch.setattr(
        cli, "clean_stale_aiter_locks", lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0}
    )
    monkeypatch.setattr(cli, "_preflight", preflight)
    monkeypatch.setattr(cli, "_check_codex_sandbox", gated.append)
    monkeypatch.setattr(cli, "_resolve_models_for_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "_start_experience_kb", lambda: None)
    monkeypatch.setattr(cli, "_preflight_agentx_backend", lambda _args: None)
    # Registered so teardown removes what the stub preflight writes to os.environ.
    for name in preflight_sets or {}:
        monkeypatch.delenv(name, raising=False)

    asyncio.run(cli._prepare_run(argparse.Namespace(resume_from="session")))
    (backend,) = gated
    return backend


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"OPENAI_API_KEY": "k"}, AGENT_BACKEND_CODEX),
        ({"ANTHROPIC_API_KEY": "k"}, AGENT_BACKEND_CLAUDE),
        ({"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, AGENT_BACKEND_CLAUDE),
        ({AGENT_BACKEND_ENV: AGENT_BACKEND_CLAUDE, "OPENAI_API_KEY": "k"}, AGENT_BACKEND_CLAUDE),
        (
            {AGENT_BACKEND_ENV: AGENT_BACKEND_CODEX, "OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"},
            AGENT_BACKEND_CODEX,
        ),
    ],
)
def test_the_gate_receives_the_orchestration_backend(monkeypatch, env, expected):
    assert _gated_backend(monkeypatch, env) == expected


def test_the_gate_follows_the_intent_captured_before_preflight_filled_the_other_side(monkeypatch):
    backend = _gated_backend(
        monkeypatch,
        {"OPENAI_BASE_URL": "https://gateway.example/v1"},
        preflight_sets={"ANTHROPIC_BASE_URL": "https://gateway.example"},
    )

    assert backend == AGENT_BACKEND_CODEX
