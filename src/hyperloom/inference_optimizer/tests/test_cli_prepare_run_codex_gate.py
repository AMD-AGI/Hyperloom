# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Codex sandbox gate takes the backend the orchestration role will run on."""

from __future__ import annotations

import argparse
import asyncio
import os

import pytest

from hyperloom_common.llm.llm_config import (
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


def _prepare_run(
    monkeypatch,
    env: dict[str, str],
    steps: list[str],
    *,
    preflight_sets: dict[str, str] | None = None,
    gate=lambda _backend: None,
) -> None:
    """Run ``_prepare_run`` under ``env``, appending ``gate:<backend>`` and ``models`` to ``steps`` as they run."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    def preflight(_args):
        os.environ.update(preflight_sets or {})
        return ("", "")

    def checked(backend):
        steps.append(f"gate:{backend}")
        gate(backend)

    monkeypatch.setattr(
        cli, "clean_stale_aiter_locks", lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0}
    )
    monkeypatch.setattr(cli, "_preflight", preflight)
    monkeypatch.setattr(cli, "_check_codex_sandbox", checked)
    monkeypatch.setattr(cli, "_resolve_models_for_run", lambda *_a, **_kw: steps.append("models"))
    monkeypatch.setattr(cli, "_start_experience_kb", lambda: None)
    monkeypatch.setattr(cli, "_preflight_agentx_backend", lambda _args: None)
    # Registered so teardown removes what the stub preflight writes to os.environ.
    for name in preflight_sets or {}:
        monkeypatch.delenv(name, raising=False)

    asyncio.run(cli._prepare_run(argparse.Namespace(resume_from="session")))


def test_the_gate_follows_the_intent_captured_before_preflight_filled_the_other_side(monkeypatch):
    steps: list[str] = []

    _prepare_run(
        monkeypatch,
        {"OPENAI_BASE_URL": "https://gateway.example/v1"},
        steps,
        preflight_sets={"ANTHROPIC_BASE_URL": "https://gateway.example"},
    )

    assert steps == [f"gate:{AGENT_BACKEND_CODEX}", "models"]


def test_a_refused_sandbox_stops_the_run_before_model_resolution(monkeypatch):
    steps: list[str] = []

    def refuse(_backend):
        raise SystemExit(2)

    with pytest.raises(SystemExit) as exc:
        _prepare_run(monkeypatch, {"OPENAI_API_KEY": "k"}, steps, gate=refuse)

    assert exc.value.code == 2
    assert steps == [f"gate:{AGENT_BACKEND_CODEX}"]
