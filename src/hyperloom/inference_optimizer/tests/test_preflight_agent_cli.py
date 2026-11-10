# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight WARN-only check on the agent CLI specialists run."""

from __future__ import annotations

import pytest

from hyperloom_common.llm.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX, AGENT_BACKEND_ENV
from hyperloom.inference_optimizer.cli import preflight
from hyperloom.inference_optimizer.cli import codex_sandbox


@pytest.mark.parametrize(
    ("backend", "named"),
    [(AGENT_BACKEND_CLAUDE, "no claude CLI"), (AGENT_BACKEND_CODEX, "no codex CLI")],
)
def test_a_missing_agent_cli_warns(monkeypatch, capsys, backend, named):
    monkeypatch.setattr(preflight, "resolve_agent_backend", lambda: backend)
    monkeypatch.setattr(preflight, "resolve_claude_executable", lambda: "")
    monkeypatch.setattr(preflight, "resolve_codex_executable", lambda: "")
    monkeypatch.setattr(preflight, "_check_codex_sandbox", lambda _backend: None)

    preflight._check_agent_cli()

    out = capsys.readouterr().out
    assert f"WARNING — {named}" in out
    assert "cannot start without it" in out


def test_a_resolvable_agent_cli_passes_quietly(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "resolve_agent_backend", lambda: AGENT_BACKEND_CLAUDE)
    monkeypatch.setattr(preflight, "resolve_claude_executable", lambda: "/usr/bin/claude")
    monkeypatch.setattr(preflight, "_check_codex_sandbox", lambda _backend: None)

    preflight._check_agent_cli()

    assert capsys.readouterr().out == ""


def test_explicit_claude_override_skips_codex_probe_with_only_openai_key(monkeypatch):
    monkeypatch.setenv(AGENT_BACKEND_ENV, AGENT_BACKEND_CLAUDE)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "resolve_claude_executable", lambda: "/usr/bin/claude")
    monkeypatch.setattr(codex_sandbox, "probe_codex_sandbox_capability", lambda: pytest.fail("Codex probe ran"))

    preflight._check_agent_cli()


def test_explicit_codex_override_gates_with_both_provider_keys(monkeypatch):
    monkeypatch.setenv(AGENT_BACKEND_ENV, AGENT_BACKEND_CODEX)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr(preflight, "resolve_codex_executable", lambda: "/usr/bin/codex")
    monkeypatch.setattr(codex_sandbox, "resolve_codex_sandbox_mode", lambda: "workspace-write")
    monkeypatch.setattr(codex_sandbox, "probe_codex_sandbox_capability", lambda: False)
    monkeypatch.setattr(codex_sandbox.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit) as exc:
        preflight._check_agent_cli()

    assert exc.value.code == 2
