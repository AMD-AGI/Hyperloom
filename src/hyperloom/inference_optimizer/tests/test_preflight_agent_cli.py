# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight WARN-only check on the agent CLI specialists run."""

from __future__ import annotations

import pytest

from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.inference_optimizer.cli import preflight


@pytest.mark.parametrize(
    ("backend", "named"),
    [(AGENT_BACKEND_CLAUDE, "no claude CLI"), (AGENT_BACKEND_CODEX, "no codex runtime")],
)
def test_a_missing_agent_cli_warns(monkeypatch, capsys, backend, named):
    monkeypatch.setattr(preflight, "preferred_agent_backend", lambda: backend)
    monkeypatch.setattr(preflight, "resolve_claude_executable", lambda: "")
    monkeypatch.setattr(preflight, "resolve_codex_executable", lambda: "")

    preflight._check_agent_cli()

    out = capsys.readouterr().out
    assert f"WARNING — {named}" in out
    assert "cannot start without it" in out


def test_a_resolvable_agent_cli_passes_quietly(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "preferred_agent_backend", lambda: AGENT_BACKEND_CLAUDE)
    monkeypatch.setattr(preflight, "resolve_claude_executable", lambda: "/usr/bin/claude")

    preflight._check_agent_cli()

    assert capsys.readouterr().out == ""
