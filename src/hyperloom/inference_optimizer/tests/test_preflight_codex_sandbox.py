# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight gate on Codex's bubblewrap sandbox."""

from __future__ import annotations

import pytest

from hyperloom_common.llm.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.inference_optimizer.cli import codex_sandbox


def _stub(monkeypatch, *, mode, probe_ok, bwrap_on_path=True):
    monkeypatch.setattr(codex_sandbox, "resolve_codex_sandbox_mode", lambda: mode)
    monkeypatch.setattr(codex_sandbox, "probe_codex_sandbox_capability", lambda: probe_ok)
    monkeypatch.setattr(codex_sandbox.shutil, "which", lambda name: "/usr/bin/bwrap" if bwrap_on_path else None)


def test_a_failed_probe_stops_preflight_with_the_fix(monkeypatch, capsys):
    _stub(monkeypatch, mode="workspace-write", probe_ok=False)

    with pytest.raises(SystemExit) as exc:
        codex_sandbox._check_codex_sandbox(AGENT_BACKEND_CODEX)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "SYS_ADMIN" in err
    assert "apparmor=unconfined" in err
    assert "HYPERLOOM_CODEX_SANDBOX_MODE=bypass" in err


def test_a_missing_bwrap_is_named(monkeypatch, capsys):
    _stub(monkeypatch, mode="workspace-write", probe_ok=False, bwrap_on_path=False)

    with pytest.raises(SystemExit):
        codex_sandbox._check_codex_sandbox(AGENT_BACKEND_CODEX)

    assert "bwrap is not on PATH" in capsys.readouterr().err


def test_a_working_sandbox_passes_quietly(monkeypatch, capsys):
    _stub(monkeypatch, mode="workspace-write", probe_ok=True)

    codex_sandbox._check_codex_sandbox(AGENT_BACKEND_CODEX)

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("backend", "mode"),
    [(AGENT_BACKEND_CLAUDE, "workspace-write"), (AGENT_BACKEND_CODEX, "bypass")],
)
def test_the_probe_is_skipped_when_no_sandbox_is_needed(monkeypatch, backend, mode):
    _stub(monkeypatch, mode=mode, probe_ok=False)

    codex_sandbox._check_codex_sandbox(backend)
