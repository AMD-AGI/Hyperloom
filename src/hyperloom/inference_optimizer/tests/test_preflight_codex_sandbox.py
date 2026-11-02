# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight gate on Codex's bubblewrap sandbox."""

from __future__ import annotations

import pytest

from hyperloom_common.llm.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX
from hyperloom.inference_optimizer.cli import preflight


def _stub(monkeypatch, *, backend, mode, probe_ok, bwrap_on_path=True):
    monkeypatch.setattr(preflight, "preferred_agent_backend", lambda: backend)
    monkeypatch.setattr(preflight, "resolve_codex_sandbox_mode", lambda: mode)
    monkeypatch.setattr(preflight, "probe_codex_sandbox_capability", lambda: probe_ok)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/bwrap" if bwrap_on_path else None)


def test_a_failed_probe_stops_preflight_with_the_fix(monkeypatch, capsys):
    _stub(monkeypatch, backend=AGENT_BACKEND_CODEX, mode="workspace-write", probe_ok=False)

    with pytest.raises(SystemExit) as exc:
        preflight._check_codex_sandbox()

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "SYS_ADMIN" in err
    assert "apparmor=unconfined" in err
    assert "HYPERLOOM_CODEX_SANDBOX_MODE=bypass" in err


def test_a_missing_bwrap_is_named(monkeypatch, capsys):
    _stub(monkeypatch, backend=AGENT_BACKEND_CODEX, mode="workspace-write", probe_ok=False, bwrap_on_path=False)

    with pytest.raises(SystemExit):
        preflight._check_codex_sandbox()

    assert "bwrap is not on PATH" in capsys.readouterr().err


def test_a_working_sandbox_passes_quietly(monkeypatch, capsys):
    _stub(monkeypatch, backend=AGENT_BACKEND_CODEX, mode="workspace-write", probe_ok=True)

    preflight._check_codex_sandbox()

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("backend", "mode"),
    [(AGENT_BACKEND_CLAUDE, "workspace-write"), (AGENT_BACKEND_CODEX, "bypass")],
)
def test_the_probe_is_skipped_when_no_sandbox_is_needed(monkeypatch, backend, mode):
    _stub(monkeypatch, backend=backend, mode=mode, probe_ok=False)

    preflight._check_codex_sandbox()
