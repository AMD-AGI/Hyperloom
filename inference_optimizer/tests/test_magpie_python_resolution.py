# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `_resolve_magpie_python` robustness.

`kernel-agent/scripts/install.sh` resolves `MAGPIE_PYTHON` BEFORE Magpie is
pip-installed, so a freshly-generated `kernel-agent.env.sh` can bake in an
interpreter that cannot `import Magpie` (observed: `/usr/bin/python3`). Trusting
that value blindly made every Magpie benchmark fail with
`ModuleNotFoundError: No module named 'Magpie'` (surfaced as
`subprocess_nonzero` / baseline_failed). The resolver must validate the env
value and fall through to auto-detection.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from inference_optimizer.orchestrator.action_executors import _grid_runner


def test_env_magpie_python_used_when_it_can_import(monkeypatch):
    monkeypatch.setenv("MAGPIE_PYTHON", "/good/python")
    monkeypatch.setattr(
        _grid_runner, "run_with_session_kill",
        lambda *a, **k: type("P", (), {"returncode": 0})(),
    )
    assert _grid_runner._resolve_magpie_python() == "/good/python"


def test_can_import_probe_uses_only_supported_run_kwargs(monkeypatch):
    """Regression guard: the ``import Magpie`` probe must call
    ``run_with_session_kill`` with ONLY kwargs that function accepts.

    The probe previously passed ``capture_output=True`` — a kwarg
    ``run_with_session_kill`` does NOT accept (it always captures via PIPE
    internally). Every probe therefore raised ``TypeError`` (swallowed by the
    broad ``except`` -> ``return False``), so a perfectly valid
    ``$MAGPIE_PYTHON`` was always rejected and the whole stale-interpreter
    self-heal degraded to "always fall through to the hard-coded fallback".

    The other tests in this module hide that bug because they patch
    ``run_with_session_kill`` with ``lambda *a, **k`` (which swallows any
    kwarg). This test patches it with a mock that mirrors the REAL signature,
    so passing an unsupported kwarg raises ``TypeError`` here just like in
    production.
    """
    monkeypatch.setenv("MAGPIE_PYTHON", "/good/python")
    probe_cmds: list[list[str]] = []

    def strict_run(
        cmd, *, env=None, cwd=None, timeout=None, text=True,
        soft_deadline_sec=None,
    ):
        # Mirror run_with_session_kill's real signature exactly — no
        # ``capture_output``. A buggy probe that passes it raises TypeError
        # at call-binding time (before this body runs).
        probe_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", strict_run)

    assert _grid_runner._resolve_magpie_python() == "/good/python"
    assert probe_cmds, "the probe must actually invoke run_with_session_kill"
    assert probe_cmds[0][:2] == ["/good/python", "-c"]


def test_stale_env_magpie_python_ignored_and_autodetected(monkeypatch, caplog):
    """A MAGPIE_PYTHON that cannot import Magpie is ignored; resolver falls
    through to a PATH python3 that can."""
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")

    def fake_run(cmd, *a, **k):
        py = cmd[0]
        # Only the auto-detected PATH python can import Magpie; the stale
        # env value cannot.
        rc = 0 if py == "/opt/venv/bin/python3" else 1
        return type("P", (), {"returncode": rc})()

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", fake_run)
    monkeypatch.setattr(_grid_runner.shutil, "which", lambda _n: "/opt/venv/bin/python3")

    with caplog.at_level(logging.WARNING):
        resolved = _grid_runner._resolve_magpie_python()

    assert resolved == "/opt/venv/bin/python3"
    assert any("MAGPIE_PYTHON" in r.message and "cannot import Magpie" in r.message
               for r in caplog.records)


def test_falls_back_to_opt_venv_when_path_python_cannot_import(monkeypatch, tmp_path):
    """When neither the (stale) env value nor PATH python3 can import Magpie,
    return /opt/venv/bin/python as the unconditional last resort.

    The resolver must NOT fall back to the PATH python3 it just proved cannot
    import Magpie (that would silently benchmark with a Magpie-less
    interpreter). It returns the canonical Magpie venv path regardless of
    whether that file physically exists on this box — so the assertion is
    deterministic in CI (where /opt/venv/bin/python may be absent) and a
    truly broken image fails loudly on an actionable path instead.
    """
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(
        _grid_runner, "run_with_session_kill",
        lambda *a, **k: type("P", (), {"returncode": 1})(),
    )
    monkeypatch.setattr(_grid_runner.shutil, "which", lambda _n: "/usr/bin/python3")
    assert _grid_runner._resolve_magpie_python() == "/opt/venv/bin/python"
