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

import pytest

from inference_optimizer.orchestrator.action_executors import _grid_runner


def test_env_magpie_python_used_when_it_can_import(monkeypatch):
    monkeypatch.setenv("MAGPIE_PYTHON", "/good/python")
    monkeypatch.setattr(
        _grid_runner, "run_with_session_kill",
        lambda *a, **k: type("P", (), {"returncode": 0})(),
    )
    assert _grid_runner._resolve_magpie_python() == "/good/python"


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
    fall back to /opt/venv/bin/python if present."""
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(
        _grid_runner, "run_with_session_kill",
        lambda *a, **k: type("P", (), {"returncode": 1})(),
    )
    monkeypatch.setattr(_grid_runner.shutil, "which", lambda _n: "/usr/bin/python3")
    # /opt/venv/bin/python exists in this image, so the resolver returns it.
    assert _grid_runner._resolve_magpie_python() == "/opt/venv/bin/python"
