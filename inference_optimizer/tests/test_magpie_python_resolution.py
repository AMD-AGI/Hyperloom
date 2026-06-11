# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `_resolve_magpie_python` robustness: validate a stale `MAGPIE_PYTHON` and fall through to auto-detection."""

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
    """Regression guard: the ``import Magpie`` probe calls ``run_with_session_kill`` with only accepted kwargs (no ``capture_output``)."""
    monkeypatch.setenv("MAGPIE_PYTHON", "/good/python")
    probe_cmds: list[list[str]] = []

    def strict_run(
        cmd, *, env=None, cwd=None, timeout=None, text=True,
        soft_deadline_sec=None,
    ):
        # Mirror run_with_session_kill's real signature exactly — no ``capture_output``.
        probe_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", strict_run)

    assert _grid_runner._resolve_magpie_python() == "/good/python"
    assert probe_cmds, "the probe must actually invoke run_with_session_kill"
    assert probe_cmds[0][:2] == ["/good/python", "-c"]


def test_stale_env_magpie_python_ignored_and_autodetected(monkeypatch, caplog):
    """A MAGPIE_PYTHON that cannot import Magpie is ignored; resolver falls through to a PATH python3 that can."""
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")

    def fake_run(cmd, *a, **k):
        py = cmd[0]
        # Only the auto-detected PATH python can import Magpie.
        rc = 0 if py == "/opt/venv/bin/python3" else 1
        return type("P", (), {"returncode": rc})()

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", fake_run)
    monkeypatch.setattr(_grid_runner.shutil, "which", lambda _n: "/opt/venv/bin/python3")

    with caplog.at_level(logging.WARNING):
        resolved = _grid_runner._resolve_magpie_python()

    assert resolved == "/opt/venv/bin/python3"
    assert any("MAGPIE_PYTHON" in r.message and "cannot import Magpie" in r.message
               for r in caplog.records)


def test_probe_requires_yaml_dependency(monkeypatch):
    """The probe must verify Magpie's runtime deps (yaml), not just that
    ``import Magpie`` works.

    Regression: an interpreter where ``import Magpie`` succeeds (editable
    .pth points at the source tree, independent of installed deps) but
    PyYAML is missing was selected, then Magpie died at startup with
    ``ModuleNotFoundError: No module named 'yaml'`` -> subprocess_nonzero /
    baseline_failed. The probe command must include ``yaml``.
    """
    monkeypatch.setenv("MAGPIE_PYTHON", "/deps-missing/python")
    probe_cmds: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        probe_cmds.append(list(cmd))
        # The deps-missing interpreter fails the (Magpie+yaml) probe; only
        # the canonical venv passes.
        rc = 0 if cmd[0] == "/opt/venv/bin/python3" else 1
        return type("P", (), {"returncode": rc})()

    monkeypatch.setattr(_grid_runner, "run_with_session_kill", fake_run)
    monkeypatch.setattr(
        _grid_runner.shutil, "which", lambda _n: "/opt/venv/bin/python3",
    )

    resolved = _grid_runner._resolve_magpie_python()

    assert resolved == "/opt/venv/bin/python3"
    # Every probe must import yaml (Magpie's top-level runtime dependency).
    assert probe_cmds, "probe must run"
    assert all("yaml" in c[-1] for c in probe_cmds), probe_cmds


def test_falls_back_to_opt_venv_when_path_python_cannot_import(monkeypatch, tmp_path):
    """When neither the env value nor PATH python3 can import Magpie, return /opt/venv/bin/python as the last resort."""
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(
        _grid_runner, "run_with_session_kill",
        lambda *a, **k: type("P", (), {"returncode": 1})(),
    )
    monkeypatch.setattr(_grid_runner.shutil, "which", lambda _n: "/usr/bin/python3")
    assert _grid_runner._resolve_magpie_python() == "/opt/venv/bin/python"
