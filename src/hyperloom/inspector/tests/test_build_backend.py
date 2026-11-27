# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The in-tree PEP 517 backend that builds the Inspector UI (build_backend/hyperloom_build.py)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[4] / "build_backend" / "hyperloom_build.py"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    if not _BACKEND.is_file():
        pytest.skip("needs a source checkout (build_backend/ is not installed)")
    spec = importlib.util.spec_from_file_location("hyperloom_build_under_test", _BACKEND)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_FRONTEND", tmp_path / "frontend")
    monkeypatch.setattr(mod, "_OUT", tmp_path / "modern")
    for name in ("HYPERLOOM_SKIP_UI_BUILD", "HYPERLOOM_REQUIRE_UI"):
        monkeypatch.delenv(name, raising=False)
    mod.calls = []

    def fake_run(argv, cwd, check, stdout):
        mod.calls.append((Path(argv[0]).name, *argv[1:]))
        if getattr(mod, "fails", False):
            if check:
                raise subprocess.CalledProcessError(1, argv)
            return subprocess.CompletedProcess(argv, 1)
        if argv[1:] == ["run", "build"] and getattr(mod, "emits", True):
            (tmp_path / "modern").mkdir()
            (tmp_path / "modern" / "index.html").write_text("<!doctype html>")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/npm")
    return mod


def _frontend(mod):
    mod._FRONTEND.mkdir()
    (mod._FRONTEND / "package.json").write_text("{}")


def _built(mod):
    mod._OUT.mkdir()
    (mod._OUT / "index.html").write_text("<!doctype html>")


def test_runs_npm_ci_then_build_in_the_frontend(backend, capsys):
    _frontend(backend)
    backend.build_ui()
    assert backend.calls == [("npm", "ci", "--no-audit", "--no-fund"), ("npm", "run", "build")]
    assert "WARNING" not in capsys.readouterr().err


def test_skip_flag_builds_nothing(backend, monkeypatch):
    _frontend(backend)
    monkeypatch.setenv("HYPERLOOM_SKIP_UI_BUILD", "1")
    backend.build_ui()
    assert backend.calls == []


def _no_npm(mod, monkeypatch):
    _frontend(mod)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)


def _npm_fails(mod, monkeypatch):
    _frontend(mod)
    mod.fails = True


#: Each way a build ends with no UI to ship, and the reason the build names for it.
_NO_UI = {
    "no npm": (_no_npm, "npm was not found"),
    "npm fails": (_npm_fails, "npm ci --no-audit --no-fund failed with exit code 1"),
    "sdist without a UI": (lambda mod, monkeypatch: None, "frontend/inspector is not here"),
}


@pytest.mark.parametrize("case", _NO_UI)
def test_without_a_ui_warns_and_ships_without_it(backend, monkeypatch, capsys, case):
    setup, reason = _NO_UI[case]
    setup(backend, monkeypatch)
    backend.build_ui()
    err = capsys.readouterr().err
    assert f"WARNING: Inspector UI: {reason}; the package ships without it" in err


@pytest.mark.parametrize("case", _NO_UI)
def test_without_a_ui_fails_when_the_ui_is_required(backend, monkeypatch, case):
    setup, reason = _NO_UI[case]
    setup(backend, monkeypatch)
    monkeypatch.setenv("HYPERLOOM_REQUIRE_UI", "1")
    with pytest.raises(RuntimeError, match=f"{reason}.*HYPERLOOM_REQUIRE_UI is set"):
        backend.build_ui()


def test_without_npm_ships_an_existing_build(backend, monkeypatch, capsys):
    _frontend(backend)
    _built(backend)
    monkeypatch.setattr(backend.shutil, "which", lambda name: None)
    backend.build_ui()
    assert "existing build" in capsys.readouterr().err
    assert (backend._OUT / "index.html").is_file()


def test_from_an_sdist_uses_the_ui_it_carries(backend, monkeypatch):
    _built(backend)
    monkeypatch.setenv("HYPERLOOM_REQUIRE_UI", "1")
    backend.build_ui()
    assert backend.calls == []


def test_a_build_that_writes_no_page_is_reported(backend):
    _frontend(backend)
    backend.emits = False
    with pytest.raises(RuntimeError, match="wrote no"):
        backend.build_ui()
