# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the specialists/selfcheck module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_run_selfcheck_command_sequence_and_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """run_selfcheck calls pip install, import check, py_compile and returns JSON with ok/steps."""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (worktree / "mymod.py").write_text("x = 1\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_create_venv(venv_dir: Path, *, timeout_sec: int = 600) -> None:
        venv_dir.mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin").mkdir()
        (venv_dir / "bin" / "python").touch()

    def _fake_run(argv: list[str], *, cwd: Path | None = None, timeout_sec: int = 3600):
        calls.append(list(argv))
        if "diff" in argv:
            return 0, "mymod.py\n"
        return 0, "ok"

    import hyperloom.orchestrator.specialists.selfcheck as sc

    monkeypatch.setattr(sc, "_run", _fake_run)

    with patch("hyperloom.agents.framework.isolation.create_venv", _fake_create_venv):
        result = sc.run_selfcheck(worktree=worktree, project_rel="")

    assert isinstance(result, dict)
    assert "ok" in result
    assert "steps" in result
    step_names = [s["step"] for s in result["steps"]]
    assert "pip_install" in step_names
    assert "py_compile" in step_names

    pip_call = next(c for c in calls if "-m" in c and "pip" in c)
    assert "--no-deps" in pip_call
    assert "--no-build-isolation" in pip_call
    assert "-e" in pip_call


def test_run_selfcheck_with_project_rel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """project_rel is passed to pip install -e."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    python_dir = worktree / "python"
    python_dir.mkdir()

    calls: list[list[str]] = []

    def _fake_create_venv(venv_dir: Path, *, timeout_sec: int = 600) -> None:
        venv_dir.mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin").mkdir()
        (venv_dir / "bin" / "python").touch()

    def _fake_run(argv: list[str], *, cwd: Path | None = None, timeout_sec: int = 3600):
        calls.append(list(argv))
        return 0, ""

    import hyperloom.orchestrator.specialists.selfcheck as sc

    monkeypatch.setattr(sc, "_run", _fake_run)

    with patch("hyperloom.agents.framework.isolation.create_venv", _fake_create_venv):
        sc.run_selfcheck(worktree=worktree, project_rel="python")

    pip_call = next((c for c in calls if "-m" in c and "pip" in c), None)
    assert pip_call is not None
    install_path = pip_call[pip_call.index("-e") + 1]
    assert install_path.endswith("python")


def test_run_selfcheck_stops_on_pip_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If pip install fails, selfcheck returns ok=False without running further steps."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[str] = []

    def _fake_create_venv(venv_dir: Path, *, timeout_sec: int = 600) -> None:
        venv_dir.mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin").mkdir()
        (venv_dir / "bin" / "python").touch()

    def _fake_run(argv: list[str], *, cwd: Path | None = None, timeout_sec: int = 3600):
        label = argv[2] if len(argv) > 2 else str(argv)
        calls.append(label)
        if "pip" in argv:
            return 1, "error: build failed"
        return 0, ""

    import hyperloom.orchestrator.specialists.selfcheck as sc

    monkeypatch.setattr(sc, "_run", _fake_run)

    with patch("hyperloom.agents.framework.isolation.create_venv", _fake_create_venv):
        result = sc.run_selfcheck(worktree=worktree)

    assert result["ok"] is False
    step_names = [s["step"] for s in result["steps"]]
    assert "py_compile" not in step_names


def test_run_selfcheck_runs_pytest_when_targets_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """pytest targets are run when provided."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    calls: list[list[str]] = []

    def _fake_create_venv(venv_dir: Path, *, timeout_sec: int = 600) -> None:
        venv_dir.mkdir(parents=True, exist_ok=True)
        (venv_dir / "bin").mkdir()
        (venv_dir / "bin" / "python").touch()

    def _fake_run(argv: list[str], *, cwd: Path | None = None, timeout_sec: int = 3600):
        calls.append(list(argv))
        return 0, ""

    import hyperloom.orchestrator.specialists.selfcheck as sc

    monkeypatch.setattr(sc, "_run", _fake_run)

    with patch("hyperloom.agents.framework.isolation.create_venv", _fake_create_venv):
        result = sc.run_selfcheck(worktree=worktree, pytest_targets=["tests/unit/test_sched.py"])

    step_names = [s["step"] for s in result["steps"]]
    assert "pytest" in step_names
    pytest_call = next(c for c in calls if "pytest" in c)
    assert "tests/unit/test_sched.py" in pytest_call


def test_main_outputs_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """main() prints valid JSON to stdout."""
    worktree = tmp_path / "wt"
    worktree.mkdir()

    def _fake_run_selfcheck(**_kw) -> dict:
        return {"ok": True, "steps": [{"step": "pip_install", "ok": True, "output_tail": ""}]}

    import hyperloom.orchestrator.specialists.selfcheck as sc

    monkeypatch.setattr(sc, "run_selfcheck", _fake_run_selfcheck)
    rc = sc.main(["--worktree", str(worktree)])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["ok"] is True
    assert rc == 0


def test_find_project_dir_rel_sglang_layout(tmp_path: Path):
    """_find_project_dir_rel returns the pyproject.toml-bearing subdirectory."""
    checkout = tmp_path / "sglang"
    checkout.mkdir()
    python_dir = checkout / "python"
    python_dir.mkdir()
    (python_dir / "pyproject.toml").write_text("[project]\nname='sglang'\n", encoding="utf-8")

    from hyperloom.orchestrator.specialists.runner import _find_project_dir_rel
    from hyperloom.inference_optimizer.framework_paths import FrameworkTree

    source = FrameworkTree(tree=python_dir / "sglang", root=checkout, checkout=True)
    rel = _find_project_dir_rel(checkout / "worktree", source)
    assert rel == "python"


def test_find_project_dir_rel_root_has_pyproject(tmp_path: Path):
    """When pyproject.toml is at the root of the tree, returns ''."""
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    from hyperloom.orchestrator.specialists.runner import _find_project_dir_rel
    from hyperloom.inference_optimizer.framework_paths import FrameworkTree

    source = FrameworkTree(tree=checkout, root=checkout, checkout=True)
    rel = _find_project_dir_rel(checkout / "worktree", source)
    assert rel == ""


def test_find_project_dir_rel_none_source(tmp_path: Path):
    """None source returns ''."""
    from hyperloom.orchestrator.specialists.runner import _find_project_dir_rel

    assert _find_project_dir_rel(tmp_path, None) == ""
