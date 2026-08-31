"""Regression test for task preparation's pristine staging (review #2).

The prepass stages newly-authored task scaffolding into pristine with
``git add -A`` so IterationLoop captures it in the base SHA. But ``-A`` would
also sweep in ``forge_experiments/`` -- the campaign's own run state, candidate
CSVs and ``workspace.lock`` -- which must never enter the pristine commit. The
fix uses the pathspec ``-- . :(exclude)forge_experiments``; this test pins that
behaviour against a real git repo.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kernelforge.loop import task_preparer

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _init_repo(root: Path) -> None:
    task_preparer._git(root, "init", "-q")
    task_preparer._git(root, "config", "user.email", "t@t")
    task_preparer._git(root, "config", "user.name", "t")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    task_preparer._git(root, "add", "-A")
    task_preparer._git(root, "commit", "-q", "-m", "seed")


def _tracked(root: Path) -> set[str]:
    code, out = task_preparer._git(root, "ls-files")
    assert code == 0, out
    return {line.strip() for line in out.splitlines() if line.strip()}


def test_prepass_add_excludes_forge_experiments(tmp_path):
    repo = tmp_path / "ws"
    repo.mkdir()
    _init_repo(repo)

    # Newly authored scaffolding that SHOULD land in pristine.
    (repo / "driver.py").write_text("# driver\n", encoding="utf-8")
    sub = repo / "task" / "kernels"
    sub.mkdir(parents=True)
    (sub / "impl.py").write_text("# impl\n", encoding="utf-8")

    # Campaign run state that must NOT land in pristine.
    fe = repo / "forge_experiments"
    (fe / "candidates").mkdir(parents=True)
    (fe / "campaign_config.json").write_text("{}", encoding="utf-8")
    (fe / "workspace.lock").write_text("pid=1\n", encoding="utf-8")
    (fe / "candidates" / "cand_0.py").write_text("# cand\n", encoding="utf-8")

    # Exactly the command the prepass runs.
    code, out = task_preparer._git(repo, "add", "-A", "--", ".", ":(exclude)forge_experiments")
    assert code == 0, out
    task_preparer._git(repo, "commit", "-q", "-m", "prepass")

    tracked = _tracked(repo)
    assert "driver.py" in tracked
    assert "task/kernels/impl.py" in tracked
    # Nothing under forge_experiments/ may be tracked.
    assert not any(p.startswith("forge_experiments/") for p in tracked), tracked


def test_plain_add_all_would_have_included_forge_experiments(tmp_path):
    """Guard: proves the exclusion is load-bearing -- a plain ``add -A`` DOES
    stage forge_experiments, so the pathspec is what prevents the leak."""
    repo = tmp_path / "ws"
    repo.mkdir()
    _init_repo(repo)

    fe = repo / "forge_experiments"
    fe.mkdir()
    (fe / "workspace.lock").write_text("pid=1\n", encoding="utf-8")

    code, out = task_preparer._git(repo, "add", "-A")
    assert code == 0, out
    task_preparer._git(repo, "commit", "-q", "-m", "plain")

    assert "forge_experiments/workspace.lock" in _tracked(repo)
