# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A checkout owned by another uid must stay operable.

The documented container recipe bind-mounts the repo (``-v $REPO_ROOT:$REPO_ROOT``),
so the trees the optimizer patches are routinely owned by a different uid than
the process. git then refuses every operation on them, including reads.

``GIT_TEST_ASSUME_DIFFERENT_OWNER`` is git's own hook for this path, so the tests
need no root and no foreign-owned directory.
"""

from __future__ import annotations

import subprocess

import pytest

from hyperloom.common import git_safety
from hyperloom.orchestrator.actions.executors import _git


def _init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t.local",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def foreign_repo(tmp_path, monkeypatch):
    """A real repo, turned foreign only after setup so init/commit can run."""
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    return repo


def test_dubious_ownership_repo_is_still_readable(foreign_repo):
    ok, out, err = _git._run_git(["-C", str(foreign_repo), "rev-parse", "--short", "HEAD"])

    assert ok, err
    assert out.strip()


def test_dubious_ownership_repo_is_still_writable(foreign_repo):
    cp = _git._run_git_cp(
        [
            "-c",
            "user.email=f@h.local",
            "-c",
            "user.name=hl",
            "-C",
            str(foreign_repo),
            "commit",
            "--allow-empty",
            "-m",
            "kept",
        ]
    )

    assert cp is not None and cp.returncode == 0, cp and cp.stderr


def test_a_subdirectory_target_marks_the_repository_root(foreign_repo):
    """git resolves ownership against the root, so naming the subdir is ignored."""
    sub = foreign_repo / "pkg" / "inner"
    sub.mkdir(parents=True)

    ok, _out, err = _git._run_git(["-C", str(sub), "rev-parse", "--short", "HEAD"])

    assert ok, err


def test_linked_worktree_git_file_is_recognised(foreign_repo, tmp_path):
    """In a linked worktree ``.git`` is a file, not a directory."""
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-c", f"safe.directory={foreign_repo}", "-C", str(foreign_repo), "worktree", "add", "-q", str(linked)],
        check=True,
        capture_output=True,
    )
    assert (linked / ".git").is_file()

    ok, _out, err = _git._run_git(["-C", str(linked), "rev-parse", "--short", "HEAD"])

    assert ok, err


def test_a_path_outside_any_repo_is_left_alone(tmp_path, monkeypatch):
    """No exception is invented for a target that is not in a checkout."""
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    plain = tmp_path / "plain"
    plain.mkdir()

    assert git_safety.repo_root(str(plain)) is None
    assert git_safety.safe_directory_args(["-C", str(plain), "status"]) == ["-C", str(plain), "status"]


def test_the_exception_precedes_the_subcommand(foreign_repo):
    """git ignores -c placed after the subcommand."""
    args = git_safety.safe_directory_args(["-C", str(foreign_repo), "rev-parse", "HEAD"])

    assert args[0] == "-c"
    assert args[1] == f"safe.directory={foreign_repo}"
    assert args.index("rev-parse") > 1


def test_caller_supplied_c_options_survive(foreign_repo):
    args = git_safety.safe_directory_args(["-c", "user.name=hl", "-C", str(foreign_repo), "commit", "-m", "x"])

    assert "user.name=hl" in args
    assert f"safe.directory={foreign_repo}" in args
