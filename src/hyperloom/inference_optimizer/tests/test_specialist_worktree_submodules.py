# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise specialist worktrees against real nested Git submodules."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.specialists.subprocess_ import _setup_worktree


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def repository(path: Path) -> Path:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Worktree Test")
    git(path, "config", "user.email", "worktree-test@example.invalid")
    (path / "README").write_text("test fixture\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "fixture")
    return path


def add_submodule(parent: Path, child: Path, relative: str) -> None:
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(child), relative)
    git(parent, "commit", "-qam", "add dependency")


@pytest.fixture
def nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    # Only local, test-owned remotes; production Git protocol policy is unchanged.
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    leaf = repository(tmp_path / "leaf")
    (leaf / "include").mkdir()
    (leaf / "include/header.hpp").write_text("pinned header\n")
    git(leaf, "add", ".")
    git(leaf, "commit", "-qm", "header")
    pinned_leaf = git(leaf, "rev-parse", "HEAD")
    middle = repository(tmp_path / "middle")
    add_submodule(middle, leaf, "dependencies/leaf")
    parent = repository(tmp_path / "parent")
    add_submodule(parent, middle, "vendor/middle")
    (leaf / "include/header.hpp").write_text("new remote head must not be selected\n")
    git(leaf, "commit", "-qam", "advance remote")
    return parent, pinned_leaf


def test_initializes_recursive_pinned_dependencies(nested, tmp_path):
    parent, pinned_leaf = nested
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert error == ""
    dependency = worktree / "vendor/middle/dependencies/leaf"
    assert (dependency / "include/header.hpp").read_text() == "pinned header\n"
    assert git(dependency, "rev-parse", "HEAD") == pinned_leaf
    assert not any(line.startswith("-") for line in git(worktree, "submodule", "status", "--recursive").splitlines())


def test_fresh_worktree_checks_out_pins_even_when_source_skips_submodule_updates(nested, tmp_path):
    parent, pinned_leaf = nested
    git(parent, "config", "submodule.vendor/middle.update", "none")
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert worktree is not None and error == ""
    dependency = worktree / "vendor/middle/dependencies/leaf"
    assert (dependency / "include/header.hpp").read_text() == "pinned header\n"
    assert git(dependency, "rev-parse", "HEAD") == pinned_leaf


def test_reuse_preserves_specialist_edits(nested, tmp_path):
    parent, _ = nested
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert not error
    header = worktree / "vendor/middle/dependencies/leaf/include/header.hpp"
    header.write_text("specialist edit\n")
    again, error = _setup_worktree(parent, worktree, "specialist-case")
    assert again == worktree and error == ""
    assert header.read_text() == "specialist edit\n"


def test_reuse_preserves_changed_submodule_commit(nested, tmp_path):
    parent, pinned_leaf = nested
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert not error
    dependency = worktree / "vendor/middle/dependencies/leaf"
    (dependency / "include/header.hpp").write_text("committed specialist edit\n")
    git(
        dependency,
        "-c",
        "user.name=Worktree Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qam",
        "specialist change",
    )
    changed = git(dependency, "rev-parse", "HEAD")
    assert changed != pinned_leaf
    again, error = _setup_worktree(parent, worktree, "specialist-case")
    assert again == worktree and not error
    assert git(dependency, "rev-parse", "HEAD") == changed


def test_retry_rejects_incomplete_worktree(nested, tmp_path):
    parent, _ = nested
    incomplete = tmp_path / "worktree"
    git(parent, "worktree", "add", "-b", "partial-initialization", str(incomplete))
    assert git(incomplete, "submodule", "status").startswith("-")
    worktree, error = _setup_worktree(parent, incomplete, "partial-initialization")
    assert worktree is None
    assert "uninitialized or conflicted submodules" in error
    assert list((incomplete / "vendor/middle").iterdir()) == []


def test_reuse_rejects_conflicted_submodule_without_changing_it(nested, tmp_path):
    parent, _ = nested
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert worktree is not None and not error
    middle = worktree / "vendor/middle"
    commit = git(middle, "rev-parse", "HEAD")
    conflict = f"0 {'0' * 40}\tvendor/middle\n" + "".join(
        f"160000 {commit} {stage}\tvendor/middle\n" for stage in (1, 2, 3)
    )
    subprocess.run(
        ["git", "-C", str(worktree), "update-index", "--index-info"],
        input=conflict,
        capture_output=True,
        text=True,
        check=True,
    )
    before = git(worktree, "ls-files", "--unmerged")
    assert git(worktree, "submodule", "status").startswith("U")
    again, error = _setup_worktree(parent, worktree, "specialist-case")
    assert again is None and "uninitialized or conflicted submodules" in error
    assert git(worktree, "ls-files", "--unmerged") == before
    assert git(middle, "rev-parse", "HEAD") == commit


@pytest.mark.parametrize("status", ["failure", "timeout", "missing_git"])
def test_reuse_rejects_unverified_submodules(nested, tmp_path, monkeypatch, status):
    parent, _ = nested
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert not error
    original_run = subprocess.run

    def run(args, **kwargs):
        if "submodule" not in args:
            return original_run(args, **kwargs)
        assert args[-2:] == ["status", "--recursive"]
        if status == "timeout":
            raise subprocess.TimeoutExpired(args, 60)
        if status == "missing_git":
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(args, 1, "", "cannot read submodule status")

    monkeypatch.setattr(subprocess, "run", run)
    again, error = _setup_worktree(parent, worktree, "specialist-case")
    assert again is None and error


def test_plain_repository_is_unchanged(tmp_path):
    parent = repository(tmp_path / "parent")
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert error == ""
    assert (worktree / "README").read_text() == "test fixture\n"
    assert git(worktree, "status", "--porcelain") == ""


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "missing_git"])
def test_submodule_initialization_failure_is_reported(nested, tmp_path, monkeypatch, failure):
    parent, _ = nested
    original_run = subprocess.run

    def run(args, **kwargs):
        if "submodule" not in args:
            return original_run(args, **kwargs)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 300)
        if failure == "missing_git":
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(args, 1, "", "unavailable pinned dependency")

    monkeypatch.setattr(subprocess, "run", run)
    worktree, error = _setup_worktree(parent, tmp_path / "worktree", "specialist-case")
    assert worktree is None
    assert "submodule initialization" in error
