# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The app's git over a linked worktree goes through the admin directory its repository records, never through the
one the worktree's ``.git`` names: an agent running as the app's own uid makes one no ownership check tells apart."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hyperloom_common.git_safety import app_git, app_git_diff, recorded_admin_dir, trusted_admin_dir, worktree_head

_IDENTITY = ("-c", "user.email=t@t", "-c", "user.name=t")


def _git(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, env=env).stdout


def _repo(where: Path) -> Path:
    where.mkdir(parents=True)
    (where / "mod.py").write_text("old = 1\n", encoding="utf-8")
    _git("init", "-q", str(where))
    _git("-C", str(where), "add", "-A")
    _git("-C", str(where), *_IDENTITY, "commit", "-qm", "base")
    return where


def _worktree(repo: Path, where: Path, *options: str) -> Path:
    _git("-C", str(repo), *options, "worktree", "add", "-q", "--detach", str(where))
    return where


def _forge(worktree: Path, where: Path, objects: Path, head: str) -> Path:
    """What an agent writing ``worktree`` as the app's uid can do: point ``.git`` at an admin dir it made in a
    repository whose config it wrote (here: diffs without the ``a/`` and ``b/`` prefixes), showing the real objects."""
    common = where / "forged.git"
    _git("init", "-q", "--bare", str(common))
    _git("--git-dir", str(common), "config", "diff.noprefix", "true")
    (common / "objects" / "info" / "alternates").write_text(f"{objects}\n", encoding="utf-8")
    admin = common / "worktrees" / "w"
    admin.mkdir(parents=True)
    (admin / "gitdir").write_text(f"{worktree / '.git'}\n", encoding="utf-8")
    (admin / "commondir").write_text("../..\n", encoding="utf-8")
    (admin / "HEAD").write_text(f"{head}\n", encoding="utf-8")
    (worktree / ".git").write_text(f"gitdir: {admin}\n", encoding="utf-8")
    return admin


def test_the_admin_dir_is_the_one_the_repository_records_for_the_worktree(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    first = _worktree(repo, tmp_path / "a" / "worktree")
    second = _worktree(repo, tmp_path / "b" / "worktree")

    admin = recorded_admin_dir(repo, second)

    assert admin == trusted_admin_dir(second) != recorded_admin_dir(repo, first)
    assert admin.parent == (repo / ".git" / "worktrees").resolve()


def test_an_agent_of_the_apps_uid_passes_every_ownership_check_and_still_never_picks_the_admin_dir(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "repo")
    worktree = _worktree(repo, tmp_path / "ws" / "worktree")
    admin = recorded_admin_dir(repo, worktree)
    head = worktree_head(worktree, admin=admin)
    (worktree / "mod.py").write_text("new = 2\n", encoding="utf-8")

    forged = _forge(worktree, tmp_path / "ws", repo / ".git" / "objects", head)

    # The premise: ownership proves nothing, and the forged repository's config decides what the app's git prints.
    assert trusted_admin_dir(worktree) == forged
    assert "--- mod.py" in app_git(worktree, "diff", "--no-ext-diff", "HEAD").stdout
    # Pinned, the worktree's .git is not read at all.
    assert recorded_admin_dir(repo, worktree) == admin
    assert app_git(worktree, "rev-parse", "--absolute-git-dir", admin=admin).stdout.strip() == str(admin)
    assert worktree_head(worktree, admin=admin) == head
    diff = app_git_diff(worktree, head, ["."], admin=admin).stdout
    assert "--- a/mod.py\n+++ b/mod.py\n" in diff and "+new = 2" in diff


def test_a_worktree_recorded_by_a_relative_path_is_found(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    worktree = _worktree(repo, tmp_path / "worktree", "-c", "worktree.useRelativePaths=true")
    admin = (repo / ".git" / "worktrees" / "worktree").resolve()
    if os.path.isabs((admin / "gitdir").read_text(encoding="utf-8").strip()):
        pytest.skip("this git records worktrees by absolute path only")

    assert recorded_admin_dir(repo, worktree) == admin


def test_a_directory_the_repository_does_not_record_has_no_admin_dir(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    _worktree(repo, tmp_path / "recorded")
    left = tmp_path / "left-over"
    left.mkdir()
    (left / ".git").write_text(f"gitdir: {repo / '.git' / 'worktrees' / 'recorded'}\n", encoding="utf-8")

    with pytest.raises(OSError, match="0 linked worktrees"):
        recorded_admin_dir(repo, left)
    with pytest.raises(OSError, match="no git directory"):
        recorded_admin_dir(tmp_path / "not-a-repo", left)


def test_two_entries_naming_one_worktree_are_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    worktree = _worktree(repo, tmp_path / "worktree")
    admin = recorded_admin_dir(repo, worktree)
    twin = admin.parent / "twin"
    twin.mkdir()
    (twin / "gitdir").write_text((admin / "gitdir").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(OSError, match="2 linked worktrees"):
        recorded_admin_dir(repo, worktree)
