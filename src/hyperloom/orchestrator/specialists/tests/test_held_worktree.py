# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The worktree the app reads after a specialist ran is the directory it gave the specialist, never what the
specialist left at its path."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.specialists.worktrees import HeldWorktree


def _worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "ws" / "worktree"
    worktree.mkdir(parents=True)
    (worktree / "mod.py").write_text("old = 1\n", encoding="utf-8")
    return worktree


def test_a_worktree_the_specialist_only_edited_or_moved_back_is_still_the_one_it_was_given(tmp_path: Path) -> None:
    worktree = _worktree(tmp_path)
    held = HeldWorktree.before_launch(worktree)

    (worktree / "mod.py").write_text("new = 2\n", encoding="utf-8")
    worktree.rename(worktree.with_name("aside"))
    worktree.with_name("aside").rename(worktree)

    assert held.refusal() == ""


def test_a_link_left_in_the_worktrees_place_is_refused_and_removed_and_what_it_named_is_untouched(
    tmp_path: Path,
) -> None:
    worktree = _worktree(tmp_path)
    host = tmp_path / "app-secrets"
    host.mkdir()
    (host / "creds.txt").write_text("secret\n", encoding="utf-8")
    held = HeldWorktree.before_launch(worktree)

    worktree.rename(worktree.with_name("worktree.orig"))
    worktree.symlink_to(host)
    refusal = held.refusal()

    assert refusal == (
        f"worktree_refused: {worktree} is no longer the directory the specialist was given; "
        "the link left in its place was removed"
    )
    assert not worktree.is_symlink() and not worktree.exists()
    assert sorted(p.name for p in host.iterdir()) == ["creds.txt"]
    assert (worktree.with_name("worktree.orig") / "mod.py").is_file()


def test_another_directory_at_the_worktrees_path_is_refused_and_kept(tmp_path: Path) -> None:
    worktree = _worktree(tmp_path)
    held = HeldWorktree.before_launch(worktree)

    worktree.rename(worktree.with_name("worktree.orig"))
    worktree.mkdir()

    assert held.refusal() == f"worktree_refused: {worktree} is no longer the directory the specialist was given"
    assert worktree.is_dir()


def test_a_worktree_that_was_a_link_when_the_specialist_started_is_never_read(tmp_path: Path) -> None:
    real = _worktree(tmp_path)
    linked = tmp_path / "ws" / "linked"
    linked.symlink_to(real)
    held = HeldWorktree.before_launch(linked, tmp_path / "admin")

    refusal = held.refusal()

    assert held.inode is None and held.git is None  # no git ran over it either
    assert refusal.startswith(f"worktree_refused: {linked} was not a directory of its own when the specialist started")
    assert not linked.is_symlink() and (real / "mod.py").is_file()
    assert HeldWorktree.before_launch(tmp_path / "missing").refusal().endswith("when the specialist started")
