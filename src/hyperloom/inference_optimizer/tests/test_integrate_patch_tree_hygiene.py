# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reverted candidate leaves the framework tree exactly at HEAD.

Every KEEP is committed, so HEAD *is* the accepted stack by construction and
"restore the working tree to HEAD" reverts a candidate exactly: kept work is in
commits and survives, candidate work is uncommitted and goes.

Reverse-applying each diff instead is textual, and it can report success while
leaving the tree a few lines away from HEAD -- the forward apply may have used
fuzz or a guessed -p level. That residue is what starts the failure:

  residue survives the revert
    -> the next candidate's auto-stash banks it as "user changes"
    -> a KEEP lands in between, touching the same functions
    -> git stash pop conflicts, leaving the tree in UU with markers in source

from where two different runs died. One benchmarked a tree whose source no
longer parsed; the other could not stash at all (git refuses while a conflict is
unresolved), so five consecutive candidates aborted with apply_failed and the
phase plateaued out of a 24h budget.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hyperloom.orchestrator.actions.executors.integrate_patch import (
    _git_restore_stash_if_needed,
    _git_stash_if_dirty,
)

from .conftest import git_commit_all, init_git_repo

_SEED = "def f():\n    return 1\n"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(repo),
            "GIT_AUTHOR_NAME": "Hyperloom Test",
            "GIT_AUTHOR_EMAIL": "hyperloom@test.local",
            "GIT_COMMITTER_NAME": "Hyperloom Test",
            "GIT_COMMITTER_EMAIL": "hyperloom@test.local",
        },
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "framework"
    init_git_repo(repo, seed_file="src.py", seed_text=_SEED)
    return repo


def _wedge_with_a_stash_conflict(repo: Path) -> None:
    """Recreate the exact state both failed runs ended in: an unresolved pop."""
    (repo / "src.py").write_text("def f():\n    return 111\n", encoding="utf-8")
    _git(repo, "stash", "push", "-u", "-m", "hyperloom-auto-stash: preserving user changes before candidate run")
    (repo / "src.py").write_text("def f():\n    return 222\n", encoding="utf-8")
    git_commit_all(repo, "hyperloom KEEP")
    _git(repo, "stash", "pop")  # conflicts; leaves UU + markers


def _is_conflicted(repo: Path) -> bool:
    return bool(_git(repo, "ls-files", "-u").stdout.strip())


def test_stash_recovers_a_tree_left_conflicted_by_an_earlier_cycle(tmp_path: Path):
    """The state that aborted five candidates in a row must self-heal."""
    repo = _repo(tmp_path)
    _wedge_with_a_stash_conflict(repo)
    assert _is_conflicted(repo), "fixture did not reproduce the wedge"

    state, note = _git_stash_if_dirty(repo)

    assert state != "failed", f"still wedged: {note}"
    assert not _is_conflicted(repo)
    assert "<<<<<<<" not in (repo / "src.py").read_text()


def test_clearing_an_unresolved_merge_banks_it_rather_than_deleting_it(tmp_path: Path):
    """Nothing here can prove whose merge it is, so it must stay recoverable.

    The tree Hyperloom is handed may be an operator's own repository, and this
    code cannot tell a previous cycle's wreckage from a merge someone started by
    hand. Clearing it is required to make progress; destroying it is not.
    """
    repo = _repo(tmp_path)
    _wedge_with_a_stash_conflict(repo)
    assert _is_conflicted(repo), "fixture did not reproduce the wedge"
    assert "111" in (repo / "src.py").read_text(), "fixture lost the conflicting side"

    state, note = _git_stash_if_dirty(repo)

    assert state != "failed", f"still wedged: {note}"
    assert not _is_conflicted(repo)
    listing = [ln for ln in _git(repo, "stash", "list").stdout.splitlines() if "hyperloom-quarantine" in ln]
    assert listing, "the merge was discarded, not banked"
    ref = listing[0].split(":", 1)[0]
    banked = _git(repo, "stash", "show", "-p", ref).stdout
    assert "111" in banked, f"quarantine {ref} does not carry the conflicting content:\n{banked}"


def _wedge_with_an_operator_merge(repo: Path) -> None:
    """A conflict the operator started by hand, not one Hyperloom created.

    Unlike a failed stash pop this leaves ``MERGE_HEAD`` standing, which is what
    makes the next commit a merge commit.
    """
    _git(repo, "checkout", "-q", "-b", "operator-branch")
    (repo / "src.py").write_text("def f():\n    return 111\n", encoding="utf-8")
    git_commit_all(repo, "operator work")
    _git(repo, "checkout", "-q", "-")
    (repo / "src.py").write_text("def f():\n    return 222\n", encoding="utf-8")
    git_commit_all(repo, "mainline work")
    _git(repo, "merge", "operator-branch")  # conflicts


def test_a_kept_candidate_after_a_quarantine_is_not_a_merge_commit(tmp_path: Path):
    """Clearing the index is not the same as ending the merge.

    ``MERGE_HEAD`` outlives the staging and the stash, and while it stands the
    next commit takes a second parent. A KEEP would then claim the whole of the
    operator's branch as accepted work, and "HEAD is the accepted stack" would
    no longer be true of the stack it reports.
    """
    repo = _repo(tmp_path)
    _wedge_with_an_operator_merge(repo)
    assert _is_conflicted(repo), "fixture did not reproduce an operator merge"
    assert (repo / ".git" / "MERGE_HEAD").exists(), "fixture did not leave MERGE_HEAD"

    state, note = _git_stash_if_dirty(repo)
    assert state != "failed", f"still wedged: {note}"
    assert not (repo / ".git" / "MERGE_HEAD").exists(), "the merge was never ended"

    (repo / "candidate.py").write_text("x = 1\n", encoding="utf-8")
    git_commit_all(repo, "hyperloom KEEP")
    parents = _git(repo, "log", "-1", "--format=%P").stdout.split()
    assert len(parents) == 1, f"the KEEP became a merge commit with parents {parents}"


def test_a_quarantined_merge_is_never_popped_back(tmp_path: Path):
    """Popping conflict markers back into source is what broke the benchmark."""
    repo = _repo(tmp_path)
    _wedge_with_a_stash_conflict(repo)

    state, ref = _git_stash_if_dirty(repo)
    _git_restore_stash_if_needed(repo, state, ref)

    assert "<<<<<<<" not in (repo / "src.py").read_text()
    assert not _is_conflicted(repo)
    assert "hyperloom-quarantine" in _git(repo, "stash", "list").stdout, "quarantine must survive the pop"


def test_a_conflicted_pop_does_not_leave_markers_in_source(tmp_path: Path):
    """On a pop that cannot merge, the tree returns to HEAD and the stash stays."""
    repo = _repo(tmp_path)
    # A candidate's residue, banked as if it were user state.
    (repo / "src.py").write_text("def f():\n    return 111\n", encoding="utf-8")
    state, ref = _git_stash_if_dirty(repo)
    assert state == "stashed"
    # A KEEP lands on the same lines while the stash is held.
    (repo / "src.py").write_text("def f():\n    return 222\n", encoding="utf-8")
    git_commit_all(repo, "hyperloom KEEP")

    _git_restore_stash_if_needed(repo, state, ref)

    assert not _is_conflicted(repo)
    assert "<<<<<<<" not in (repo / "src.py").read_text()
    assert "return 222" in (repo / "src.py").read_text(), "the KEEP must survive"
    assert "hyperloom-auto-stash" in _git(repo, "stash", "list").stdout, "user work must remain recoverable"


def test_a_stash_that_pops_cleanly_still_restores_the_work(tmp_path: Path):
    """The behaviour the stash exists for is unchanged."""
    repo = _repo(tmp_path)
    (repo / "other.py").write_text("x = 1\n", encoding="utf-8")
    state, ref = _git_stash_if_dirty(repo)
    assert state == "stashed"
    assert not (repo / "other.py").exists()

    note = _git_restore_stash_if_needed(repo, state, ref)

    assert note == ""
    assert (repo / "other.py").read_text() == "x = 1\n"


def test_a_clean_tree_is_left_alone(tmp_path: Path):
    """No stash, no repair, no surprise."""
    repo = _repo(tmp_path)
    state, note = _git_stash_if_dirty(repo)
    assert state == "clean"
    assert note == ""
    assert (repo / "src.py").read_text() == _SEED
