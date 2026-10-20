# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keeping the operator's uncommitted work out of the way of an ungraded candidate.

Every integration mutates a tree the operator may have work in. These banks that
work in a stash entry of this project's own, restore the tree to the commit the
attempt recorded, and hand the work back afterwards -- the obligations
``restore_pending_integrate`` discharges.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._git import _run_git_cp

log = logging.getLogger(__name__)


_HYPERLOOM_AUTO_STASH_MSG = "hyperloom-auto-stash: preserving user changes before candidate run"
# Deliberately shares no substring with the auto-stash tag: _find_hyperloom_auto_stash
# matches by message, and a quarantined merge must never be picked up and popped back.
_HYPERLOOM_QUARANTINE_STASH_MSG = "hyperloom-quarantine: unresolved merge cleared before candidate run"


def _find_hyperloom_auto_stash(framework_root: Path) -> str:
    """Return the newest Hyperloom auto-stash ref, or ``""`` if absent."""
    cp = _run_git_cp(
        ["-C", str(framework_root), "stash", "list", "--format=%gd:%gs"],
        timeout=30.0,
    )
    if cp is None or cp.returncode != 0:
        return ""
    for line in cp.stdout.splitlines():
        ref, _sep, msg = line.partition(":")
        if ref and _HYPERLOOM_AUTO_STASH_MSG in msg:
            return ref
    return ""


def _git_quarantine_unmerged(framework_root: Path, paths: list[str]) -> tuple[bool, str]:
    """Bank unresolved-merge paths in a stash entry that is never popped back.

    ``git stash`` refuses to run while the index carries unmerged entries, so they
    are staged first and the working-tree content, conflict markers and all, is
    what gets banked; recover it with ``git stash list | grep hyperloom-quarantine``.
    ``MERGE_HEAD`` outlives both the staging and the stash, and while it stands the
    next ``git commit`` is a merge commit -- a KEEP would then claim the whole of
    the other side as accepted work -- so ``git merge --quit`` forgets the merge
    without touching the index or the working tree.
    """
    add = _run_git_cp(["-C", str(framework_root), "add", "--", *paths], timeout=60.0)
    if add is None:
        return False, "git add failed"
    if add.returncode != 0:
        return False, f"git add rc={add.returncode}: {add.stderr.strip()}"
    cp = _run_git_cp(
        [
            "-C",
            str(framework_root),
            "stash",
            "push",
            "-m",
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "--",
            *paths,
        ],
        timeout=60.0,
    )
    if cp is None:
        return False, "git stash push failed"
    if cp.returncode != 0:
        return False, f"git stash push rc={cp.returncode}: {cp.stderr.strip()}"
    quit_cp = _run_git_cp(["-C", str(framework_root), "merge", "--quit"], timeout=30.0)
    if quit_cp is None or quit_cp.returncode != 0:
        # Nothing was banked that a later revert cannot reach, but leaving
        # MERGE_HEAD standing would mislabel the next KEEP, so refuse rather
        # than proceed on a repo that is still mid-merge.
        detail = "git merge --quit failed" if quit_cp is None else quit_cp.stderr.strip()
        return False, f"merge state could not be cleared: {detail}"
    return True, ""


def _git_unmerged_paths(framework_root: Path) -> list[str]:
    """Return repo-relative paths left in an unresolved merge state, if any."""
    cp = _run_git_cp(["-C", str(framework_root), "ls-files", "-u", "--full-name"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        return []
    seen: list[str] = []
    for line in (cp.stdout or "").splitlines():
        _meta, _sep, path = line.partition("\t")
        path = path.strip()
        if path and path not in seen:
            seen.append(path)
    return seen


def _git_restore_to_head(framework_root: Path, paths: list[str] | None = None) -> tuple[bool, str]:
    """Force ``paths`` (or the whole tree) back to HEAD, clearing any merge state.

    Every KEEP is committed, so HEAD is the accepted stack: restoring to it drops
    candidate work and keeps everything the loop has accepted.
    """
    target = paths if paths else ["."]
    # `checkout --force HEAD --` resolves an unmerged index, which plain
    # `checkout -- .` refuses to touch.
    cp = _run_git_cp(
        ["-C", str(framework_root), "checkout", "--force", "HEAD", "--", *target],
        timeout=60.0,
    )
    if cp is None:
        return False, "git checkout spawn failed"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    return True, ""


def _git_stash_if_dirty(framework_root: Path) -> tuple[str, str]:
    """Stash uncommitted user changes so destructive resets don't lose them.

    The returned state is ``clean``, ``stashed`` (``note`` is the stash ref to
    restore) or ``failed`` -- on ``failed`` the tree is dirty and callers must not
    proceed with destructive operations.
    """
    cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
    if cp is None:
        return "failed", "git status check failed"
    if cp.returncode != 0:
        # Non-git directory or other git status errors: treat as clean.
        log.debug(
            "integrate_patch: git status rc=%d in %s (not a git repo?), treating as clean",
            cp.returncode,
            framework_root,
        )
        return "clean", ""
    if not cp.stdout.strip():
        return "clean", ""
    # git refuses to stash while an unresolved merge stands, so every later
    # candidate would abort here forever. Clear it — but bank the content
    # instead of overwriting it. Usually this is wreckage from an earlier cycle;
    # nothing here can prove that, and a merge the operator started themselves
    # is not ours to throw away. Quarantine keeps both cases recoverable at the
    # cost of one stash entry. It is deliberately not the auto-stash tag, so it
    # is never popped back: restoring conflict markers into the source is what
    # made every benchmark fail to parse the model.
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_quarantine_unmerged(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s had %d path(s) in an unresolved merge (%s); moved to a '%s' stash entry%s",
            framework_root,
            len(unmerged),
            ", ".join(unmerged[:5]),
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "" if ok else f" FAILED: {err}",
        )
        if not ok:
            return "failed", f"unresolved merge could not be quarantined: {err}"
        cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
        if cp is None:
            return "failed", "git status check failed"
        if not cp.stdout.strip():
            return "clean", ""
    cp2 = _run_git_cp(
        ["-C", str(framework_root), "stash", "push", "-u", "-m", _HYPERLOOM_AUTO_STASH_MSG],
        timeout=60.0,
    )
    if cp2 is None:
        return "failed", "git stash push failed"
    if cp2.returncode == 0:
        stash_ref = _find_hyperloom_auto_stash(framework_root) or "stash@{0}"
        log.info(
            "integrate_patch: stashed user changes in %s as %s",
            framework_root,
            stash_ref,
        )
        return "stashed", stash_ref
    return "failed", f"git stash push rc={cp2.returncode}: {cp2.stderr.strip()}"


def _git_restore_stash_if_needed(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
) -> str:
    """Restore the user-change stash created before candidate mutation."""
    if stash_state != "stashed":
        return ""
    ref = stash_ref or _find_hyperloom_auto_stash(framework_root)
    if not ref:
        return "auto-stash ref not found; user changes remain in git stash"
    cp = _run_git_cp(["-C", str(framework_root), "stash", "pop", "--index", ref], timeout=120.0)
    if cp is None:
        return f"git stash pop failed; user changes remain in {ref}"
    if cp.returncode == 0:
        log.info("integrate_patch: restored user changes from %s", ref)
        return ""
    # A pop that cannot merge leaves conflict markers in source. Left there they
    # break every later measurement silently -- the benchmark fails to parse the
    # file, and git will not stash again while the merge is unresolved. Put the
    # tree back at HEAD instead; the stash is deliberately NOT dropped, so the
    # work is still there under `git stash list`.
    detail = (cp.stderr or "").strip()
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_restore_to_head(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s did not merge back into %s; restored %d path(s) to HEAD and kept the stash%s",
            ref,
            framework_root,
            len(unmerged),
            "" if ok else f" (restore FAILED: {err})",
        )
        if not ok:
            detail = f"{detail}; tree left conflicted: {err}"
    return f"git stash pop {ref} rc={cp.returncode}: {detail}; user changes remain in git stash"


def _git_reset_clean(framework_root: Path, base: str, *, exclude: list[str] | None = None) -> tuple[bool, str]:
    """Return the tree to ``base`` exactly: tracked, staged and untracked candidate files all go.

    ``base`` is the HEAD recorded before the candidate, which is the accepted stack
    because every KEEP is committed. User changes must already have been stashed.
    """
    cp = _run_git_cp(["-C", str(framework_root), "reset", "--hard", base], timeout=60.0)
    if cp is None:
        return False, "git reset spawn failed"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    exclusions = [arg for path in exclude or [] for arg in ("-e", path)]
    cp2 = _run_git_cp(["-C", str(framework_root), "clean", "-fd", *exclusions], timeout=60.0)
    if cp2 is None:
        return False, "git clean spawn failed"
    return cp2.returncode == 0, cp2.stderr.strip()
