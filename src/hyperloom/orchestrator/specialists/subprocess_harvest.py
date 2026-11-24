# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Harvest of what a reaped specialist left: its worktree diff or patch files, and its done file."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

from hyperloom_common.fs_utils import read_text_nofollow, regular_files_nofollow, write_text_nofollow
from hyperloom_common.git_safety import app_git_diff

from .worktrees import HeldWorktree, PinnedGit

log = logging.getLogger(__name__)

#: Directories a specialist writes for its own use inside the worktree, never
#: part of a deliverable.
_SPECIALIST_SCRATCH_DIRS: tuple[str, ...] = ("patches", "artifacts", ".hyperloom")


def _harvest_pathspec(targets: Sequence[str] = ()) -> list[str]:
    """Return the ``git diff`` / ``git add`` pathspec that scopes a harvest.

    With no target declared the whole worktree is in scope, minus
    :data:`_SPECIALIST_SCRATCH_DIRS`, whose whole-file copies would
    otherwise be harvested as file creations, and minus the work artifacts
    ``vet_patches`` refuses -- task-owned done files and bytecode caches.
    Excluding exactly what vetting rejects keeps the harvest from authoring
    a patch that is guaranteed to be dropped downstream.
    """
    from .patch_safety import SPECIALIST_WORK_ARTIFACT_PATHSPECS

    declared = [str(t).strip().lstrip("/") for t in targets if str(t).strip()]
    if declared:
        return declared
    return [
        ".",
        *(f":(exclude){name}" for name in _SPECIALIST_SCRATCH_DIRS),
        *SPECIALIST_WORK_ARTIFACT_PATHSPECS,
    ]


def _harvest_worktree_diff(worktree: Path, *, admin: Path, base: str = "HEAD", targets: Sequence[str] = ()) -> str:
    """Return the worktree's edits as one ``-p1`` diff, or ``""``.

    The comparison is ``base``-against-working-tree, since a specialist is
    not required to commit. ``app_git_diff`` marks untracked paths
    intent-to-add so they render as creations, ``git diff`` being blind to
    them otherwise, which keeps "edited a file and added one" from
    harvesting a patch that silently omits the addition. Proven Python
    comment-only edits produce no installable patch at all.

    Args:
        worktree: Per-task worktree.
        admin: Its admin directory as its repository records it.
        base: The recorded pre-round commit to diff against.
        targets: Worktree-relative paths the round declared.

    Returns:
        str: The diff text, or ``""`` when there is nothing to harvest or
            git could not be run.
    """

    from .patch_safety import patch_is_annotation_only

    try:
        diff = app_git_diff(worktree, base, _harvest_pathspec(targets), admin=admin)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("specialist: git diff in %s failed: %r", worktree, exc)
        return ""
    if diff.returncode != 0 or not diff.stdout.strip():
        return ""
    if patch_is_annotation_only(diff.stdout, worktree, reverse=True):
        return ""
    return diff.stdout


def collect_patches(
    worktree: Path | None,
    workspace: Path,
    worktree_base: Path | None = None,
    *,
    pinned: PinnedGit | None,
    targets: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Harvest the specialist's edits, else collect the patch files it wrote.

    Editing the worktree is the primary contract: ``git diff HEAD`` renders
    those edits as one canonical ``-p1`` patch and names its apply root, so
    nothing downstream has to infer which tree it belongs to. Scanning
    ``patches/`` remains for dispatches that never got a worktree, and for
    any git failure here -- a harvest that raises would otherwise discard
    the whole specialist result, done-file included.

    Args:
        worktree: Per-task worktree, or None.
        workspace: Task workspace.
        worktree_base: The tree the worktree stands for -- the checkout it
            was branched off, or the directory it holds a snapshot of --
            which is the apply root of anything harvested from it.
        pinned: The worktree's admin directory as its repository records
            it, and the commit recorded before the round, so the harvest
            stays anchored there even if the specialist committed; ``None``
            harvests nothing by git.
        targets: Worktree-relative paths the round declared, which scope the
            harvest pathspec.

    Returns:
        ``(patch_paths, patch_roots)``; the latter is empty for scanned files.
    """
    if worktree is not None and pinned is not None:
        harvested_diff = _harvest_worktree_diff(
            worktree, admin=pinned.admin, base=pinned.start or "HEAD", targets=targets
        )
        if harvested_diff:
            try:
                harvested = write_text_nofollow(worktree, ("patches", "_worktree_diff.patch"), harvested_diff)
            except OSError as exc:
                log.warning("specialist: refusing to write the harvested patch under %s: %r", worktree, exc)
            else:
                return [str(harvested)], {str(harvested): str(worktree_base or worktree)}

    out: list[str] = []
    for base in (worktree, workspace):
        if base is None:
            continue
        for ext in (".patch", ".diff"):
            out.extend(
                str(p) for p in regular_files_nofollow(base, "patches", (ext,)) if p.name != "_worktree_diff.patch"
            )
    return out, {}


def read_done(done_file: Path, *, held: HeldWorktree | None = None) -> dict[str, Any] | None:
    """Parse a ``specialist_done.json`` file, unwrapping intent envelopes.

    Tolerates missing files and parse errors (logged, returns None).
    When the file holds a ``specialist_done`` intent envelope, the
    inner ``payload`` is merged with the outer keys so callers always
    see a flat dict.

    Args:
        done_file (Path): Path to the candidate done-file.
        held (HeldWorktree | None): The worktree as recorded before the run; a file directly in it is read only
            through that directory (:meth:`~.worktrees.HeldWorktree.read_text`).

    Returns:
        dict[str, Any] | None: The parsed (and possibly unwrapped)
            payload, or None when missing / unparseable / not a dict.
    """
    if not done_file.exists():
        return None
    try:
        if held is not None and done_file.parent == held.path:
            text = held.read_text(done_file.name)
        else:
            text = read_text_nofollow(done_file)
        data = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning(
            "specialist_done.json parse failed at %s: %r",
            done_file,
            exc,
        )
        return None
    if not isinstance(data, dict):
        log.warning(
            "specialist_done.json at %s is not a dict (%r); ignoring",
            done_file,
            type(data).__name__,
        )
        return None
    if str(data.get("intent_type") or "") == "specialist_done" and isinstance(data.get("payload"), dict):
        inner = data["payload"]
        merged: dict[str, Any] = {}
        for k, v in data.items():
            if k in ("intent_type", "payload"):
                continue
            merged[k] = v
        for k, v in inner.items():
            merged[k] = v
        log.info(
            "_read_done: unwrapped specialist_done intent envelope at %s (proposal_set_len=%d)",
            done_file,
            len(inner.get("proposal_set") or []) if isinstance(inner.get("proposal_set"), list) else 0,
        )
        return merged
    return data
