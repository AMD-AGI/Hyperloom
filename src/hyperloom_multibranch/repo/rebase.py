# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch that no longer applies where a merge puts it, rebased by a three-way merge of the files it touches.

Each file's merge base is its content under the stack the patch was written on; one side is the file under the stack
it now lands on, the other the file with the patch's own change. Only a merge in which no line is changed by both sides
yields a patch: an overlap is a choice between two edits, which a merge of text cannot make.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

#: A file's content by its path in the component's upstream repository, ``None`` where there is no file.
Files = Mapping[str, bytes | None]
# Both commands work on the few files one patch touches; this only bounds a wedged filesystem.
_GIT_TIMEOUT_S = 60.0
# git merge-file exits with the number of conflicts, capped at 127; above that it failed.
_MAX_CONFLICTS = 127


class _Overlap(Exception):
    """Both sides changed one line of a file."""


def merge_files(
    base: Files, ours: Files, theirs: Files, *, theirs_wins: bool = False
) -> dict[str, bytes | None] | None:
    """``ours`` with what ``theirs`` changed from ``base`` in each path of ``theirs``; ``None`` when a path overlaps.

    ``base`` and ``ours`` hold every path of ``theirs``. ``theirs_wins`` takes ``theirs``'s side of every overlap
    instead (``git merge-file --theirs``), and of a file one side lacks: only to test whether ``theirs`` supersedes
    what ``ours`` changed, never to rebase.
    """
    merged = dict(ours)
    try:
        for path, change in theirs.items():
            merged[path] = _merge_file(base[path], ours[path], change, theirs_wins)
    except _Overlap:
        return None
    return merged


def _merge_file(base: bytes | None, ours: bytes | None, theirs: bytes | None, theirs_wins: bool) -> bytes | None:
    if theirs in (base, ours):
        return ours
    if base == ours:
        return theirs
    if base is None or ours is None or theirs is None:
        if theirs_wins:
            return theirs
        raise _Overlap
    side_flag = ["--theirs"] if theirs_wins else []
    with tempfile.TemporaryDirectory(prefix="hl-rebase-") as scratch:
        sides = [Path(scratch) / name for name in ("ours", "base", "theirs")]
        for side, data in zip(sides, (ours, base, theirs)):
            side.write_bytes(data)
        done = subprocess.run(
            ["git", "merge-file", "-p", "--quiet", *side_flag, *map(str, sides)],
            env=_env(Path(scratch)),
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    if done.returncode > _MAX_CONFLICTS:
        raise subprocess.CalledProcessError(done.returncode, "git merge-file", done.stdout, done.stderr)
    if done.returncode:
        raise _Overlap
    return done.stdout


def patch_between(before: Files, after: Files) -> str:
    """A git patch that turns ``before`` into ``after``, named ``a/<path>`` as the patches the toolbox applies are;
    empty when nothing differs.

    ``after`` holds every path of ``before``.
    """
    if all(before.get(path) == content for path, content in after.items()):
        return ""
    with tempfile.TemporaryDirectory(prefix="hl-rebase-") as scratch:
        tree = Path(scratch)
        _git(tree, "init", "--quiet")
        trees = [_write_tree(tree, files) for files in (before, after)]
        return _git(tree, "diff", "--no-color", "--no-ext-diff", "--no-renames", *trees)


def _write_tree(tree: Path, files: Files) -> str:
    for path, content in files.items():
        target = tree / path
        if content is None:
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _git(tree, "add", "--all")
    return _git(tree, "write-tree").strip()


def _git(tree: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=tree, env=_env(tree), capture_output=True, timeout=_GIT_TIMEOUT_S, check=True
    )
    return done.stdout.decode()


def _env(scratch: Path) -> dict[str, str]:
    """The environment of a git command in ``scratch``: neither the operator's configuration nor an inherited
    ``GIT_DIR`` may shape the merge or the patch."""
    inherited = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    return {
        **inherited,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CEILING_DIRECTORIES": str(scratch.parent),
        "LC_ALL": "C",
    }


__all__ = ["Files", "merge_files", "patch_between"]
