# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch that no longer applies where a merge puts it, rebased by a three-way merge of the files it touches.

Each file's merge base is its content under the stack the patch was written on; one side is the file under the stack
it now lands on, the other the file with the patch's own change. Only a merge in which no line is changed by both sides
yields a patch: an overlap is a choice between two edits, which a merge of text cannot make.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from hyperloom_multibranch.repo.isolated_git import INIT, git_argv, git_env

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
            git_argv("merge-file", "-p", "--quiet", *side_flag, *map(str, sides)),
            cwd=scratch,
            env=git_env(GIT_CEILING_DIRECTORIES=str(Path(scratch).parent)),
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
        root = Path(scratch)
        _git(root, *INIT, "--bare", "--quiet")
        blobs = _hash_objects(
            root, {content for files in (before, after) for content in files.values() if content is not None}
        )
        trees = [_write_tree(root, name, files, blobs) for name, files in (("before", before), ("after", after))]
        return _git(root, "diff", "--no-color", "--no-ext-diff", "--no-renames", *trees)


def _hash_objects(root: Path, contents: set[bytes]) -> dict[bytes, str]:
    """Each of ``contents`` written as a blob, by its object id."""
    ordered = list(contents)
    sources = [root / f"blob-{number}" for number in range(len(ordered))]
    for source, content in zip(sources, ordered):
        source.write_bytes(content)
    listing = "".join(f"{source}\n" for source in sources).encode()
    ids = _git(root, "hash-object", "-w", "--no-filters", "--stdin-paths", stdin=listing).split()
    return dict(zip(ordered, ids, strict=True))


def _write_tree(root: Path, name: str, files: Files, blobs: Mapping[bytes, str]) -> str:
    """The tree of ``files`` built in an index of its own: with no working tree, no ignore or attributes file in
    ``files`` decides what the tree holds."""
    index = str(root / f"{name}.index")
    entries = b"".join(
        f"100644 {blobs[content]}\t{path}".encode() + b"\0" for path, content in files.items() if content is not None
    )
    _git(root, "update-index", "-z", "--index-info", stdin=entries, GIT_INDEX_FILE=index)
    return _git(root, "write-tree", GIT_INDEX_FILE=index).strip()


def _git(root: Path, *args: str, stdin: bytes = b"", **pinned: str) -> str:
    done = subprocess.run(
        git_argv(*args),
        cwd=root,
        env=git_env(GIT_DIR=str(root / "scratch.git"), **pinned),
        input=stdin,
        capture_output=True,
        timeout=_GIT_TIMEOUT_S,
        check=True,
    )
    return done.stdout.decode()


__all__ = ["Files", "merge_files", "patch_between"]
