# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch that no longer applies where a merge puts it, rebased by a three-way merge of the files it touches.

Each file's merge base is its content under the stack the patch was written on; one side is the file under the stack
it now lands on, the other the file with the patch's own change. Only a merge in which no line is changed by both sides
yields a patch: an overlap is a choice between two edits, which a merge of text cannot make. Whether one side's
change rewrites every line of the other's is what lets a merge pick one of them instead.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from hyperloom_multibranch.repo.isolated_git import INIT, git_argv, git_env

#: A file's content by its path in the component's upstream repository, ``None`` where there is no file.
Files = Mapping[str, bytes | None]
# Each git command works on the few files one patch touches; this only bounds a wedged filesystem.
_SCRATCH_PREFIX = "hl-rebase-"
_GIT_TIMEOUT_S = 60.0
# git merge-file exits with the number of conflicts, capped at 127; above that it failed.
_MAX_CONFLICTS = 127
#: ``git diff`` with no colour or external diff driver, whatever the environment asks for.
_DIFF = ("diff", "--no-color", "--no-ext-diff")
# The old side of a hunk header; with no count it is one line, and with a count of 0 the hunk inserts after that line.
_HUNK_OLD_SIDE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? ", re.MULTILINE)


class _Overlap(Exception):
    """Both sides changed one line of a file."""


def merge_files(base: Files, ours: Files, theirs: Files) -> dict[str, bytes | None] | None:
    """``ours`` with what ``theirs`` changed from ``base`` in each path of ``theirs``; ``None`` when a path overlaps.

    ``base`` and ``ours`` hold every path of ``theirs``.
    """
    merged = dict(ours)
    try:
        for path, change in theirs.items():
            merged[path] = _merge_file(base[path], ours[path], change)
    except _Overlap:
        return None
    return merged


def _merge_file(base: bytes | None, ours: bytes | None, theirs: bytes | None) -> bytes | None:
    if theirs in (base, ours):
        return ours
    if base == ours:
        return theirs
    if base is None or ours is None or theirs is None:
        raise _Overlap
    done = _git_on(
        {"ours": ours, "base": base, "theirs": theirs}, "merge-file", "-p", "--quiet", "ours", "base", "theirs"
    )
    if done.returncode > _MAX_CONFLICTS:
        raise subprocess.CalledProcessError(done.returncode, "git merge-file", done.stdout, done.stderr)
    if done.returncode:
        raise _Overlap
    return done.stdout


def rewrites(base: bytes | None, ours: bytes | None, theirs: bytes | None) -> bool:
    """Whether ``theirs`` changes from ``base`` every line ``ours`` changes, so nothing of ``ours``'s change is left
    where ``theirs`` stands in for it.

    A line ``ours`` inserts counts as changed only inside a run of lines ``theirs`` rewrites, so a file ``ours``
    creates never does: two insertions at one place are two changes, which git reports as a conflict. A file ``ours``
    deletes is rewritten only by deleting it, and one ``ours`` edits never by deleting it.
    """
    if ours == base:
        return True
    if base is None or ours is None or theirs is None:
        return base is not None and ours is None and theirs is None
    rewritten = {line for start, end in _hunks(base, theirs) for line in range(start, end)}
    return all(
        set(range(start, end)) <= rewritten if start < end else {start - 1, start} <= rewritten
        for start, end in _hunks(base, ours)
    )


def _hunks(before: bytes, after: bytes) -> list[tuple[int, int]]:
    """The runs of lines of ``before`` that ``after`` changes, as half-open line ranges; an empty one is an insertion
    there.

    Repeated lines can be aligned more than one way; git merge-file aligns them as this diff does, with the same
    algorithm and no indent heuristic, so a line counts as changed exactly where the merge sees it changed.
    """
    done = _git_on(
        {"before": before, "after": after},
        *_DIFF,
        "--no-index",
        "--no-textconv",
        "--text",
        "--diff-algorithm=myers",
        "--no-indent-heuristic",
        "-U0",
        "before",
        "after",
    )
    # git diff --no-index exits 1 when the files differ.
    if done.returncode > 1:
        raise subprocess.CalledProcessError(done.returncode, "git diff --no-index", done.stdout, done.stderr)
    hunks = []
    for match in _HUNK_OLD_SIDE.finditer(done.stdout):
        start, count = int(match[1]), 1 if match[2] is None else int(match[2])
        hunks.append((start, start) if count == 0 else (start - 1, start - 1 + count))
    return hunks


def _git_on(files: Mapping[str, bytes], *args: str) -> subprocess.CompletedProcess[bytes]:
    """``git <args>`` run on ``files`` written by name into a scratch directory outside any repository."""
    with tempfile.TemporaryDirectory(prefix=_SCRATCH_PREFIX) as scratch:
        for name, data in files.items():
            (Path(scratch) / name).write_bytes(data)
        return subprocess.run(
            git_argv(*args),
            cwd=scratch,
            env=git_env(GIT_CEILING_DIRECTORIES=str(Path(scratch).parent)),
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )


def patch_between(before: Files, after: Files) -> str:
    """A git patch that turns ``before`` into ``after``, named ``a/<path>`` as the patches the toolbox applies are;
    empty when nothing differs.

    ``after`` holds every path of ``before``.
    """
    if all(before.get(path) == content for path, content in after.items()):
        return ""
    with tempfile.TemporaryDirectory(prefix=_SCRATCH_PREFIX) as scratch:
        root = Path(scratch)
        return _git(root, *_DIFF, "--no-renames", *_trees(root, before, after))


def _hash_objects(root: Path, contents: set[bytes]) -> dict[bytes, str]:
    """Each of ``contents`` written as a blob, by its object id."""
    ordered = list(contents)
    sources = [root / f"blob-{number}" for number in range(len(ordered))]
    for source, content in zip(sources, ordered):
        source.write_bytes(content)
    listing = "".join(f"{source}\n" for source in sources).encode()
    ids = _git(root, "hash-object", "-w", "--no-filters", "--stdin-paths", stdin=listing).split()
    return dict(zip(ordered, ids, strict=True))


def renames(before: Files, after: Files) -> dict[str, str]:
    """The files ``after`` moves, as ``git diff -M`` finds them: each new path with the path it was moved from.

    ``after`` holds every path of ``before``.
    """
    with tempfile.TemporaryDirectory(prefix=_SCRATCH_PREFIX) as scratch:
        root = Path(scratch)
        fields = _git(root, "diff", "--name-status", "-z", "-M", "--diff-filter=R", *_trees(root, before, after))
    # Each rename is three fields: its status (R and a similarity), the old path and the new one.
    parts = fields.split("\0")
    return {new: old for old, new in zip(parts[1::3], parts[2::3], strict=True)}


def _trees(root: Path, before: Files, after: Files) -> list[str]:
    """``before`` and ``after`` written as trees of a scratch repository in ``root``."""
    _git(root, *INIT, "--bare", "--quiet")
    blobs = _hash_objects(
        root, {content for files in (before, after) for content in files.values() if content is not None}
    )
    return [_write_tree(root, name, files, blobs) for name, files in (("before", before), ("after", after))]


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


__all__ = ["Files", "merge_files", "patch_between", "renames", "rewrites"]
