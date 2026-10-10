# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the durable-publication primitives promise their callers."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from kernelforge.durable_io import (
    atomic_write_bytes,
    atomic_write_text,
    fsync_tree,
    fsync_tree_directories,
)


def _tree(root: Path) -> Path:
    (root / "nested" / "deeper").mkdir(parents=True)
    (root / "top.json").write_text("{}")
    (root / "nested" / "mid.txt").write_text("mid")
    (root / "nested" / "deeper" / "leaf.txt").write_text("leaf")
    return root


def _opened_under(root: Path, walk: Callable[[Path], None]) -> set[str]:
    """The paths under ``root`` that ``walk(root)`` opened.

    ``os.open`` is patched for the whole process, so a finalizer or thread left over from another test can open its own
    paths while the walk runs; only the paths under ``root`` are the walk's.
    """
    opened: list[str] = []
    real_open = os.open

    def _record(path, flags, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "open", _record)
        walk(root)
    return {path for path in opened if Path(path).is_relative_to(root)}


def test_atomic_write_replaces_prior_content_in_one_step(tmp_path):
    target = tmp_path / "out" / "result.json"
    atomic_write_text(target, '{"a": 1}')
    atomic_write_bytes(target, b'{"a": 2}')

    assert target.read_text() == '{"a": 2}'
    assert [p.name for p in target.parent.iterdir()] == ["result.json"]


def test_fsync_tree_visits_every_file_and_directory(tmp_path):
    root = _tree(tmp_path / "bundle")

    assert _opened_under(root, fsync_tree) == {
        str(root),
        str(root / "top.json"),
        str(root / "nested"),
        str(root / "nested" / "mid.txt"),
        str(root / "nested" / "deeper"),
        str(root / "nested" / "deeper" / "leaf.txt"),
    }


def test_fsync_tree_directories_skips_the_files(tmp_path):
    root = _tree(tmp_path / "bundle")

    assert _opened_under(root, fsync_tree_directories) == {
        str(root),
        str(root / "nested"),
        str(root / "nested" / "deeper"),
    }


@pytest.mark.parametrize("walk_tree", [fsync_tree, fsync_tree_directories])
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1000)() == 0, reason="root can enumerate a directory with no read bit"
)
def test_a_directory_that_cannot_be_enumerated_fails_the_durability_claim(tmp_path, walk_tree):
    """The caller renames this tree into place next, so a skipped subtree is data loss."""
    root = _tree(tmp_path / "bundle")
    unreadable = root / "nested"
    original_mode = stat.S_IMODE(unreadable.stat().st_mode)
    unreadable.chmod(0o000)
    try:
        with pytest.raises(OSError):
            walk_tree(root)
    finally:
        unreadable.chmod(original_mode)
