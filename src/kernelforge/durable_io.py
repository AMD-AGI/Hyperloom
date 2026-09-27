# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Crash-safe publication of a single file."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)


def fsync_directory(path: str | Path) -> None:
    """Flush one directory's metadata so a rename survives a crash."""
    descriptor = os.open(str(path), _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Publish bytes at ``path``, replacing any prior content in one step."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_file():
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_text(path: str | Path, content: str) -> None:
    """Publish UTF-8 text at ``path``, replacing any prior content in one step."""
    atomic_write_bytes(path, content.encode("utf-8"))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _walk_depth_first(root: Path) -> Iterator[tuple[Path, list[str]]]:
    """Yield ``(directory, filenames)`` under ``root``, children before parents.

    A directory that cannot be enumerated raises rather than being skipped: the
    caller is about to rename this tree into place and its durability claim only
    holds if the whole tree was visited.
    """

    def _reraise(error: OSError) -> None:
        raise error

    for directory, _subdirectories, filenames in os.walk(root, topdown=False, onerror=_reraise):
        yield Path(directory), filenames


def fsync_tree(root: Path) -> None:
    """Flush every file and directory under ``root`` before it is renamed."""
    for directory, filenames in _walk_depth_first(root):
        for filename in filenames:
            _fsync_file(directory / filename)
        fsync_directory(directory)


def fsync_tree_directories(root: Path) -> None:
    """Flush every directory under ``root`` for a tree whose files were fsynced as they were written."""
    for directory, _filenames in _walk_depth_first(root):
        fsync_directory(directory)
