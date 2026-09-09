# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Crash-safe publication of a single file."""

from __future__ import annotations

import os
import shutil
import tempfile
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


def fsync_tree(root: Path) -> None:
    """Flush every file and directory under ``root`` before it is renamed."""
    for directory, _subdirectories, filenames in os.walk(root):
        current = Path(directory)
        for filename in filenames:
            descriptor = os.open(str(current / filename), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(current)
