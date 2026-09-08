# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Crash-safe publication of a single file.

Every artifact a run is resumed, scored or audited from is published through
here, so a crash between the write and the rename leaves the prior version
intact rather than a truncated one. Serialization stays with the caller: the
exact bytes of a published payload are that caller's contract with its readers.
"""

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
    """Publish bytes at ``path``, replacing any prior content in one step.

    A replaced file keeps the permissions it had. The temp file this publishes
    through is created owner-only, so without carrying them over a file would
    come back more restricted than the one it replaced.
    """
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
    """Flush every file and directory under ``root`` before it is renamed.

    For a caller that stages a whole directory and then publishes it with one
    ``os.replace``: the rename is only crash-safe if the contents reached disk
    first, and a directory built from several writes has no single point to flush.

    Files published through :func:`atomic_write_bytes` are already durable, so
    this exists for the ones that are not -- a ``shutil.copy2`` of an agent's
    file, for instance -- and re-flushing the rest costs a no-op syscall rather
    than a second write.
    """
    for directory, _subdirectories, filenames in os.walk(root):
        current = Path(directory)
        for filename in filenames:
            descriptor = os.open(str(current / filename), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(current)
