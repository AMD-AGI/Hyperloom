# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Symlink-proof file access under a directory the caller does not control.

A staging directory sits in a container that other processes can write to, so a verified file must be read through the
same descriptor that was checked: every path component is opened with ``O_NOFOLLOW`` relative to its parent descriptor,
which makes a symlink anywhere on the path a refusal instead of a redirect. Linux only (``dir_fd`` support).
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Iterator
from pathlib import Path

CHUNK = 1024 * 1024
_RDONLY = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


class UnsafeEntry(Exception):
    """A path holds a symlink or something that is not a regular file or directory."""

    def __init__(self, path: str, kind: str) -> None:
        super().__init__(f"{path}: {kind}")
        self.path = path
        self.kind = kind


class MissingEntry(Exception):
    """A path that should exist does not."""


def _open_step(name: str, flags: int, dir_fd: int, shown: str) -> int:
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        raise MissingEntry(shown) from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
            raise UnsafeEntry(shown, "symlink or not a directory on the path") from None
        raise


def open_root(root: Path) -> int:
    """A descriptor for the directory itself; refuses a symlink in the last component."""
    try:
        return os.open(os.fspath(root), _RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        raise MissingEntry(os.fspath(root)) from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafeEntry(os.fspath(root), "symlink or not a directory") from None
        raise


def open_beneath(root_fd: int, rel: str) -> int:
    """Open the regular file ``rel`` below ``root_fd`` without following any symlink; the caller closes the result."""
    parts = rel.split("/")
    held: list[int] = []
    try:
        current = root_fd
        for part in parts[:-1]:
            current = _open_step(part, _RDONLY | os.O_DIRECTORY, current, rel)
            held.append(current)
        fd = _open_step(parts[-1], _RDONLY | os.O_NONBLOCK, current, rel)
    finally:
        for descriptor in held:
            os.close(descriptor)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeEntry(rel, "not a regular file")
    return fd


def stream(fd: int) -> Iterator[bytes]:
    """The remaining bytes of ``fd`` in chunks."""
    while chunk := os.read(fd, CHUNK):
        yield chunk


def sha256_and_size(fd: int) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    for chunk in stream(fd):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def walk_regular_files(root_fd: int, prefix: str = "") -> Iterator[str]:
    """Relative paths of every regular file below ``root_fd``; a symlink or special file raises :class:`UnsafeEntry`."""
    with os.scandir(root_fd) as entries:
        listing = sorted(entries, key=lambda entry: entry.name)
    for entry in listing:
        rel = f"{prefix}{entry.name}"
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise UnsafeEntry(rel, "symlink")
        if stat.S_ISDIR(mode):
            child = _open_step(entry.name, _RDONLY | os.O_DIRECTORY, root_fd, rel)
            try:
                yield from walk_regular_files(child, f"{rel}/")
            finally:
                os.close(child)
        elif stat.S_ISREG(mode):
            yield rel
        else:
            raise UnsafeEntry(rel, "special file")
