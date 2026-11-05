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
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

CHUNK = 1024 * 1024
#: Modes of what this module creates, stated so the result never depends on the umask: owner only.
FILE_MODE = 0o600
DIR_MODE = 0o700
#: How many directories deep :func:`walk_regular_files` goes; a directory below that is refused, so whoever wrote the
#: tree bounds neither the walk's memory nor the descriptors it holds open (one per level).
MAX_DEPTH = 128
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
    *parents, name = rel.split("/")
    directory = open_dir_beneath(root_fd, parents) if parents else None
    try:
        fd = _open_step(name, _RDONLY | os.O_NONBLOCK, root_fd if directory is None else directory, rel)
    finally:
        if directory is not None:
            os.close(directory)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise UnsafeEntry(rel, "not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_dir_beneath(root_fd: int, parts: list[str], *, created: list[str] | None = None) -> int:
    """Open the directory ``parts`` below ``root_fd``, creating missing ones when ``created`` is given.

    A symlink (or a file) in any component raises :class:`UnsafeEntry`; a missing one raises :class:`MissingEntry`
    unless ``created`` is given, in which case the directory is made and its relative path appended to ``created``.
    The caller closes the result.
    """
    current = os.dup(root_fd)
    shown: list[str] = []
    try:
        for part in parts:
            shown.append(part)
            rel = "/".join(shown)
            try:
                child = _open_step(part, _RDONLY | os.O_DIRECTORY, current, rel)
            except MissingEntry:
                if created is None:
                    raise
                os.mkdir(part, DIR_MODE, dir_fd=current)
                created.append(rel)
                child = _open_step(part, _RDONLY | os.O_DIRECTORY, current, rel)
            os.close(current)
            current = child
    except BaseException:
        os.close(current)
        raise
    return current


def create_exclusive(dir_fd: int, name: str) -> int:
    """Create a new file ``name`` in ``dir_fd`` for writing; never follows a symlink, never replaces anything."""
    return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, FILE_MODE, dir_fd=dir_fd)


def exists_beneath(root_fd: int, rel: str) -> bool:
    """Whether ``rel`` exists below ``root_fd`` (a dangling symlink counts); a symlinked parent raises."""
    *parents, name = rel.split("/")
    try:
        parent = open_dir_beneath(root_fd, parents)
    except MissingEntry:
        return False
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    finally:
        os.close(parent)
    return True


def write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` to ``fd``.

    ``os.write`` may write fewer bytes than it was given and still succeed: a file that reaches the size limit of the
    process, or a filesystem that fills up, takes what fits and returns the count. The rest is written again, which
    then fails with the error that stopped it, so a short write is never taken for a whole one.
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "a write made no progress")
        view = view[written:]


def stream(fd: int, limit: int | None = None) -> Iterator[bytes]:
    """The remaining bytes of ``fd`` in chunks; with ``limit``, no more than ``limit + 1`` of them.

    One byte past ``limit`` is enough for the caller to refuse a file that is longer than it should be, so a file that
    grows (or keeps growing) is never read further than that.
    """
    left = sys.maxsize if limit is None else limit + 1
    while left > 0 and (chunk := os.read(fd, min(CHUNK, left))):
        left -= len(chunk)
        yield chunk


def sha256_and_size(fd: int, limit: int | None = None) -> tuple[str, int]:
    """The sha256 and size of the remaining bytes of ``fd``, read as :func:`stream` reads them."""
    digest, size = hashlib.sha256(), 0
    for chunk in stream(fd, limit):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def walk_regular_files(root_fd: int) -> Iterator[str]:
    """Relative paths of every regular file below ``root_fd``, each directory's entries in name order.

    A symlink or special file raises :class:`UnsafeEntry`, and so does a directory more than :data:`MAX_DEPTH` levels
    down; an entry that goes away while it is walked raises :class:`MissingEntry`. The walk keeps its own stack instead
    of recursing, so no depth of the tree reaches Python's recursion limit.
    """
    levels = [_Level(root_fd, "", _entries(root_fd))]
    try:
        while levels:
            level = levels[-1]
            entry = next(level.entries, None)
            if entry is None:
                levels.pop()
                if levels:  # the caller's own descriptor stays open
                    os.close(level.fd)
                continue
            rel = f"{level.prefix}{entry.name}"
            mode = _mode(entry, rel)
            if stat.S_ISLNK(mode):
                raise UnsafeEntry(rel, "symlink")
            if stat.S_ISDIR(mode):
                if len(levels) > MAX_DEPTH:
                    raise UnsafeEntry(rel, f"more than {MAX_DEPTH} directories deep")
                levels.append(_open_level(level.fd, entry.name, rel))
            elif stat.S_ISREG(mode):
                yield rel
            else:
                raise UnsafeEntry(rel, "special file")
    finally:
        for level in levels[1:]:
            os.close(level.fd)


class _Level(NamedTuple):
    """A directory the walk is in: its descriptor, the relative path its entries start with, and those left to see."""

    fd: int
    prefix: str
    entries: Iterator[os.DirEntry[str]]


def _entries(dir_fd: int) -> Iterator[os.DirEntry[str]]:
    with os.scandir(dir_fd) as entries:
        return iter(sorted(entries, key=lambda entry: entry.name))


def _mode(entry: os.DirEntry[str], rel: str) -> int:
    try:
        return entry.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise MissingEntry(rel) from None


def _open_level(parent_fd: int, name: str, rel: str) -> _Level:
    child = _open_step(name, _RDONLY | os.O_DIRECTORY, parent_fd, rel)
    try:
        return _Level(child, f"{rel}/", _entries(child))
    except BaseException:
        os.close(child)
        raise
