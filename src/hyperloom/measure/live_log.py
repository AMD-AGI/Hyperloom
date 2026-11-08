# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The last thing a running test said, read from the logs it writes.

The local toolbox reads it from the sandbox's directory; the runner reads it from its own output directory and tells the
application through ``progress.json``, and the application cleans what it is told again, as the runner's output is
whatever the measured program printed.
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LINE_CHARS", "LIVE_LOGS", "clean_line", "last_output", "last_output_below"]

#: The logs of a running test that show whether it still makes progress; the one written last is reported.
LIVE_LOGS = frozenset({"server.log", "benchmark_stderr.log", "benchmark_stdout.log"})
#: How much of the end of a log is read: a progress line has its newest figures last.
_TAIL_BYTES = 64 * 1024
LINE_CHARS = 300
#: How a directory below the top is opened: as a directory only, never through a link and never waiting, so one the
#: measured program swaps for a link, a pipe or a file after it was listed fails to open (a pipe opened to read would
#: wait for a writer).
_BELOW = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
#: How many levels below the top are walked: the logs lie a few down (Magpie's server in a directory of its own below
#: the run's), and the walk holds a descriptor for each level of a tree the measured program may have made deep.
_MAX_DEPTH = 16
_ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")


def clean_line(text: str) -> str:
    """The last non-empty line of ``text`` with terminal escapes and control characters removed, shortened to its last
    ``LINE_CHARS`` characters; empty when there is none."""
    for line in reversed(_ESCAPES.sub("", text).splitlines()):
        cleaned = _CONTROL_CHARS.sub(" ", line).strip()
        if cleaned:
            return cleaned[-LINE_CHARS:]
    return ""


def last_output(directory: Path, now: float) -> tuple[float, str] | None:
    """``(seconds since it was written, its last non-empty line)`` of the live log below ``directory`` written last.

    ``None`` when there is none or it cannot be read. The line is cleaned as :func:`clean_line` does. Only regular files
    are read, never a link: the logs are written by the measured program, which could name any file the reader can read.
    ``directory`` itself is opened by its path; below it, as :func:`last_output_below` walks.
    """
    try:
        top = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        return last_output_below(top, now)
    except OSError:
        return None
    finally:
        os.close(top)


def last_output_below(dir_fd: int, now: float) -> tuple[float, str] | None:
    """:func:`last_output` of the directory open at ``dir_fd`` (left open), walked by descriptor (:func:`_directories`):
    each directory is opened from the one above it as a directory only, following no link and never waiting, so one
    the measured program swaps for a link, a pipe or a file during the walk is skipped, and the log is opened from its
    own directory's descriptor.

    Raises:
        OSError: A directory could not be listed, or this process ran out of descriptors.
    """
    newest: _Log | None = None
    try:
        with contextlib.closing(_directories(dir_fd, ".", 0)) as walk:
            for root, names, directory in walk:
                newest = _newer(newest, root, names, directory)
        tail = None if newest is None else _tail(newest.name, dir_fd=newest.directory)
    finally:
        if newest is not None:
            os.close(newest.directory)
    if newest is None or tail is None:
        return None
    return max(0.0, now - newest.written), clean_line(tail)


def _directories(directory: int, root: str, depth: int) -> Generator[tuple[str, list[str], int], None, None]:
    """``(root, the names in it, its descriptor)`` of the directory open at ``directory`` (at ``root``, ``depth``
    levels below the top) and of each directory below it down to :data:`_MAX_DEPTH`, top-down. Each is opened from
    the one above it by :data:`_BELOW` and listed by its descriptor, which is open until the next step; one that is
    gone or no longer a directory when it is opened is skipped."""
    names, below = _listed(directory)
    yield root, names, directory
    if depth >= _MAX_DEPTH:
        return
    for name in below:
        try:
            opened = os.open(name, _BELOW, dir_fd=directory)
        except OSError:  # gone, or a link, a pipe or a file put in its place since it was listed
            continue
        try:
            yield from _directories(opened, os.path.join(root, name), depth + 1)
        finally:
            os.close(opened)


def _listed(directory: int) -> tuple[list[str], list[str]]:
    """The names in the directory open at ``directory``, and those of them that were directories as it was listed."""
    with os.scandir(directory) as entries:
        listed = list(entries)
    below = sorted(entry.name for entry in listed if entry.is_dir(follow_symlinks=False))
    return [entry.name for entry in listed], below


@dataclass(frozen=True)
class _Log:
    """A live log found by the walk: when it was ``written``, its ``path`` below the top, and its ``name`` in the
    directory open at ``directory`` (a descriptor of its own, which its finder closes)."""

    written: float
    path: str
    directory: int
    name: str


def _newer(newest: _Log | None, root: str, names: list[str], directory: int) -> _Log | None:
    """``newest``, or the regular live log among ``names`` (in the directory open at ``directory``, at ``root``)
    written after it, then holding a descriptor of its own of ``directory`` (``newest``'s is closed)."""
    for name in sorted(LIVE_LOGS.intersection(names)):
        try:
            status = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except OSError:
            continue
        found = (status.st_mtime, os.path.join(root, name))
        if stat.S_ISREG(status.st_mode) and (newest is None or found > (newest.written, newest.path)):
            held = os.dup(directory)
            if newest is not None:
                os.close(newest.directory)
            newest = _Log(*found, held, name)
    return newest


def _tail(path: Path | str, *, dir_fd: int | None = None) -> str | None:
    """The last ``_TAIL_BYTES`` of the regular file at ``path`` (relative to ``dir_fd`` when given); ``None`` for a
    link, anything else or an unreadable one."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)  # a pipe must not block
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
