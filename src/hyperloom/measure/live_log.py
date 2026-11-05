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
from pathlib import Path

__all__ = ["LINE_CHARS", "LIVE_LOGS", "clean_line", "last_output"]

#: The logs of a running test that show whether it still makes progress; the one written last is reported.
LIVE_LOGS = frozenset({"server.log", "benchmark_stderr.log", "benchmark_stdout.log"})
#: How much of the end of a log is read: a progress line has its newest figures last.
_TAIL_BYTES = 64 * 1024
LINE_CHARS = 300
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
    """
    found = []
    for root, _, names in os.walk(directory):  # a directory that goes away during the walk is skipped, not raised
        for path in (Path(root) / name for name in LIVE_LOGS.intersection(names)):
            with contextlib.suppress(OSError):
                status = path.lstat()
                if stat.S_ISREG(status.st_mode):
                    found.append((status.st_mtime, path))
    if not found:
        return None
    written, newest = max(found)
    tail = _tail(newest)
    return None if tail is None else (max(0.0, now - written), clean_line(tail))


def _tail(path: Path) -> str | None:
    """The last ``_TAIL_BYTES`` of the regular file at ``path``; ``None`` for a link, anything else or an unreadable one."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)  # a pipe put there must not block
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
