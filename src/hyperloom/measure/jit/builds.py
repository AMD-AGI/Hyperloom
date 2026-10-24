# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a server's log says aiter built while it ran: the hit counter and the compile time of a test (design 4.21).

aiter prints ``start build <path>`` and ``finish build <path>, cost <seconds>s`` for every module or library it compiles
and nothing for one it finds already built, so a test that found its modules in the cache has no build lines. Ranks of
one server build the same module at the same time (one compiles, the others wait for its lock), so the compile time is
the longest cost seen for each module, summed over modules; the count is of distinct modules.

The compile time is not a phase of the test: aiter builds while the server captures its graphs, so a phase would count
those seconds twice.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

__all__ = ["BuildLog", "scan_builds", "server_logs"]

SERVER_LOG = "server.log"
_MAX_LOG_BYTES = 256 * 1024**2
_MAX_LOGS = 64
_NAME = r"(?:\[([A-Za-z0-9_.-]+)\]|(\S+?))"
_START = re.compile(rf"start build {_NAME}(?:\s|$)")
_FINISH = re.compile(rf"finish build {_NAME},? cost:? (\d+(?:\.\d+)?)s")
#: aiter reports a failed build as one log record that opens and closes with its tag: ``failed jit build [<name>]`` and a
#: run of down arrows, the traceback with the compiler's output, then the tag again with up arrows.
_FAILED_OPEN = re.compile(r"failed jit build \[([A-Za-z0-9_.-]+)\]\u2193")
_FAILED_CLOSE = re.compile(r"failed jit build \[([A-Za-z0-9_.-]+)\]\u2191")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
#: A clang diagnostic (``file:line:column: error: message``) and, for a failed code generator, the exception it ends with.
_DIAGNOSTIC = re.compile(r"(?P<file>\S+\.\w+):(?P<line>\d+)(?::\d+)?: (?:fatal )?error: (?P<message>.+)")
_EXCEPTION = re.compile(r"^\s*(?:[\w.]+\.)?\w*(?:Error|Exception)\b: .+")
_MAX_DIAGNOSTICS = 3
_MAX_RECORD_LINES = 5000
_MAX_EXCERPT_LINE = 240


@dataclass(frozen=True)
class BuildLog:
    """The modules aiter built in a test, and the seconds each took, as the longest the log says."""

    seconds: dict[str, float]
    started: frozenset[str]
    #: The modules whose build failed, each with what the compiler said: ``file:line: message`` of its first errors.
    failed: Mapping[str, str] = field(default_factory=dict)

    @property
    def builds_seen(self) -> int:
        return len(self.started | set(self.seconds))

    @property
    def compile_seconds(self) -> float:
        return sum(self.seconds.values())


def server_logs(root: Path) -> list[Path]:
    """The server logs below ``root`` (one per run of each round), in a stable order and no more than a test can have."""
    found = sorted(path for path in Path(root).rglob(SERVER_LOG) if path.is_file() and not path.is_symlink())
    return found[:_MAX_LOGS]


def scan_builds(logs: Iterable[Path]) -> BuildLog:
    """Read the build lines of ``logs``, line by line, each log up to 256 MiB."""
    seconds: dict[str, float] = {}
    started: set[str] = set()
    failed: dict[str, str] = {}
    for log in logs:
        try:
            with open(log, "rb") as handle:
                _scan(handle, seconds, started, failed)
        except OSError:
            continue
    return BuildLog(seconds, frozenset(started), failed)


def _scan(handle: BinaryIO, seconds: dict[str, float], started: set[str], failed: dict[str, str]) -> None:
    budget = _MAX_LOG_BYTES
    record: _FailedBuild | None = None
    for raw in handle:
        budget -= len(raw)
        if budget < 0:
            return
        line = raw.decode("utf-8", errors="replace")
        if record is not None:
            if record.take(line):
                failed.setdefault(record.module, record.excerpt())
                record = None
            continue
        if "build" not in line:
            continue
        opened = _FAILED_OPEN.search(_ANSI.sub("", line))
        if opened:
            record = _FailedBuild(opened.group(1))
            continue
        begun = _START.search(line)
        if begun:
            started.add(_module(begun))
        done = _FINISH.search(line)
        if done:
            name = _module(done)
            seconds[name] = max(seconds.get(name, 0.0), float(done.group(3)))


class _FailedBuild:
    """One failed-build record being read: what the compiler said, until the record closes."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._said: list[str] = []
        self._lines = 0

    def take(self, raw: str) -> bool:
        """Read a line of the record; ``True`` once the record is over (its closing tag, or more lines than a build says)."""
        line = _ANSI.sub("", raw)
        self._lines += 1
        found = _DIAGNOSTIC.search(line)
        if found and len(self._said) < _MAX_DIAGNOSTICS:
            self._said.append(f"{_short(found['file'])}:{found['line']}: {found['message']}")
        elif _EXCEPTION.match(line) and len(self._said) < _MAX_DIAGNOSTICS:
            self._said.append(line.strip())
        closed = _FAILED_CLOSE.search(line)
        return bool(closed and closed.group(1) == self.module) or self._lines >= _MAX_RECORD_LINES

    def excerpt(self) -> str:
        said = "; ".join(_UNPRINTABLE.sub(" ", text)[:_MAX_EXCERPT_LINE] for text in self._said)
        return said or "the build failed and the log holds no compiler message"


def _short(path: str) -> str:
    """The last two components of a file the compiler named: its directory is the sandbox's, the rest says which file."""
    return posixpath.join(*Path(path).parts[-2:])


def _module(match: re.Match[str]) -> str:
    """The module's name: the bracketed ``md_name`` aiter's builder prints, or the last directory of a cpp_itfs path."""
    return match.group(1) or Path(match.group(2)).name
