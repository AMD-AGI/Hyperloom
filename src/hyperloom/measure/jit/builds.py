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

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

__all__ = ["BuildLog", "scan_builds", "server_logs"]

SERVER_LOG = "server.log"
_MAX_LOG_BYTES = 256 * 1024**2
_MAX_LOGS = 64
_NAME = r"(?:\[([A-Za-z0-9_.-]+)\]|(\S+?))"
_START = re.compile(rf"start build {_NAME}(?:\s|$)")
_FINISH = re.compile(rf"finish build {_NAME},? cost:? (\d+(?:\.\d+)?)s")


@dataclass(frozen=True)
class BuildLog:
    """The modules aiter built in a test, and the seconds each took, as the longest the log says."""

    seconds: dict[str, float]
    started: frozenset[str]

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
    for log in logs:
        try:
            with open(log, "rb") as handle:
                _scan(handle, seconds, started)
        except OSError:
            continue
    return BuildLog(seconds, frozenset(started))


def _scan(handle: BinaryIO, seconds: dict[str, float], started: set[str]) -> None:
    budget = _MAX_LOG_BYTES
    for raw in handle:
        budget -= len(raw)
        if budget < 0:
            return
        line = raw.decode("utf-8", errors="replace")
        if "build" not in line:
            continue
        begun = _START.search(line)
        if begun:
            started.add(_module(begun))
        done = _FINISH.search(line)
        if done:
            name = _module(done)
            seconds[name] = max(seconds.get(name, 0.0), float(done.group(3)))


def _module(match: re.Match[str]) -> str:
    """The module's name: the bracketed ``md_name`` aiter's builder prints, or the last directory of a cpp_itfs path."""
    return match.group(1) or Path(match.group(2)).name
