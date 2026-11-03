# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A limit on how long a test's inference server may take to report ready.

The silence watchdog of the benchmark launcher starts only when the server's ready banner appears. A server that never
gets ready (a boot that tunes kernels for an hour) is therefore ended by the round's hard cap alone, and the same
candidate repeats it on the rerun. The guard here ends such a round at a share of that cap: it watches for the stamp the
launcher writes beside a round's server log the moment the banner appears and, when none has come within the limit,
cancels the scope the launcher's watchdog listens to, which reaps the whole process tree.

A server that builds an aiter JIT module is not tuning: aiter writes ``start build [<module>]`` and nothing more until
the module is built, and a CK-tile GEMM took up to 1347 s with eight tests building at once (#169), which with a CK GEMM
and the rest of a boot overran half of a 3600 s round. So the time the newest server log's last line says aiter is
building a module is not counted against the limit, each build for at most as long as a module of its kind may take
(:func:`~hyperloom.measure.jit.modules.module_build_limit_sec`: 2700 s for a CK-tile module, 600 s for any other). A
build that hangs counts again from there, and a boot that does anything else is bounded as before.

The guard watches a round's server boot only, so a test that hangs anywhere else before ready (a runner that stopped
making progress, a round that never launched its server) waited out the whole ``timeout_sec``. :class:`PreReadySilence`
bounds, on the application's side, how long a hosted test may go without output before its server is ready. Measured: one integration test sat 60 minutes with an empty attempt directory, holding the session's only integration
lane, and a second one 54 minutes until it was cancelled by hand.

The application reads the runner's ``progress.json`` on every poll. Each change of what it says the test did (its phase,
the last output it saw, the newest live log's write time or last line) is activity; once the runner says a round's server
is ready (:data:`SERVER_READY_KEY`), or the test is past running, the bound no longer applies. A test with no activity
for :data:`PRE_READY_SILENCE_SEC` before that ends as ``hung_before_ready``, an infrastructure failure a new sandbox
retries. A server whose last line says aiter started building a JIT module is working, as aiter writes nothing until the
module is built: with eight tests of a session each building its own copy, the FP8 GEMM module
``module_gemm_a8w8_bpreshuffle_cktile`` took up to 844 s in the tests that finished it. While that line stands, only
the boot guard bounds the boot.

The limit comes from the server logs of 1857 test rounds (1711 reached ready): the longest stretch before ready with
no output line, timestamped or carrying a progress figure, was 262 s, an aiter JIT build that reported ``cost 273``
seconds itself; the 99.9th percentile was 144 s, and the slowest phase before a server starts (``env_rebuild``) took
at most 27 s in 599 tests. 900 s is over three times the longest legitimate silence and ends a hang 45 minutes sooner.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.measure.jit.modules import module_build_limit_sec
from hyperloom.orchestrator.actions.cancel_channel import CancelScope

__all__ = [
    "BOOT_CANCEL_REASON",
    "BOOT_SHARE",
    "BuildTime",
    "HUNG_BEFORE_READY",
    "PRE_READY_SILENCE_SEC",
    "SERVER_READY_KEY",
    "BootGuard",
    "PreReadySilence",
    "boot_limit_sec",
    "boot_seconds",
    "building_module",
    "read_ready_stamp",
    "server_ready",
]

#: The ``reason`` the guard cancels the scope with.
BOOT_CANCEL_REASON = "server_boot_timeout"
#: The share of a round's time a boot may take. Boot, the warmup, the measured round and the accuracy evaluation all
#: run inside one cap, so a boot that has used half of it leaves the rest less than the boot took.
BOOT_SHARE = 0.5
#: The frameworks whose server prints a ready banner the launcher detects; another framework has no stamp to wait for.
_SERVER_FRAMEWORKS = ("sglang", "vllm", "atom")
#: Written by the launcher next to ``server.log`` (``_subprocess_kill``); its first field is the wall-clock second the
#: server reported ready, its second the seconds the boot took from the spawn (0 for a server the round found ready).
_READY_STAMP = "server_ready_at"
_MAX_POLL_SEC = 2.0
#: The launcher's log of a round's server, beside its ready stamp.
_SERVER_LOG = "server.log"
#: How much of the end of a server log is read for its last line.
_TAIL_BYTES = 8192


def boot_limit_sec(benchmark: Mapping[str, Any], timeout_sec: float) -> float | None:
    """How long the server of a round on ``benchmark`` may take to be ready; ``None`` when no stamp can say it is.

    The budget is the one the point carries (``server_lifecycle.server_ready_timeout_s``), never more than the round's
    ``timeout_sec``; a point that carries none gets the cap itself, which is what the launcher writes into that field.
    """
    if str(benchmark.get("framework") or "").lower() not in _SERVER_FRAMEWORKS:
        return None
    lifecycle = benchmark.get("server_lifecycle")
    carried = lifecycle.get("server_ready_timeout_s") if isinstance(lifecycle, Mapping) else None
    budget = float(carried) if isinstance(carried, (int, float)) and carried > 0 else timeout_sec
    return BOOT_SHARE * min(budget, timeout_sec)


class BootGuard:
    """Cancels ``scope`` when a watched round's server is not ready within its limit."""

    def __init__(self, output_dir: Path, scope: CancelScope) -> None:
        self._output_dir = output_dir
        self._scope = scope
        self._expired_after: float | None = None
        self._waited = 0.0
        self._excused = 0.0

    @property
    def expired_after(self) -> float | None:
        """The limit, in seconds, that ended a round; ``None`` while none has."""
        return self._expired_after

    @property
    def waited_sec(self) -> float:
        """The seconds the round that expired had waited for its server, aiter's module builds included."""
        return self._waited

    @property
    def building_sec(self) -> float:
        """Of :attr:`waited_sec`, the seconds that were not counted against the limit because aiter was building a module."""
        return self._excused

    @contextlib.contextmanager
    def watching(self, limit_sec: float) -> Iterator[None]:
        """Watch the round that runs inside the block."""
        started = time.time()
        stop = threading.Event()
        poll = min(_MAX_POLL_SEC, max(0.05, limit_sec / 20))
        builds = BuildTime(started)

        def watch() -> None:
            while not stop.wait(poll):
                if self._ready_since(started):
                    return
                now = time.time()
                builds.observe(now, self._building(started))
                if now - started - builds.excused >= limit_sec:
                    self._expired_after, self._waited, self._excused = limit_sec, now - started, builds.excused
                    self._scope.cancel(reason=BOOT_CANCEL_REASON)
                    return

        thread = threading.Thread(target=watch, name="hl-boot-watch", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=poll + 1)

    def _ready_since(self, started: float) -> bool:
        try:
            stamps = list(self._output_dir.rglob(_READY_STAMP))
        except OSError:
            return False
        return any(_stamped_at(stamp) >= started for stamp in stamps)

    def _building(self, started: float) -> str | None:
        """The module aiter is building, as the last line of the newest server log written since ``started`` says (an
        earlier round's is not this boot's); ``None`` when it says no build started, or no such log can be read."""
        try:
            logs = [(path.stat().st_mtime, str(path), path) for path in self._output_dir.rglob(_SERVER_LOG)]
        except OSError:
            return None
        logs = [log for log in logs if log[0] >= started]
        if not logs:
            return None
        return building_module(_last_line(max(logs)[2]))


@dataclass
class BuildTime:
    """The seconds of a boot that aiter spent building modules, as readings of what it is building say, each build
    excused for at most :func:`~hyperloom.measure.jit.modules.module_build_limit_sec` of its module."""

    last: float
    excused: float = 0.0
    module: str | None = None
    since: float = 0.0

    def observe(self, now: float, building: str | None) -> None:
        """Take one reading, at ``now``: the module being built, or ``None``. The time since the previous reading is
        excused when a build was under way through it, as far as the build has not yet had its time."""
        if building is not None:
            if building != self.module:
                self.module, self.since = building, self.last
            until = self.since + module_build_limit_sec(building)
            self.excused += max(0.0, min(now, until) - self.last)
        else:
            self.module = None
        self.last = now


def building_module(line: str) -> str | None:
    """The module a line in which aiter says it started a build names; ``""`` for one it names by a path that is not a
    module's name, ``None`` for a line that says no build started."""
    found = _JIT_BUILD_MODULE.search(line)
    if found is None:
        return None
    return found.group(1) or ""


def _last_line(path: Path) -> str:
    """The last line with text of the file at ``path``; empty when it has none or cannot be read."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _TAIL_BYTES))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _stamped_at(stamp: Path) -> float:
    """The wall-clock second a stamp says its server was ready; ``0.0`` for one that cannot be read."""
    try:
        return float(stamp.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def boot_seconds(directory: Path) -> float | None:
    """The longest server boot the ready stamps below ``directory`` record, or ``None`` when no round booted a server.

    A test boots its server once, in its first round, and its later rounds reuse it and stamp no boot, so the longest
    one is the boot of the test. A stamp that cannot be read is no boot.
    """
    try:
        stamps = list(directory.rglob(_READY_STAMP))
    except OSError:
        return None
    booted = [seconds for seconds in map(_booted_in, stamps) if seconds > 0.0]
    return max(booted, default=None)


def server_ready(directory: Path) -> bool:
    """Whether a round below ``directory`` stamped its server ready; a stamp that cannot be read is not ready."""
    try:
        stamps = list(directory.rglob(_READY_STAMP))
    except OSError:
        return False
    return any(_stamped_at(stamp) > 0.0 for stamp in stamps)


def read_ready_stamp(directory: Path) -> tuple[float, float] | None:
    """The ``(ready_unix, boot_seconds)`` the stamp beside a round's server log records, or ``None`` when the round left
    none or it cannot be read; ``boot_seconds`` is 0 for a round that found its server ready."""
    stamp = directory / _READY_STAMP
    ready = _stamped_at(stamp)
    return (ready, _booted_in(stamp)) if ready > 0.0 else None


def _booted_in(stamp: Path) -> float:
    """The seconds the boot a stamp records took; ``0.0`` for a stamp that records none or cannot be read."""
    try:
        seconds = float(stamp.read_text(encoding="utf-8").split()[1])
    except (OSError, ValueError, IndexError):
        return 0.0
    return seconds if math.isfinite(seconds) else 0.0


#: The ``progress.json`` key the runner sets once a round's server has reported ready.
SERVER_READY_KEY = "server_ready"
#: The ``error_class`` of a test that went silent before its server was ready.
HUNG_BEFORE_READY = "hung_before_ready"
PRE_READY_SILENCE_SEC = 900.0
#: The runner's phase once the measurement has ended; nothing after it is bounded here.
_COLLECTING = "collecting"
#: A log written within this many seconds of the last write seen is the same write, seen again through the clock jitter
#: of ``updated_unix - last_log_age_sec``.
_WRITE_RESOLUTION_SEC = 1.0
#: The line aiter writes as it starts building a JIT module, with or without the process tag it puts first (``[aiter]
#: [pid=829812 pname=Process-1] start build [module_gemm_a8w8_bpreshuffle_cktile] under ...``), searched for anywhere in
#: the line, as the runner keeps only the end of a long one.
_JIT_BUILD_STARTED = re.compile(r"\[aiter\] (?:\[[^\]]*\] )*start build ")
#: The same, with the module it names in brackets (aiter's own form); a path in place of the name gives none.
_JIT_BUILD_MODULE = re.compile(r"\[aiter\] (?:\[[^\]]*\] )*start build (?:\[([A-Za-z0-9_.-]+)\])?")


def _log_written_at(progress: Mapping[str, Any]) -> float | None:
    updated, age = progress.get("updated_unix"), progress.get("last_log_age_sec")
    if isinstance(updated, (int, float)) and isinstance(age, (int, float)):
        return float(updated) - float(age)
    return None


@dataclass
class PreReadySilence:
    """Watches one attempt's progress, on the application's clock, from ``started``."""

    started: float
    limit_sec: float = PRE_READY_SILENCE_SEC
    _active_at: float = field(init=False)
    _seen: tuple[object, ...] | None = field(default=None, init=False)
    _written_at: float | None = field(default=None, init=False)
    _armed: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self._active_at = self.started

    def observe(self, progress: Mapping[str, Any] | None, now: float) -> None:
        """Take one reading of ``progress.json``; ``None`` for one that could not be read, which shows no activity."""
        if not self._armed or progress is None:
            return
        if progress.get(SERVER_READY_KEY) is True or progress.get("phase") == _COLLECTING:
            self._armed = False
            return
        seen = (progress.get("phase"), progress.get("last_output_unix"), progress.get("last_line"))
        written = _log_written_at(progress)
        rewritten = written is not None and (
            self._written_at is None or written > self._written_at + _WRITE_RESOLUTION_SEC
        )
        building = _JIT_BUILD_STARTED.search(str(progress.get("last_line") or "")) is not None
        if seen != self._seen or rewritten or building:
            self._active_at = now
        self._seen = seen
        if rewritten:
            self._written_at = written

    def silent_for(self, now: float) -> float | None:
        """Seconds without activity once that is over the limit before ready; ``None`` while it is not."""
        silent = now - self._active_at
        return silent if self._armed and silent >= self.limit_sec else None
