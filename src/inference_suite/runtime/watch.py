# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's stop gates: what ends a round, or the server it runs against, before it finishes.

:class:`RoundGate` holds the round to its cancel, its absolute deadline and its budget. :class:`ServerWatch` reads the
server's log as it grows: before ready, an exit, the boot budget and pre-ready silence end the boot; once live, a
death or a silence ends the round. Each gate that trips names a :class:`FailedStep`. A server whose log's last line
says aiter started building a JIT module is working, for as long as a module of its kind may take to build, and that
time does not count against the boot budget.

Before ready, a server whose processes keep using CPU is working too, however long its log stays quiet: a
large-checkpoint load can spend half an hour filling host memory without a line. Only the boot budget bounds such a
boot; a server that neither writes nor uses CPU for the pre-ready silence limit is hung.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inference_suite.contracts.config import Deadlines
from inference_suite.contracts.report import FailedStep
from inference_suite.runtime.process import Spawned
from inference_suite.server_log import LogCursor, death_excerpt

#: How often a wait looks up from what it waits on to check its gates.
POLL_SEC = 0.5

#: Asked at each poll; a non-empty answer is why the round was cancelled.
Cancel = Callable[[], str]

_AITER_BUILD = re.compile(r"\[aiter\] (?:\[[^\]]*\] )*start build (?:\[([A-Za-z0-9_.-]+)\])?")
#: How long one quiet build counts as activity: ``MODULE_BUILD_LIMIT_SEC`` and ``CKTILE_MODULE_BUILD_LIMIT_SEC`` of
#: ``hyperloom.measure.jit.modules``.
_BUILD_LIMIT_SEC = 600.0
_CKTILE_BUILD_LIMIT_SEC = 2700.0
_CKTILE = "cktile"

#: Reads the CPU seconds, user and system, the server's processes have used so far; ``None`` when they cannot be read.
CpuReader = Callable[[], "float | None"]
#: How often, at most, a boot's CPU time is read: each read lists ``/proc``.
CPU_SAMPLE_SEC = 10.0
#: The share of one CPU a boot's processes must use, averaged between two reads, for the stretch to count as work.
#: A server rank that only waits (its event loop, heartbeat and watchdog threads polling) uses a few percent of a CPU,
#: and a boot runs one to a few dozen such processes, the larger counts mostly helpers that sleep; a phase that loads a
#: checkpoint keeps at least one CPU per rank busy. A quarter of a CPU sits above the one and well below the other.
CPU_ACTIVE_SHARE = 0.25


@dataclass(frozen=True)
class Stop:
    """Why a round, or its server, stopped before it finished."""

    step: FailedStep
    detail: str


@dataclass(frozen=True)
class Exited:
    returncode: int


@dataclass(frozen=True)
class RoundGate:
    """The gates every wait in a round checks: cancel, the absolute deadline and the round budget."""

    deadlines: Deadlines
    cancel: Cancel
    started: float

    def check(self, now: float) -> Stop | None:
        reason = self.cancel()
        if reason:
            return Stop(FailedStep.CANCELLED, reason)
        deadline = self.deadlines.deadline_unix
        if deadline is not None and time.time() >= deadline:
            return Stop(FailedStep.DEADLINE_EXHAUSTED, f"the absolute deadline {deadline} has passed")
        elapsed = now - self.started
        if elapsed >= self.deadlines.round_sec:
            return Stop(
                FailedStep.ROUND_TIMEOUT, f"the round ran {elapsed:.1f}s of a {self.deadlines.round_sec}s budget"
            )
        return None


@dataclass(frozen=True)
class _Build:
    module: str
    since: float

    def counted(self, now: float) -> float:
        """The seconds of this build that count as the server's activity."""
        limit = _CKTILE_BUILD_LIMIT_SEC if _CKTILE in self.module else _BUILD_LIMIT_SEC
        return min(now - self.since, limit)


class ServerWatch:
    """What the server's log has said since ``offset``, and the server gates it decides."""

    def __init__(
        self, log_path: Path, offset: int, deadlines: Deadlines, now: float, cpu: CpuReader | None = None
    ) -> None:
        self._log = str(log_path)
        self._cursor = LogCursor({self._log: offset})
        self._deadlines = deadlines
        self._started = now
        self._last_output = now
        self._build: _Build | None = None
        self._built_sec = 0.0
        self.ready_logged = False
        self._cpu = cpu
        self._cpu_read_at = now
        self._cpu_used = None if cpu is None else cpu()
        self._cpu_unread = cpu is None or self._cpu_used is None
        #: When the server's processes last used CPU as a working boot does; ``None`` while they have not.
        self.cpu_active_at: float | None = None
        self._live = False

    def advance(self, now: float) -> None:
        scan = self._cursor.advance(self._log)
        self.ready_logged = self.ready_logged or scan.saw_ready
        if scan.grew:
            self._last_output = now
        if scan.server_line is not None:
            self._follow_build(scan.server_line, now)
        if not self._live:
            self._sample_cpu(now)

    def _sample_cpu(self, now: float) -> None:
        """Read the CPU time every :data:`CPU_SAMPLE_SEC`; a stretch that used :data:`CPU_ACTIVE_SHARE` of a CPU is
        activity. A read that fails, or a total that fell (a process gone before its time was counted), is none."""
        if self._cpu is None or now - self._cpu_read_at < CPU_SAMPLE_SEC:
            return
        used, previous, since = self._cpu(), self._cpu_used, self._cpu_read_at
        self._cpu_used, self._cpu_read_at, self._cpu_unread = used, now, used is None
        if used is not None and previous is not None and used - previous >= CPU_ACTIVE_SHARE * (now - since):
            self.cpu_active_at = now

    def note_output(self, now: float) -> None:
        """Count output the round's client saw as the server's activity."""
        self._last_output = max(self._last_output, now)

    def _follow_build(self, line: str, now: float) -> None:
        found = _AITER_BUILD.search(line)
        module = None if found is None else (found.group(1) or "")
        if self._build is not None and module == self._build.module:
            return
        if self._build is not None:
            self._built_sec += self._build.counted(now)
        self._build = None if module is None else _Build(module, now)

    def _quiet_sec(self, now: float) -> float:
        active = self._last_output
        if self._build is not None:
            active = max(active, self._build.since + self._build.counted(now))
        return now - active

    def _boot_sec(self, now: float) -> float:
        building = 0.0 if self._build is None else self._build.counted(now)
        return now - self._started - self._built_sec - building

    def _death(self, fallback: str) -> str:
        return death_excerpt(self._log) or fallback

    def before_ready(self, now: float, exit_code: int | None) -> Stop | None:
        if exit_code is not None:
            detail = self._death(f"the server exited with code {exit_code} before it was ready")
            return Stop(FailedStep.SERVER_EXITED_BEFORE_READY, detail)
        boot = self._boot_sec(now)
        if boot >= self._deadlines.boot_sec:
            return Stop(FailedStep.SERVER_BOOT_TIMEOUT, f"not ready after {boot:.1f}s of boot, aiter builds excluded")
        quiet = self._quiet_sec(now)
        cpu_quiet = now - (self._started if self.cpu_active_at is None else self.cpu_active_at)
        limit = self._deadlines.pre_ready_silence_sec
        if quiet >= limit and cpu_quiet >= limit:
            detail = f"the server log was silent for {quiet:.1f}s and {self._cpu_evidence(cpu_quiet)}"
            return Stop(FailedStep.SERVER_SILENT_BEFORE_READY, detail)
        return None

    def _cpu_evidence(self, cpu_quiet: float) -> str:
        if self._cpu is None:
            return "its CPU time was not watched"
        if self._cpu_unread:
            return "its CPU time could not be read"
        return f"its processes used under {CPU_ACTIVE_SHARE:g} of a CPU for {cpu_quiet:.1f}s"

    def mark_ready(self, now: float) -> None:
        """Start the live silence window at ``now``: a boot's last quiet stretch is not the served round's, and a live
        server's CPU is no longer activity."""
        self._last_output = now
        self._live = True

    def died(self) -> Stop:
        return Stop(FailedStep.SERVER_DIED, self._death("the server exited after it was ready"))

    def stalled(self, now: float) -> Stop | None:
        quiet = self._quiet_sec(now)
        if quiet >= self._deadlines.silence_sec:
            return Stop(FailedStep.SERVER_STALLED, f"no server log or client output for {quiet:.1f}s")
        return None


#: A live server's gate, checked at each poll of a wait; returns why the round has to stop, if it has to.
LiveCheck = Callable[[float], "Stop | None"]


def wait_child(child: Spawned, gate: RoundGate, live: LiveCheck | None = None) -> Exited | Stop:
    """Wait for ``child`` to exit while the gates hold; it and whatever carries its mark are reaped either way."""

    try:
        while True:
            returncode = child.proc.poll()
            if returncode is not None:
                return Exited(returncode)
            now = time.monotonic()
            stop = gate.check(now) or (None if live is None else live(now))
            if stop is not None:
                return stop
            try:
                child.proc.wait(timeout=POLL_SEC)
            except subprocess.TimeoutExpired:
                continue
    finally:
        child.reap()


__all__ = [
    "CPU_ACTIVE_SHARE",
    "CPU_SAMPLE_SEC",
    "POLL_SEC",
    "Cancel",
    "CpuReader",
    "Exited",
    "LiveCheck",
    "RoundGate",
    "ServerWatch",
    "Stop",
    "wait_child",
]
