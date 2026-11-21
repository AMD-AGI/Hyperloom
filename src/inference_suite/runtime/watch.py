# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's stop gates: what ends a round, or the server it runs against, before it finishes.

:class:`RoundGate` holds the round to its cancel, its absolute deadline and its budget. :class:`ServerWatch` reads the
server's log as it grows: before ready, an exit, the boot budget and pre-ready silence end the boot; once live, a
death or a silence ends the round. Each gate that trips names a :class:`FailedStep`. A server whose log's last line
says aiter started building a JIT module is working, for as long as a module of its kind may take to build, and that
time does not count against the boot budget.

Before ready, a server whose processes keep using CPU is working too, however long its log stays quiet: a
large-checkpoint load can spend half an hour without a line, CPU-busy or IO-bound. A collective that spins uses CPU as
well, so CPU use extends the silence only up to :data:`PRE_READY_BUSY_SILENCE_SEC`. A server whose resident memory or
reads keep growing is making progress, which a spin does not: that restarts the silence like a log line, and only the
boot budget bounds it. A server that neither writes, grows nor uses CPU for the pre-ready silence limit is hung.
"""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from inference_suite.contracts.config import PRE_READY_BUSY_SILENCE_SEC, Deadlines
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
#: and a boot runs one to a few dozen such processes, the larger counts mostly helpers that sleep, so a quarter of a CPU
#: sits above a waiting boot. A checkpoint load may be CPU-busy or IO-bound: pre-faulting a file mapped from a network
#: filesystem was measured at about 0.06 of a CPU per rank, and shows only in resident memory, which
#: :data:`RSS_GROWTH_BYTES` reads.
CPU_ACTIVE_SHARE = 0.25

_MIB = 1 << 20
#: Reads a per-process counter of the server's processes (resident bytes, bytes read), keyed by process; ``None`` when
#: it cannot be read.
TreeReader = Callable[[], Mapping[object, int] | None]
#: How much the server's processes must grow their resident memory, summed over the processes seen in both reads, for
#: a :data:`CPU_SAMPLE_SEC` stretch to count as progress (scaled to the stretch's length). A process that waits holds
#: its pages (allocator churn moves it by a few MiB), while a boot that fills host memory, even from a slow network
#: filesystem, adds tens of MiB a second: 64 MiB per 10 s (6.4 MiB/s) sits above the one and well below the other.
#: Memory is read because pre-faulting a file mapped from a network filesystem shows in neither IO counter.
RSS_GROWTH_BYTES = 64 * _MIB
#: The same for the bytes the processes read (``rchar`` + ``read_bytes``): a waiting server reads a heartbeat or a
#: ``/proc`` file now and then, KiB a second, and a checkpoint read moves tens of MiB a second at the least.
IO_GROWTH_BYTES = 64 * _MIB


@dataclass(frozen=True)
class BootReaders:
    """What a boot's processes are read for between log lines: their CPU time, and the per-process counters whose
    growth is progress; ``None`` for one not watched."""

    cpu: CpuReader | None = None
    rss: TreeReader | None = None
    io: TreeReader | None = None


class _Growth:
    """One per-process counter of the server's processes whose growth is progress."""

    def __init__(self, what: str, read: TreeReader | None, per_sample: int) -> None:
        self.what = what
        self._read = read
        self._per_sample = per_sample
        self.watched = read is not None
        self.last = None if read is None else read()

    def grew(self, elapsed: float) -> bool:
        """Read again; whether the processes seen in both reads grew by the threshold over ``elapsed`` seconds. A read
        that fails is no growth, and neither is a process seen only once (a helper that starts is not progress)."""
        if self._read is None:
            return False
        current, previous = self._read(), self.last
        self.last = current
        if current is None or previous is None:
            return False
        grown = sum(current[key] - previous[key] for key in current.keys() & previous.keys())
        return grown >= self._per_sample * elapsed / CPU_SAMPLE_SEC

    def evidence(self, flat: float) -> str:
        if self.last is None:
            return f"its {self.what} could not be read"
        return f"its {self.what} grew under {self._per_sample // _MIB} MiB per {CPU_SAMPLE_SEC:g}s for {flat:.1f}s"


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
        self,
        log_path: Path,
        offset: int,
        deadlines: Deadlines,
        now: float,
        readers: BootReaders = BootReaders(),
    ) -> None:
        self._log = str(log_path)
        self._cursor = LogCursor({self._log: offset})
        self._deadlines = deadlines
        self._started = now
        self._last_output = now
        self._build: _Build | None = None
        self._built_sec = 0.0
        self.ready_logged = False
        cpu = self._cpu = readers.cpu
        self._cpu_read_at = now
        self._cpu_used = None if cpu is None else cpu()
        self._cpu_unread = cpu is None or self._cpu_used is None
        #: When the server's processes last used CPU as a working boot does; ``None`` while they have not.
        self.cpu_active_at: float | None = None
        self._growth = (
            _Growth("resident memory", readers.rss, RSS_GROWTH_BYTES),
            _Growth("IO", readers.io, IO_GROWTH_BYTES),
        )
        #: When the server's resident memory or reads last grew as a working boot's do; ``None`` while they have not.
        self.progress_at: float | None = None
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
        if now - self._cpu_read_at < CPU_SAMPLE_SEC:
            return
        since = self._cpu_read_at
        self._cpu_read_at = now
        # Every counter is read each sample, so each compares two readings one sample apart.
        grew = [growth.grew(now - since) for growth in self._growth]
        if any(grew):
            self.progress_at = now
        if self._cpu is None:
            return
        used, previous = self._cpu(), self._cpu_used
        self._cpu_used, self._cpu_unread = used, used is None
        if used is not None and previous is not None and used - previous >= CPU_ACTIVE_SHARE * (now - since):
            self.cpu_active_at = now

    @property
    def boot_active_at(self) -> float | None:
        """When the boot was last seen working without a line, by CPU use or by growth; ``None`` while it has not."""
        seen = [at for at in (self.cpu_active_at, self.progress_at) if at is not None]
        return max(seen) if seen else None

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
        flat = now - (self._started if self.progress_at is None else self.progress_at)
        # Growth restarts the silence as a log line does: neither bound below holds a boot that makes progress.
        still = min(quiet, flat)
        cpu_quiet = now - (self._started if self.cpu_active_at is None else self.cpu_active_at)
        limit = self._deadlines.pre_ready_silence_sec
        if still >= limit and cpu_quiet >= limit:
            detail = f"the server log was silent for {quiet:.1f}s{self._growth_evidence(flat, ',')} and "
            return Stop(FailedStep.SERVER_SILENT_BEFORE_READY, detail + self._cpu_evidence(cpu_quiet))
        if still >= max(limit, PRE_READY_BUSY_SILENCE_SEC):
            detail = f"the server log was silent for {quiet:.1f}s{self._growth_evidence(flat, ' and')}"
            return Stop(
                FailedStep.SERVER_SILENT_BEFORE_READY, f"{detail} while its processes stayed busy, which may be a spin"
            )
        return None

    def _growth_evidence(self, flat: float, joint: str) -> str:
        """What the memory and IO readings showed, after ``joint``; empty when neither was watched."""
        parts = [growth.evidence(flat) for growth in self._growth if growth.watched]
        return "".join(f"{joint} {part}" for part in parts)

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
    "BootReaders",
    "CPU_ACTIVE_SHARE",
    "CPU_SAMPLE_SEC",
    "IO_GROWTH_BYTES",
    "POLL_SEC",
    "RSS_GROWTH_BYTES",
    "Cancel",
    "CpuReader",
    "Exited",
    "LiveCheck",
    "RoundGate",
    "ServerWatch",
    "Stop",
    "TreeReader",
    "wait_child",
]
