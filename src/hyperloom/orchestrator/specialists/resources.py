# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resource sampling for specialists and their container.

Per specialist: the process-tree resident memory peak and cumulative CPU seconds,
read from ``/proc``, and the disk the worktree and its self-check venv use.
Per container: cgroup v2 memory limit, peak, OOM-kill count and CPU limit. Every
reader returns ``None`` for what it cannot read instead of guessing, and nothing
here writes to or signals the processes it observes.

Records go to ``agents/resources.jsonl``; their purpose is to size the hosted
application container and to decide when CPU specialists need a sub-run, so a
missing reading must stay distinguishable from a zero.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.proctree import descendants

__all__ = [
    "CONTAINER_RECORD_KIND",
    "RESOURCES_RELATIVE_PATH",
    "SPECIALIST_RECORD_KIND",
    "ContainerSnapshot",
    "ProcessTreeSampler",
    "ResourceLog",
    "TreeSample",
    "dir_usage_bytes",
    "read_container_cgroup",
]

UTF8 = "utf-8"
DECODE_ERRORS = "replace"
RESOURCES_RELATIVE_PATH = Path("agents") / "resources.jsonl"
SPECIALIST_RECORD_KIND = "specialist_resources"
CONTAINER_RECORD_KIND = "container_resources"


@dataclass(frozen=True)
class TreeSample:
    """One reading of a process tree."""

    rss_bytes: int
    """Resident memory summed over the live processes of the tree."""
    cpu_seconds: float
    """The tree's CPU seconds so far: the largest sum yet read of its live processes' and their reaped children's."""
    live_pids: int


def _clock_ticks() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return 100


class ProcessTreeSampler:
    """Track the memory peak and CPU time of one process tree.

    The tree is the root and its descendants as :func:`hyperloom.common.proctree.descendants`
    finds them in ``/proc``; ``proc_root`` is where each member's readings are read.

    A process that exits stops counting towards memory the moment it is gone. Its CPU
    seconds reach the total through its parent: the kernel adds a child's user and
    system time, and its own reaped children's, to the parent's ``cutime``/``cstime``
    when the parent waits for it. A sample sums ``utime + stime + cutime + cstime``
    over the live processes, so a compiler that starts and exits between two samples
    still counts, and the total keeps the largest sum read, so it never goes back when
    a process leaves the tree without a member of it waiting for it.
    """

    def __init__(self, root_pid: int, *, proc_root: Path = Path("/proc"), clock_ticks: int | None = None) -> None:
        self._root = int(root_pid)
        self._proc = Path(proc_root)
        self._ticks = clock_ticks or _clock_ticks()
        self._cpu_seconds = 0.0
        self._peak_rss = 0

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss

    @property
    def cpu_seconds(self) -> float:
        return self._cpu_seconds

    def sample(self) -> TreeSample:
        """Read the tree once and fold the reading into the peak and the CPU total."""
        rss = 0
        cpu = 0.0
        live = 0
        for pid, start_time in self._tree():
            reading = self._read_process(pid, start_time)
            if reading is None:
                continue
            live += 1
            rss += reading[0]
            cpu += reading[1]
        self._peak_rss = max(self._peak_rss, rss)
        self._cpu_seconds = max(self._cpu_seconds, cpu)
        return TreeSample(rss_bytes=rss, cpu_seconds=self.cpu_seconds, live_pids=live)

    def _read_process(self, pid: int, start_time: int | None) -> tuple[int, float] | None:
        """Read one process; ``None`` when it is gone or ``pid`` now names a process started at another time."""
        try:
            status = (self._proc / str(pid) / "status").read_text(encoding=UTF8, errors=DECODE_ERRORS)
            stat = (self._proc / str(pid) / "stat").read_text(encoding=UTF8, errors=DECODE_ERRORS)
            fields = stat.rsplit(")", 1)[-1].split()
            if start_time is not None and int(fields[19]) != start_time:
                return None
            utime, stime, cutime, cstime = (int(value) for value in fields[11:15])
            cpu = (utime + stime + cutime + cstime) / self._ticks
        except (OSError, ValueError, IndexError):
            return None
        rss = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    rss = int(parts[1]) * 1024
                break
        return rss, cpu

    def _tree(self) -> list[tuple[int, int | None]]:
        """The root, then its descendants with every parent before its children, each with its start time.

        A child reaped between the read of its parent and its own is then missed by one
        sample instead of counted twice, and a pid reused after the scan is not read as
        the process the scan found; the kept maximum would make either count permanent.
        The root carries no start time, so it is read by its pid alone.
        """
        root: list[tuple[int, int | None]] = [(self._root, None)]
        try:
            below = descendants(self._root)
        except OSError:
            return root
        return [*root, *reversed(below)]


@dataclass(frozen=True)
class ContainerSnapshot:
    """cgroup v2 readings of the container this process runs in; ``None`` = unreadable or unlimited."""

    memory_max_bytes: int | None = None
    memory_peak_bytes: int | None = None
    oom_kill: int | None = None
    cpu_quota_us: int | None = None
    cpu_period_us: int | None = None

    def as_fields(self) -> dict[str, int | None]:
        return {
            "memory_max_bytes": self.memory_max_bytes,
            "memory_peak_bytes": self.memory_peak_bytes,
            "oom_kill": self.oom_kill,
            "cpu_quota_us": self.cpu_quota_us,
            "cpu_period_us": self.cpu_period_us,
        }


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding=UTF8).strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _read_oom_kill(path: Path) -> int | None:
    try:
        text = path.read_text(encoding=UTF8)
    except OSError:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill" and value.strip().isdigit():
            return int(value.strip())
    return None


def read_container_cgroup(root: Path = Path("/sys/fs/cgroup")) -> ContainerSnapshot:
    """Read the cgroup v2 files of ``root``; a v1 or unmounted hierarchy reads as all ``None``."""
    base = Path(root)
    if not (base / "cgroup.controllers").exists():
        return ContainerSnapshot()
    try:
        parts = (base / "cpu.max").read_text(encoding=UTF8).split()
    except OSError:
        parts = []  # an unreadable cpu.max leaves both CPU fields None: unknown, not unlimited
    quota: int | None = None
    period: int | None = None
    if len(parts) == 2 and parts[1].isdigit():
        period = int(parts[1])
        quota = int(parts[0]) if parts[0].isdigit() else None
    return ContainerSnapshot(
        memory_max_bytes=_read_int(base / "memory.max"),
        memory_peak_bytes=_read_int(base / "memory.peak"),
        oom_kill=_read_oom_kill(base / "memory.events"),
        cpu_quota_us=quota,
        cpu_period_us=period,
    )


def _entry_blocks(entry: os.DirEntry[str]) -> tuple[int, Path | None]:
    """Disk bytes of one directory entry, and the directory to descend into, if it is one."""
    try:
        info = entry.stat(follow_symlinks=False)
        is_dir = entry.is_dir(follow_symlinks=False)
    except OSError:
        return 0, None
    return info.st_blocks * 512, Path(entry.path) if is_dir else None


def _directory_blocks(directory: Path, subdirs: list[Path]) -> int:
    """Disk bytes of the entries directly in ``directory``, and its subdirectories appended to ``subdirs``.

    A directory that cannot be read, or fails part way, counts what was read before the failure.
    """
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                size, subdir = _entry_blocks(entry)
                total += size
                if subdir is not None:
                    subdirs.append(subdir)
    except OSError:
        # Keep the partial count: an unreadable directory adds nothing, a failed scan what it read first.
        pass
    return total


def dir_usage_bytes(path: Path) -> int | None:
    """Disk blocks used under ``path`` (symlinks not followed); ``None`` when it does not exist."""
    root = Path(path)
    if not root.exists():
        return None
    total = 0
    stack = [root]
    while stack:
        total += _directory_blocks(stack.pop(), stack)
    return total


@dataclass
class ResourceLog:
    """Append-only ``agents/resources.jsonl``; one JSON object per line."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
