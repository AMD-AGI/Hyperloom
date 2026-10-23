# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resource sampling for specialists and their container.

Per specialist: the process-tree resident memory peak and cumulative CPU seconds,
read from ``/proc``, and the disk the worktree and its self-check venv use.
Per container: cgroup v2 memory limit, peak, OOM-kill count and CPU limit. Every
reader returns ``None`` for what it cannot read instead of guessing, and nothing
here writes to or signals the processes it observes.

Records go to ``agents/resources.jsonl`` (detailed design 4.14); their purpose is
to size the hosted application container and to decide when CPU specialists need a
sub-run, so a missing reading must stay distinguishable from a zero.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = [
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
    """CPU seconds (user + system) summed over every process ever seen in the tree."""
    live_pids: int


def _clock_ticks() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return 100


class ProcessTreeSampler:
    """Track the memory peak and CPU time of one process tree.

    A process that exits stops counting towards memory the moment it is gone, but
    the CPU seconds it was last seen with stay in the total, so a short-lived
    compiler child still shows up in the CPU figure.
    """

    def __init__(self, root_pid: int, *, proc_root: Path = Path("/proc"), clock_ticks: int | None = None) -> None:
        self._root = int(root_pid)
        self._proc = Path(proc_root)
        self._ticks = clock_ticks or _clock_ticks()
        self._cpu_by_pid: dict[int, float] = {}
        self._peak_rss = 0
        self._children_files_ok = True

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss

    @property
    def cpu_seconds(self) -> float:
        return sum(self._cpu_by_pid.values())

    def sample(self) -> TreeSample:
        """Read the tree once and fold the reading into the peak and the CPU total."""
        pids = self._tree_pids()
        rss = 0
        live = 0
        for pid in pids:
            reading = self._read_process(pid)
            if reading is None:
                continue
            live += 1
            rss += reading[0]
            self._cpu_by_pid[pid] = max(self._cpu_by_pid.get(pid, 0.0), reading[1])
        self._peak_rss = max(self._peak_rss, rss)
        return TreeSample(rss_bytes=rss, cpu_seconds=self.cpu_seconds, live_pids=live)

    def _read_process(self, pid: int) -> tuple[int, float] | None:
        try:
            status = (self._proc / str(pid) / "status").read_text(encoding=UTF8, errors=DECODE_ERRORS)
            stat = (self._proc / str(pid) / "stat").read_text(encoding=UTF8, errors=DECODE_ERRORS)
            fields = stat.rsplit(")", 1)[-1].split()
            cpu = (int(fields[11]) + int(fields[12])) / self._ticks
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

    def _tree_pids(self) -> list[int]:
        """The root and every descendant, by the kernel's children files when it has them."""
        if self._children_files_ok:
            found = self._walk_children_files()
            if found is not None:
                return found
            self._children_files_ok = False
        return self._walk_by_ppid()

    def _children_of(self, pid: int) -> list[int] | None:
        """Direct children of ``pid`` from the kernel's children files; ``None`` when the kernel has none."""
        task_dir = self._proc / str(pid) / "task"
        try:
            tids = [entry.name for entry in os.scandir(task_dir)]
        except OSError:
            return []  # the process is gone
        children: list[int] = []
        for tid in tids:
            try:
                text = (task_dir / tid / "children").read_text(encoding=UTF8)
            except FileNotFoundError:
                return None
            except OSError:
                continue
            children.extend(int(token) for token in text.split() if token.isdigit())
        return children

    def _walk_children_files(self) -> list[int] | None:
        seen = [self._root]
        queue = [self._root]
        while queue:
            pid = queue.pop()
            children = self._children_of(pid)
            if children is None:
                if pid == self._root:
                    return None
                continue
            for child in children:
                if child not in seen:
                    seen.append(child)
                    queue.append(child)
        return seen

    def _walk_by_ppid(self) -> list[int]:
        parent_of: dict[int, int] = {}
        try:
            names = [entry.name for entry in os.scandir(self._proc) if entry.name.isdigit()]
        except OSError:
            return [self._root]
        for name in names:
            try:
                text = (self._proc / name / "stat").read_text(encoding=UTF8, errors=DECODE_ERRORS)
                parent_of[int(name)] = int(text.rsplit(")", 1)[-1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
        seen = [self._root]
        grew = True
        while grew:
            grew = False
            for pid, ppid in parent_of.items():
                if ppid in seen and pid not in seen:
                    seen.append(pid)
                    grew = True
        return seen


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
    quota: int | None = None
    period: int | None = None
    try:
        parts = (base / "cpu.max").read_text(encoding=UTF8).split()
        if len(parts) == 2 and parts[1].isdigit():
            period = int(parts[1])
            quota = int(parts[0]) if parts[0].isdigit() else None
    except OSError:
        pass
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
