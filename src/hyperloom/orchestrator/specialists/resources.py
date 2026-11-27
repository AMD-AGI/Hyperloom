# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only resource sampling for specialists and their container.

Per specialist: the process-tree resident memory peak and cumulative CPU seconds,
read from ``/proc``, and the disk the worktree and its self-check venv use.
Per container: cgroup v2 memory limit, peak, OOM-kill count and CPU limit. Every
reader returns ``None`` for what it cannot read instead of guessing, and nothing
here writes anything or signals the processes it observes.

The readings are for a specialist's resource record; their purpose is to size the
hosted application container and to decide when CPU specialists need a sub-run, so
a missing reading must stay distinguishable from a zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR

from hyperloom_common.proctree import cpu_ticks, descendants, rss_bytes

__all__ = [
    "ContainerSnapshot",
    "ProcessTreeSampler",
    "dir_usage_bytes",
    "read_container_cgroup",
]

UTF8 = "utf-8"


class ProcessTreeSampler:
    """Track the memory peak and CPU time of one process tree.

    The tree is the root and its descendants as :func:`hyperloom_common.proctree.descendants`
    finds them in ``/proc``.

    A process that exits stops counting towards memory the moment it is gone. Its CPU
    seconds reach the total through its parent: the kernel adds a child's user and
    system time, and its own reaped children's, to the parent's ``cutime``/``cstime``
    when the parent waits for it. A sample sums ``utime + stime + cutime + cstime``
    over the live processes, so a compiler that starts and exits between two samples
    still counts, and the total keeps the largest sum read, so it never goes back when
    a process leaves the tree without a member of it waiting for it.
    """

    def __init__(self, root_pid: int, *, clock_ticks: int | None = None) -> None:
        self._root = int(root_pid)
        self._ticks = clock_ticks or os.sysconf("SC_CLK_TCK")
        self._cpu_seconds = 0.0
        self._peak_rss = 0

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss

    @property
    def cpu_seconds(self) -> float:
        return self._cpu_seconds

    def sample(self) -> None:
        """Read the tree once and fold the reading into the peak and the CPU total.

        A member that is gone, or whose pid now names a process started at another time, is skipped.
        """
        rss = 0
        ticks = 0
        for pid, start_time in self._tree():
            own = cpu_ticks(pid, start_time)
            if own is None:
                continue
            ticks += own
            rss += rss_bytes(pid) or 0
        self._peak_rss = max(self._peak_rss, rss)
        self._cpu_seconds = max(self._cpu_seconds, ticks / self._ticks)

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
    """Disk blocks used under ``path``, following no symlink (one at ``path`` neither); ``None`` (unknown) when ``path``
    is not a directory of its own: a link a specialist left at its worktree's path would have the app walk whatever
    directory it names."""
    root = Path(path)
    try:
        if not S_ISDIR(os.lstat(root).st_mode):
            return None
    except OSError:
        return None
    total = 0
    stack = [root]
    while stack:
        total += _directory_blocks(stack.pop(), stack)
    return total
