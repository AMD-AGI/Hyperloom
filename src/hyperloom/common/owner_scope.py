# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The execution scope a process records on the ownership it takes: this boot, PID
namespace and container start."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

_PROC = Path("/proc")
_PROC_ENCODING = "ascii"


def local_owner_scope(proc: Path = _PROC) -> str:
    """Identify this boot, PID namespace and the start of its init, or leave ownership
    unobservable.

    The boot and PID-namespace inode alone repeat: a container started on the same node
    after another one exited can get the same namespace inode, and then a lost
    container's ownership reads as this one's. The start time of the namespace's PID 1,
    in clock ticks since boot, differs between two containers of one boot.
    """
    try:
        boot = (proc / "sys/kernel/random/boot_id").read_text(encoding=_PROC_ENCODING).strip()
        namespace = (proc / "self/ns/pid").stat().st_ino
        init_started = _start_ticks((proc / "1/stat").read_text(encoding=_PROC_ENCODING))
    except (OSError, ValueError, IndexError):
        return ""
    return f"{boot}:{namespace}:{init_started}" if boot else ""


def _start_ticks(stat: str) -> int:
    """Field 22 of ``/proc/<pid>/stat``: when the process started, in ticks since boot.

    The command name in field 2 may hold spaces and parentheses, so fields are counted
    from its closing parenthesis.
    """
    return int(stat[stat.rindex(")") + 2 :].split()[19])


def same_environment(recorded: str, local: str) -> bool:
    """Whether a recorded scope names this environment: equal to ``local``, or the two-part
    scope earlier versions recorded for this boot and PID namespace.

    A two-part match cannot tell this container from an earlier one that repeated its
    namespace inode; a caller that releases on it first checks the holder process.
    """
    if not recorded or not local:
        return False
    return recorded == local or (recorded.count(":") == 1 and local.rsplit(":", 1)[0] == recorded)


def process_started_at(pid: int, proc: Path = _PROC) -> datetime | None:
    """When ``pid`` started, or None when it is not running here or cannot be read."""
    try:
        ticks = _start_ticks((proc / str(pid) / "stat").read_text(encoding=_PROC_ENCODING))
        booted = next(
            int(line.split()[1])
            for line in (proc / "stat").read_text(encoding=_PROC_ENCODING).splitlines()
            if line.startswith("btime ")
        )
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    return datetime.fromtimestamp(booted + ticks / os.sysconf("SC_CLK_TCK"), timezone.utc)
