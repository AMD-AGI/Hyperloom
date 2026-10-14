# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The execution scope a process records on the ownership it takes: this boot, PID
namespace and container start."""

from __future__ import annotations

from pathlib import Path

_PROC = Path("/proc")


def local_owner_scope(proc: Path = _PROC) -> str:
    """Identify this boot, PID namespace and the start of its init, or leave ownership
    unobservable.

    The boot and PID-namespace inode alone repeat: a container started on the same node
    after another one exited can get the same namespace inode, and then a lost
    container's ownership reads as this one's. The start time of the namespace's PID 1,
    in clock ticks since boot, differs between two containers of one boot.
    """
    try:
        boot = (proc / "sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        namespace = (proc / "self/ns/pid").stat().st_ino
        init_started = _start_ticks((proc / "1/stat").read_text(encoding="ascii"))
    except (OSError, ValueError, IndexError):
        return ""
    return f"{boot}:{namespace}:{init_started}" if boot else ""


def _start_ticks(stat: str) -> int:
    """Field 22 of ``/proc/<pid>/stat``: when the process started, in ticks since boot.

    The command name in field 2 may hold spaces and parentheses, so fields are counted
    from its closing parenthesis.
    """
    return int(stat[stat.rindex(")") + 2 :].split()[19])
