# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A PID namespace as the lifetime boundary for one task's processes.

A lane keeps two rounds off the same GPUs. It is retained when a specialist ends
without confirming teardown, because its processes may still run -- and nothing
could afterwards establish that they had stopped, so the lane was held for the
life of the session. On 2026-09-21 that stranded six lanes for two hours behind
work that had already finished.

Three identities were tried as proof a lane was free, and each was refuted by
probe: a terminal task says nothing about its children; the spawn process group
is left behind by any descendant calling ``setsid``, which served inference
servers do by design; and a pidfile names a server only after it answers,
leaving the whole model-load window unnamed. Each describes a process. None
constrains one.

A PID namespace does. A process may change its session, process group, parent,
executable, uid and open files without leaving it: ``setns`` cannot move a
caller into an ancestor PID namespace, only into its own or a descendant, and
even then only for children created afterwards. And when the namespace's init
exits, the kernel kills every remaining member before the namespace can go away.
So "is anybody still running" stops being a search and becomes one question with
a definite answer: has that init exited.

Measured on the deployment's own pod, which holds no ``CAP_SYS_ADMIN``:
``unshare(CLONE_NEWUSER|CLONE_NEWPID)`` succeeds, a survivor that calls
``setsid`` twice and outlives its root is gone once init exits, and ROCm still
sees every GPU from inside.

Out of contract: a workload that hands an already-open device descriptor to a
process outside the namespace through external IPC. Neither this nor a cgroup
can retract an exported descriptor.

See ``docs/task-containment.md``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CLONE_NEWPID",
    "CLONE_NEWUSER",
    "SupervisorIdentity",
    "identify_supervisor",
    "supervisor_state",
    "unshare_supported",
]

CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000

#: Field index of ``starttime`` in ``/proc/<pid>/stat``, counting from the field
#: after the comm parenthesis. A pid is reused; a pid at a start time is not.
_STARTTIME_FIELD = 19


@dataclass(frozen=True)
class SupervisorIdentity:
    """Durable identity of the namespace init owning one task's processes.

    A bare pid cannot name it: pids are reused, and after a coordinator restart
    an unrelated process could be sitting on the recorded number. Start time
    pins the incarnation, and the namespace inode pins which namespace that
    process is init of, so a mismatch is proof rather than suspicion.

    Attributes:
        pid: The supervisor's pid in the coordinator's own namespace.
        start_time: ``starttime`` from ``/proc/<pid>/stat``, in clock ticks.
        ns_inode: Inode of ``/proc/<pid>/ns/pid``, identifying the namespace.
    """

    pid: int
    start_time: int
    ns_inode: int

    def as_row(self) -> dict[str, int]:
        """Return the identity as columns to commit with the lane row."""
        return {
            "supervisor_pid": self.pid,
            "supervisor_start_time": self.start_time,
            "supervisor_ns_inode": self.ns_inode,
        }


def unshare_supported() -> tuple[bool, str]:
    """Whether this host lets an unprivileged process own a PID namespace.

    Established by performing the call in a throwaway child rather than by
    reading a sysctl, because seccomp and LSM policy can refuse what the sysctl
    permits, and a wrong answer here would re-enable automatic release on a
    guarantee that does not hold.

    Returns:
        tuple[bool, str]: Whether it is supported, and why not when it is not.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        rc = libc.unshare(CLONE_NEWUSER | CLONE_NEWPID)
        err = ctypes.get_errno() if rc != 0 else 0
        os.write(w, str(err).encode())
        os.close(w)
        os._exit(0)
    os.close(w)
    try:
        raw = os.read(r, 32).decode() or "0"
    finally:
        os.close(r)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    code = int(raw or 0)
    if code == 0:
        return True, ""
    return False, (
        f"unshare(CLONE_NEWUSER|CLONE_NEWPID) refused: {os.strerror(code)}. "
        "Namespace containment is unavailable; lanes stay fail-closed."
    )


def _start_time(pid: int) -> int | None:
    """Read ``starttime`` for ``pid``, or ``None`` when it cannot be read.

    Args:
        pid: Process to inspect.

    Returns:
        int | None: The start time in clock ticks.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    # comm may contain spaces and parentheses; everything after the last ')'
    # is positional.
    try:
        tail = raw[raw.rindex(")") + 2 :].split()
        return int(tail[_STARTTIME_FIELD])
    except (ValueError, IndexError):
        return None


def _ns_inode(pid: int) -> int | None:
    """Inode of ``pid``'s PID namespace, or ``None`` when unreadable."""
    try:
        return os.stat(f"/proc/{pid}/ns/pid").st_ino
    except OSError:
        return None


def identify_supervisor(pid: int) -> SupervisorIdentity | None:
    """Capture the durable identity of a live supervisor.

    Args:
        pid: The supervisor's pid, as the coordinator sees it.

    Returns:
        SupervisorIdentity | None: The identity, or ``None`` when the process is
        gone or ``/proc`` cannot be read for it.
    """
    start = _start_time(pid)
    inode = _ns_inode(pid)
    if start is None or inode is None:
        return None
    return SupervisorIdentity(pid=pid, start_time=start, ns_inode=inode)


def supervisor_state(identity: SupervisorIdentity) -> bool | None:
    """Whether the recorded supervisor is still running.

    Returns:
        bool | None: ``True`` while it runs, ``False`` once it is definitely
        gone -- which means the kernel has killed every remaining member of its
        namespace -- and ``None`` when ``/proc`` gives no usable answer, in
        which case the caller must retain the lane. An unreadable answer is not
        evidence of absence; treating it as one is the mistake this module
        exists to retire.
    """
    try:
        os.stat(f"/proc/{identity.pid}")
    except FileNotFoundError:
        return False
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            return None
        return None

    start = _start_time(identity.pid)
    inode = _ns_inode(identity.pid)
    if start is None or inode is None:
        # The pid exists but says nothing about itself: it may be a zombie
        # mid-teardown, or /proc may be restricted. Either way, not proof.
        return None
    if start != identity.start_time or inode != identity.ns_inode:
        # The number was reused by something unrelated, which is definite
        # evidence that the recorded incarnation is gone.
        return False
    return True
