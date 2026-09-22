# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-enforced containment for the processes a task may run on its GPUs.

A lane keeps two rounds off the same cards. When a specialist ends without
confirming its teardown the lane is deliberately retained, because its process
tree may still be running -- and nothing afterwards could establish that those
processes had gone, so the lane was held for the life of the session. On
2026-09-21 that stranded six lanes for two hours behind work that had in fact
already finished.

Three process identities were tried as proof that a lane was free and each was
refuted: a terminal task says nothing about its children; a spawn process group
is left behind by any descendant calling ``setsid``, which the served inference
servers do by design; and a pidfile appears only after the server answers,
leaving the whole model-load window unnamed. Every one of them is an observation
of expected behaviour rather than an ownership boundary.

A cgroup is such a boundary. ``setsid`` changes PID-session and process-group
membership only, never cgroup membership, and a descendant cannot leave the
cgroup it was born into unless it is allowed to write ``cgroupfs``. Emptiness
then becomes a fact this module can read -- ``populated 0`` -- rather than a
failure to find anything.

This module is the gate in front of that: it reports whether the host can
actually deliver the guarantee. Where it cannot, automatic reclamation stays
off and lanes stay fail-closed, because silently falling back to a refuted proxy
is worse than a lane an operator can see and clear.

See ``docs/task-containment.md``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import signal
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CLONE_INTO_CGROUP",
    "CGROUP2_MAGIC",
    "ContainmentSupport",
    "MIN_CLONE3_CGROUP_KERNEL",
    "cgroup_v2_root",
    "probe_containment_support",
]

#: ``clone3`` flag placing the child directly in a cgroup, so no instruction of
#: the workload runs outside it. Adding a cgroup from ``preexec_fn``, or moving
#: the pid after ``Popen`` returns, both leave a window and do not close the gap.
CLONE_INTO_CGROUP = 0x200000000

#: ``statfs`` magic identifying a unified (v2) hierarchy.
CGROUP2_MAGIC = 0x63677270

#: ``CLONE_INTO_CGROUP`` landed in 5.7. Detected by trying it rather than by
#: parsing a release string, but reported so an operator sees the requirement.
MIN_CLONE3_CGROUP_KERNEL = (5, 7)

#: Capability that would let a workload write ``cgroupfs`` and migrate itself or
#: a descendant out of its domain, which breaks the guarantee outright.
_CAP_SYS_ADMIN = 21

_SYS_clone3 = 435


@dataclass(frozen=True)
class ContainmentSupport:
    """What this host can actually enforce.

    Attributes:
        cgroup_v2: A unified hierarchy is mounted.
        delegated_root: A subtree this process may create task cgroups under,
            or ``None`` when there is none.
        clone_into_cgroup: ``clone3(CLONE_INTO_CGROUP)`` is accepted by this
            kernel, established by calling it rather than by version string.
        kill_and_events: The reclaimed domain can be drained with ``cgroup.kill``
            and attested empty through ``cgroup.events``.
        can_drop_cap_sys_admin: This process holds the capability it must be able
            to drop from a workload, so a spawned child can be denied it.
        reason: Why containment is unavailable, for the operator log. Empty when
            :attr:`available` is true.
    """

    cgroup_v2: bool
    delegated_root: Path | None
    clone_into_cgroup: bool
    kill_and_events: bool
    can_drop_cap_sys_admin: bool
    reason: str

    @property
    def available(self) -> bool:
        """Whether containment may be relied on for automatic lane release."""
        return (
            self.cgroup_v2
            and self.delegated_root is not None
            and self.clone_into_cgroup
            and self.kill_and_events
            and self.can_drop_cap_sys_admin
        )


class _CloneArgs(ctypes.Structure):
    """``struct clone_args`` as of the ``CLONE_INTO_CGROUP`` extension."""

    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("pidfd", ctypes.c_uint64),
        ("child_tid", ctypes.c_uint64),
        ("parent_tid", ctypes.c_uint64),
        ("exit_signal", ctypes.c_uint64),
        ("stack", ctypes.c_uint64),
        ("stack_size", ctypes.c_uint64),
        ("tls", ctypes.c_uint64),
        ("set_tid", ctypes.c_uint64),
        ("set_tid_size", ctypes.c_uint64),
        ("cgroup", ctypes.c_uint64),
    ]


def cgroup_v2_root() -> Path | None:
    """Return the unified cgroup mount point, or ``None`` when there is none.

    Returns:
        Path | None: The mount point, when ``statfs`` reports a v2 hierarchy.
    """
    root = Path("/sys/fs/cgroup")
    try:
        return root if os.statvfs(root) and _fs_magic(root) == CGROUP2_MAGIC else None
    except OSError:
        return None


def _fs_magic(path: Path) -> int | None:
    """Read ``f_type`` for ``path``, which :mod:`os` does not expose.

    Args:
        path: Directory to inspect.

    Returns:
        int | None: The filesystem magic, or ``None`` when it cannot be read.
    """

    class _Statfs(ctypes.Structure):
        _fields_ = [("f_type", ctypes.c_long)] + [(f"_pad{i}", ctypes.c_long) for i in range(16)]

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buf = _Statfs()
    if libc.statfs(str(path).encode(), ctypes.byref(buf)) != 0:
        return None
    return int(buf.f_type)


def _effective_caps() -> int:
    """Return this process's effective capability mask, or 0 when unreadable."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("CapEff:"):
                return int(line.split()[1], 16)
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _probe_clone_into_cgroup(cgroup: Path) -> bool:
    """Whether this kernel accepts ``clone3(CLONE_INTO_CGROUP)``.

    Established by making the call: a version string says what the kernel claims,
    not what the seccomp profile in front of it permits.

    Args:
        cgroup: An existing cgroup directory to clone into.

    Returns:
        bool: True when a child was created inside ``cgroup``.
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    try:
        fd = os.open(str(cgroup), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return False
    try:
        args = _CloneArgs(flags=CLONE_INTO_CGROUP, exit_signal=signal.SIGCHLD, cgroup=fd)
        pid = libc.syscall(_SYS_clone3, ctypes.byref(args), ctypes.sizeof(args))
        if pid == 0:
            # Child: nothing may run here that could fail and be reported twice.
            os._exit(0)
        if pid < 0:
            # ENOSYS/EINVAL on an older kernel, EPERM under a seccomp profile.
            return False
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        return True
    except OSError as exc:
        if exc.errno in (errno.ENOSYS, errno.EINVAL, errno.EPERM):
            return False
        raise
    finally:
        os.close(fd)


def probe_containment_support(*, parent: Path | None = None) -> ContainmentSupport:
    """Establish what this host can enforce, by exercising it.

    Every check is a real operation. A host that merely looks capable -- the
    right kernel, a mounted hierarchy -- can still refuse the calls that matter
    under a seccomp profile or an undelegated subtree, and a wrong answer here
    would re-enable automatic release on a guarantee that does not hold.

    Args:
        parent: Where to create the throwaway probe cgroup. Defaults to the
            unified mount point.

    Returns:
        ContainmentSupport: The findings, with :attr:`ContainmentSupport.reason`
        naming the first missing requirement.
    """
    root = cgroup_v2_root()
    if root is None:
        return ContainmentSupport(False, None, False, False, False, "no cgroup v2 hierarchy is mounted")

    base = parent or root
    probe = base / f"hyperloom_probe_{os.getpid()}"
    try:
        probe.mkdir(exist_ok=True)
    except OSError as exc:
        return ContainmentSupport(
            True, None, False, False, False, f"no delegated cgroup subtree under {base}: {exc.strerror or exc}"
        )

    try:
        kill_and_events = (probe / "cgroup.kill").exists() and (probe / "cgroup.events").exists()
        clone_ok = _probe_clone_into_cgroup(probe)
        can_drop = bool(_effective_caps() & (1 << _CAP_SYS_ADMIN))

        reason = ""
        if not clone_ok:
            reason = (
                "clone3(CLONE_INTO_CGROUP) was refused; a workload cannot be placed in its domain "
                f"atomically (needs Linux {MIN_CLONE3_CGROUP_KERNEL[0]}.{MIN_CLONE3_CGROUP_KERNEL[1]}+ "
                "and a seccomp profile that permits it)"
            )
        elif not kill_and_events:
            reason = "this cgroup exposes no cgroup.kill/cgroup.events, so a domain cannot be drained and attested"
        elif not can_drop:
            # Without it here, a child cannot be denied it either -- and a
            # workload that keeps CAP_SYS_ADMIN can migrate itself out.
            reason = "this process lacks CAP_SYS_ADMIN, so it cannot drop that capability from a workload"

        return ContainmentSupport(True, base, clone_ok, kill_and_events, can_drop, reason)
    finally:
        try:
            probe.rmdir()
        except OSError:
            pass
