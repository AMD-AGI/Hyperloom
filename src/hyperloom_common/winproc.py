# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Windows process control: the half of tree management POSIX gives for free.

Linux names a process tree twice over -- by parentage under ``/proc`` and by process
group -- and ends one with a signal to either. Windows has neither: it names a tree by
``taskkill /T`` walking the live parent chain, and it has no catchable termination
signal, so a graceful ask and a forced end are the same call with or without ``/F``.

This module is the Windows side of that: liveness through an opened process handle, and
a whole-tree kill through ``taskkill``. ``hyperloom.common.proctree`` delegates to it,
so every caller that reaps a server or a Ray worker keeps one interface on both hosts.

A handle is what separates a live pid from a reused one: Windows keeps a pid allocated
until the last handle on it closes, and ``GetExitCodeProcess`` answers ``STILL_ACTIVE``
only while the process has not exited.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from ctypes import wintypes

from hyperloom_common.platform import is_windows

log = logging.getLogger(__name__)

#: What ``GetExitCodeProcess`` reports for a process that has not exited.
STILL_ACTIVE = 259
#: The job limit that ends every process in the job when its last handle closes -- the
#: Windows form of "this child dies with the process that started it".
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
#: Enough access to ask whether a process is still running; anything wider is refused for
#: a process this account does not own.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
#: ``taskkill`` walk the parent chain. Without it only the named process dies, and a
#: server's engine ranks outlive it holding VRAM.
_TREE = "/T"
_FORCE = "/F"

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if is_windows() else None


class _IoCounters(ctypes.Structure):
    """The I/O counters a job's extended limits carry; nothing here reads them."""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimits(ctypes.Structure):
    """The scheduling limits of a job; only ``LimitFlags`` is set here."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    """What ``SetInformationJobObject`` writes for the extended-limit class."""

    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _configure_kernel32() -> None:
    """Declare every signature used here, so a 64-bit handle is not truncated by the default ``int``."""
    if _kernel32 is None:
        return
    for name in ("OpenProcess", "CreateJobObjectW", "AssignProcessToJobObject"):
        getattr(_kernel32, name).restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    for name in ("GetExitCodeProcess", "AssignProcessToJobObject", "TerminateJobObject"):
        getattr(_kernel32, name).restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL


if _kernel32 is not None:
    _configure_kernel32()


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a process that has not exited.

    A pid still listed by Windows after its process exited is not alive: the entry
    belongs to a handle somebody holds, and no signal reaches it.

    Args:
        pid: The process to inspect.

    Returns:
        bool: True while the process runs; False for an exited or unknown pid, and on
        a host where this cannot be asked.
    """
    if _kernel32 is None or pid <= 0:
        return False
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # A pid no process holds is refused here; one this account may not open is
        # refused too, and either way nothing of ours is left to reap.
        return False
    try:
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        _kernel32.CloseHandle(handle)


def terminate_tree(pid: int, *, force: bool) -> bool:
    """End ``pid`` and every process below it, reporting whether the call was accepted.

    Args:
        pid: The root of the tree.
        force: Whether to force the end. Without it ``taskkill`` asks the tree to close;
            a console process has no window to close and routinely ignores the ask, which
            is why the forced call is what actually ends a server.

    Returns:
        bool: True when ``taskkill`` reported success, including when the tree was
        already gone.
    """
    if not is_windows() or pid <= 0:
        return False
    # The pid travels as its own argument: taskkill rejects ``/PID=<n>``.
    command = ["taskkill", _TREE, "/PID", str(int(pid))]
    if force:
        command.insert(1, _FORCE)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("winproc: taskkill %d failed: %s", pid, exc)
        return False
    if completed.returncode != 0:
        # 128 is "no such process": the tree was already gone, which is the outcome
        # asked for. Anything else is worth a line, since VRAM may still be held.
        detail = (completed.stderr or completed.stdout or "").strip()
        if "not found" not in detail.lower() and completed.returncode != 128:
            log.warning("winproc: taskkill %d exited %d: %s", pid, completed.returncode, detail)
            return False
    return True


class OwnedJob:
    """A job object that ends its processes when this handle closes.

    This is how a child is made to die with the process that started it, which POSIX
    spells ``PR_SET_PDEATHSIG``: the kernel closes every handle of a dying process, the
    job's last handle goes with them, and the limit armed below kills what is left.

    The handle must outlive the child. Whenever a reference to the instance is dropped,
    Python closes it and the child is killed, so the owner keeps it beside the process it
    gave the job to.
    """

    def __init__(self) -> None:
        """Create the job and arm it; the handle is ``None`` when the host refused."""
        self._handle: int | None = None
        if _kernel32 is None:
            return
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            log.warning("winproc: a job object could not be created; the tree is reaped by pid only")
            return
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            log.warning("winproc: a job object could not be armed; the tree is reaped by pid only")
            _kernel32.CloseHandle(handle)
            return
        self._handle = handle

    @property
    def armed(self) -> bool:
        """Whether the job exists and will kill its processes when this handle closes."""
        return self._handle is not None

    def assign(self, pid: int) -> bool:
        """Put the running process ``pid`` -- and so everything it starts -- in the job."""
        if _kernel32 is None or self._handle is None or pid <= 0:
            return False
        process = _kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        if not process:
            return False
        try:
            return bool(_kernel32.AssignProcessToJobObject(self._handle, process))
        finally:
            _kernel32.CloseHandle(process)

    def terminate(self) -> None:
        """End every process in the job, leaving the handle open."""
        if _kernel32 is not None and self._handle is not None:
            _kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Drop the handle; with the kill-on-close limit armed this ends the job."""
        if _kernel32 is not None and self._handle is not None:
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "OwnedJob":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def spawn_flags() -> int:
    """The creation flags a child that will be reaped as a tree needs.

    ``CREATE_NEW_PROCESS_GROUP`` keeps a Ctrl-C aimed at this process from reaching the
    server, which is what ``start_new_session`` buys on POSIX.
    """
    return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
