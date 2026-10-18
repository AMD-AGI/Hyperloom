# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End every process a sandbox started, including ones that left the tree."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

from hyperloom.common import proctree

__all__ = ["kill_launchers", "reap_sandbox_processes", "sweep_tagged"]

_WINDOW_SEC = 5.0
_POLL_SEC = 0.05


def _live_descendants(root_pids: Iterable[int]) -> list[proctree.ProcessId]:
    """Every running descendant of the launchers; zombies are already gone for our purposes."""
    found: list[proctree.ProcessId] = []
    for root in root_pids:
        found.extend(d for d in proctree.descendants(root) if proctree.running(d[0]))
    return found


def _drain(root_pids: list[int], sig: int, window_sec: float) -> bool:
    """Signal descendants of the launchers until none is left, or the window ends.

    Each pass walks ``/proc`` again, so a process forked while we signal is
    caught on the next pass. The launchers stay alive throughout and are
    subreapers, so nothing can leave the tree by detaching.
    """
    deadline = time.monotonic() + window_sec
    while True:
        alive = _live_descendants(root_pids)
        if not alive:
            return True
        if time.monotonic() >= deadline:
            return False
        proctree.signal_processes(alive, sig)
        time.sleep(_POLL_SEC)


def kill_launchers(procs: list[subprocess.Popen[bytes]], *, grace_sec: float = 2.0) -> bool:
    """Reap everything that runs under the given launchers, then the launchers.

    Order: freeze the launchers, SIGTERM the descendants and give them ``grace_sec``, SIGKILL what is
    left until a fresh scan finds nothing, and only then kill the launchers.

    Args:
        procs: Launcher processes started by the sandbox.
        grace_sec: How long descendants get to exit on SIGTERM.

    Returns:
        bool: True when no descendant is left running.
    """
    roots = [p.pid for p in procs]
    for pid in roots:
        # Freeze each launcher first so it cannot fork a command we have not seen yet.
        # A stopped subreaper still receives orphans; it only stops reaping zombies.
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
    try:
        # SIGTERM once, then wait the grace; the SIGKILL passes re-scan until empty.
        proctree.signal_processes(_live_descendants(roots), signal.SIGTERM)
        _drain(roots, 0, grace_sec)
        clean = _drain(roots, signal.SIGKILL, _WINDOW_SEC)
    except OSError:
        clean = False  # /proc could not be listed; nothing can be confirmed
    for proc in procs:
        proctree.signal_group(proc.pid, signal.SIGKILL, what="sandbox launcher")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=_WINDOW_SEC)
        except subprocess.TimeoutExpired:
            clean = False
    return clean


def _tagged_pids(tag: bytes) -> list[int]:
    """Pids whose initial environment carries ``tag``; unreadable processes are skipped."""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if tag in environ.split(b"\0"):
            found.append(int(entry.name))
    return found


def sweep_tagged(sandbox_id: str) -> bool:
    """SIGKILL every process started with this sandbox's id in its environment.

    A fallback for the case the launchers themselves were lost; it cannot see
    a process that cleared its environment.

    Returns:
        bool: True when no tagged process is left within the window.
    """
    tag = f"E2B_SANDBOX_ID={sandbox_id}".encode()
    deadline = time.monotonic() + _WINDOW_SEC
    while True:
        pids = [pid for pid in _tagged_pids(tag) if proctree.running(pid)]
        if not pids:
            return True
        if time.monotonic() >= deadline:
            return False
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
        time.sleep(_POLL_SEC)


def reap_sandbox_processes(procs: list[subprocess.Popen[bytes]], sandbox_id: str) -> bool:
    """End all processes of a sandbox and confirm nothing is left.

    Args:
        procs: The sandbox's launcher processes.
        sandbox_id: The id the sandbox put in its processes' environment.

    Returns:
        bool: True when no process of the sandbox is left running.
    """
    clean = kill_launchers(procs)
    return sweep_tagged(sandbox_id) and clean
