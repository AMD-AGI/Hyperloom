# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End every process a sandbox started, including ones that left the tree."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from hyperloom.common import proctree

__all__ = ["reap_sandbox_processes", "sweep_tagged"]

_SWEEP_WINDOW_SEC = 5.0


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
    """SIGKILL every process that was started with this sandbox's id in its environment.

    This catches processes that detached from the tree (a daemon that called
    ``setsid`` and was re-parented to init), which tree enumeration cannot see.

    Args:
        sandbox_id: The id carried in ``E2B_SANDBOX_ID``; it is unique per sandbox.

    Returns:
        bool: True when no tagged process is left within the window.
    """
    tag = f"E2B_SANDBOX_ID={sandbox_id}".encode()
    deadline = time.monotonic() + _SWEEP_WINDOW_SEC
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
        time.sleep(0.05)


def reap_sandbox_processes(
    roots: list[subprocess.Popen[str]] | list[subprocess.Popen[bytes]],
    sandbox_id: str,
    *,
    grace_sec: float = 2.0,
) -> bool:
    """End the whole process tree of a sandbox and confirm nothing is left.

    Order: SIGTERM then SIGKILL the enumerated trees (and their process
    groups), then kill any remaining process tagged with the sandbox id.

    Args:
        roots: Every process the sandbox spawned directly.
        sandbox_id: The id the sandbox put in its processes' environment.
        grace_sec: How long a tree gets to exit on SIGTERM.

    Returns:
        bool: True when no process of the sandbox is left running.
    """
    try:
        tree = proctree.collect_tree(p.pid for p in roots)
        proctree.kill_tree(tree, grace_sec=grace_sec, confirm_sec=_SWEEP_WINDOW_SEC)
    except OSError:
        # /proc could not be listed; the group kill and the tag sweep below still run.
        pass
    for proc in roots:
        # The root may be dead while members of its group live on.
        proctree.signal_group(proc.pid, signal.SIGKILL, what="sandbox reap")
    swept = sweep_tagged(sandbox_id)
    for proc in roots:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return False
    return swept
