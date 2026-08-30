# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Process-group lifecycle helpers for driver and profiler subprocesses."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


async def kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill and reap a subprocess's isolated process group."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=10)
    # AITER's zero-byte FileBaton lock is not released when the child receives
    # SIGKILL. The cache is per-attempt, so after the whole group is reaped it is
    # safe to remove only locks owned by this Forge process.
    with contextlib.suppress(Exception):
        from kernelforge.loop.aiter_cache import cleanup_current_owned_aiter_locks

        cleanup_current_owned_aiter_locks()


async def communicate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    timeout: float,
) -> tuple[bytes, bytes]:
    """Communicate with an isolated subprocess and clean up on cancellation.

    Every caller must create ``proc`` with ``start_new_session=True``.
    Timeout and cancellation semantics are preserved after the whole process
    group has been killed and reaped.
    """
    try:
        return await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        await kill_process_group(proc)
        raise


__all__ = ["communicate_process_group", "kill_process_group"]
