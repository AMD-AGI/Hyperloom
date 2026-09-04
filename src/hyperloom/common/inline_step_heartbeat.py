# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep a long inline step visible to the KERNEL idle guard.

An inline step never becomes a task row, so the task-progress heartbeat cannot
cover it: only a re-stamped timestamp on shared state keeps the idle guard from
counting a working phase as idle. A phase-entry step that awaits a subprocess for
an hour has no such stamp today, leaving the whole phase unobservable and the
guard blind to the difference between busy and stuck.

Re-stamped per beat rather than once at the start, so a stamp that outlives its
process expires instead of muting the guard forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable


@contextlib.asynccontextmanager
async def inline_step_heartbeat(
    *,
    stamp: Callable[[float], None],
    interval_sec: float,
    now: Callable[[], float] = time.time,
    on_beat: Callable[[int], None] | None = None,
) -> AsyncIterator[None]:
    """Stamp progress every ``interval_sec`` for as long as the block runs.

    Stamps once on entry so a step shorter than one interval is still visible,
    then again per beat. Cancellation is the normal exit and is swallowed.

    Args:
        stamp: Records a progress timestamp; called with the current time.
        interval_sec: Seconds between beats; a non-positive value disables
            beating but keeps the entry stamp.
        now: Clock, injected for testability.
        on_beat: Optional observer receiving the 1-based beat number.

    Yields:
        None, for the duration of the guarded step.
    """
    stamp(now())
    task: asyncio.Task[None] | None = None
    if interval_sec > 0:

        async def _beat() -> None:
            beats = 0
            while True:
                await asyncio.sleep(interval_sec)
                beats += 1
                stamp(now())
                if on_beat is not None:
                    on_beat(beats)

        task = asyncio.create_task(_beat())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
