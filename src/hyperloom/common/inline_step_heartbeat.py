# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep a long inline step visible to the KERNEL idle guard."""

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
    clear: Callable[[], None] | None = None,
) -> AsyncIterator[None]:
    """Stamp progress every ``interval_sec`` for as long as the block runs."""
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
        # Retiring the stamp gets a finally of its own: awaiting the cancelled beat re-raises anything it died of, and
        # a beat that died is exactly the case where a stamp is left behind.
        try:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            if clear is not None:
                with contextlib.suppress(Exception):
                    clear()
