# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduler's time source: wall-clock seconds and a way to wait, both replaceable by a virtual clock in tests."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Clock:
    """``now`` is Unix seconds, the time base of the blackboard's records; ``sleep`` waits that many seconds."""

    now: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    async def wait_for(self, work: Awaitable[T], seconds: float) -> T:
        """:func:`asyncio.wait_for` timed by this clock: ``work`` is cancelled and this raises
        :class:`asyncio.TimeoutError` once ``seconds`` of clock time pass first."""

        task = asyncio.ensure_future(work)
        timer = asyncio.ensure_future(self.sleep(max(seconds, 0.0)))
        try:
            await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            timer.cancel()
            timed_out = not task.done()
            if timed_out:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if timed_out:
            raise asyncio.TimeoutError(f"no result within {seconds} clock seconds")
        return task.result()


__all__ = ["Clock"]
