# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``Clock.wait_for`` times work on the clock's own time, so a virtual clock bounds it as a real one would."""

from __future__ import annotations

import asyncio

import pytest

from hyperloom_multibranch.scheduling.clock import Clock

A_DAY = 86_400.0


async def _instant(seconds: float) -> None:
    await asyncio.sleep(0)


async def test_work_that_finishes_first_returns_its_result() -> None:
    async def work() -> str:
        return "done"

    assert await Clock().wait_for(work(), A_DAY) == "done"


async def test_a_day_of_virtual_time_times_out_and_cancels_the_work() -> None:
    cancelled = asyncio.Event()

    async def forever() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(Clock(sleep=_instant).wait_for(forever(), A_DAY), timeout=5.0)
    assert cancelled.is_set()


async def test_the_work_errors_propagate() -> None:
    async def broken() -> None:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await Clock().wait_for(broken(), A_DAY)
