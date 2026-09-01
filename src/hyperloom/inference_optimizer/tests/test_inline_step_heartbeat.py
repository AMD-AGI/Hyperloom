# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A long inline step must keep stamping progress, or the idle guard goes blind."""

from __future__ import annotations

import asyncio

import pytest

from hyperloom.common.inline_step_heartbeat import inline_step_heartbeat


async def test_entry_stamps_once_even_for_an_instant_step() -> None:
    """A step shorter than one interval must still be visible."""
    stamps: list[float] = []
    async with inline_step_heartbeat(stamp=stamps.append, interval_sec=10.0, now=lambda: 42.0):
        pass
    assert stamps == [42.0]


async def test_a_long_step_keeps_stamping() -> None:
    stamps: list[float] = []
    beats: list[int] = []
    clock = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    async with inline_step_heartbeat(
        stamp=stamps.append,
        interval_sec=0.01,
        now=lambda: next(clock),
        on_beat=beats.append,
    ):
        await asyncio.sleep(0.05)
    assert len(stamps) >= 2
    assert beats[:2] == [1, 2]


async def test_stamps_are_re_taken_not_reused() -> None:
    """Re-stamped per beat, so a stamp outliving its process expires."""
    stamps: list[float] = []
    ticks = iter(range(100))
    async with inline_step_heartbeat(stamp=stamps.append, interval_sec=0.01, now=lambda: float(next(ticks))):
        await asyncio.sleep(0.04)
    assert len(set(stamps)) == len(stamps)


async def test_a_disabled_interval_still_stamps_on_entry() -> None:
    stamps: list[float] = []
    async with inline_step_heartbeat(stamp=stamps.append, interval_sec=0.0, now=lambda: 7.0):
        await asyncio.sleep(0.02)
    assert stamps == [7.0]


async def test_the_beat_task_is_cancelled_on_exit() -> None:
    """A leaked beat task would keep muting the guard after the step ended."""
    stamps: list[float] = []
    async with inline_step_heartbeat(stamp=stamps.append, interval_sec=0.01, now=lambda: 1.0):
        await asyncio.sleep(0.03)
    before = len(stamps)
    await asyncio.sleep(0.05)
    assert len(stamps) == before


async def test_an_exception_inside_the_block_propagates_and_stops_the_beat() -> None:
    stamps: list[float] = []
    with pytest.raises(RuntimeError, match="boom"):
        async with inline_step_heartbeat(stamp=stamps.append, interval_sec=0.01, now=lambda: 1.0):
            raise RuntimeError("boom")
    before = len(stamps)
    await asyncio.sleep(0.03)
    assert len(stamps) == before
