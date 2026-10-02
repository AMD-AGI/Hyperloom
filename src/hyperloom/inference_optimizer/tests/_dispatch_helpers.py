# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive the dispatcher until the work it started has finished and been booked."""

from __future__ import annotations

import asyncio
from typing import Any


async def pump_until_settled(coord: Any, *, timeout: float = 10.0) -> None:
    """Pump until nothing the dispatcher started is running or awaiting bookkeeping."""

    async def _drive() -> None:
        await coord._pump_dispatcher_once()
        while coord._inflight_actions or coord.has_unbooked_completions():
            await coord.wait_for_running_work(timeout=0.05)
            await coord._pump_dispatcher_once()

    await asyncio.wait_for(_drive(), timeout)
