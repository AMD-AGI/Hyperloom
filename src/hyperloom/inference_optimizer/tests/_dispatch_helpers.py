# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive the dispatcher until the work it started has finished and been booked."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator


async def pump_until_settled(dispatcher: "DispatcherCollaborator", *, timeout: float = 10.0) -> None:
    """Pump until nothing the dispatcher started is running or awaiting bookkeeping."""

    async def _drive() -> None:
        await dispatcher.pump_dispatcher_once()
        while dispatcher._inflight_actions or dispatcher.has_unbooked_completions():
            await dispatcher.wait_for_running_work(timeout=0.05)
            await dispatcher.pump_dispatcher_once()

    await asyncio.wait_for(_drive(), timeout)
