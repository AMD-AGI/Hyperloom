# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dispatcher pump that keeps running while a reactor turn waits on its LLM call."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatcher import DispatcherCollaborator

log = logging.getLogger(__name__)


class PumpBesideTurn:
    """Pump the dispatcher at its poll cadence for as long as the ``async with`` body runs.

    A reactor turn holds its tick for as long as its LLM call takes, which only the backend's turn bound limits;
    without this pump every finished task would wait that long to be booked, and every queued one to start.

    Construct it before the turn enters its agent's trajectory scope: the pump runs in the context captured here, so
    the work it starts is attributed to the tick rather than to the agent. A pump in flight when the body ends is
    finished, never cancelled, so a completion it took off the queue is never lost and what follows the body (routing
    the turn's intents) never interleaves with booking.
    """

    def __init__(self, dispatcher: DispatcherCollaborator, stop_requested: Callable[[], bool]) -> None:
        self._dispatcher = dispatcher
        self._stop_requested = stop_requested
        self._context = contextvars.copy_context()
        self._ended = asyncio.Event()
        self._pumping: asyncio.Task[None] | None = None

    async def __aenter__(self) -> None:
        self._pumping = self._context.run(asyncio.create_task, self._pump_until_ended())

    async def __aexit__(self, *_exc_info: object) -> None:
        self._ended.set()
        if self._pumping is not None:
            await self._pumping

    async def _pump_until_ended(self) -> None:
        while True:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._ended.wait(), timeout=self._dispatcher.poll_sec)
            if self._ended.is_set() or self._stop_requested():
                return
            try:
                await self._dispatcher.pump_dispatcher_once()
            except Exception:  # outermost handler on this pump's path; the tick pumps again once the turn ends
                log.exception("turn_pump: the dispatcher pump beside a reactor turn raised; paused until the turn ends")
                return


__all__ = ["PumpBesideTurn"]
