# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ambient progress heartbeat for long-running tasks."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

import logging

log = logging.getLogger(__name__)

# Well below the 300s a consumer waits before calling an agent silent, so a step that is genuinely working is never
# one missed tick away from an accusation.
_OUTPUT_HEARTBEAT_INTERVAL_S: float = 60.0

# How long teardown waits for the driver to finish the note it is in.
_DRIVER_STOP_GRACE_S: float = 5.0

ProgressReporter = Callable[..., Awaitable[None]]

_REPORTER: ContextVar[ProgressReporter | None] = ContextVar(
    "hyperloom_task_progress_reporter",
    default=None,
)


@contextmanager
def progress_scope(reporter: ProgressReporter | None) -> Iterator[None]:
    """Bind ``reporter`` as the ambient progress sink for the enclosed work."""
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


async def report_progress(**note: Any) -> None:
    """Report that the enclosing task finished a unit of work."""
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        await reporter(**note)
    except Exception as exc:  # noqa: BLE001 — a heartbeat never breaks its caller
        log.debug("task progress note dropped: %r", exc)


class OutputActivity:
    """Thread-safe tally of the output a child process has produced."""

    def __init__(self) -> None:
        self._lines = 0
        self._lock = threading.Lock()

    def note(self) -> None:
        """Record one more line of child output. Callable from any thread."""
        with self._lock:
            self._lines += 1

    def count(self) -> int:
        """Read the tally."""
        with self._lock:
            return self._lines


@asynccontextmanager
async def heartbeat_while_output_flows(
    *,
    interval_s: float | None = None,
    **note: Any,
) -> AsyncIterator[OutputActivity]:
    """Keep reporting a long step alive for as long as its child keeps talking."""
    activity = OutputActivity()
    stop = asyncio.Event()
    tick_s = _OUTPUT_HEARTBEAT_INTERVAL_S if interval_s is None else interval_s
    driver = asyncio.create_task(_report_new_output(activity, tick_s, stop, note))
    try:
        yield activity
    finally:
        await _stop_driver(driver, stop)


async def _stop_driver(driver: asyncio.Task, stop: asyncio.Event) -> None:
    """Stop the heartbeat driver cooperatively, cancelling only if it overruns."""
    stop.set()
    try:
        done, _pending = await asyncio.wait({driver}, timeout=_DRIVER_STOP_GRACE_S)
    except asyncio.CancelledError:
        driver.cancel()
        raise
    if not done:
        # Deliberately not awaited: the driver is stuck in a sink that already overran its grace, and the step it was
        # reporting for has returned.
        driver.cancel()


async def _report_new_output(
    activity: OutputActivity,
    interval_s: float,
    stop: asyncio.Event,
    note: dict[str, Any],
) -> None:
    """Report one heartbeat per interval in which new output arrived."""
    seen = 0
    while True:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        if stop.is_set():
            return
        current = activity.count()
        if current == seen:
            continue
        seen = current
        await report_progress(status="running", output_lines=current, **note)


__all__ = [
    "OutputActivity",
    "ProgressReporter",
    "heartbeat_while_output_flows",
    "progress_scope",
    "report_progress",
]
