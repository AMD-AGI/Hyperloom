# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ambient progress heartbeat for long-running tasks.

A composite action — an explore grid over a dozen variants, a baseline
double-run, a profile plus its analysis — is a single ``tasks`` row that
internally completes many units of work over hours. Between dispatch and
return it emits nothing durable, so a healthy 80-minute run and a wedged one
leave identical evidence, and every consumer downstream (stall detection, the
journal, an operator reading the session) is blind for the duration.

The reporter is ambient rather than a parameter because the emitters sit deep
inside call chains that already carry a dozen arguments — the grid runner is
eight frames below the executor entry point, reached through several call
sites that have no business knowing about task bookkeeping. A
:class:`~contextvars.ContextVar` set once at the task boundary reaches all of
them, follows ``asyncio.create_task`` and ``asyncio.to_thread``, and stays
correctly scoped when tasks run concurrently.

Emitters call :func:`report_progress` unconditionally. Outside a scope — a
unit test, a CLI entry point driving an executor directly — it is a no-op.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

import logging

log = logging.getLogger(__name__)

# Well below the 300s a consumer waits before calling an agent silent, so a
# step that is genuinely working is never one missed tick away from an
# accusation.
_OUTPUT_HEARTBEAT_INTERVAL_S: float = 60.0

ProgressReporter = Callable[..., Awaitable[None]]

_REPORTER: ContextVar[ProgressReporter | None] = ContextVar(
    "hyperloom_task_progress_reporter",
    default=None,
)


@contextmanager
def progress_scope(reporter: ProgressReporter | None) -> Iterator[None]:
    """Bind ``reporter`` as the ambient progress sink for the enclosed work.

    Args:
        reporter (ProgressReporter | None): ``async (**note) -> None`` sink,
            typically bound to one task row. ``None`` clears any outer scope,
            which is what nested execution of an unrelated task wants.

    Yields:
        None: For the duration of the ``with`` block.
    """
    token = _REPORTER.set(reporter)
    try:
        yield
    finally:
        _REPORTER.reset(token)


async def report_progress(**note: Any) -> None:
    """Report that the enclosing task finished a unit of work.

    Best-effort in every direction: no scope, a sink that raises, a task row
    reaped underneath — none of it is worth failing the work being reported.

    Args:
        **note (Any): Structured detail for the unit that landed. ``unit``
            (e.g. ``"variant"``, ``"baseline_round"``) and a human-readable
            ``label`` are the conventional keys; consumers treat the rest as
            opaque.
    """
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        await reporter(**note)
    except Exception as exc:  # noqa: BLE001 — a heartbeat never breaks its caller
        log.debug("task progress note dropped: %r", exc)


class OutputActivity:
    """Thread-safe tally of the output a child process has produced.

    The reader lives in a subprocess pump thread while the heartbeat lives on
    the event loop, so the two sides share nothing but this counter.
    """

    def __init__(self) -> None:
        self._lines = 0
        self._lock = threading.Lock()

    def note(self) -> None:
        """Record one more line of child output. Callable from any thread."""
        with self._lock:
            self._lines += 1

    def count(self) -> int:
        """Read the tally.

        Returns:
            int: Lines recorded so far; never decreases.
        """
        with self._lock:
            return self._lines


@asynccontextmanager
async def heartbeat_while_output_flows(
    *,
    interval_s: float = _OUTPUT_HEARTBEAT_INTERVAL_S,
    **note: Any,
) -> AsyncIterator[OutputActivity]:
    """Keep reporting a long step alive for as long as its child keeps talking.

    Never a bare timer: a tick that saw no new output reports nothing, so a
    wedged child falls silent here too and stays accusable. Faking the
    heartbeat would disarm the very signal it feeds.

    Args:
        interval_s (float): Seconds between ticks.
        **note (Any): Fields stamped on every heartbeat, e.g. ``unit`` and
            ``label``.

    Yields:
        OutputActivity: Handle the subprocess reader calls :meth:`note` on.
    """
    activity = OutputActivity()
    driver = asyncio.create_task(_report_new_output(activity, interval_s, note))
    try:
        yield activity
    finally:
        driver.cancel()
        with suppress(asyncio.CancelledError):
            await driver


async def _report_new_output(
    activity: OutputActivity,
    interval_s: float,
    note: dict[str, Any],
) -> None:
    """Report one heartbeat per interval in which new output arrived.

    Args:
        activity (OutputActivity): Counter the reader thread advances.
        interval_s (float): Seconds between ticks.
        note (dict[str, Any]): Fields stamped on every heartbeat.
    """
    seen = 0
    while True:
        await asyncio.sleep(interval_s)
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
