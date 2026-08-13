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

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Iterator

import logging

log = logging.getLogger(__name__)

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


__all__ = ["ProgressReporter", "progress_scope", "report_progress"]
