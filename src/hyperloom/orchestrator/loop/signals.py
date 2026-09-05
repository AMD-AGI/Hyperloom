# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record an operator's stop the instant it arrives, whatever the loop is doing.

``loop.add_signal_handler`` does not record the stop until the loop is free to
run its callback, which is the wrong order for the one signal whose purpose is
to interrupt something taking too long. :class:`SignalDrain` reads the
interpreter's wakeup pipe from a dedicated thread and records first, dispatches
second.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
from types import FrameType
from typing import Any, Iterable

log = logging.getLogger(__name__)

__all__ = ["SignalDrain"]

#: Signals an operator uses to ask for a graceful stop.
STOP_SIGNALS: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)


class SignalDrain:
    """Capture stop signals into a pipe and fan them out to loop and tick.

    Handlers can only be installed from the main thread of the main
    interpreter. Construction elsewhere installs nothing and reports
    :attr:`armed` as False rather than raising, since a coordinator running off
    the main thread is a supported shape.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        stop_event: asyncio.Event,
        signals: Iterable[int] = STOP_SIGNALS,
    ) -> None:
        """Install the handlers and start the draining thread.

        Args:
            loop: The loop whose ``stop_event`` should be set on arrival.
            stop_event: The asyncio event the run loop waits on.
            signals: Signal numbers to capture.
        """
        self._loop = loop
        self._stop_event = stop_event
        self.requested = threading.Event()
        """threading.Event: Set by the reading thread the instant a stop arrives,
        so synchronous code on the tick's stack can see it without the loop."""
        self._read_fd = -1
        self._write_fd = -1
        self._previous: dict[int, Any] = {}
        self._previous_wakeup: int = -1
        self._thread: threading.Thread | None = None
        self.armed = False
        """bool: Whether handlers were installed."""

        try:
            self._arm(tuple(signals))
        except (ValueError, OSError, RuntimeError) as exc:
            # ValueError is what signal.signal raises off the main thread.
            log.info("SignalDrain: not armed (%s); stop signals will not be captured", exc)
            self.close()

    def _arm(self, signals: tuple[int, ...]) -> None:
        """Create the pipe, install handlers, and start the reader thread."""
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._write_fd, False)
        # A Python-level handler must exist for the interpreter to treat the
        # signal as ours; the wakeup byte is what actually carries it.
        for sig in signals:
            self._previous[sig] = signal.signal(sig, self._noop_handler)
        self._previous_wakeup = signal.set_wakeup_fd(self._write_fd)
        self._thread = threading.Thread(
            target=self._drain,
            name="hyperloom-signal-drain",
            daemon=True,
        )
        self._thread.start()
        self.armed = True

    @staticmethod
    def _noop_handler(_signum: int, _frame: FrameType | None) -> None:
        """Accept the signal so the interpreter writes its wakeup byte."""

    def _drain(self) -> None:
        """Block on the wakeup pipe, publishing every arrival until it closes."""
        while True:
            try:
                data = os.read(self._read_fd, 64)
            except (OSError, ValueError):
                return
            if not data:
                return
            self.requested.set()
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                # The loop is closed; the threading event still stands.
                return

    def _close_fd(self, fd: int, what: str) -> int:
        """Close ``fd`` if it is open, returning the closed sentinel.

        Args:
            fd: The descriptor, or ``-1`` when already closed.
            what: Name of the descriptor, for the failure log line.

        Returns:
            int: ``-1``, whether or not the close succeeded.
        """
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                log.debug("SignalDrain: closing %s failed", what, exc_info=True)
        return -1

    def close(self) -> None:
        """Restore the previous handlers and stop the draining thread.

        Idempotent, and safe to call when arming failed part-way: teardown of a
        stop mechanism must never be the thing that raises.
        """
        if self._previous_wakeup != -1 or self._previous:
            try:
                signal.set_wakeup_fd(self._previous_wakeup if self._previous_wakeup != -1 else -1)
            except (ValueError, OSError):
                log.debug("SignalDrain: restoring the wakeup fd failed", exc_info=True)
            self._previous_wakeup = -1
        for sig, handler in list(self._previous.items()):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, TypeError):
                log.debug("SignalDrain: restoring handler for signal %s failed", sig, exc_info=True)
        self._previous.clear()
        # Closing the write end ends the reader's blocking read.
        self._write_fd = self._close_fd(self._write_fd, "the write end")
        self._read_fd = self._close_fd(self._read_fd, "the read end")
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        self.armed = False
