# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A clock a whole round can be played on without waiting for it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, Callable

__all__ = ["VirtualClock", "installed_clock"]

#: Where the virtual monotonic clock starts. Non-zero, because 0 reads as "unset".
_MONOTONIC_ORIGIN = 10_000.0

#: Where the virtual wall clock starts, as a Unix timestamp. Fixed, so two runs match.
_WALL_ORIGIN = 1_800_000_000.0


class VirtualClock:
    """Monotonic and wall time on one timeline, moving only when charged."""

    def __init__(self, *, start_monotonic: float = _MONOTONIC_ORIGIN, start_wall: float = _WALL_ORIGIN) -> None:
        """Start a clock at rest.

        Args:
            start_monotonic: Initial monotonic reading.
            start_wall: Initial wall reading, as a Unix timestamp.
        """
        self._elapsed = 0.0
        self._start_monotonic = float(start_monotonic)
        self._start_wall = float(start_wall)
        self._marks: list[tuple[float, str]] = []

    @property
    def elapsed(self) -> float:
        """float: Seconds charged to this clock since it was created."""
        return self._elapsed

    @property
    def marks(self) -> tuple[tuple[float, str], ...]:
        """tuple: ``(elapsed, label)`` for every :meth:`mark`, in order."""
        return tuple(self._marks)

    @property
    def installed(self) -> bool:
        """bool: Whether ``time.monotonic`` currently reads this clock.

        A deadline is an instant on whichever clock computed it.
        """
        return time.monotonic == self.monotonic

    def monotonic(self) -> float:
        """Return the virtual ``time.monotonic()`` reading."""
        return self._start_monotonic + self._elapsed

    def wall(self) -> float:
        """Return the virtual ``time.time()`` reading."""
        return self._start_wall + self._elapsed

    def advance(self, seconds: float) -> float:
        """Charge ``seconds`` to the clock.

        Args:
            seconds: Seconds to advance by; negatives are ignored.

        Returns:
            float: The new monotonic reading.
        """
        self._elapsed += max(0.0, float(seconds))
        return self.monotonic()

    def mark(self, label: str) -> None:
        """Record that ``label`` happened at the current reading."""
        self._marks.append((self._elapsed, label))

    def deadline_in(self, seconds: float) -> float:
        """Return the monotonic instant ``seconds`` from now.

        Args:
            seconds: How far ahead the deadline sits.

        Returns:
            float: An absolute reading suitable for ``session_deadline_sec``.
        """
        return self.monotonic() + float(seconds)

    async def sleep(self, seconds: float, real_sleep: Callable[..., Awaitable[Any]]) -> None:
        """Advance the clock instead of waiting.

        Args:
            seconds: Virtual seconds the caller meant to wait.
            real_sleep: The real ``asyncio.sleep``, awaited with ``0`` to yield.
        """
        self.advance(seconds)
        await real_sleep(0)


@contextmanager
def installed_clock(clock: VirtualClock, *modules: Any) -> Iterator[VirtualClock]:
    """Make ``clock`` the time every listed module reads.

    Args:
        clock: The clock to install.
        *modules: Modules that bound ``monotonic`` / ``time`` / ``sleep`` as
            their own name and so do not see the patch to ``time``/``asyncio``.

    Yields:
        VirtualClock: ``clock``, unchanged.
    """
    real_sleep = asyncio.sleep

    async def _sleep(seconds: float, *args: Any, **kwargs: Any) -> None:
        await clock.sleep(seconds, real_sleep)

    with ExitStack() as stack:
        stack.enter_context(_swapped(time, "monotonic", clock.monotonic))
        stack.enter_context(_swapped(time, "time", clock.wall))
        stack.enter_context(_swapped(asyncio, "sleep", _sleep))
        for module in modules:
            for name, replacement in (
                ("monotonic", clock.monotonic),
                ("time", clock.wall),
                ("sleep", _sleep),
            ):
                if callable(getattr(module, name, None)):
                    stack.enter_context(_swapped(module, name, replacement))
        yield clock


@contextmanager
def _swapped(target: Any, name: str, replacement: Any) -> Iterator[None]:
    """Rebind ``target.name`` to ``replacement`` for the duration of the block."""
    original = getattr(target, name)
    setattr(target, name, replacement)
    try:
        yield
    finally:
        setattr(target, name, original)
