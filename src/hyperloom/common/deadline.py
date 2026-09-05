# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An absolute instant work must stop by, and the arithmetic that keeps it absolute.

:class:`Deadline` is how a bound is held in this process: ``None`` is the only
spelling of unbounded, and one with ``<= 0`` remaining is expired and every
consumer must honour it as a stop. ``time.monotonic()`` has a per-process
origin, so a bound that crosses a process boundary travels as
:meth:`Deadline.remaining` seconds and is re-anchored on the far side with
:meth:`Deadline.after`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = [
    "Deadline",
    "seconds_until",
]


@dataclass(frozen=True)
class Deadline:
    """An absolute ``time.monotonic()`` instant that bounds a piece of work."""

    at: float
    """The monotonic reading the work must have stopped by."""

    @classmethod
    def after(cls, seconds: float, *, now: float | None = None) -> "Deadline":
        """Anchor a deadline ``seconds`` from now.

        Args:
            seconds: Seconds of budget; a non-positive value anchors in the past.
            now: Monotonic reading to anchor from; defaults to the live clock.

        Returns:
            Deadline: The absolute instant the budget runs out.
        """
        origin = time.monotonic() if now is None else float(now)
        return cls(at=origin + float(seconds))

    def remaining(self) -> float:
        """Seconds left before this deadline, signed: negative once it has passed."""
        return self.at - time.monotonic()

    def expired(self) -> bool:
        """Whether no time is left."""
        return self.remaining() <= 0.0

    def tightened_to(self, other: "Deadline | None") -> "Deadline":
        """Return whichever of the two stops sooner.

        Combining bounds only ever shortens; no spelling here grants time.

        Args:
            other: A second bound, or ``None`` for unbounded.

        Returns:
            Deadline: The earlier of ``self`` and ``other``.
        """
        if other is None:
            return self
        return self if self.at <= other.at else other


def seconds_until(deadline: "Deadline | None", *, unbounded_cap: float) -> float:
    """Convert a bound into the finite, non-negative wait a poller may use.

    Args:
        deadline: The bound, or ``None`` for unbounded.
        unbounded_cap: Finite seconds to allow when nothing bounds the work.

    Returns:
        float: ``unbounded_cap`` when ``deadline`` is ``None``, otherwise the
            time left, floored at ``0.0``.

    Raises:
        ValueError: If ``unbounded_cap`` is not a positive finite number.
    """
    cap = float(unbounded_cap)
    if not cap > 0.0 or cap == float("inf"):
        raise ValueError(f"unbounded_cap must be a positive finite number, got {unbounded_cap!r}")
    if deadline is None:
        return cap
    return max(0.0, deadline.remaining())
