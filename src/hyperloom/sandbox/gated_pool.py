# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A card pool that looks at what holds a card before it hands the card out, and under an enforcing guard does not hand it out.

A card nobody leased is not necessarily a card nobody uses: an agent's command, a tool that outlived its lease or a server
a teardown missed can hold it. Every hand-out of this pool, to a measurement sandbox or to a lease, looks at what holds
the free cards first (:mod:`hyperloom.sandbox.card_occupancy`). Enforcing, it takes only cards nobody holds; a card that is
held stays in the pool and is looked at again at the next try. Observing, which is the default until the probe has been shown
right on the machines that run tests, it hands cards out as a plain pool does and the log says who holds the ones that are
held. Where the probe cannot say who holds the cards they are handed out in both modes and the reason is logged: the runner
of a test states for itself, in the signed result, what it could check.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence

from hyperloom.sandbox.card_occupancy import CardGuard, CardReading, Holder, read_holders
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom.sandbox.socket import SandboxUnavailable

__all__ = ["CardsHeld", "GatedPool"]

log = logging.getLogger(__name__)

_POLL_SEC = 1.0
#: How many holders a refusal names.
_NAMED = 3


class CardsHeld(SandboxUnavailable):
    """Enough cards are free in the pool but processes hold them; ``holders`` are those processes."""

    def __init__(self, message: str, holders: tuple[Holder, ...]) -> None:
        super().__init__(message)
        self.holders = holders


def _described(holders: Sequence[Holder]) -> str:
    named = [f"card {h.card} by pid {h.pid}" + (f" ({h.command})" if h.command else "") for h in holders[:_NAMED]]
    more = len(holders) - len(named)
    return "held " + "; ".join(named) + (f"; and {more} more" if more > 0 else "")


class GatedPool(GpuPool):
    """A :class:`GpuPool` that, when it enforces, hands out only cards no process holds.

    Args:
        gpus: The card numbers, as for :class:`GpuPool`; they are numbered the way ``ROCR_VISIBLE_DEVICES`` numbers cards.
        guard: What is done about a held card: refused under ``ENFORCE``, handed out and logged under ``OBSERVE``.
        read: Who holds some cards: :func:`hyperloom.sandbox.card_occupancy.read_holders` unless a test says otherwise.
        poll_sec: How often a wait looks again, since a card is let go without any call to this pool.
    """

    def __init__(
        self,
        gpus: Iterable[int],
        guard: CardGuard,
        *,
        read: Callable[[Sequence[int]], CardReading] = read_holders,
        poll_sec: float = _POLL_SEC,
    ) -> None:
        super().__init__(gpus)
        self._guard = guard
        self._read = read
        self._poll_sec = poll_sec
        self._unavailable = ""

    def acquire(self, count: int, wait_sec: float) -> tuple[int, ...]:
        """Lease ``count`` cards nobody holds, waiting up to ``wait_sec`` for them to be free and let go.

        Raises:
            CardsHeld: When the guard enforces, enough cards are free but processes hold them, once the wait ran out.
            SandboxUnavailable: When the pool is smaller than ``count`` or too few cards were free within the wait.
        """
        if count <= 0 or count > self.total:
            return super().acquire(count, wait_sec)
        deadline = time.monotonic() + max(0.0, wait_sec)
        while True:
            refusal = None
            try:
                taken = self._take_unheld(count)
            except CardsHeld as held:
                refusal, taken = held, None
            if taken is not None:
                return taken
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise refusal or SandboxUnavailable(
                    f"no {count} free GPUs within {wait_sec}s ({len(self.free_cards())} free)"
                )
            with self._cond:
                self._cond.wait(min(remaining, self._poll_sec))

    def _take_unheld(self, count: int) -> tuple[int, ...] | None:
        """Take ``count`` free cards, none that a process holds when enforcing; ``None`` when too few are free or another
        taker was first.

        Raises:
            CardsHeld: When enforcing, enough cards are free but too few of them are unheld.
        """
        free = self.free_cards()
        if len(free) < count:
            return None
        reading = self._read(free)
        if reading.unavailable is not None:
            self._say_unchecked(reading.unavailable)
            chosen = free[:count]
        elif self._guard is CardGuard.OBSERVE:
            chosen = free[:count]
            held = [holder for holder in reading.holders if holder.card in chosen]
            if held:
                log.warning(
                    "cards a process holds were handed out (the card guard observes only): %s", _described(held)
                )
        else:
            held_cards = reading.held_cards()
            chosen = tuple(card for card in free if card not in held_cards)[:count]
            if len(chosen) < count:
                raise CardsHeld(
                    f"no {count} free GPUs that no process holds: {_described(reading.holders)}", reading.holders
                )
        with self._cond:
            if not all(card in self._free for card in chosen):
                return None
            for card in chosen:
                self._free.remove(card)
        return chosen

    def _say_unchecked(self, reason: str) -> None:
        if reason != self._unavailable:
            self._unavailable = reason
            log.warning("cards are handed out without a check for processes on them: %s", reason)
