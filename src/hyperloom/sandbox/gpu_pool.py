# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A pool of GPU cards leased to sandboxes."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable

from hyperloom.sandbox.socket import SandboxUnavailable

__all__ = ["GpuPool"]


class GpuPool:
    """Hands out disjoint sets of cards and takes them back.

    A card is leased to at most one holder. Cards that are never released
    (because the holder's processes could not be confirmed gone) stay leased:
    a leak is visible as a shrunken pool, never as two tests on one card.
    """

    def __init__(self, gpus: Iterable[int]) -> None:
        cards = tuple(gpus)
        if len(set(cards)) != len(cards):
            raise ValueError(f"duplicate GPU ids in pool: {cards}")
        self._all = tuple(sorted(cards))
        self._free = list(self._all)
        self._cond = threading.Condition()

    @property
    def total(self) -> int:
        """int: How many cards the pool owns."""
        return len(self._all)

    def free_cards(self) -> tuple[int, ...]:
        """tuple[int, ...]: The cards nobody holds right now."""
        with self._cond:
            return tuple(self._free)

    def acquire(self, count: int, wait_sec: float) -> tuple[int, ...]:
        """Lease ``count`` cards, waiting up to ``wait_sec`` for them.

        Args:
            count: How many cards; zero returns an empty lease at once.
            wait_sec: The longest to wait for enough free cards.

        Returns:
            tuple[int, ...]: The leased card ids, ascending.

        Raises:
            SandboxUnavailable: When the pool is smaller than ``count`` or the
                wait ran out.
        """
        if count <= 0:
            return ()
        if count > len(self._all):
            raise SandboxUnavailable(f"template needs {count} GPUs but the pool owns {len(self._all)}")
        deadline = time.monotonic() + max(0.0, wait_sec)
        with self._cond:
            while len(self._free) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxUnavailable(f"no {count} free GPUs within {wait_sec}s ({len(self._free)} free)")
                self._cond.wait(remaining)
            lease = tuple(self._free[:count])
            del self._free[:count]
            return lease

    def release(self, cards: Iterable[int]) -> None:
        """Return leased cards to the pool.

        Args:
            cards: Cards previously returned by :meth:`acquire`.

        Raises:
            ValueError: When a card is not one this pool leased out.
        """
        with self._cond:
            for card in cards:
                if card not in self._all or card in self._free:
                    raise ValueError(f"GPU {card} is not leased from this pool")
                self._free.append(card)
            self._free.sort()
            self._cond.notify_all()
