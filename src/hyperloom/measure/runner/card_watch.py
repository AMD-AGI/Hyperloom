# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's watch on its own cards: nothing but the test may hold them, before it starts, while it runs and after it ends.

A throughput measured on a card another process was using is not the candidate's number, and nothing in the result would
say so. The runner therefore reads who holds its cards (:mod:`hyperloom.sandbox.card_occupancy`) before the test starts,
at a fixed interval while it runs, and once more when it has ended. A holder that is not a process of this sandbox is
remembered with the time it was first seen.

What is done about it is the card guard's say. Enforcing, the result is refused as ``foreign_process_on_card``: an
infrastructure failure, whatever the test itself did, because a failure it showed may be the intruder's doing as much as a
number may. Observing, which is the default until the probe has been shown right on the machines that run tests, nothing is
refused: the signed result says ``foreign`` with the process, the card and when, and the runner leaves a note.

Where the probe cannot say who holds the cards (no driver list, a pid namespace of its own), the test goes on in both modes
and the signed result says ``unknown`` and why; it is never written as clean.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from hyperloom.contracts.measurement import CARD_CLEAN, CARD_FOREIGN, CARD_UNKNOWN
from hyperloom.measure.failures import FOREIGN_PROCESS_ON_CARD
from hyperloom.measure.runner.service import RunnerFailure
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CARD_GUARD_ENV, CardGuard, CardReading, Holder, OwnProcesses

__all__ = [
    "AFTER_TEST",
    "BEFORE_TEST",
    "CARD_POLL_SEC",
    "DURING_TEST",
    "CardStatement",
    "CardWatch",
]

#: How often a running test's cards are looked at. A look is a few small file reads, so it is made every second: a
#: process that holds a card and is gone before the next look is not seen, which is the limit of looking at all.
CARD_POLL_SEC = 1.0
BEFORE_TEST = "before the test"
DURING_TEST = "during the test"
AFTER_TEST = "after the test"
#: How many intruders the failure names, and how much of each one's command line.
_NAMED = 2
_COMMAND_CHARS = 80
_STOP_WAIT_SEC = 10.0
_NEVER_LOOKED = "the cards were not looked at"


@dataclass(frozen=True)
class Sighting:
    """An intruder: a process outside the test that held one of its cards, with when it was first seen there."""

    holder: Holder
    first_seen_unix: float
    when: str

    def describe(self) -> str:
        seen = datetime.fromtimestamp(self.first_seen_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        command = f": {self.holder.command[:_COMMAND_CHARS]}" if self.holder.command else ""
        return f"pid {self.holder.pid} on card {self.holder.card}, first seen {seen} {self.when}{command}"


@dataclass(frozen=True)
class CardStatement:
    """What the signed result says of the test's cards: ``clean``, ``foreign`` with the process that was seen, or ``unknown``
    with the reason a look could not say. ``checks`` is how many looks read the list."""

    state: str
    reason: str | None
    checks: int

    def to_document(self) -> dict[str, object]:
        return {"state": self.state, "reason": self.reason, "checks": self.checks}


class CardWatch:
    """Watches ``cards`` for processes that are not ``own``'s, and applies ``guard`` to what it sees.

    ``cards`` are numbered as ``ROCR_VISIBLE_DEVICES`` numbers them. With none (a test that has no card) there is nothing
    to watch and the result states nothing about its cards.
    """

    def __init__(
        self,
        cards: Sequence[int],
        own: OwnProcesses,
        guard: CardGuard,
        *,
        poll_sec: float = CARD_POLL_SEC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cards = tuple(cards)
        self._own = own
        self._guard = guard
        self._poll_sec = poll_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._sightings: dict[tuple[int, int], Sighting] = {}
        self._unavailable = ""
        self._checks = 0

    def check(self, when: str) -> None:
        """Look at who holds the cards now; ``when`` says which stage of the test this is, for the record."""
        if not self._cards:
            return
        found: tuple[Holder, ...] = ()
        try:
            reading = card_occupancy.read_holders(self._cards)
            if reading.unavailable is None:
                found = reading.foreign(self._own)
        except Exception as exc:  # noqa: BLE001 - a look that failed must leave the result unknown, never clean
            reading = CardReading(unavailable=f"a look at the cards failed ({type(exc).__name__})")
        with self._lock:
            if reading.unavailable is not None:
                self._unavailable = self._unavailable or reading.unavailable
                return
            self._checks += 1
            for holder in found:
                self._sightings.setdefault((holder.pid, holder.card), Sighting(holder, self._clock(), when))

    @contextlib.contextmanager
    def watching(self) -> Iterator[None]:
        """Look at the cards every ``poll_sec`` for as long as the block runs, then once more when it has ended."""
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(self._poll_sec):
                self.check(DURING_TEST)

        thread = threading.Thread(target=watch, name="hl-card-watch", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=_STOP_WAIT_SEC)
            self.check(AFTER_TEST)

    def enforce_guard(self) -> None:
        """Raise the failure that refuses the result when the guard enforces and a process outside the test held a card
        at any look; observing, do nothing (:meth:`observation` and :meth:`statement` carry what was seen)."""
        found = self._seen()
        if self._guard is CardGuard.ENFORCE and found:
            raise RunnerFailure(FOREIGN_PROCESS_ON_CARD, _describe(found))

    def observation(self) -> str | None:
        """The note a runner that only observes leaves when a process outside the test held a card, else ``None``."""
        found = self._seen()
        if self._guard is CardGuard.OBSERVE and found:
            return (
                f"CARD GUARD OBSERVES ONLY ({CARD_GUARD_ENV}=observe): this result was kept although {_describe(found)}"
            )
        return None

    def _seen(self) -> list[Sighting]:
        with self._lock:
            return sorted(self._sightings.values(), key=lambda s: (s.first_seen_unix, s.holder.card, s.holder.pid))

    def statement(self) -> CardStatement | None:
        """What the signed result says of its cards: ``foreign`` and who when a process outside the test held one,
        ``clean`` for a test every look of which read the list and found none, ``unknown`` and why when a look could not
        read it; ``None`` for a test with no card."""
        if not self._cards:
            return None
        found = self._seen()
        with self._lock:
            checks, unavailable = self._checks, self._unavailable
        if found:
            return CardStatement(CARD_FOREIGN, _describe(found), checks)
        if not checks:
            return CardStatement(CARD_UNKNOWN, unavailable or _NEVER_LOOKED, 0)
        if unavailable:
            return CardStatement(CARD_UNKNOWN, unavailable, checks)
        return CardStatement(CARD_CLEAN, None, checks)


def _describe(found: Sequence[Sighting]) -> str:
    named = "; ".join(sighting.describe() for sighting in found[:_NAMED])
    more = f"; and {len(found) - _NAMED} more" if len(found) > _NAMED else ""
    return f"{len(found)} held the test's cards: {named}{more}"
