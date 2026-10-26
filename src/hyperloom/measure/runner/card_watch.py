# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's watch on its own cards: nothing but the test may hold them, before it starts, while it runs and after it ends.

A throughput measured on a card another process was using is not the candidate's number, and nothing in the result would
say so. The runner therefore reads who holds its cards (:mod:`hyperloom.sandbox.card_occupancy`) before the test starts,
at a fixed interval while it runs, and once more when it has ended. A holder that is not a process of this sandbox is
remembered with the time it was first seen. Before the test, a holder is read again for up to ``EXCESS_SETTLE_SEC`` and
counts only if it is still there: the previous test's server, as it exits, stays listed for a few seconds.

While the test runs and after it, the test's own server stops: its descriptors close and the kernel still lists its memory
for as long as the driver takes to free it, which on a GPU pod was seconds and can be longer for hundreds of GiB. In a pod
that memory is then accounted for by no process the container sees, as a foreign holder's would be. The watch therefore
remembers every holder a look found accounted for on a card (the test's own) and does not count memory on a card all of
whose holders it remembers; a holder it never saw accounted for, or a second one, still counts, however long the first one
takes to go.

What is done about it is the card guard's say. Enforcing, which is the default, the result is refused as
``foreign_process_on_card``: an infrastructure failure, whatever the test itself did, because a failure it showed may be the
intruder's doing as much as a number may. The first look during the test that finds such a process ends the test then, as
a boot limit does, rather than at its end: measured on a GPU pod, a test that ran beside a matrix multiply was slowed about
13 times and ran 33 minutes before it could be refused. Observing, nothing is refused and the test runs to its end: the
signed result says ``foreign`` with the process, the card and when, and the runner leaves a note.

Where the probe cannot say who holds the cards (no driver list, ``/proc`` hiding other accounts' processes), the test goes
on in both modes and the signed result says ``unknown`` and why; it is never written as clean. In a pid namespace of its
own the probe counts instead of naming (before the test nothing may hold a card, during it one process may hold memory on
each, and no memory may be held that no process of this container accounts for): the result says so in its ``method``.
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
from hyperloom.sandbox.card_occupancy import (
    CARD_GUARD_ENV,
    EXCESS_SETTLE_SEC,
    METHOD_CENSUS,
    METHOD_CENSUS_FDINFO,
    METHOD_PID_MATCH,
    CardGuard,
    CardReading,
    Intrusion,
    OwnProcesses,
)

__all__ = [
    "AFTER_TEST",
    "BEFORE_TEST",
    "CARD_POLL_SEC",
    "DURING_TEST",
    "ENDED_AT_SIGHTING",
    "CardStatement",
    "CardWatch",
]

#: How often a running test's cards are looked at. A look is a few small file reads, so it is made every second: a
#: process that holds a card and is gone before the next look is not seen, which is the limit of looking at all.
CARD_POLL_SEC = 1.0
BEFORE_TEST = "before the test"
DURING_TEST = "during the test"
AFTER_TEST = "after the test"
#: What a refusal adds when the enforcing watch ended the test at the look that found the intruder.
ENDED_AT_SIGHTING = "the test was ended at that look"
#: How many intruders the failure names, and how much of each one's command line.
_NAMED = 2
_COMMAND_CHARS = 80
_STOP_WAIT_SEC = 10.0
_SETTLE_STEP_SEC = 0.5
_NEVER_LOOKED = "the cards were not looked at"


#: The methods of a reading from the one that says least to the one that says most; a result states the least any look used.
_METHOD_RANK = {METHOD_CENSUS: 0, METHOD_CENSUS_FDINFO: 1, METHOD_PID_MATCH: 2}


@dataclass(frozen=True)
class Sighting:
    """An intruder: something outside the test on one of its cards, with when it was first seen there."""

    intrusion: Intrusion
    first_seen_unix: float
    when: str

    def describe(self) -> str:
        seen = datetime.fromtimestamp(self.first_seen_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        command = f": {self.intrusion.command[:_COMMAND_CHARS]}" if self.intrusion.command else ""
        return f"{self.intrusion.what}, first seen {seen} {self.when}{command}"


@dataclass(frozen=True)
class CardStatement:
    """What the signed result says of the test's cards: ``clean``, ``foreign`` with what was seen, or ``unknown`` with the
    reason a look could not say. ``checks`` is how many looks read the list, and ``method`` how they told a holder from the
    test's own (the least one a look used; ``None`` when none read the list)."""

    state: str
    reason: str | None
    checks: int
    method: str | None = None

    def to_document(self) -> dict[str, object]:
        return {"state": self.state, "reason": self.reason, "checks": self.checks, "method": self.method}


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
        self._sightings: dict[tuple[str | int, ...], Sighting] = {}
        self._unavailable = ""
        self._checks = 0
        self._method: str | None = None
        self._accounted: frozenset[tuple[int, int]] = frozenset()
        self._ended_early = False

    def check(self, when: str) -> None:
        """Look at who holds the cards now; ``when`` says which stage of the test this is, for the record."""
        if not self._cards:
            return
        first_seen = self._clock()
        with self._lock:
            accounted = self._accounted
        reading, found = self._look(when, accounted)
        if when == BEFORE_TEST:
            reading, found = self._settled(reading, found)
        with self._lock:
            if reading.unavailable is not None:
                self._unavailable = self._unavailable or reading.unavailable
                return
            if when != BEFORE_TEST:
                self._accounted |= reading.accounted_holders()
            self._checks += 1
            self._method = _weaker(self._method, reading.method)
            for intrusion in found:
                self._sightings.setdefault(intrusion.key, Sighting(intrusion, first_seen, when))

    def _look(
        self, when: str, accounted: frozenset[tuple[int, int]] = frozenset()
    ) -> tuple[CardReading, tuple[Intrusion, ...]]:
        """One reading of the cards and what on them is not the test's; ``accounted`` are the holders earlier looks found
        to be the test's own."""
        try:
            reading = card_occupancy.read_holders(self._cards)
        except Exception as exc:  # noqa: BLE001 - a look that failed must leave the result unknown, never clean
            return CardReading(unavailable=f"a look at the cards failed ({type(exc).__name__})"), ()
        if reading.unavailable is not None:
            return reading, ()
        return reading, reading.intrusions(self._own, before_test=when == BEFORE_TEST, accounted=accounted)

    def _settled(self, reading: CardReading, found: tuple[Intrusion, ...]) -> tuple[CardReading, tuple[Intrusion, ...]]:
        """Read the cards again while something is on them, until it is gone or ``EXCESS_SETTLE_SEC`` has passed; the
        last reading decides. A process that is exiting (the previous test's server) stays listed for a few seconds and
        is not an intruder; one that is still there after that is."""
        deadline = time.monotonic() + EXCESS_SETTLE_SEC
        while found and time.monotonic() < deadline:
            time.sleep(_SETTLE_STEP_SEC)
            reading, found = self._look(BEFORE_TEST)
        return reading, found

    @contextlib.contextmanager
    def watching(self, end_test: Callable[[], None] | None = None) -> Iterator[None]:
        """Look at the cards every ``poll_sec`` for as long as the block runs, then once more when it has ended.

        Enforcing, the first look that finds a process outside the test calls ``end_test``, once, and the watch stops
        looking until the block has ended: the result is refused whatever else is seen.
        """
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(self._poll_sec):
                self.check(DURING_TEST)
                if end_test is not None and self._guard is CardGuard.ENFORCE and self._seen():
                    self._ended_early = True
                    end_test()
                    return

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
            ended = f"; {ENDED_AT_SIGHTING}" if self._ended_early else ""
            raise RunnerFailure(FOREIGN_PROCESS_ON_CARD, _describe(found) + ended)

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
            return sorted(self._sightings.values(), key=lambda s: (s.first_seen_unix, str(s.intrusion.key)))

    def statement(self) -> CardStatement | None:
        """What the signed result says of its cards: ``foreign`` and who when a process outside the test held one,
        ``clean`` for a test every look of which read the list and found none, ``unknown`` and why when a look could not
        read it; ``None`` for a test with no card."""
        if not self._cards:
            return None
        found = self._seen()
        with self._lock:
            checks, unavailable, method = self._checks, self._unavailable, self._method
        if found:
            return CardStatement(CARD_FOREIGN, _describe(found), checks, method)
        if not checks:
            return CardStatement(CARD_UNKNOWN, unavailable or _NEVER_LOOKED, 0)
        if unavailable:
            return CardStatement(CARD_UNKNOWN, unavailable, checks, method)
        return CardStatement(CARD_CLEAN, None, checks, method)


def _weaker(known: str | None, method: str) -> str:
    return method if known is None or _METHOD_RANK.get(method, 0) < _METHOD_RANK.get(known, 0) else known


def _describe(found: Sequence[Sighting]) -> str:
    named = "; ".join(sighting.describe() for sighting in found[:_NAMED])
    more = f"; and {len(found) - _NAMED} more" if len(found) > _NAMED else ""
    return f"{len(found)} held the test's cards: {named}{more}"
