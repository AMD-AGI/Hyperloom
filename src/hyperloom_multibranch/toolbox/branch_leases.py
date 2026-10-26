# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A lease on the local backend: cards out of the measurement pool and a directory of their own.

An autonomous branch runs its own agent, which serves its own models on cards of its own; a tool job (``kernel_tune``)
tunes on a card of its own. The lease takes those cards from the provider's :class:`~hyperloom.sandbox.gpu_pool.GpuPool`,
the pool every measurement sandbox takes its cards from, so a test never runs on a leased card and a leased card is never
measured on, whoever holds it. The cards are promised from the moment the request is checked, while it waits for running
tests to give some back, so the toolbox starts no test the cards would not run. The lease owns only the cards and the
directory: there is no PID namespace on this backend, and confining the agent's processes is its sandbox wrapper's job.

A lease's cards go back only once nothing is listed on them. The holder's processes are stopped before the release, but
the kernel lists the memory of a process that ended until the driver has freed it, and a card handed out before that is
found held by the next test. A card that still holds memory is read again every second for up to :data:`CARD_DRAIN_SEC`,
and one that still does then is kept out of the pool, with a warning naming what is on it, and read again every
:data:`CARD_RECHECK_SEC` until it is free. The kernel's list is read by :func:`hyperloom.sandbox.card_occupancy.read_listed`;
where it cannot be read the cards go back at once, as they did before anything was read.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from hyperloom.common.env_safety import GPU_MASK_ENV_NAMES, is_secret_shaped_env_name, scrub_benchmark_process_env
from hyperloom.measure.sandboxes import DEFAULT_RUNNER_START_TIMEOUT_SEC
from hyperloom.sandbox.card_occupancy import CardReading, read_listed
from hyperloom.sandbox.gated_pool import CardsHeld
from hyperloom.sandbox.local import LocalSandboxProvider
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.toolbox import BranchLease, MeasurementCapacity, RequestRejected

__all__ = [
    "BranchLeases",
    "CARD_DRAIN_SEC",
    "CARD_RECHECK_SEC",
    "CardDrain",
    "LEASE_WAIT_SECONDS",
    "lease_wait",
    "refuse_unfit_cards",
]

logger = logging.getLogger(__name__)

#: As long as a measurement sandbox waits for its cards.
LEASE_WAIT_SECONDS = float(DEFAULT_RUNNER_START_TIMEOUT_SEC)
_POLL_SECONDS = 0.2
_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"
#: How long a released lease's cards are read again for the memory still listed on them before they are kept out: a
#: server that ended frees hundreds of GiB in seconds, one that crashed has taken longer.
CARD_DRAIN_SEC = 60.0
#: How often a card kept out after that is read again, until nothing is listed on it.
CARD_RECHECK_SEC = 30.0
_DRAIN_STEP_SEC = 1.0
_GIB = 1 << 30


@dataclass(frozen=True)
class CardDrain:
    """How a released lease's cards are waited on (see the module's description): ``read`` lists what the kernel has on
    some cards, ``drain_sec`` and ``recheck_sec`` are :data:`CARD_DRAIN_SEC` and :data:`CARD_RECHECK_SEC` unless a test
    says otherwise."""

    read: Callable[[Sequence[int]], CardReading] = read_listed
    drain_sec: float = CARD_DRAIN_SEC
    recheck_sec: float = CARD_RECHECK_SEC


class BranchLeases:
    """The leases on the cards of one local provider, at most one per holder: a branch, or a tool job (``tool:<job id>``).

    ``area`` is the directory the leases' working directories are made in, ``cards_per_test`` the cards one test of
    the session needs, ``image_env`` the environment the session's tests start from. All state is guarded by one lock,
    so a lease and its release may race from any task or thread: a lease's cards go back exactly once. ``on_release`` is
    called after cards went back (on the thread that released, or on the one that waited for them to be freed), so the
    toolbox can start the tests they now let run. ``drain`` says how a released lease's cards are waited on.
    """

    def __init__(
        self,
        provider: LocalSandboxProvider,
        area: Path,
        cards_per_test: int,
        image_env: Mapping[str, str],
        drain: CardDrain = CardDrain(),
    ) -> None:
        self._provider = provider
        self._area = area
        self.cards_per_test = cards_per_test
        self._env = _agent_env(image_env)
        self._guard = threading.Lock()
        self._promised: dict[str, int] = {}
        self._granted: dict[str, BranchLease] = {}
        self._closed = False
        self.on_release: Callable[[], None] = _nothing
        self._read = drain.read
        self._drain_sec = drain.drain_sec
        self._recheck_sec = drain.recheck_sec
        self._held_out: set[int] = set()
        self._stopped = threading.Event()
        self._unread = ""

    def capacity(self) -> MeasurementCapacity:
        """The provider's cards, less the ones lost for good, with the promised ones counted as leased."""

        with self._guard:
            return self._capacity()

    def tests_at_once(self) -> int:
        """How many tests the cards left for measurement run at once, at least one so that a toolbox which lost its
        cards fails its tests instead of parking them."""

        return max(1, self.capacity().measurement_cards // self.cards_per_test)

    def can_lease(self, cards: int) -> bool:
        """Whether a lease of ``cards`` would leave as many cards as one test needs, as far as the pool's size goes."""

        with self._guard:
            return self._leaves_a_test(cards)

    def check(self, cards: int) -> None:
        """Refuse, as :meth:`lease` would now, a lease of ``cards`` that would leave fewer cards than one test needs.

        Nothing is held: a caller that must refuse at once, and lease when its work starts, asks first.

        Raises:
            RequestRejected: ``lease_leaves_no_measurement``, or ``invalid_request`` for fewer than one card.
        """

        with self._guard:
            self._check(cards)

    async def lease(
        self, holder: str, cards: int, root: Path | None = None, *, wait_sec: float | None = None
    ) -> BranchLease:
        """Hold ``cards`` cards and a working directory for ``holder``, waiting for running tests to give cards back.

        ``holder`` is the branch whose agent the cards are for, or ``tool:<job id>`` for a tool job; the lease says it as
        its ``branch_id``. ``root`` is the directory the lease works in, made if it is not there; one of its own below
        the area when omitted. ``wait_sec`` bounds the wait for cards (:func:`lease_wait`): ``None`` is
        :data:`LEASE_WAIT_SECONDS`, ``0`` takes only cards that are free now.

        Raises:
            RequestRejected: ``lease_exists`` when the holder holds a lease, ``lease_leaves_no_measurement`` when
                fewer than one test's cards would be left, ``invalid_request`` for fewer than one card or a
                ``wait_sec`` that is negative or not a number.
            SandboxUnavailable: When the cards were not free within the wait or the toolbox closed.
        """

        wait = lease_wait(wait_sec, LEASE_WAIT_SECONDS)
        self._promise(holder, cards)
        gpu_ids: tuple[int, ...] = ()
        try:
            gpu_ids = await self._take(cards, wait)
            return self._grant(holder, gpu_ids, root)
        except BaseException:
            self._forget(holder, gpu_ids)
            raise

    def release(self, lease_id: str) -> None:
        """Return a lease's cards to the pool once nothing is listed on them; an unknown or released lease changes
        nothing. Cards that are free go back now; the others are read again on a thread of their own (see the module's
        description), so the call does not wait for them."""

        with self._guard:
            lease = self._granted.pop(lease_id, None)
            if lease is None:
                return
            del self._promised[lease.branch_id]
            self._held_out.update(lease.gpu_ids)
        busy = self._busy(lease.gpu_ids)
        self._give_back([card for card in lease.gpu_ids if card not in busy])
        if busy:
            name = f"hl-lease-drain-{lease_id}"
            threading.Thread(target=self._drain, args=(lease_id, busy), name=name, daemon=True).start()

    def held_out(self) -> frozenset[int]:
        """The cards of released leases not yet back in the pool, because memory is still listed on them."""

        with self._guard:
            return frozenset(self._held_out)

    def close(self) -> None:
        """Release every lease and refuse the waiting ones their cards."""

        with self._guard:
            self._closed = True
            granted = tuple(self._granted)
        for lease_id in granted:
            self.release(lease_id)
        self._stopped.set()

    def _capacity(self) -> MeasurementCapacity:
        total = self._provider.pool.total
        usable = total - len(self._provider.quarantined)
        leased = sum(self._promised.values()) + len(self._held_out)
        return MeasurementCapacity(total, usable, leased_cards=min(leased, usable))

    def _leaves_a_test(self, cards: int) -> bool:
        return self._capacity().measurement_cards - cards >= self.cards_per_test

    def _check(self, cards: int) -> None:
        refuse_unfit_cards(cards, self._capacity().measurement_cards, self.cards_per_test)

    def _promise(self, holder: str, cards: int) -> None:
        with self._guard:
            if holder in self._promised:
                raise RequestRejected(f"{holder!r} already holds a lease", "lease_exists")
            self._check(cards)
            self._promised[holder] = cards

    async def _take(self, cards: int, wait: float) -> tuple[int, ...]:
        """The cards, taken as soon as they are free and no process holds them; the pool is asked at least once, whatever
        ``wait`` is."""

        pool = self._provider.pool
        until = time.monotonic() + wait
        held = ""
        while not self._closed:
            try:
                return pool.acquire(cards, 0.0)
            except SandboxUnavailable as refusal:
                held = f": {refusal}" if isinstance(refusal, CardsHeld) else ""
                left = until - time.monotonic()
                if left <= 0:
                    break
                await asyncio.sleep(min(_POLL_SECONDS, left))
        reason = "the toolbox is closing" if self._closed else f"not freed within {wait:g} s{held}"
        raise SandboxUnavailable(f"no {cards} cards for the branch sandbox lease: {reason}")

    def _grant(self, holder: str, gpu_ids: tuple[int, ...], root: Path | None) -> BranchLease:
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        directory = self._area / lease_id if root is None else root
        directory.mkdir(parents=True, exist_ok=root is not None)
        env = {**self._env, _VISIBLE_DEVICES: ",".join(map(str, gpu_ids))}
        with self._guard:
            if self._closed:
                raise SandboxUnavailable("the toolbox is closing")
            lease = BranchLease(
                lease_id, holder, gpu_ids, directory, MappingProxyType(env), self._capacity().measurement_cards
            )
            self._granted[lease_id] = lease
        return lease

    def _busy(self, cards: Sequence[int]) -> dict[int, str]:
        """The cards among ``cards`` the kernel lists memory on, each with what is listed; none when the list cannot be
        read (said once in the log)."""

        reading = self._read(cards)
        if reading.unavailable is not None:
            if reading.unavailable != self._unread:
                self._unread = reading.unavailable
                logger.warning("lease cards go back without a check for memory left on them: %s", reading.unavailable)
            return {}
        busy: dict[int, list[str]] = {}
        for holder in reading.holders:
            if holder.vram > 0 and holder.card in cards:
                busy.setdefault(holder.card, []).append(f"kernel id {holder.pid} with {holder.vram / _GIB:.1f} GiB")
        return {card: ", ".join(held) for card, held in busy.items()}

    def _give_back(self, cards: Sequence[int]) -> None:
        """Put ``cards``, held out since their lease's release, back in the pool and tell the toolbox."""

        if not cards:
            return
        with self._guard:
            self._held_out.difference_update(cards)
            self._provider.pool.release(cards)
        self.on_release()

    def _drain(self, lease_id: str, busy: dict[int, str]) -> None:
        """Read ``busy`` again until nothing is listed on them, giving each card back as it is freed: every second for
        ``drain_sec``, then, the cards kept out with a warning, every ``recheck_sec``; until the leases close."""

        deadline = time.monotonic() + self._drain_sec
        kept_out = False
        while busy and not self._stopped.wait(self._recheck_sec if kept_out else min(_DRAIN_STEP_SEC, self._drain_sec)):
            still = self._busy(tuple(busy))
            freed = [card for card in busy if card not in still]
            self._give_back(freed)
            if freed and kept_out:
                logger.info("cards %s of lease %s are free again and back in the pool", freed, lease_id)
            busy = still
            if busy and not kept_out and time.monotonic() >= deadline:
                kept_out = True
                held = "; ".join(f"card {card}: {what}" for card, what in sorted(busy.items()))
                logger.warning(
                    "cards of lease %s still hold memory %g s after its release; kept out of the pool and read again "
                    "every %g s: %s",
                    lease_id,
                    self._drain_sec,
                    self._recheck_sec,
                    held,
                )

    def _forget(self, holder: str, gpu_ids: tuple[int, ...]) -> None:
        with self._guard:
            del self._promised[holder]
            if gpu_ids:
                self._provider.pool.release(gpu_ids)
        self.on_release()


def lease_wait(wait_sec: float | None, default: float) -> float:
    """How long a lease waits for cards: ``default`` for ``None``, else ``wait_sec``, where ``0`` waits not at all.

    Every backend's lease takes the wait the same way, so a caller reads one meaning of it.

    Raises:
        RequestRejected: ``invalid_request`` for a negative wait or one that is not a finite number.
    """

    if wait_sec is None:
        return default
    if not math.isfinite(wait_sec) or wait_sec < 0:
        raise RequestRejected(f"wait_sec is a finite number of seconds, not negative: {wait_sec}")
    return float(wait_sec)


def refuse_unfit_cards(cards: int, measuring: int, cards_per_test: int) -> None:
    """The refusals every backend's lease shares: ``measuring`` is the cards tests can run on now.

    Raises:
        RequestRejected: ``invalid_request`` for fewer than one card, ``lease_leaves_no_measurement`` when fewer than
            one test's cards would be left.
    """

    if cards < 1:
        raise RequestRejected(f"a lease needs at least one card, not {cards}")
    if measuring - cards < cards_per_test:
        raise RequestRejected(
            f"leasing {cards} of the {measuring} cards tests run on would leave fewer than the "
            f"{cards_per_test} one test needs",
            "lease_leaves_no_measurement",
        )


def _nothing() -> None:
    """The release hook of a holder of cards that has nothing to start when they come back."""


def _agent_env(image_env: Mapping[str, str]) -> dict[str, str]:
    """``image_env`` without credentials, hooks that run code at start-up, or a mask of the visible cards."""

    kept = {
        name: value
        for name, value in image_env.items()
        if name not in GPU_MASK_ENV_NAMES and not is_secret_shaped_env_name(name)
    }
    return scrub_benchmark_process_env(kept)
