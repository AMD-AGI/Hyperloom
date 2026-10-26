# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A released lease's cards go back to the pool only once the kernel lists no memory on them: at once when nothing is
there, when it is freed within the drain time, and otherwise kept out with a warning and read again until it is."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.sandbox.card_occupancy import CardReading, Holder
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases, CardDrain

_GIB = 1 << 30
_DRAIN_SEC = 0.3
_RECHECK_SEC = 0.05
_LOGGER = "hyperloom_multibranch.toolbox.branch_leases"


class Listing:
    """The kernel's list as the leases read it: the test says how many bytes are on which card."""

    def __init__(self) -> None:
        self.vram: dict[int, int] = {}
        self.unavailable: str | None = None
        self.reads = 0
        self._lock = threading.Lock()

    def __call__(self, cards: Sequence[int]) -> CardReading:
        with self._lock:
            self.reads += 1
            if self.unavailable is not None:
                return CardReading(unavailable=self.unavailable)
            held = tuple(Holder(card, 1_827_000 + card, "", vram) for card, vram in sorted(self.vram.items()))
        return CardReading(holders=tuple(h for h in held if h.card in cards))

    def free(self, card: int) -> None:
        with self._lock:
            self.vram.pop(card, None)


_MADE: list[BranchLeases] = []


@pytest.fixture(autouse=True)
def _closed_after() -> Iterator[None]:
    """A test's leases are closed after it, so a card one of them still reads cannot log into the next test."""
    yield
    while _MADE:
        _MADE.pop().close()


def _leases(tmp_path: Path, listing: Listing) -> tuple[BranchLeases, list[float]]:
    provider = SimpleNamespace(pool=GpuPool(range(4)), quarantined=set())
    leases = BranchLeases(
        provider,  # type: ignore[arg-type]
        tmp_path / "leases",
        1,
        {},
        CardDrain(listing, _DRAIN_SEC, _RECHECK_SEC),
    )
    _MADE.append(leases)
    released: list[float] = []
    leases.on_release = lambda: released.append(time.monotonic())
    return leases, released


def _until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


async def test_cards_nothing_is_listed_on_go_back_at_once(tmp_path: Path) -> None:
    listing = Listing()
    leases, released = _leases(tmp_path, listing)
    lease = await leases.lease("b1", 2, wait_sec=0)

    leases.release(lease.lease_id)

    assert set(lease.gpu_ids) <= set(leases._provider.pool.free_cards())
    assert leases.held_out() == frozenset() and len(released) == 1 and listing.reads == 1


async def test_a_card_whose_memory_is_freed_within_the_drain_time_is_not_handed_out_until_it_is(tmp_path: Path) -> None:
    listing = Listing()
    leases, released = _leases(tmp_path, listing)
    lease = await leases.lease("b1", 2, wait_sec=0)
    busy, free = lease.gpu_ids
    listing.vram[busy] = 238 * _GIB  # the agent's server ended; the driver has not freed its memory yet

    leases.release(lease.lease_id)

    assert leases.held_out() == frozenset({busy})
    assert free in leases._provider.pool.free_cards() and busy not in leases._provider.pool.free_cards()
    assert leases.capacity().leased_cards == 1
    with pytest.raises(SandboxUnavailable):
        leases._provider.pool.acquire(4, 0.0)
    other = await leases.lease("b2", 2, wait_sec=0)
    assert busy not in other.gpu_ids
    leases.release(other.lease_id)
    listing.free(busy)
    assert _until(lambda: busy in leases._provider.pool.free_cards())
    assert leases.held_out() == frozenset() and len(released) == 3 and leases.capacity().leased_cards == 0


async def test_a_card_still_holding_memory_after_the_drain_time_is_kept_out_with_a_warning_and_comes_back_once_free(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=_LOGGER)
    listing = Listing()
    leases, _ = _leases(tmp_path, listing)
    lease = await leases.lease("b1", 1, wait_sec=0)
    (card,) = lease.gpu_ids
    listing.vram[card] = 12 * _GIB

    leases.release(lease.lease_id)

    assert _until(lambda: any("kept out of the pool" in r.getMessage() for r in caplog.records))
    warning = next(r for r in caplog.records if f"lease {lease.lease_id} still hold" in r.getMessage())
    assert warning.levelno == logging.WARNING
    assert f"card {card}: kernel id {1_827_000 + card} with 12.0 GiB" in warning.getMessage()
    assert f"read again every {_RECHECK_SEC:g} s" in warning.getMessage()
    reads = listing.reads
    assert _until(lambda: listing.reads >= reads + 3)  # read again while it is kept out
    assert card not in leases._provider.pool.free_cards() and leases.held_out() == frozenset({card})
    listing.free(card)
    assert _until(lambda: card in leases._provider.pool.free_cards())
    assert any("free again and back in the pool" in r.getMessage() for r in caplog.records)
    assert leases.held_out() == frozenset()


async def test_cards_go_back_at_once_when_the_kernels_list_cannot_be_read_and_the_log_says_so_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    listing = Listing()
    listing.unavailable = "no AMD GPU driver is loaded"
    leases, _ = _leases(tmp_path, listing)
    for holder in ("b1", "b2"):
        lease = await leases.lease(holder, 1, wait_sec=0)
        leases.release(lease.lease_id)

    assert leases.held_out() == frozenset() and len(leases._provider.pool.free_cards()) == 4
    unchecked = [r for r in caplog.records if "without a check for memory left on them" in r.getMessage()]
    assert len(unchecked) == 1 and "no AMD GPU driver" in unchecked[0].getMessage()


async def test_closing_the_leases_stops_reading_a_card_that_is_kept_out(tmp_path: Path) -> None:
    listing = Listing()
    leases, _ = _leases(tmp_path, listing)
    lease = await leases.lease("b1", 1, wait_sec=0)
    listing.vram[lease.gpu_ids[0]] = _GIB

    leases.release(lease.lease_id)
    leases.close()

    assert _until(lambda: not [t for t in threading.enumerate() if t.name == f"hl-lease-drain-{lease.lease_id}"])
    assert leases.held_out() == frozenset(lease.gpu_ids)
