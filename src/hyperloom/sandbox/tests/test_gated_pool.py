# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pool hands out only cards no process holds, and says why when it does not."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from hyperloom.sandbox.card_occupancy import CardReading, Holder, read_holders
from hyperloom.sandbox.gated_pool import CardsHeld, GatedPool
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_INTRUDER = Holder(card=1, pid=4242, command="python3 -c import torch")


class _Probe:
    """A probe whose answer the test sets; it records which cards it was asked about."""

    def __init__(self, reading: CardReading = CardReading()) -> None:
        self.reading = reading
        self.asked: list[tuple[int, ...]] = []

    def __call__(self, cards: Sequence[int]) -> CardReading:
        self.asked.append(tuple(cards))
        held = tuple(holder for holder in self.reading.holders if holder.card in cards)
        return CardReading(holders=held, unavailable=self.reading.unavailable)


def test_cards_nobody_holds_are_handed_out_lowest_first_like_a_plain_pool() -> None:
    pool = GatedPool(range(4), read=_Probe())

    assert pool.acquire(2, 0.0) == (0, 1)
    assert pool.free_cards() == (2, 3)


def test_a_held_card_is_skipped_and_stays_in_the_pool() -> None:
    pool = GatedPool(range(4), read=_Probe(CardReading((_INTRUDER,))))

    assert pool.acquire(2, 0.0) == (0, 2)
    assert pool.free_cards() == (1, 3)


def test_when_too_few_cards_are_unheld_the_refusal_names_who_holds_them() -> None:
    pool = GatedPool(range(2), read=_Probe(CardReading((_INTRUDER,))))

    with pytest.raises(CardsHeld) as refused:
        pool.acquire(2, 0.0)

    assert "card 1 by pid 4242 (python3 -c import torch)" in str(refused.value)
    assert refused.value.holders == (_INTRUDER,)
    assert pool.free_cards() == (0, 1)


def test_a_refusal_names_at_most_three_holders_and_counts_the_rest() -> None:
    holders = tuple(Holder(card=card, pid=100 + card, command="") for card in range(5))
    pool = GatedPool(range(5), read=_Probe(CardReading(holders)))

    with pytest.raises(CardsHeld, match=r"card 0 by pid 100; card 1 by pid 101; card 2 by pid 102; and 2 more"):
        pool.acquire(1, 0.0)


def test_a_wait_ends_when_the_holder_lets_go() -> None:
    probe = _Probe(CardReading((Holder(0, 9, "x"),)))
    pool = GatedPool(range(1), read=probe, poll_sec=0.01)
    timer = threading.Timer(0.1, lambda: setattr(probe, "reading", CardReading()))
    timer.start()
    try:
        assert pool.acquire(1, 5.0) == (0,)
    finally:
        timer.cancel()
    assert len(probe.asked) > 1


def test_a_wait_that_runs_out_while_the_card_is_held_is_a_refusal_not_a_hand_out() -> None:
    pool = GatedPool(range(1), read=_Probe(CardReading((Holder(0, 9, "x"),))), poll_sec=0.01)

    with pytest.raises(CardsHeld, match="within|no 1 free GPUs that no process holds"):
        pool.acquire(1, 0.05)


def test_only_the_free_cards_are_looked_at_a_leased_card_may_be_in_use() -> None:
    probe = _Probe()
    pool = GatedPool(range(4), read=probe)
    pool.acquire(2, 0.0)
    probe.asked.clear()

    pool.acquire(1, 0.0)

    assert probe.asked == [(2, 3)]


def test_too_few_free_cards_is_the_plain_pools_refusal_not_a_held_one() -> None:
    pool = GatedPool(range(2), read=_Probe())
    pool.acquire(2, 0.0)

    with pytest.raises(SandboxUnavailable, match=r"no 1 free GPUs within 0.0s \(0 free\)") as refused:
        pool.acquire(1, 0.0)

    assert not isinstance(refused.value, CardsHeld)


def test_a_pool_smaller_than_the_request_refuses_as_a_plain_pool_does() -> None:
    with pytest.raises(SandboxUnavailable, match="needs 3 GPUs but the pool owns 2"):
        GatedPool(range(2), read=_Probe()).acquire(3, 0.0)


def test_a_card_released_while_a_process_still_holds_it_is_not_handed_out_again() -> None:
    """A lease that ends with its agent's process still on the card must not give that card to a test."""
    probe = _Probe()
    pool = GatedPool(range(2), read=probe)
    leased = pool.acquire(1, 0.0)
    probe.reading = CardReading((Holder(leased[0], 77, "agent-tool"),))
    pool.release(leased)

    assert pool.acquire(1, 0.0) == (1,)
    with pytest.raises(CardsHeld, match="pid 77"):
        pool.acquire(1, 0.0)


def test_where_the_probe_cannot_say_cards_are_handed_out_and_the_reason_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = GatedPool(range(4), read=_Probe(CardReading(unavailable="no AMD GPU driver is loaded")))

    with caplog.at_level(logging.WARNING, logger="hyperloom.sandbox.gated_pool"):
        assert pool.acquire(1, 0.0) == (0,)
        assert pool.acquire(1, 0.0) == (1,)

    assert [record.getMessage() for record in caplog.records] == [
        "cards are handed out without a check for processes on them: no AMD GPU driver is loaded"
    ]


def test_two_takers_never_get_the_same_card() -> None:
    pool = GatedPool(range(8), read=_Probe(), poll_sec=0.01)
    taken: list[tuple[int, ...]] = []
    lock = threading.Lock()

    def take() -> None:
        cards = pool.acquire(1, 5.0)
        with lock:
            taken.append(cards)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(card for cards in taken for card in cards) == list(range(8))


def test_the_default_probe_reads_the_kernels_list(tmp_path: Path) -> None:
    tree = KernelTree(tmp_path, gpus=4)
    tree.process(900, command=("python3", "intruder.py"))
    tree.holds(900, 0)
    pool = GatedPool(range(4), read=lambda cards: read_holders(cards, kfd=tree.kfd, procfs=tree.procfs))

    assert pool.acquire(2, 0.0) == (1, 2)
    with pytest.raises(CardsHeld, match="card 0 by pid 900 .python3 intruder.py."):
        pool.acquire(2, 0.0)
    assert pool.acquire(1, 0.0) == (3,)
