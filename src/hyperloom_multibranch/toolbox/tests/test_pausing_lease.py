# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A pausing lease takes a card even when the tests need every card: it waits for the running test to end, no new test
starts meanwhile, and the tests start again once it is released."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.sandbox.card_occupancy import CardReading
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import LeaseCharge, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases, CardDrain

_CHARGE = LeaseCharge(LeaseKind.TOOL_JOB, "b1", "roofline")
_MADE: list[BranchLeases] = []


@pytest.fixture(autouse=True)
def _closed_after() -> Iterator[None]:
    yield
    while _MADE:
        _MADE.pop().close()


def _leases(tmp_path: Path, *, cards: int, cards_per_test: int) -> tuple[BranchLeases, GpuPool]:
    pool = GpuPool(range(cards))
    provider = SimpleNamespace(pool=pool, quarantined=set())
    leases = BranchLeases(
        provider,  # type: ignore[arg-type]
        tmp_path / "leases",
        cards_per_test,
        {},
        CardDrain(lambda _cards: CardReading(), 0.1, 0.05),
    )
    _MADE.append(leases)
    return leases, pool


async def test_a_pausing_lease_waits_for_the_test_on_every_card_and_holds_back_the_next(tmp_path: Path) -> None:
    leases, pool = _leases(tmp_path, cards=2, cards_per_test=2)
    test_cards = pool.acquire(2, 0.0)
    assert leases.tests_at_once() == 1

    lease = asyncio.create_task(leases.lease("tool:job-1", 1, charge=_CHARGE, wait_sec=30, pausing=True))
    await asyncio.sleep(0.3)
    assert not lease.done()
    assert leases.tests_at_once() == 0

    pool.release(test_cards)
    held = await asyncio.wait_for(lease, 5)
    assert len(held.gpu_ids) == 1
    assert leases.tests_at_once() == 0

    leases.release(held.lease_id)
    assert leases.tests_at_once() == 1
    assert len(pool.free_cards()) == 2


async def test_a_lease_that_is_not_pausing_is_still_refused_when_it_leaves_no_test(tmp_path: Path) -> None:
    leases, _pool = _leases(tmp_path, cards=2, cards_per_test=2)

    with pytest.raises(RequestRejected):
        await leases.lease("b1", 1, charge=LeaseCharge(LeaseKind.DEV, "b1"), wait_sec=0)

    assert leases.tests_at_once() == 1


async def test_a_pausing_lease_beside_room_for_tests_leaves_them_running(tmp_path: Path) -> None:
    leases, _pool = _leases(tmp_path, cards=4, cards_per_test=1)

    held = await leases.lease("tool:job-1", 1, charge=_CHARGE, wait_sec=0, pausing=True)

    assert leases.tests_at_once() == 3
    leases.release(held.lease_id)
    assert leases.tests_at_once() == 4


async def test_a_pausing_lease_that_never_got_its_card_stops_pausing(tmp_path: Path) -> None:
    leases, pool = _leases(tmp_path, cards=2, cards_per_test=2)
    pool.acquire(2, 0.0)

    with pytest.raises(Exception, match="not freed"):
        await leases.lease("tool:job-1", 1, charge=_CHARGE, wait_sec=0.3, pausing=True)

    assert leases.tests_at_once() == 1
