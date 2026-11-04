# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The blackboard's record of the local toolbox's leases, read back through the real blackboard: the grant, each card's
return as the drain gives it back, the close, a restart after a toolbox died, and what a failed write does.

The toolbox is the real local one of ``test_branch_leases`` (``local``), its blackboard's lineage opened for the branches
the tests lease for; ``Listing`` is the kernel's memory list a released lease's cards are read for.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import CardHold, LeaseEnd, LeaseStart
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseEndReason, LeaseKind
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.lease_ledger import LeaseLedger
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.tests.test_branch_leases import Make, local  # noqa: F401 - fixture
from hyperloom_multibranch.toolbox.tests.test_lease_drain import Listing

_GIB = 1 << 30
_LEDGER = "hyperloom_multibranch.toolbox.lease_ledger"
Holds = tuple[CardHold, ...]


def _reader(tmp_path: Path, session: int = 0) -> LocalBlackboard:
    return LocalBlackboard(tmp_path / f"session-{session}" / "blackboard", Writer(Role.SCHEDULER, "scheduler"))


async def _holds(blackboard: LocalBlackboard, until: Callable[[Holds], bool]) -> Holds:
    """The blackboard's holds once ``until`` says they are what the test waits for: ends are written on the toolbox's
    loop after the cards went back."""

    deadline = time.monotonic() + 10.0
    while True:
        holds = await blackboard.card_holds()
        if until(holds):
            return holds
        assert time.monotonic() < deadline, f"the holds never got there: {holds}"
        await asyncio.sleep(0.02)


async def _held(blackboard: LocalBlackboard, branch: str) -> float:
    (row,) = [row for row in await blackboard.progress() if row.branch == branch]
    return row.held_card_seconds


async def test_a_lease_charges_its_branch_from_the_grant_until_each_card_is_back_in_the_pool(
    local: Make, tmp_path: Path
) -> None:
    listing = Listing()
    box = await local(cards=3, card_memory=listing)
    reader = _reader(tmp_path)
    asked = time.time()
    lease = await box.lease_branch_sandbox("agent", 2)
    granted = time.time()
    busy, free = lease.gpu_ids
    listing.vram[busy] = 8 * _GIB  # the agent's server ended; the driver has not freed its memory yet

    await box.release_branch_sandbox(lease.lease_id)
    (partly,) = await _holds(reader, lambda holds: bool(holds) and free in holds[0].returned)
    assert busy not in partly.returned and partly.end_reason is None, "the busy card still charges its holder"
    freed = time.time()
    listing.free(busy)
    (hold,) = await _holds(reader, lambda holds: holds[0].closed)

    start = hold.start
    assert (start.holder, start.branch_id, start.kind, start.tool, start.cards) == (
        "agent",
        "agent",
        LeaseKind.DEV,
        "",
        2,
    )
    assert start.gpu_ids == lease.gpu_ids and start.sandbox_id == "" and asked <= start.started_at <= granted
    assert hold.returned[free] <= freed <= hold.returned[busy], "each card is charged until its own return"
    assert hold.end_reason is LeaseEndReason.RELEASED
    charged = (hold.returned[free] - start.started_at) + (hold.returned[busy] - start.started_at)
    assert await _held(reader, "agent") == pytest.approx(charged)


async def test_the_close_ends_every_lease_still_out_with_session_close(local: Make, tmp_path: Path) -> None:
    listing = Listing()
    box = await local(cards=4, card_memory=listing)
    reader = _reader(tmp_path)
    held = await box.lease_branch_sandbox("b02", 1)
    draining = await box.lease_branch_sandbox("b03", 2)
    busy, free = draining.gpu_ids
    listing.vram[busy] = 8 * _GIB  # kept out past the close: the drain time is a minute
    await box.release_branch_sandbox(draining.lease_id)
    await _holds(reader, lambda holds: len(holds) == 2 and free in holds[1].returned)

    closing = time.time()
    await box.close()

    holds = {hold.start.holder: hold for hold in await reader.card_holds()}
    assert all(hold.closed and hold.end_reason is LeaseEndReason.SESSION_CLOSE for hold in holds.values()), holds
    assert holds["b02"].returned[held.gpu_ids[0]] >= closing
    assert holds["b03"].returned[free] < closing <= holds["b03"].returned[busy], "a card's first return stands"


async def test_a_restarted_toolbox_ends_its_predecessors_open_leases_at_its_last_sign_of_life(
    local: Make, tmp_path: Path
) -> None:
    dead = await local(cards=4)  # it grants two leases and dies: no release, no close
    agent = await dead.lease_branch_sandbox("agent", 1)
    await asyncio.sleep(0.05)
    await dead.lease_branch_sandbox("b02", 2)
    reader = _reader(tmp_path)
    last_seen = await reader.toolbox_seen_at()
    assert last_seen is not None

    await local(cards=4, blackboard=tmp_path / "session-0" / "blackboard")

    holds = {hold.start.holder: hold for hold in await reader.card_holds()}
    for hold in holds.values():
        assert hold.end_reason is LeaseEndReason.TOOLBOX_LOST
        assert set(hold.returned.values()) == {last_seen}, "closed when the blackboard last saw the toolbox"
    started = holds["agent"].start.started_at
    assert started < last_seen and holds["agent"].start.gpu_ids == agent.gpu_ids
    assert await _held(reader, "agent") == pytest.approx(last_seen - started)


async def test_a_lease_whose_branch_the_lineage_has_not_opened_is_refused_and_holds_nothing(
    local: Make, tmp_path: Path
) -> None:
    box = await local(cards=2)

    with pytest.raises(RequestRejected) as refused:
        await box.lease_branch_sandbox("never-opened", 1)

    assert refused.value.code == "invalid_request" and "has not opened branch 'never-opened'" in refused.value.reason
    assert await box.measurement_capacity() == MeasurementCapacity(2, 2) and box.provider.pool.free_cards() == (0, 1)
    assert await _reader(tmp_path).card_holds() == ()
    assert (await box.lease_branch_sandbox("agent", 1)).branch_id == "agent", "nothing is left promised"


class _Flaky:
    """The blackboard, failing the first ``starts`` lease starts and ``ends`` lease ends it is asked to keep."""

    def __init__(self, blackboard: LocalBlackboard, *, starts: int = 0, ends: int = 0) -> None:
        self._blackboard, self._starts, self._ends = blackboard, starts, ends

    async def record_lease_start(self, start: LeaseStart) -> None:
        if self._starts:
            self._starts -= 1
            raise OSError("disk full")
        await self._blackboard.record_lease_start(start)

    async def record_lease_end(self, end: LeaseEnd) -> None:
        if self._ends:
            self._ends -= 1
            raise OSError("disk full")
        await self._blackboard.record_lease_end(end)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._blackboard, name)


def _flaky_ledger(box: LocalHostedToolbox, tmp_path: Path, **failing: int) -> None:
    """Give ``box`` a ledger whose blackboard fails as ``failing`` says."""

    blackboard = LocalBlackboard(tmp_path / "session-0" / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    ledger = LeaseLedger(_Flaky(blackboard, **failing), time.time)  # type: ignore[arg-type]
    ledger.bind(asyncio.get_running_loop())
    box._ledger = box.leases.ledger = ledger


async def test_a_grant_the_blackboard_could_not_record_is_refused_and_its_cards_go_back(
    local: Make, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    box = await local(cards=2)
    _flaky_ledger(box, tmp_path, starts=1)

    with pytest.raises(SandboxUnavailable, match="could not be recorded: disk full"):
        await box.lease_branch_sandbox("agent", 1)

    assert await box.measurement_capacity() == MeasurementCapacity(2, 2) and box.provider.pool.free_cards() == (0, 1)
    assert "it is refused" in caplog.text and await _reader(tmp_path).card_holds() == ()
    await box.close()  # the refused lease is not among the ones the close gives back
    assert box.provider.pool.free_cards() == (0, 1)


async def test_a_return_the_blackboard_could_not_record_still_frees_the_cards_and_the_close_ends_the_lease(
    local: Make, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger=_LEDGER)
    box = await local(cards=2)
    _flaky_ledger(box, tmp_path, ends=1)
    reader = _reader(tmp_path)
    lease = await box.lease_branch_sandbox("agent", 1)

    await box.release_branch_sandbox(lease.lease_id)
    await _until_logged(caplog, "could not be recorded")

    assert box.provider.pool.free_cards() == (0, 1), "the cards are back whatever the blackboard says"
    (hold,) = await reader.card_holds()
    assert not hold.closed, "the lease keeps charging until something ends it"
    closing = time.time()
    await box.close()
    (hold,) = await reader.card_holds()
    assert hold.end_reason is LeaseEndReason.SESSION_CLOSE and hold.returned[lease.gpu_ids[0]] >= closing


async def _until_logged(caplog: pytest.LogCaptureFixture, text: str) -> None:
    deadline = time.monotonic() + 10.0
    while text not in caplog.text:
        assert time.monotonic() < deadline, f"never logged {text!r}"
        await asyncio.sleep(0.02)
