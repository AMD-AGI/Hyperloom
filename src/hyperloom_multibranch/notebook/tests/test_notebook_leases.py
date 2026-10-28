# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Card leases in the notebook: the toolbox's starts and ends, folded into holds and charged to their branches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, WriterNotAllowed
from hyperloom_multibranch.contracts.notebook.records import LeaseEnd, LeaseStart, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import LeaseEndReason, LeaseKind, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import START, FakeClock

Open = Callable[..., LocalNotebook]

#: An Arbor branch's sandbox on two local cards, and a Forge job on a hosted sandbox of two.
ARBOR = LeaseStart("lease-1", "b01", "b01", LeaseKind.AUTONOMOUS, 2, START, gpu_ids=(4, 5))
FORGE = LeaseStart("lease-2", "tool:job-7", "b01", LeaseKind.TOOL_JOB, 2, START, tool="forge", sandbox_id="sb-1")


async def _opened(open_as: Open) -> LocalNotebook:
    """A toolbox handle on a notebook whose lineage opened ``b01`` from the root."""

    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b01", from_commit="c0"))
    return open_as(Role.TOOLBOX)


async def test_a_lease_charges_its_branch_each_card_from_the_grant_until_it_goes_back(
    open_as: Open, clock: FakeClock
) -> None:
    toolbox = await _opened(open_as)
    await toolbox.record_lease_start(ARBOR)
    await toolbox.record_lease_end(LeaseEnd("lease-1", START + 100.0, LeaseEndReason.RELEASED, (4,)))
    clock.now = START + 300.0

    (row,) = [item for item in await toolbox.progress() if item.branch == "b01"]
    (open_hold,) = await toolbox.card_holds()

    # Card 4 went back at 100 s; card 5 is still out, so it counts up to the reading.
    assert row.held_card_seconds == pytest.approx(100.0 + 300.0, abs=0.01)
    assert not open_hold.closed and open_hold.end_reason is None

    await toolbox.record_lease_end(LeaseEnd("lease-1", START + 400.0, LeaseEndReason.SESSION_CLOSE))
    clock.now = START + 900.0
    snapshot = await toolbox.snapshot()

    (hold,) = snapshot.card_holds
    assert hold.returned == {4: START + 100.0, 5: START + 400.0}
    assert hold.closed and hold.end_reason is LeaseEndReason.SESSION_CLOSE
    assert snapshot.held_card_hours == pytest.approx(500.0 / 3600.0)
    assert {item.branch: item.held_card_seconds for item in snapshot.progress} == {
        INTEGRATION_BRANCH: 0.0,
        "b01": pytest.approx(500.0),
    }


async def test_a_hosted_lease_goes_back_whole_and_a_cards_first_return_stands(open_as: Open) -> None:
    toolbox = await _opened(open_as)
    await toolbox.record_lease_start(FORGE)
    await toolbox.record_lease_end(LeaseEnd("lease-2", START + 50.0, LeaseEndReason.RELEASED))
    await toolbox.record_lease_start(replace(ARBOR, lease_id="lease-3", cards=1, gpu_ids=(0,)))
    await toolbox.record_lease_end(LeaseEnd("lease-3", START + 10.0, LeaseEndReason.RELEASED, (0,)))
    await toolbox.record_lease_end(LeaseEnd("lease-3", START + 20.0, LeaseEndReason.TOOLBOX_LOST, (0,)))

    forge, retried = await toolbox.card_holds()

    assert forge.returned == {0: START + 50.0, 1: START + 50.0} and forge.card_seconds(START + 999.0) == 100.0
    assert retried.returned == {0: START + 10.0} and retried.end_reason is LeaseEndReason.RELEASED


@pytest.mark.parametrize(
    "start",
    [
        replace(ARBOR, branch_id="b09"),
        replace(ARBOR, cards=3),
        replace(ARBOR, cards=0, gpu_ids=()),
        replace(FORGE, tool=""),
        replace(ARBOR, tool="forge"),
        replace(ARBOR, kind="sandbox"),
    ],
)
async def test_a_lease_start_the_notebook_cannot_charge_or_count_is_refused(open_as: Open, start: LeaseStart) -> None:
    toolbox = await _opened(open_as)

    with pytest.raises(ValueError):
        await toolbox.record_lease_start(start)
    assert await toolbox.card_holds() == ()


async def test_a_lease_end_of_no_lease_or_card_and_a_lease_started_twice_are_refused(open_as: Open) -> None:
    toolbox = await _opened(open_as)
    await toolbox.record_lease_start(ARBOR)

    with pytest.raises(ValueError, match="already started"):
        await toolbox.record_lease_start(replace(ARBOR, gpu_ids=(6, 7)))
    with pytest.raises(ValueError, match="never started"):
        await toolbox.record_lease_end(LeaseEnd("lease-9", START + 1.0, LeaseEndReason.RELEASED))
    with pytest.raises(ValueError, match="holds no card"):
        await toolbox.record_lease_end(LeaseEnd("lease-1", START + 1.0, LeaseEndReason.RELEASED, (4, 6)))
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SCHEDULER).record_lease_start(replace(ARBOR, lease_id="lease-4"))
    (hold,) = await toolbox.card_holds()
    assert hold.returned == {}


async def test_the_toolbox_is_last_seen_at_its_own_last_entry(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    assert await scheduler.toolbox_seen_at() is None
    toolbox = await _opened(open_as)

    await toolbox.record_lease_start(ARBOR)
    seen = await toolbox.toolbox_seen_at()
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b02", from_commit="c0"))

    entries = toolbox._state().entries
    assert seen == max(entry.at for entry in entries if entry.role is Role.TOOLBOX)
    assert await scheduler.toolbox_seen_at() == seen
