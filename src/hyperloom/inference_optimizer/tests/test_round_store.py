# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bring-up mutex, exercised across ticks on a clock nobody waits for.

The store's whole job is to answer one question -- may another round start? --
and every interesting answer is separated from the acquire that caused it by
minutes of lease, so the behaviour only exists across ticks. The virtual clock
supplies those ticks as numbers, so a lease that runs out an hour after the
round opened is one line rather than an hour.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.bus.resource_lock import BRINGUP_ROUND_LANE, ROUND_LEASE_PID
from hyperloom.orchestrator.bus.storage import SqliteConnection
from hyperloom.orchestrator.state import round_store as rs
from hyperloom.orchestrator.state.round_store import (
    ABANDONED,
    BOOTED,
    EXPIRED_REAPED,
    EXPIRED_UNREAPED,
    STALE_FENCE,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TerminalTaskReuse, create_in_cursor

_LEASE = 600.0


@pytest.fixture
def store(tmp_path):
    """A :class:`RoundStore` over a real temp session database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    yield RoundStore(db)
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [EXPIRED_UNREAPED, EXPIRED_REAPED, ABANDONED, BOOTED])
async def test_a_settled_round_releases_the_machine_whatever_it_settled_as(store, virtual_clock, outcome):
    """Settling releases. No outcome buys a round exclusion it did not pay a lease for.

    An exclusion that outlives every reader is what trapped a session before:
    the row said the machine was held and nothing could say otherwise.
    """
    clock = virtual_clock
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok

    clock.advance(30.0)
    settled_at = clock.wall()
    assert (
        await store.settle("r", holder_task_id="t-1", fence=1, outcome=outcome, now_unix=settled_at, request_id="q2")
    ).ok

    row = await store.get("r")
    assert row is not None and row.outcome == outcome
    assert row.excludes_at(settled_at) is False
    assert await store.excluding(settled_at) == []
    # And the next round is admitted at once, on the same instant.
    assert (await store.open("next", holder_task_id="t-2", lease_sec=_LEASE, now_unix=settled_at, request_id="q3")).ok


@pytest.mark.asyncio
async def test_an_open_round_holds_the_machine_only_while_its_lease_is_live(store, virtual_clock):
    """The exclusion is time-bounded, so a round nobody settles frees itself."""
    clock = virtual_clock
    opened_at = clock.wall()
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=opened_at, request_id="q1")).ok

    row = await store.get("r")
    assert row is not None
    assert row.excludes_at(opened_at + _LEASE - 1.0) is True
    assert row.excludes_at(opened_at + _LEASE + 1.0) is False
    assert [r.round_id for r in await store.excluding(opened_at)] == ["r"]
    assert await store.excluding(opened_at + _LEASE + 1.0) == []

    # A holder that never settled cannot keep the next round out for good.
    assert (
        await store.open(
            "next", holder_task_id="t-2", lease_sec=_LEASE, now_unix=opened_at + _LEASE + 1.0, request_id="q2"
        )
    ).ok


@pytest.mark.asyncio
async def test_the_open_round_is_read_back_from_the_table_not_from_a_field(store, virtual_clock):
    """``held`` is how a later lifecycle call finds the round it must address."""
    clock = virtual_clock
    assert await store.held() is None
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok
    held = await store.held()
    assert held is not None and held.round_id == "r" and held.holder_task_id == "t-1"

    clock.advance(30.0)
    assert (
        await store.settle(
            "r", holder_task_id="t-1", fence=held.fence, outcome=BOOTED, now_unix=clock.wall(), request_id="q2"
        )
    ).ok
    assert await store.held() is None


@pytest.mark.asyncio
async def test_only_one_of_two_contending_acquires_wins_and_the_loser_is_told_why(store, virtual_clock):
    """Two rounds race for a machine one of them gets."""
    clock = virtual_clock
    first = await store.open("round-a", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    second = await store.open("round-b", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q2")
    assert first.ok and first.fence == 1
    assert second.ok is False
    assert second.reason == rs.EXCLUDED
    assert await store.get("round-b") is None

    # The winner's lease is the loser's opening: once it runs out with nothing
    # settling it, the round it excluded may finally start.
    clock.advance(_LEASE + 1.0)
    retry = await store.open("round-b", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q3")
    assert retry.ok


@pytest.mark.asyncio
async def test_the_holder_task_row_commits_with_the_acquire_and_never_adopts_a_finished_task(store, virtual_clock):
    """The joined creation path shares the acquiring transaction."""
    clock = virtual_clock
    tasks = TaskRegistry(store.db)

    def _join(cur):
        create_in_cursor(cur, kind="baseline", params={}, idempotency_key="round-a", task_id="t-1")

    assert (
        await store.open(
            "round-a",
            holder_task_id="t-1",
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id="q1",
            join=_join,
        )
    ).ok
    assert (await tasks.get("t-1")).state == "queued"

    # A loser rolls back whatever its join wrote, so no task row outlives the
    # acquire that failed.
    def _loser(cur):
        create_in_cursor(cur, kind="baseline", params={}, idempotency_key="round-b", task_id="t-2")

    denied = await store.open(
        "round-b",
        holder_task_id="t-2",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="q2",
        join=_loser,
    )
    assert denied.ok is False
    assert await tasks.find_by_idempotency_key("round-b") is None

    # And a retry under a key whose task already finished is refused rather
    # than silently held by work that is over.
    await tasks.transition("t-1", "running")
    await tasks.transition("t-1", "succeeded")
    clock.advance(_LEASE + 1.0)
    with pytest.raises(TerminalTaskReuse):
        await store.open(
            "round-c",
            holder_task_id="t-1",
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id="q3",
            join=_join,
        )


@pytest.mark.asyncio
async def test_renewing_extends_the_lease_without_invalidating_the_holders_settle(store, virtual_clock):
    """A heartbeat is not a change of holder, so it leaves the fence alone."""
    clock = virtual_clock
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok

    for tick in range(3):
        clock.advance(_LEASE / 2.0)
        renewed = await store.renew(
            "r",
            holder_task_id="t-1",
            fence=1,
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id=f"hb-{tick}",
        )
        assert renewed.ok
        assert renewed.fence == 1

    row = await store.get("r")
    assert row is not None
    assert row.fence == 1
    assert row.expires_unix == pytest.approx(clock.wall() + _LEASE)

    # The token the holder acquired under still settles the round it holds.
    assert (
        await store.settle("r", holder_task_id="t-1", fence=1, outcome=BOOTED, now_unix=clock.wall(), request_id="q2")
    ).ok


@pytest.mark.asyncio
async def test_a_handoff_advances_the_fence_and_the_old_holders_settle_is_rejected(store, virtual_clock):
    """The fence names a holder, and only a handoff can change either."""
    clock = virtual_clock
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok

    clock.advance(120.0)
    handed = await store.handoff(
        "r",
        holder_task_id="t-1",
        fence=1,
        new_holder_task_id="t-2",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="q2",
    )
    assert handed.ok and handed.fence == 2
    row = await store.get("r")
    assert row is not None and row.holder_task_id == "t-2"

    # The old holder is no longer the holder at all.
    clock.advance(60.0)
    displaced = await store.settle(
        "r", holder_task_id="t-1", fence=1, outcome=BOOTED, now_unix=clock.wall(), request_id="q3"
    )
    assert displaced.ok is False
    assert displaced.reason == rs.NOT_OWNER

    # And the token minted before the handoff no longer settles the round even
    # in the hands of the task that now holds it, which is what makes a fence a
    # fence rather than a second name for the holder.
    stale = await store.settle(
        "r",
        holder_task_id="t-2",
        fence=1,
        outcome=BOOTED,
        now_unix=clock.wall(),
        request_id="q4",
        evidence={"tput": 12.5},
    )
    assert stale.ok is False
    assert stale.reason == STALE_FENCE
    assert (await store.get("r")).state == rs.OPEN

    # A rejected settle is a settle still owed, and everything needed to make
    # it again survived the rejection.
    pending = await store.redrivable_settles()
    assert [e.request_id for e in pending] == ["q3", "q4"]
    assert pending[-1].outcome == BOOTED
    assert pending[-1].evidence == {"tput": 12.5}

    redriven = await store.settle(
        "r",
        holder_task_id="t-2",
        fence=2,
        outcome=pending[-1].outcome,
        now_unix=clock.wall(),
        request_id=pending[-1].request_id,
        evidence=pending[-1].evidence,
    )
    assert redriven.ok
    assert await store.redrivable_settles() == []


@pytest.mark.asyncio
async def test_settling_twice_records_the_replay_without_changing_the_round(store, virtual_clock):
    """A retried settle is idempotent; a different one is refused."""
    clock = virtual_clock
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok
    clock.advance(45.0)
    first_settled_at = clock.wall()
    first = await store.settle(
        "r", holder_task_id="t-1", fence=1, outcome=BOOTED, now_unix=first_settled_at, request_id="q2"
    )
    assert first.ok and first.duplicate is False

    clock.advance(5.0)
    replay = await store.settle(
        "r", holder_task_id="t-1", fence=1, outcome=BOOTED, now_unix=clock.wall(), request_id="q2"
    )
    assert replay.ok and replay.duplicate is True

    row = await store.get("r")
    assert row is not None
    assert row.outcome == BOOTED
    # The replay left the round exactly where the first settle put it: the
    # settle instant is the first one, not the retry's.
    assert row.settled_unix == pytest.approx(first_settled_at)

    contradicting = await store.settle(
        "r", holder_task_id="t-1", fence=1, outcome=EXPIRED_UNREAPED, now_unix=clock.wall(), request_id="q3"
    )
    assert contradicting.ok is False
    assert contradicting.reason == rs.ALREADY_SETTLED


@pytest.mark.asyncio
async def test_a_non_owner_cannot_settle_a_round_it_does_not_hold(store, virtual_clock):
    """Ownership is checked separately from the fence, and both are recorded."""
    clock = virtual_clock
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok
    refused = await store.settle(
        "r", holder_task_id="impostor", fence=1, outcome=BOOTED, now_unix=clock.wall(), request_id="q2"
    )
    assert refused.ok is False
    assert refused.reason == rs.NOT_OWNER
    rows = await store.db.fetchall(
        "SELECT op, result, reason FROM round_events WHERE round_id = ? ORDER BY event_id",
        ("r",),
    )
    recorded = [(r["op"], r["result"], r["reason"]) for r in rows]
    assert recorded == [("open", "applied", ""), ("settle", "rejected", rs.NOT_OWNER)]


async def _lane_row(store, round_id: str):
    """The lane row an open round holds, or ``None`` once it has let go."""
    return await store.db.fetchone(
        "SELECT * FROM leases WHERE lane = ? AND holder_id = ?",
        (BRINGUP_ROUND_LANE, round_id),
    )


@pytest.mark.asyncio
async def test_an_open_round_holds_the_lane_and_a_settled_one_does_not(store, virtual_clock):
    """The round and its lease are one write, so the lease reaper sees the round.

    Without this the round is a holder of the machine that ``lane_holders`` has
    no row for, and the lease sweep and the round sweep are two clocks over the
    same fact.
    """
    clock = virtual_clock
    opened_at = clock.wall()
    assert (await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=opened_at, request_id="q1")).ok

    row = await _lane_row(store, "r")
    assert row is not None
    assert row["task_id"] == "t-1"
    # No pid: the holder is a task, and only the task registry can prove a
    # task's process dead, so the dead-holder sweep must skip this row.
    assert int(row["pid"]) == ROUND_LEASE_PID
    acquired = row["acquired_at"]

    clock.advance(60.0)
    renewed_at = clock.wall()
    assert (
        await store.renew("r", holder_task_id="t-1", fence=1, lease_sec=_LEASE, now_unix=renewed_at, request_id="q2")
    ).ok
    row = await _lane_row(store, "r")
    assert row["acquired_at"] == acquired, "a renewal moves the lease, not the acquire"
    assert row["expires_at"] > acquired

    clock.advance(60.0)
    moved = await store.handoff(
        "r",
        holder_task_id="t-1",
        fence=1,
        new_holder_task_id="t-2",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="q3",
    )
    assert moved.ok
    assert (await _lane_row(store, "r"))["task_id"] == "t-2", "the round stays open, so it keeps the lane"

    clock.advance(60.0)
    assert (
        await store.settle("r", holder_task_id="t-2", fence=2, outcome=BOOTED, now_unix=clock.wall(), request_id="q4")
    ).ok
    assert await _lane_row(store, "r") is None


@pytest.mark.asyncio
async def test_a_refused_acquire_leaves_no_lane_behind(store, virtual_clock):
    """A round that was never opened holds nothing; the two roll back together."""
    clock = virtual_clock
    assert (await store.open("r1", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")).ok
    refused = await store.open("r2", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q2")
    assert refused.ok is False and refused.reason == rs.EXCLUDED
    assert await _lane_row(store, "r2") is None
    assert await _lane_row(store, "r1") is not None
