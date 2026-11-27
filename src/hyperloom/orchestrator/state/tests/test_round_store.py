# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bring-up mutex, exercised across ticks on a clock nobody waits for."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
from hyperloom.orchestrator.state import round_store as rs
from hyperloom.orchestrator.state.round_store import (
    ABANDONED,
    BOOTED,
    EXPIRED_REAPED,
    STALE_FENCE,
    Round,
    RoundResult,
    RoundStore,
)
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec, TerminalTaskReuse, create_in_cursor

_LEASE = 600.0


@pytest.fixture
def store(tmp_path):
    """A :class:`RoundStore` over a real temp session database."""
    db = SqliteConnection(tmp_path / "coordinator.db")
    yield RoundStore(db)
    db.close()


def _claim(round_id: str, holder_task_id: str, fence: int) -> Round:
    """The round as a caller holding it under ``fence`` presents it; the store reads no other field."""
    return Round(
        round_id=round_id,
        state=rs.OPEN,
        outcome="",
        holder_task_id=holder_task_id,
        fence=fence,
        opened_unix=0.0,
        renewed_unix=0.0,
        expires_unix=0.0,
        settled_unix=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [EXPIRED_REAPED, ABANDONED, BOOTED])
async def test_a_settled_round_releases_the_machine_whatever_it_settled_as(store, virtual_clock, outcome):
    """Settling releases. No outcome buys a round exclusion it did not pay a lease for."""
    clock = virtual_clock
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok

    clock.advance(30.0)
    settled_at = clock.wall()
    settled = await store.settle(_claim("r", "t-1", 1), now_unix=settled_at, outcome=outcome)
    assert settled.ok

    row = await store.get("r")
    assert row is not None and row.outcome == outcome
    assert row.state == rs.SETTLED
    assert await store.open_rounds() == []
    # And the next round is admitted at once, on the same instant.
    opened = await store.open("next", now_unix=settled_at, holder_task_id="t-2", lease_sec=_LEASE)
    assert opened.ok


@pytest.mark.asyncio
async def test_an_open_round_holds_the_machine_until_explicit_settlement(store, virtual_clock):
    """Elapsed budget is not evidence that the round's owner stopped."""
    clock = virtual_clock
    opened_at = clock.wall()
    opened = await store.open("r", now_unix=opened_at, holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok

    row = await store.get("r")
    assert row is not None
    assert row.state == rs.OPEN
    assert [r.round_id for r in await store.open_rounds()] == ["r"]

    opened = await store.open("next", now_unix=opened_at + _LEASE + 1.0, holder_task_id="t-2", lease_sec=_LEASE)
    assert not opened.ok


@pytest.mark.asyncio
async def test_the_open_round_is_read_back_from_the_table_not_from_a_field(store, virtual_clock):
    """``held`` is how a later lifecycle call finds the round it must address."""
    clock = virtual_clock
    assert await store.held() is None
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok
    held = await store.held()
    assert held is not None and held.round_id == "r" and held.holder_task_id == "t-1"

    clock.advance(30.0)
    settled = await store.settle(held, now_unix=clock.wall(), outcome=BOOTED)
    assert settled.ok
    assert await store.held() is None


@pytest.mark.asyncio
async def test_only_one_of_two_contending_acquires_wins_and_the_loser_is_told_why(store, virtual_clock):
    """Two rounds race for a machine one of them gets."""
    clock = virtual_clock
    first = await store.open("round-a", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    second = await store.open("round-b", now_unix=clock.wall(), holder_task_id="t-2", lease_sec=_LEASE)
    assert first.ok
    assert (await store.get("round-a")).fence == 1
    assert second.ok is False
    assert second.reason == rs.EXCLUDED
    assert await store.get("round-b") is None

    clock.advance(_LEASE + 1.0)
    retry = await store.open("round-b", now_unix=clock.wall(), holder_task_id="t-2", lease_sec=_LEASE)
    assert not retry.ok
    await store.settle(_claim("round-a", "t-1", 1), now_unix=clock.wall(), outcome=BOOTED)
    acquired = await store.open("round-b", now_unix=clock.wall(), holder_task_id="t-2", lease_sec=_LEASE)
    assert acquired.ok


@pytest.mark.asyncio
async def test_the_holder_task_row_commits_with_the_acquire_and_never_adopts_a_finished_task(store, virtual_clock):
    """The joined creation path shares the acquiring transaction."""
    clock = virtual_clock
    tasks = TaskRegistry(store.db)

    def _join(cur):
        create_in_cursor(cur, TaskSpec(kind="baseline", params={}, idempotency_key="round-a", task_id="t-1"))

    opened = await store.open(
        "round-a",
        now_unix=clock.wall(),
        holder_task_id="t-1",
        lease_sec=_LEASE,
        join=_join,
    )
    assert opened.ok
    assert (await tasks.get("t-1")).state == "queued"

    # A loser rolls back whatever its join wrote, so no task row outlives the
    # acquire that failed.
    def _loser(cur):
        create_in_cursor(cur, TaskSpec(kind="baseline", params={}, idempotency_key="round-b", task_id="t-2"))

    denied = await store.open(
        "round-b",
        now_unix=clock.wall(),
        holder_task_id="t-2",
        lease_sec=_LEASE,
        join=_loser,
    )
    assert denied.ok is False
    assert await tasks.find_by_idempotency_key("round-b") is None

    # And a retry under a key whose task already finished is refused rather
    # than silently held by work that is over.
    await tasks.transition("t-1", "running")
    await tasks.transition("t-1", "succeeded")
    await store.settle(_claim("round-a", "t-1", 1), now_unix=clock.wall(), outcome=BOOTED)
    clock.advance(_LEASE + 1.0)
    with pytest.raises(TerminalTaskReuse):
        await store.open(
            "round-c",
            now_unix=clock.wall(),
            holder_task_id="t-1",
            lease_sec=_LEASE,
            join=_join,
        )


@pytest.mark.asyncio
async def test_renewing_extends_the_lease_without_invalidating_the_holders_settle(store, virtual_clock):
    """A heartbeat is not a change of holder, so it leaves the fence alone."""
    clock = virtual_clock
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok

    for tick in range(3):
        clock.advance(_LEASE / 2.0)
        renewed = await store.renew(_claim("r", "t-1", 1), now_unix=clock.wall(), lease_sec=_LEASE)
        assert renewed.ok

    row = await store.get("r")
    assert row is not None
    assert row.fence == 1
    assert row.renewed_unix == clock.wall()
    assert row.expires_unix == clock.wall() + _LEASE

    # The token the holder acquired under still settles the round it holds.
    settled = await store.settle(_claim("r", "t-1", 1), now_unix=clock.wall(), outcome=BOOTED)
    assert settled.ok


@pytest.mark.asyncio
async def test_a_handoff_advances_the_fence_and_the_old_holders_settle_is_rejected(store, virtual_clock):
    """The fence names a holder, and only a handoff can change either."""
    clock = virtual_clock
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok

    clock.advance(120.0)
    handed = await store.handoff(
        _claim("r", "t-1", 1),
        now_unix=clock.wall(),
        new_holder_task_id="t-2",
        lease_sec=_LEASE,
    )
    assert handed.ok
    row = await store.get("r")
    assert row is not None and row.holder_task_id == "t-2" and row.fence == 2

    # The old holder is no longer the holder at all.
    clock.advance(60.0)
    displaced = await store.settle(_claim("r", "t-1", 1), now_unix=clock.wall(), outcome=BOOTED)
    assert displaced.ok is False
    assert displaced.reason == rs.NOT_OWNER

    # And the token minted before the handoff no longer settles the round even
    # in the hands of the task that now holds it, which is what makes a fence a
    # fence rather than a second name for the holder.
    stale = await store.settle(
        _claim("r", "t-2", 1),
        now_unix=clock.wall(),
        outcome=BOOTED,
    )
    assert stale.ok is False
    assert stale.reason == STALE_FENCE
    assert (await store.get("r")).state == rs.OPEN

    # The holder the fence names can still settle it.
    settled = await store.settle(_claim("r", "t-2", 2), now_unix=clock.wall(), outcome=BOOTED)
    assert settled.ok


@pytest.mark.asyncio
async def test_settling_twice_answers_the_replay_without_changing_the_round(store, virtual_clock):
    """A retried settle is idempotent; a different one is refused."""
    clock = virtual_clock
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok
    clock.advance(45.0)
    first_settled_at = clock.wall()
    first = await store.settle(_claim("r", "t-1", 1), now_unix=first_settled_at, outcome=BOOTED)
    assert first.ok

    clock.advance(5.0)
    replay = await store.settle(_claim("r", "t-1", 1), now_unix=clock.wall(), outcome=BOOTED)
    assert replay.ok

    row = await store.get("r")
    assert row is not None
    assert row.outcome == BOOTED
    # The replay left the round exactly where the first settle put it: the
    # settle instant is the first one, not the retry's.
    assert row.settled_unix == first_settled_at

    contradicting = await store.settle(_claim("r", "t-1", 1), now_unix=clock.wall(), outcome=rs.FAILED)
    assert contradicting.ok is False
    assert contradicting.reason == rs.ALREADY_SETTLED


@pytest.mark.asyncio
async def test_a_non_owner_cannot_settle_a_round_it_does_not_hold(store, virtual_clock):
    """Ownership is checked separately from the fence."""
    clock = virtual_clock
    opened = await store.open("r", now_unix=clock.wall(), holder_task_id="t-1", lease_sec=_LEASE)
    assert opened.ok
    refused = await store.settle(_claim("r", "impostor", 1), now_unix=clock.wall(), outcome=BOOTED)
    assert refused.ok is False
    assert refused.reason == rs.NOT_OWNER
    assert (await store.get("r")).state == rs.OPEN


@pytest.mark.asyncio
async def test_every_path_gives_the_same_answer_and_leaves_the_same_rows(store):
    """Each attempt's ``RoundResult`` and the round rows it leaves, column for column; the lease table stays empty."""

    async def snapshot():
        rounds = await store.db.fetchall("SELECT * FROM bringup_rounds ORDER BY round_id")
        assert await store.db.fetchall("SELECT * FROM leases") == []
        return [tuple(r) for r in rounds]

    results = [
        await store.open(
            "a",
            now_unix=1000.0,
            holder_task_id="t-1",
            lease_sec=600.0,
        ),
        await store.open("b", now_unix=1001.0, holder_task_id="t-2", lease_sec=600.0),
        await store.open("a", now_unix=1002.0, holder_task_id="t-3", lease_sec=600.0),
        await store.renew(_claim("a", "t-1", 1), now_unix=1010.0, lease_sec=600.0),
    ]
    after_renew = await snapshot()
    results += [
        await store.renew(_claim("a", "t-9", 1), now_unix=1011.0, lease_sec=600.0),
        await store.renew(_claim("a", "t-1", 7), now_unix=1012.0, lease_sec=600.0),
        await store.renew(_claim("ghost", "t-1", 1), now_unix=1013.0, lease_sec=600.0),
        await store.handoff(
            _claim("a", "t-1", 1),
            now_unix=1020.0,
            new_holder_task_id="t-2",
            lease_sec=300.0,
        ),
    ]
    after_handoff = await snapshot()
    results += [
        await store.handoff(
            _claim("a", "t-1", 1),
            now_unix=1021.0,
            new_holder_task_id="t-3",
            lease_sec=1.0,
        ),
        await store.handoff(
            _claim("a", "t-2", 1),
            now_unix=1022.0,
            new_holder_task_id="t-3",
            lease_sec=1.0,
        ),
        await store.handoff(
            _claim("ghost", "t-2", 2),
            now_unix=1023.0,
            new_holder_task_id="t-3",
            lease_sec=1.0,
        ),
        await store.settle(_claim("a", "t-2", 1), now_unix=1024.0, outcome=BOOTED),
        await store.settle(_claim("a", "t-9", 2), now_unix=1025.0, outcome=BOOTED),
        await store.settle(
            _claim("a", "t-2", 2),
            now_unix=1030.0,
            outcome=BOOTED,
        ),
        await store.settle(_claim("a", "t-2", 2), now_unix=1040.0, outcome=BOOTED),
        await store.settle(_claim("a", "t-9", 2), now_unix=1046.0, outcome=BOOTED),
        await store.settle(_claim("a", "t-2", 1), now_unix=1047.0, outcome=BOOTED),
        await store.settle(_claim("a", "t-2", 2), now_unix=1041.0, outcome=rs.FAILED),
        await store.renew(_claim("a", "t-2", 2), now_unix=1042.0, lease_sec=600.0),
        await store.handoff(
            _claim("a", "t-2", 2),
            now_unix=1043.0,
            new_holder_task_id="t-3",
            lease_sec=1.0,
        ),
        await store.settle(_claim("ghost", "t-2", 2), now_unix=1044.0, outcome=BOOTED),
        await store.open("a", now_unix=1050.0, holder_task_id="t-4", lease_sec=600.0),
        await store.open("b", now_unix=1051.0, holder_task_id="t-5", lease_sec=600.0),
    ]

    ok = RoundResult(ok=True)

    def refused(reason):
        return RoundResult(ok=False, reason=reason)

    assert results == (
        [
            ok,
            refused(rs.EXCLUDED),
            refused(rs.ALREADY_EXISTS),
            ok,
            refused(rs.NOT_OWNER),
            refused(rs.STALE_FENCE),
            refused(rs.UNKNOWN_ROUND),
            ok,
            refused(rs.NOT_OWNER),
            refused(rs.STALE_FENCE),
            refused(rs.UNKNOWN_ROUND),
            refused(rs.STALE_FENCE),
            refused(rs.NOT_OWNER),
            ok,
            ok,
            refused(rs.ALREADY_SETTLED),
            refused(rs.ALREADY_SETTLED),
            refused(rs.ALREADY_SETTLED),
            refused(rs.NOT_OPEN),
            refused(rs.NOT_OPEN),
            refused(rs.UNKNOWN_ROUND),
            refused(rs.ALREADY_EXISTS),
            ok,
        ]
    )
    assert after_renew == [("a", "open", "", "t-1", 1, 1000.0, 1010.0, 1610.0, None)]
    assert after_handoff == [("a", "open", "", "t-2", 2, 1000.0, 1020.0, 1320.0, None)]
    assert await snapshot() == [
        ("a", "settled", "booted", "t-2", 2, 1000.0, 1020.0, 1320.0, 1030.0),
        ("b", "open", "", "t-5", 1, 1051.0, 1051.0, 1651.0, None),
    ]
