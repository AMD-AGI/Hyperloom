# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bring-up mutex, exercised across ticks on a clock nobody waits for."""

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
    Round,
    RoundResult,
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
@pytest.mark.parametrize("outcome", [EXPIRED_UNREAPED, EXPIRED_REAPED, ABANDONED, BOOTED])
async def test_a_settled_round_releases_the_machine_whatever_it_settled_as(store, virtual_clock, outcome):
    """Settling releases. No outcome buys a round exclusion it did not pay a lease for."""
    clock = virtual_clock
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok

    clock.advance(30.0)
    settled_at = clock.wall()
    settled = await store.settle(_claim("r", "t-1", 1), outcome=outcome, now_unix=settled_at, request_id="q2")
    assert settled.ok

    row = await store.get("r")
    assert row is not None and row.outcome == outcome
    assert row.excludes_at(settled_at) is False
    assert await store.excluding(settled_at) == []
    # And the next round is admitted at once, on the same instant.
    opened = await store.open("next", holder_task_id="t-2", lease_sec=_LEASE, now_unix=settled_at, request_id="q3")
    assert opened.ok


@pytest.mark.asyncio
async def test_an_open_round_holds_the_machine_until_explicit_settlement(store, virtual_clock):
    """Elapsed budget is not evidence that the round's owner stopped."""
    clock = virtual_clock
    opened_at = clock.wall()
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=opened_at, request_id="q1")
    assert opened.ok

    row = await store.get("r")
    assert row is not None
    assert row.excludes_at(opened_at + _LEASE - 1.0) is True
    assert row.excludes_at(opened_at + _LEASE + 1.0) is True
    assert [r.round_id for r in await store.excluding(opened_at)] == ["r"]
    assert [r.round_id for r in await store.excluding(opened_at + _LEASE + 1.0)] == ["r"]

    opened = await store.open(
        "next", holder_task_id="t-2", lease_sec=_LEASE, now_unix=opened_at + _LEASE + 1.0, request_id="q2"
    )
    assert not opened.ok


@pytest.mark.asyncio
async def test_the_open_round_is_read_back_from_the_table_not_from_a_field(store, virtual_clock):
    """``held`` is how a later lifecycle call finds the round it must address."""
    clock = virtual_clock
    assert await store.held() is None
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok
    held = await store.held()
    assert held is not None and held.round_id == "r" and held.holder_task_id == "t-1"

    clock.advance(30.0)
    settled = await store.settle(held, outcome=BOOTED, now_unix=clock.wall(), request_id="q2")
    assert settled.ok
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

    clock.advance(_LEASE + 1.0)
    retry = await store.open("round-b", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q3")
    assert not retry.ok
    await store.settle(_claim("round-a", "t-1", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="q4")
    acquired = await store.open(
        "round-b", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q5"
    )
    assert acquired.ok


@pytest.mark.asyncio
async def test_the_holder_task_row_commits_with_the_acquire_and_never_adopts_a_finished_task(store, virtual_clock):
    """The joined creation path shares the acquiring transaction."""
    clock = virtual_clock
    tasks = TaskRegistry(store.db)

    def _join(cur):
        create_in_cursor(cur, kind="baseline", params={}, idempotency_key="round-a", task_id="t-1")

    opened = await store.open(
        "round-a",
        holder_task_id="t-1",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="q1",
        join=_join,
    )
    assert opened.ok
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
    await store.settle(_claim("round-a", "t-1", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="settle")
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
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok

    for tick in range(3):
        clock.advance(_LEASE / 2.0)
        renewed = await store.renew(
            _claim("r", "t-1", 1),
            lease_sec=_LEASE,
            now_unix=clock.wall(),
            request_id=f"hb-{tick}",
        )
        assert renewed.ok
        assert renewed.fence == 1

    row = await store.get("r")
    assert row is not None
    assert row.fence == 1
    assert row.renewed_unix == clock.wall()
    assert row.expires_unix == clock.wall() + _LEASE

    # The token the holder acquired under still settles the round it holds.
    settled = await store.settle(_claim("r", "t-1", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="q2")
    assert settled.ok


@pytest.mark.asyncio
async def test_a_handoff_advances_the_fence_and_the_old_holders_settle_is_rejected(store, virtual_clock):
    """The fence names a holder, and only a handoff can change either."""
    clock = virtual_clock
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok

    clock.advance(120.0)
    handed = await store.handoff(
        _claim("r", "t-1", 1),
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
    displaced = await store.settle(_claim("r", "t-1", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="q3")
    assert displaced.ok is False
    assert displaced.reason == rs.NOT_OWNER

    # And the token minted before the handoff no longer settles the round even
    # in the hands of the task that now holds it, which is what makes a fence a
    # fence rather than a second name for the holder.
    stale = await store.settle(
        _claim("r", "t-2", 1),
        outcome=BOOTED,
        now_unix=clock.wall(),
        request_id="q4",
        evidence={"tput": 12.5},
    )
    assert stale.ok is False
    assert stale.reason == STALE_FENCE
    assert (await store.get("r")).state == rs.OPEN

    # The refusal is recorded with what was asked for, which is how a fence
    # firing is ever noticed.
    rows = await store.db.fetchall(
        "SELECT request_id, result, reason FROM round_events WHERE round_id = ? AND op = 'settle' ORDER BY event_id",
        ("r",),
    )
    assert [(r["request_id"], r["result"], r["reason"]) for r in rows][-1] == ("q4", "rejected", STALE_FENCE)

    # The holder the fence names can still settle it.
    settled = await store.settle(_claim("r", "t-2", 2), outcome=BOOTED, now_unix=clock.wall(), request_id="q5")
    assert settled.ok


@pytest.mark.asyncio
async def test_settling_twice_records_the_replay_without_changing_the_round(store, virtual_clock):
    """A retried settle is idempotent; a different one is refused."""
    clock = virtual_clock
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok
    clock.advance(45.0)
    first_settled_at = clock.wall()
    first = await store.settle(_claim("r", "t-1", 1), outcome=BOOTED, now_unix=first_settled_at, request_id="q2")
    assert first.ok and first.duplicate is False

    clock.advance(5.0)
    replay = await store.settle(_claim("r", "t-1", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="q2")
    assert replay.ok and replay.duplicate is True

    row = await store.get("r")
    assert row is not None
    assert row.outcome == BOOTED
    # The replay left the round exactly where the first settle put it: the
    # settle instant is the first one, not the retry's.
    assert row.settled_unix == first_settled_at

    contradicting = await store.settle(
        _claim("r", "t-1", 1), outcome=EXPIRED_UNREAPED, now_unix=clock.wall(), request_id="q3"
    )
    assert contradicting.ok is False
    assert contradicting.reason == rs.ALREADY_SETTLED


@pytest.mark.asyncio
async def test_a_non_owner_cannot_settle_a_round_it_does_not_hold(store, virtual_clock):
    """Ownership is checked separately from the fence, and both are recorded."""
    clock = virtual_clock
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok
    refused = await store.settle(_claim("r", "impostor", 1), outcome=BOOTED, now_unix=clock.wall(), request_id="q2")
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
    """The round and its lease are one write, so the lease reaper sees the round."""
    clock = virtual_clock
    opened_at = clock.wall()
    opened = await store.open("r", holder_task_id="t-1", lease_sec=_LEASE, now_unix=opened_at, request_id="q1")
    assert opened.ok

    row = await _lane_row(store, "r")
    assert row is not None
    assert row["task_id"] == "t-1"
    # No pid: the holder is a task, and only the task registry can prove a
    # task's process dead, so the dead-holder sweep must skip this row.
    assert int(row["pid"]) == ROUND_LEASE_PID
    acquired = row["acquired_at"]

    clock.advance(60.0)
    renewed_at = clock.wall()
    renewed = await store.renew(_claim("r", "t-1", 1), lease_sec=_LEASE, now_unix=renewed_at, request_id="q2")
    assert renewed.ok
    row = await _lane_row(store, "r")
    assert row["acquired_at"] == acquired, "a renewal moves the lease, not the acquire"
    assert row["expires_at"] > acquired

    clock.advance(60.0)
    moved = await store.handoff(
        _claim("r", "t-1", 1),
        new_holder_task_id="t-2",
        lease_sec=_LEASE,
        now_unix=clock.wall(),
        request_id="q3",
    )
    assert moved.ok
    assert (await _lane_row(store, "r"))["task_id"] == "t-2", "the round stays open, so it keeps the lane"

    clock.advance(60.0)
    settled = await store.settle(_claim("r", "t-2", 2), outcome=BOOTED, now_unix=clock.wall(), request_id="q4")
    assert settled.ok
    assert await _lane_row(store, "r") is None


@pytest.mark.asyncio
async def test_every_path_writes_the_same_outbox_row_and_answer(store):
    """Each attempt's ``RoundResult``, its ``round_events`` row and the rows it leaves, column for column."""

    async def snapshot():
        rounds = await store.db.fetchall("SELECT * FROM bringup_rounds ORDER BY round_id")
        lanes = await store.db.fetchall("SELECT * FROM leases WHERE lane = ? ORDER BY holder_id", (BRINGUP_ROUND_LANE,))
        return repr([tuple(r) for r in rounds]), repr([tuple(r) for r in lanes])

    results = [
        await store.open(
            "a", holder_task_id="t-1", lease_sec=600.0, now_unix=1000.0, request_id="o1", evidence={"z": 1, "a": "x"}
        ),
        await store.open("b", holder_task_id="t-2", lease_sec=600.0, now_unix=1001.0, request_id="o2"),
        await store.open("a", holder_task_id="t-3", lease_sec=600.0, now_unix=1002.0, request_id="o3"),
        await store.renew(_claim("a", "t-1", 1), lease_sec=600.0, now_unix=1010.0, request_id="r1"),
    ]
    after_renew = await snapshot()
    results += [
        await store.renew(_claim("a", "t-9", 1), lease_sec=600.0, now_unix=1011.0, request_id="r2"),
        await store.renew(_claim("a", "t-1", 7), lease_sec=600.0, now_unix=1012.0, request_id="r3"),
        await store.renew(_claim("ghost", "t-1", 1), lease_sec=600.0, now_unix=1013.0, request_id="r4"),
        await store.handoff(
            _claim("a", "t-1", 1),
            new_holder_task_id="t-2",
            lease_sec=300.0,
            now_unix=1020.0,
            request_id="h1",
            evidence={"k": "v"},
        ),
    ]
    after_handoff = await snapshot()
    results += [
        await store.handoff(
            _claim("a", "t-1", 1),
            new_holder_task_id="t-3",
            lease_sec=1.0,
            now_unix=1021.0,
            request_id="h2",
        ),
        await store.handoff(
            _claim("a", "t-2", 1),
            new_holder_task_id="t-3",
            lease_sec=1.0,
            now_unix=1022.0,
            request_id="h3",
        ),
        await store.handoff(
            _claim("ghost", "t-2", 2),
            new_holder_task_id="t-3",
            lease_sec=1.0,
            now_unix=1023.0,
            request_id="h4",
        ),
        await store.settle(_claim("a", "t-2", 1), outcome=BOOTED, now_unix=1024.0, request_id="s1"),
        await store.settle(_claim("a", "t-9", 2), outcome=BOOTED, now_unix=1025.0, request_id="s2"),
        await store.settle(
            _claim("a", "t-2", 2),
            outcome=BOOTED,
            now_unix=1030.0,
            request_id="s3",
            evidence={"reason": "r"},
        ),
        await store.settle(_claim("a", "t-2", 2), outcome=BOOTED, now_unix=1040.0, request_id="s3"),
        await store.settle(_claim("a", "t-9", 2), outcome=BOOTED, now_unix=1046.0, request_id="s6"),
        await store.settle(_claim("a", "t-2", 1), outcome=BOOTED, now_unix=1047.0, request_id="s7"),
        await store.settle(_claim("a", "t-2", 2), outcome=rs.FAILED, now_unix=1041.0, request_id="s4"),
        await store.renew(_claim("a", "t-2", 2), lease_sec=600.0, now_unix=1042.0, request_id="r5"),
        await store.handoff(
            _claim("a", "t-2", 2),
            new_holder_task_id="t-3",
            lease_sec=1.0,
            now_unix=1043.0,
            request_id="h5",
        ),
        await store.settle(_claim("ghost", "t-2", 2), outcome=BOOTED, now_unix=1044.0, request_id="s5"),
        await store.open("a", holder_task_id="t-4", lease_sec=600.0, now_unix=1050.0, request_id="o4"),
        await store.open("b", holder_task_id="t-5", lease_sec=600.0, now_unix=1051.0, request_id="o5"),
    ]

    def ok(round_id, fence, state, outcome="", *, event_id, duplicate=False):
        return RoundResult(
            ok=True,
            round_id=round_id,
            fence=fence,
            state=state,
            outcome=outcome,
            duplicate=duplicate,
            event_id=event_id,
        )

    def refused(round_id, fence, state, outcome, reason, *, event_id):
        return RoundResult(
            ok=False,
            round_id=round_id,
            fence=fence,
            state=state,
            outcome=outcome,
            reason=reason,
            event_id=event_id,
        )

    # repr, not ==, so an int where a float was written (or the reverse) also fails.
    assert repr(results) == repr(
        [
            ok("a", 1, rs.OPEN, event_id=1),
            refused("b", 0, "", "", rs.EXCLUDED, event_id=2),
            refused("a", 0, "", "", rs.ALREADY_EXISTS, event_id=3),
            ok("a", 1, rs.OPEN, event_id=4),
            refused("a", 1, rs.OPEN, "", rs.NOT_OWNER, event_id=5),
            refused("a", 1, rs.OPEN, "", rs.STALE_FENCE, event_id=6),
            refused("ghost", 0, "", "", rs.UNKNOWN_ROUND, event_id=7),
            ok("a", 2, rs.OPEN, event_id=8),
            refused("a", 2, rs.OPEN, "", rs.NOT_OWNER, event_id=9),
            refused("a", 2, rs.OPEN, "", rs.STALE_FENCE, event_id=10),
            refused("ghost", 0, "", "", rs.UNKNOWN_ROUND, event_id=11),
            refused("a", 2, rs.OPEN, "", rs.STALE_FENCE, event_id=12),
            refused("a", 2, rs.OPEN, "", rs.NOT_OWNER, event_id=13),
            ok("a", 2, rs.SETTLED, BOOTED, event_id=14),
            ok("a", 2, rs.SETTLED, BOOTED, event_id=15, duplicate=True),
            refused("a", 2, rs.SETTLED, BOOTED, rs.ALREADY_SETTLED, event_id=16),
            refused("a", 2, rs.SETTLED, BOOTED, rs.ALREADY_SETTLED, event_id=17),
            refused("a", 2, rs.SETTLED, BOOTED, rs.ALREADY_SETTLED, event_id=18),
            refused("a", 2, rs.SETTLED, BOOTED, rs.NOT_OPEN, event_id=19),
            refused("a", 2, rs.SETTLED, BOOTED, rs.NOT_OPEN, event_id=20),
            refused("ghost", 0, "", "", rs.UNKNOWN_ROUND, event_id=21),
            refused("a", 0, "", "", rs.ALREADY_EXISTS, event_id=22),
            ok("b", 1, rs.OPEN, event_id=23),
        ]
    )
    events = await store.db.fetchall("SELECT * FROM round_events ORDER BY event_id")
    assert repr([tuple(r) for r in events]) == repr(
        [
            (1, "a", "o1", "open", "applied", "", 1, "t-1", "", '{"a": "x", "z": 1}', 1000.0),
            (2, "b", "o2", "open", "rejected", "", 0, "t-2", "excluded", "{}", 1001.0),
            (3, "a", "o3", "open", "rejected", "", 0, "t-3", "already_exists", "{}", 1002.0),
            (4, "a", "r1", "renew", "applied", "", 1, "t-1", "", "{}", 1010.0),
            (5, "a", "r2", "renew", "rejected", "", 1, "t-9", "not_owner", "{}", 1011.0),
            (6, "a", "r3", "renew", "rejected", "", 7, "t-1", "stale_fence", "{}", 1012.0),
            (7, "ghost", "r4", "renew", "rejected", "", 1, "t-1", "unknown_round", "{}", 1013.0),
            (8, "a", "h1", "handoff", "applied", "", 2, "t-2", "", '{"k": "v"}', 1020.0),
            (9, "a", "h2", "handoff", "rejected", "", 1, "t-1", "not_owner", "{}", 1021.0),
            (10, "a", "h3", "handoff", "rejected", "", 1, "t-2", "stale_fence", "{}", 1022.0),
            (11, "ghost", "h4", "handoff", "rejected", "", 2, "t-2", "unknown_round", "{}", 1023.0),
            (12, "a", "s1", "settle", "rejected", "booted", 1, "t-2", "stale_fence", "{}", 1024.0),
            (13, "a", "s2", "settle", "rejected", "booted", 2, "t-9", "not_owner", "{}", 1025.0),
            (14, "a", "s3", "settle", "applied", "booted", 2, "t-2", "", '{"reason": "r"}', 1030.0),
            (15, "a", "s3", "settle", "duplicate", "booted", 2, "t-2", "", "{}", 1040.0),
            (16, "a", "s6", "settle", "rejected", "booted", 2, "t-9", "already_settled", "{}", 1046.0),
            (17, "a", "s7", "settle", "rejected", "booted", 1, "t-2", "already_settled", "{}", 1047.0),
            (18, "a", "s4", "settle", "rejected", "failed", 2, "t-2", "already_settled", "{}", 1041.0),
            (19, "a", "r5", "renew", "rejected", "", 2, "t-2", "not_open", "{}", 1042.0),
            (20, "a", "h5", "handoff", "rejected", "", 2, "t-2", "not_open", "{}", 1043.0),
            (21, "ghost", "s5", "settle", "rejected", "booted", 2, "t-2", "unknown_round", "{}", 1044.0),
            (22, "a", "o4", "open", "rejected", "", 0, "t-4", "already_exists", "{}", 1050.0),
            (23, "b", "o5", "open", "applied", "", 1, "t-5", "", "{}", 1051.0),
        ]
    )
    lane_a = (BRINGUP_ROUND_LANE, "a")
    lane_b = (BRINGUP_ROUND_LANE, "b")
    round_lease = (BRINGUP_ROUND_LANE, ROUND_LEASE_PID, "")
    assert after_renew == (
        repr([("a", "open", "", "t-1", 1, 1000.0, 1010.0, 1610.0, None)]),
        repr(
            [
                (
                    *lane_a,
                    "t-1",
                    *round_lease,
                    "1970-01-01T00:16:40.000000+00:00",
                    "1970-01-01T00:26:50.000000+00:00",
                    "1970-01-01T00:16:50.000000+00:00",
                )
            ]
        ),
    )
    assert after_handoff == (
        repr([("a", "open", "", "t-2", 2, 1000.0, 1020.0, 1320.0, None)]),
        repr(
            [
                (
                    *lane_a,
                    "t-2",
                    *round_lease,
                    "1970-01-01T00:16:40.000000+00:00",
                    "1970-01-01T00:22:00.000000+00:00",
                    "1970-01-01T00:17:00.000000+00:00",
                )
            ]
        ),
    )
    assert await snapshot() == (
        repr(
            [
                ("a", "settled", "booted", "t-2", 2, 1000.0, 1020.0, 1320.0, 1030.0),
                ("b", "open", "", "t-5", 1, 1051.0, 1051.0, 1651.0, None),
            ]
        ),
        repr(
            [
                (
                    *lane_b,
                    "t-5",
                    *round_lease,
                    "1970-01-01T00:17:31.000000+00:00",
                    "1970-01-01T00:27:31.000000+00:00",
                    "1970-01-01T00:17:31.000000+00:00",
                )
            ]
        ),
    )


@pytest.mark.asyncio
async def test_a_refused_acquire_leaves_no_lane_behind(store, virtual_clock):
    """A round that was never opened holds nothing; the two roll back together."""
    clock = virtual_clock
    opened = await store.open("r1", holder_task_id="t-1", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q1")
    assert opened.ok
    refused = await store.open("r2", holder_task_id="t-2", lease_sec=_LEASE, now_unix=clock.wall(), request_id="q2")
    assert refused.ok is False and refused.reason == rs.EXCLUDED
    assert await _lane_row(store, "r2") is None
    assert await _lane_row(store, "r1") is not None
