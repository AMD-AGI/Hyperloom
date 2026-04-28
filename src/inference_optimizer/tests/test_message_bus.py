"""Tests for orchestrator/message_bus.py."""

from __future__ import annotations

import asyncio

import pytest

from inference_optimizer.orchestrator.message_bus import Message, MessageBus


@pytest.mark.asyncio
async def test_append_assigns_monotonic_seq(db):
    bus = MessageBus(db)
    seqs = []
    for i in range(10):
        m = Message.new("executor", "critic", "proposal",
                        {"i": i})
        seqs.append(await bus.append_and_seq(m))
    assert seqs == sorted(seqs)
    assert seqs[0] == 1
    assert seqs[-1] == 10


@pytest.mark.asyncio
async def test_unknown_topic_rejected(db):
    bus = MessageBus(db)
    bad = Message.new("executor", "critic", "wat_is_this",
                      {})
    with pytest.raises(ValueError, match="topic"):
        await bus.append_and_seq(bad)


@pytest.mark.asyncio
async def test_invalid_priority_rejected(db):
    bus = MessageBus(db)
    msg = Message.new("executor", "critic", "proposal",
                      {})
    msg.priority = 99
    with pytest.raises(ValueError, match="priority"):
        await bus.append_and_seq(msg)


@pytest.mark.asyncio
async def test_concurrent_appenders_get_unique_seq(db):
    bus = MessageBus(db)
    N = 50

    async def appender(idx):
        msg = Message.new("executor", "*", "event", {"i": idx})
        return await bus.append_and_seq(msg)

    seqs = await asyncio.gather(*(appender(i) for i in range(N)))
    assert len(set(seqs)) == N
    assert min(seqs) == 1
    assert max(seqs) == N


@pytest.mark.asyncio
async def test_tail_returns_recent_first(db):
    bus = MessageBus(db)
    for i in range(5):
        await bus.append_and_seq(
            Message.new("executor", "critic", "event", {"i": i})
        )
    tail = await bus.tail(n=3)
    assert [m.payload["i"] for m in tail] == [4, 3, 2]


@pytest.mark.asyncio
async def test_tail_filter_by_topic(db):
    bus = MessageBus(db)
    await bus.append_and_seq(
        Message.new("e", "c", "proposal", {"k": "p"})
    )
    await bus.append_and_seq(
        Message.new("e", "c", "objection", {"k": "o"})
    )
    await bus.append_and_seq(
        Message.new("e", "c", "proposal", {"k": "p2"})
    )
    proposals = await bus.tail(topic="proposal")
    assert len(proposals) == 2
    assert all(m.topic == "proposal" for m in proposals)


@pytest.mark.asyncio
async def test_replay_for_returns_sequence(db):
    bus = MessageBus(db)
    for i in range(3):
        await bus.append_and_seq(
            Message.new("executor", "critic", "event", {"i": i})
        )
    msgs = await bus.replay_for("critic", after_seq=0)
    assert [m.payload["i"] for m in msgs] == [0, 1, 2]


@pytest.mark.asyncio
async def test_replay_skips_already_processed(db):
    bus = MessageBus(db)
    for i in range(4):
        await bus.append_and_seq(
            Message.new("executor", "critic", "event", {"i": i})
        )
    msgs = await bus.replay_for("critic", after_seq=2)
    assert [m.payload["i"] for m in msgs] == [2, 3]


@pytest.mark.asyncio
async def test_lookup_by_id(db):
    bus = MessageBus(db)
    msg = Message.new("e", "c", "proposal", {"k": "v"})
    seq = await bus.append_and_seq(msg)
    fetched = await bus.lookup_by_id(msg.msg_id)
    assert fetched is not None
    assert fetched.seq == seq
    assert fetched.payload == {"k": "v"}


@pytest.mark.asyncio
async def test_count(db):
    bus = MessageBus(db)
    assert await bus.count() == 0
    await bus.append_and_seq(Message.new("e", "c", "event", {}))
    assert await bus.count() == 1


@pytest.mark.asyncio
async def test_append_batch_atomic(db):
    bus = MessageBus(db)
    msgs = [Message.new("e", "c", "event", {"i": i}) for i in range(5)]
    seqs = await bus.append_batch(msgs)
    assert seqs == sorted(seqs)
    assert len(seqs) == 5
    rows = await db.fetchall("SELECT COUNT(*) AS c FROM events")
    assert rows[0]["c"] == 5
