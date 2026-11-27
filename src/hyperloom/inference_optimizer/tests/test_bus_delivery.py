# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inbox delivery: an agent's inbox holds every message but its own.

``replay_for`` and ``has_actionable_inbox`` are the only inbox paths that filter; ``tail`` is a diagnostic read and
must stay unfiltered.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.bus.message_bus import LANES_BUSY_SOURCE, Message, MessageBus
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection


@pytest.fixture
def bus(tmp_path) -> MessageBus:
    sc = SqliteConnection(tmp_path / "bus.db")
    yield MessageBus(sc)
    sc.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("to_agent", "topic"),
    [("*", "observation"), ("kernel_agent", "request"), ("orchestration", "observation")],
    ids=["broadcast", "request", "addressed-to-itself"],
)
async def test_sender_does_not_receive_own_message(bus: MessageBus, to_agent: str, topic: str) -> None:
    await bus.append_and_seq(Message.new("orchestration", to_agent, topic, {"k": "v"}))

    assert await bus.replay_for("orchestration", after_seq=0) == []
    assert not await bus.has_actionable_inbox("orchestration", after_seq=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("from_agent", "to_agent", "topic"),
    [("coordinator", "*", "delegated_result"), ("kernel_agent", "orchestration", "response")],
    ids=["broadcast", "addressed"],
)
async def test_another_senders_message_is_delivered(
    bus: MessageBus, from_agent: str, to_agent: str, topic: str
) -> None:
    await bus.append_and_seq(Message.new(from_agent, to_agent, topic, {"task_id": "t1"}))

    msgs = await bus.replay_for("orchestration", after_seq=0)

    assert [m.topic for m in msgs] == [topic]
    assert await bus.has_actionable_inbox("orchestration", after_seq=0)


@pytest.mark.asyncio
async def test_a_delivered_message_is_not_delivered_again_past_its_seq(bus: MessageBus) -> None:
    seq = await bus.append_and_seq(Message.new("coordinator", "*", "event", {"kind": "task_queued"}))

    assert await bus.replay_for("orchestration", after_seq=seq) == []
    assert not await bus.has_actionable_inbox("orchestration", after_seq=seq)


@pytest.mark.asyncio
async def test_a_lanes_busy_reply_is_delivered_but_asks_nothing(bus: MessageBus) -> None:
    await bus.append_and_seq(
        Message.new(
            "kernel_agent", "orchestration", "response", {"kind": "integrate_done", "source": LANES_BUSY_SOURCE}
        )
    )

    assert len(await bus.replay_for("orchestration", after_seq=0)) == 1
    assert not await bus.has_actionable_inbox("orchestration", after_seq=0)


@pytest.mark.asyncio
async def test_tail_shows_the_senders_own_rows(bus: MessageBus) -> None:
    await bus.append_and_seq(Message.new("orchestration", "*", "observation", {"y": 2}))
    assert len(await bus.tail(n=100)) == 1
