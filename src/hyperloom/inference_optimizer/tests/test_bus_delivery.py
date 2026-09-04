# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inbox delivery: per-role topic subscriptions and self-echo exclusion.

``replay_for`` is the only inbox path that filters; ``lookup_by_id`` and
``tail`` are diagnostic reads and must stay unfiltered.
"""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.bus.message_bus import Message, MessageBus
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection


@pytest.fixture
def bus(tmp_path) -> MessageBus:
    sc = SqliteConnection(tmp_path / "bus.db")
    yield MessageBus(sc)
    sc.close()


@pytest.mark.asyncio
async def test_sender_does_not_receive_own_message(bus: MessageBus) -> None:
    await bus.append_and_seq(Message.new("orchestration", "*", "observation", {"k": "v"}))
    msgs = await bus.replay_for("orchestration", after_seq=0)
    assert msgs == []


@pytest.mark.asyncio
async def test_unsubscribed_topic_not_delivered(bus: MessageBus) -> None:
    """``request`` is subscribed by kernel_agent only."""
    await bus.append_and_seq(Message.new("coordinator", "*", "request", {"kind": "integrate"}))
    assert await bus.replay_for("orchestration", after_seq=0) == []


@pytest.mark.asyncio
async def test_subscribed_topic_is_delivered(bus: MessageBus) -> None:
    await bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", {"task_id": "t1"}))
    msgs = await bus.replay_for("orchestration", after_seq=0)
    assert [m.topic for m in msgs] == ["delegated_result"]


@pytest.mark.asyncio
async def test_unknown_agent_receives_nothing(bus: MessageBus) -> None:
    """Absent from ROLE_SUBSCRIPTIONS means default-deny, not default-allow."""
    await bus.append_and_seq(Message.new("coordinator", "*", "observation", {"k": "v"}))
    assert await bus.replay_for("ghost", after_seq=0) == []


@pytest.mark.asyncio
async def test_lookup_by_id_bypasses_subscription(bus: MessageBus) -> None:
    msg = Message.new("orchestration", "*", "observation", {"x": 1})
    await bus.append_and_seq(msg)
    found = await bus.lookup_by_id(msg.msg_id)
    assert found is not None and found.msg_id == msg.msg_id


@pytest.mark.asyncio
async def test_tail_bypasses_subscription(bus: MessageBus) -> None:
    await bus.append_and_seq(Message.new("orchestration", "*", "observation", {"y": 2}))
    assert len(await bus.tail(n=100)) == 1
