"""Tests for per-role topic subscriptions and self-echo exclusion."""
import asyncio
import pytest
from hyperloom.orchestrator.bus.message_bus import Message, MessageBus, ROLE_SUBSCRIPTIONS

# Use in-memory SQLite for all tests
@pytest.fixture
def bus(tmp_path):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.bus.storage.schema import create_tables
    import asyncio
    async def _make():
        conn = SqliteConnection(tmp_path / "bus.db")
        await conn.connect()
        await create_tables(conn)
        return MessageBus(conn)
    return asyncio.get_event_loop().run_until_complete(_make())

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def test_sender_does_not_receive_own_message(bus):
    """A message sent by orchestration must not appear in orchestration's inbox."""
    run(bus.append_and_seq(Message.new("orchestration", "*", "observation", {"k": "v"})))
    msgs = run(bus.replay_for("orchestration", after_seq=0))
    assert all(m.from_agent != "orchestration" for m in msgs)

def test_unsubscribed_topic_not_delivered(bus):
    """A topic not in orchestration's subscription set is not delivered."""
    # 'request' is in kernel_agent subscriptions but not orchestration
    run(bus.append_and_seq(Message.new("coordinator", "*", "request", {"kind": "integrate"})))
    msgs = run(bus.replay_for("orchestration", after_seq=0))
    assert not any(m.topic == "request" for m in msgs)

def test_subscribed_topic_is_delivered(bus):
    """A topic in orchestration's subscription set IS delivered."""
    run(bus.append_and_seq(Message.new("coordinator", "*", "delegated_result", {"task_id": "t1"})))
    msgs = run(bus.replay_for("orchestration", after_seq=0))
    assert any(m.topic == "delegated_result" for m in msgs)

def test_lookup_by_id_bypasses_subscription(bus):
    """lookup_by_id returns the message regardless of subscription."""
    msg = Message.new("orchestration", "*", "observation", {"x": 1})
    run(bus.append_and_seq(msg))
    found = run(bus.lookup_by_id(msg.msg_id))
    assert found is not None
    assert found.msg_id == msg.msg_id

def test_tail_bypasses_subscription(bus):
    """tail() returns all messages regardless of subscription."""
    run(bus.append_and_seq(Message.new("orchestration", "*", "observation", {"y": 2})))
    all_msgs = run(bus.tail(n=100))
    assert len(all_msgs) == 1
