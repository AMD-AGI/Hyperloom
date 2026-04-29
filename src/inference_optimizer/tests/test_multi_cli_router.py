"""Tests for orchestrator/multi_cli/router.py — JSONL ↔ SQLite bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.feature_flags import build_feature_flags
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import Message, MessageBus
from inference_optimizer.orchestrator.multi_cli.agent_card import (
    AgentCard,
    RestartPolicy,
)
from inference_optimizer.orchestrator.multi_cli.envelope import (
    Envelope,
    EnvelopeKind,
    _SEQ_ALLOCATOR,
    read_envelopes,
    write_envelope,
)
from inference_optimizer.orchestrator.multi_cli.router import (
    MultiCLIRouter,
    agent_inbox_path,
    agent_outbox_path,
    envelope_to_intent,
)
from inference_optimizer.orchestrator.policy import (
    DEFAULT_QUICK_ACTION_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_seq_allocator():
    _SEQ_ALLOCATOR.reset()
    yield
    _SEQ_ALLOCATOR.reset()


@pytest.fixture
def card_executor() -> AgentCard:
    return AgentCard(
        name="executor",
        role="executor",
        backend="mock",
        card_path=Path("/dev/null"),
        card_dir=Path("/dev/null"),
        allowed_modes=(),
        restart_policy=RestartPolicy(),
    )


@pytest.fixture
def card_critic() -> AgentCard:
    return AgentCard(
        name="critic",
        role="critic",
        backend="mock",
        card_path=Path("/dev/null"),
        card_dir=Path("/dev/null"),
        allowed_modes=(),
        restart_policy=RestartPolicy(continue_flag=False),
    )


# ---------------------------------------------------------------------------
# envelope_to_intent
# ---------------------------------------------------------------------------
def test_envelope_to_intent_round_trip():
    env = Envelope.intent(
        from_agent="executor",
        intent_type="delegate",
        payload={"action_name": "baseline"},
    )
    intent = envelope_to_intent(env)
    assert intent.type is IntentType.DELEGATE
    assert intent.payload == {"action_name": "baseline"}


def test_envelope_to_intent_rejects_message_envelope():
    env = Envelope.message(msg_id="m", seq=1, from_agent="conductor",
                           to_agent="executor", topic="event", payload={})
    with pytest.raises(Exception):
        envelope_to_intent(env)


def test_envelope_to_intent_rejects_unknown_intent_type():
    env = Envelope.intent(from_agent="executor", intent_type="time_travel",
                          payload={})
    with pytest.raises(Exception, match="unknown"):
        envelope_to_intent(env)


# ---------------------------------------------------------------------------
# Bus → inbox mirror
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mirror_writes_one_envelope_per_event(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(
        session_dir=session_dir,
        bus=bus,
        agents=[card_executor],
    )
    await bus.append_and_seq(Message.new("conductor", "executor", "event",
                                         {"kind": "run_started"}))
    await bus.append_and_seq(Message.new("conductor", "*", "alert",
                                         {"summary": "ok"}))

    written = await router.mirror_bus_tick()
    assert written == 2

    inbox = agent_inbox_path(session_dir, "executor")
    envs = read_envelopes(inbox)
    assert len(envs) == 2
    assert all(e.kind is EnvelopeKind.MESSAGE for e in envs)
    assert envs[0].topic == "event"
    assert envs[1].topic == "alert"


@pytest.mark.asyncio
async def test_mirror_does_not_echo_self_messages(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus, agents=[card_executor])
    # Executor talked to *: shouldn't appear in its own inbox.
    await bus.append_and_seq(Message.new("executor", "*", "event", {}))
    await bus.append_and_seq(Message.new("conductor", "executor", "event", {}))
    await router.mirror_bus_tick()
    envs = read_envelopes(agent_inbox_path(session_dir, "executor"))
    # Only the conductor->executor message should be mirrored.
    assert len(envs) == 1
    assert envs[0].from_agent == "conductor"


@pytest.mark.asyncio
async def test_mirror_is_idempotent_across_ticks(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus, agents=[card_executor])
    await bus.append_and_seq(Message.new("conductor", "executor", "event", {}))
    await router.mirror_bus_tick()
    await router.mirror_bus_tick()
    envs = read_envelopes(agent_inbox_path(session_dir, "executor"))
    assert len(envs) == 1


@pytest.mark.asyncio
async def test_mirror_picks_up_new_events_after_first_tick(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus, agents=[card_executor])
    await bus.append_and_seq(Message.new("conductor", "executor", "event", {"i": 1}))
    await router.mirror_bus_tick()
    await bus.append_and_seq(Message.new("conductor", "executor", "event", {"i": 2}))
    await router.mirror_bus_tick()
    envs = read_envelopes(agent_inbox_path(session_dir, "executor"))
    assert [e.payload.get("i") for e in envs] == [1, 2]


@pytest.mark.asyncio
async def test_mirror_persists_seq_cursor(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus, agents=[card_executor])
    await bus.append_and_seq(Message.new("conductor", "executor", "event", {}))
    await router.mirror_bus_tick()
    # Cursor file should exist with the last seq.
    inbox = agent_inbox_path(session_dir, "executor")
    cursor = inbox.with_suffix(inbox.suffix + ".seq")
    assert cursor.is_file()
    assert int(cursor.read_text()) == 1


@pytest.mark.asyncio
async def test_mirror_skips_disabled_agents(db, session_dir):
    bus = MessageBus(db)
    disabled = AgentCard(
        name="critic", role="critic", backend="codex",
        card_path=Path("/dev/null"), card_dir=Path("/dev/null"),
        enabled=False,
    )
    router = MultiCLIRouter(session_dir=session_dir, bus=bus, agents=[disabled])
    await bus.append_and_seq(Message.new("conductor", "critic", "event", {}))
    await router.mirror_bus_tick()
    inbox = agent_inbox_path(session_dir, "critic")
    assert not inbox.is_file()


# ---------------------------------------------------------------------------
# Outbox → handler drain
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_drain_calls_intent_handler(db, session_dir, card_executor):
    bus = MessageBus(db)
    captured: list[tuple[str, Intent]] = []

    async def handler(agent: str, intent: Intent) -> None:
        captured.append((agent, intent))

    router = MultiCLIRouter(
        session_dir=session_dir, bus=bus,
        agents=[card_executor], intent_handler=handler,
    )
    outbox = agent_outbox_path(session_dir, "executor")
    write_envelope(outbox, Envelope.intent(
        from_agent="executor", intent_type="send_message",
        payload={"topic": "heartbeat"},
    ))
    processed = await router.drain_outbox_tick()
    assert processed == 1
    assert captured == [("executor", Intent(
        type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat"}
    ))]


@pytest.mark.asyncio
async def test_drain_advances_offset_and_is_idempotent(db, session_dir, card_executor):
    bus = MessageBus(db)
    seen: list[Intent] = []

    async def handler(agent: str, intent: Intent) -> None:
        seen.append(intent)

    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], intent_handler=handler)
    outbox = agent_outbox_path(session_dir, "executor")
    write_envelope(outbox, Envelope.intent(
        from_agent="executor", intent_type="send_message",
        payload={"topic": "heartbeat"},
    ))
    await router.drain_outbox_tick()
    await router.drain_outbox_tick()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_drain_processes_new_lines_after_subsequent_writes(
    db, session_dir, card_executor
):
    bus = MessageBus(db)
    seen: list[Intent] = []

    async def handler(agent: str, intent: Intent) -> None:
        seen.append(intent)

    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], intent_handler=handler)
    outbox = agent_outbox_path(session_dir, "executor")
    write_envelope(outbox, Envelope.intent(
        from_agent="executor", intent_type="send_message",
        payload={"topic": "heartbeat", "i": 1},
    ))
    await router.drain_outbox_tick()
    write_envelope(outbox, Envelope.intent(
        from_agent="executor", intent_type="send_message",
        payload={"topic": "heartbeat", "i": 2},
    ))
    await router.drain_outbox_tick()
    assert [i.payload["i"] for i in seen] == [1, 2]


@pytest.mark.asyncio
async def test_drain_enforces_policy_gate(db, session_dir, card_critic):
    """A Codex critic role is *not* allowed to delegate; PolicyGate must
    deny the intent and the handler must NOT be called.
    """
    from inference_optimizer.orchestrator.agent_role import default_role_registry

    bus = MessageBus(db)
    flags = build_feature_flags(ExecutionMode.GUIDED_KERNEL_OPT)
    policy = PolicyGate(
        flags=flags,
        mode=ExecutionMode.GUIDED_KERNEL_OPT,
        role_registry=default_role_registry(),
    )

    handler_called: list[Intent] = []

    async def handler(agent: str, intent: Intent) -> None:
        handler_called.append(intent)

    deny_records: list[tuple[str, str, str]] = []

    async def denier(agent: str, intent: Intent, rule: str, reason: str) -> None:
        deny_records.append((agent, rule, reason))

    router = MultiCLIRouter(
        session_dir=session_dir, bus=bus,
        agents=[card_critic], policy=policy,
        intent_handler=handler, deny_recorder=denier,
    )

    outbox = agent_outbox_path(session_dir, "critic")
    write_envelope(outbox, Envelope.intent(
        from_agent="critic", intent_type="delegate",
        payload={"action_name": "baseline"},
    ))
    await router.drain_outbox_tick()
    assert handler_called == []
    assert deny_records and deny_records[0][0] == "critic"
    assert deny_records[0][1] == "role"


@pytest.mark.asyncio
async def test_drain_skips_message_envelopes(db, session_dir, card_executor):
    bus = MessageBus(db)
    handler_called: list[Intent] = []

    async def handler(agent: str, intent: Intent) -> None:
        handler_called.append(intent)

    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], intent_handler=handler)
    outbox = agent_outbox_path(session_dir, "executor")
    # A misplaced "message" envelope in an outbox is logged + ignored.
    write_envelope(outbox, Envelope.message(
        msg_id="m1", seq=1, from_agent="executor", to_agent="conductor",
        topic="event", payload={},
    ))
    processed = await router.drain_outbox_tick()
    assert processed == 1  # consumed but didn't dispatch
    assert handler_called == []


@pytest.mark.asyncio
async def test_drain_records_malformed_intent(db, session_dir, card_executor):
    bus = MessageBus(db)
    deny_records: list[tuple[str, str, str]] = []

    async def denier(agent, intent, rule, reason) -> None:
        deny_records.append((agent, rule, reason))

    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], deny_recorder=denier)
    outbox = agent_outbox_path(session_dir, "executor")
    bad = Envelope.intent(from_agent="executor", intent_type="warp_drive",
                          payload={})
    write_envelope(outbox, bad)
    await router.drain_outbox_tick()
    assert deny_records and deny_records[0][1] == "payload"


@pytest.mark.asyncio
async def test_drain_persists_byte_cursor(db, session_dir, card_executor):
    bus = MessageBus(db)

    async def handler(agent, intent):
        return None

    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], intent_handler=handler)
    outbox = agent_outbox_path(session_dir, "executor")
    write_envelope(outbox, Envelope.intent(from_agent="executor",
                                           intent_type="send_message",
                                           payload={}))
    await router.drain_outbox_tick()
    cur = outbox.with_suffix(outbox.suffix + ".cursor")
    assert cur.is_file()
    assert int(cur.read_text()) > 0


# ---------------------------------------------------------------------------
# register / unregister
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_register_then_unregister(db, session_dir, card_executor):
    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus)
    assert router.agents == {}
    router.register(card_executor)
    assert "executor" in router.agents
    router.unregister("executor")
    assert router.agents == {}


# ---------------------------------------------------------------------------
# run loop is cancelable via request_stop
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_loop_cancellable(db, session_dir, card_executor):
    import asyncio

    bus = MessageBus(db)
    router = MultiCLIRouter(session_dir=session_dir, bus=bus,
                            agents=[card_executor], tick_s=0.05)
    task = asyncio.create_task(router.run())
    await asyncio.sleep(0.15)
    router.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
