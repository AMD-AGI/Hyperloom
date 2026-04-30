"""v0.4 MVP — every agent (executor / critic / triage / kernel) running
as an independent CLI through the MultiCLIRouter.

These tests do *not* spawn real Claude CLIs. Instead they:

  1. Set the Conductor up in MULTI_CLI mode + marathon execution.
  2. Use the bundled agent_cards to validate discovery + Router setup.
  3. Simulate each agent emitting a representative intent into its
     outbox.jsonl and assert the Router → PolicyGate → handle_intent
     path lands the result on the bus exactly as the legacy reactor
     would have done.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from inference_optimizer.agents import agents_root
from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import Conductor, TransportMode
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import IntentType
from inference_optimizer.orchestrator.multi_cli.envelope import (
    Envelope,
    write_envelope,
    _SEQ_ALLOCATOR,
)
from inference_optimizer.orchestrator.multi_cli.router import (
    agent_inbox_path,
    agent_outbox_path,
)
from inference_optimizer.storage.connection import SqliteConnection


@pytest.fixture(autouse=True)
def _reset_seq_allocator():
    _SEQ_ALLOCATOR.reset()
    yield
    _SEQ_ALLOCATOR.reset()


@pytest.fixture
def marathon_env(tmp_path):
    return {
        "MODEL_PATH": str(tmp_path / "fake-model"),
        "MAX_HOURS": "10.0",  # marathon (>6h)
        "TARGET_GAIN_PCT": "100",
    }


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_marathon_multi_cli_loads_all_four_v04_agents(
    session_dir, marathon_env,
):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**marathon_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()
        # v0.4 — guided/marathon roster collapsed to executor + critic +
        # kernel + triage (per standalone §13.2).
        assert set(ctx.cli_agents) == {
            "executor", "critic", "kernel", "triage",
        }
        assert ctx.in_proc_roles == []
        # Router should have all four agent cards.
        assert ctx.multi_cli_router is not None
        registered = set(ctx.multi_cli_router.agents.keys())
        assert registered == {
            "executor", "critic", "kernel", "triage",
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_each_agent_can_emit_their_allowed_intents(session_dir, marathon_env):
    """Execute the 'happy path' intent for each of the four roles and
    confirm PolicyGate accepts + bus records them.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**marathon_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()

        # Per-role happy-path intent envelopes.
        cases = [
            ("executor", "send_message",
             {"topic": "heartbeat", "body_md": "from executor CLI"}),
            ("critic", "send_message",
             {"topic": "heartbeat", "body_md": "from critic CLI"}),
            ("triage", "alert",
             {"severity": "low", "summary": "diagnostic ping",
              "detail": "no action needed"}),
            # Plan A — kernel agent's happy-path intent is a heartbeat
            # send_message (RESPONSE requires an in_reply_to to a real
            # request, which we don't seed here).
            ("kernel", "send_message",
             {"topic": "heartbeat", "body_md": "from kernel CLI"}),
        ]
        for agent, itype, payload in cases:
            outbox = agent_outbox_path(session_dir, agent)
            write_envelope(outbox, Envelope.intent(
                from_agent=agent, intent_type=itype, payload=payload,
            ))

        await ctx.multi_cli_router.drain_outbox_tick()

        msgs = await ctx.bus.tail(n=200)
        for agent, _itype, _payload in cases:
            relevant = [
                m for m in msgs
                if m.from_agent == agent
                and m.topic in ("heartbeat", "alert")
            ]
            assert relevant, f"no bus event from {agent}: {msgs!r}"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_critic_cannot_delegate(session_dir, marathon_env):
    """v0.4 — critic must be denied 'delegate' intents (PolicyGate).
    The Router + Conductor pipeline should record a policy_denied
    observation rather than queueing a task.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**marathon_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()

        outbox = agent_outbox_path(session_dir, "critic")
        write_envelope(outbox, Envelope.intent(
            from_agent="critic",
            intent_type="delegate",
            payload={"action_name": "baseline"},
        ))
        await ctx.multi_cli_router.drain_outbox_tick()

        denied = [
            m for m in await ctx.bus.tail(n=200)
            if m.topic == "observation"
            and m.payload.get("kind") == "policy_denied"
            and m.payload.get("agent") == "critic"
        ]
        assert denied, "expected a policy_denied observation for critic delegate"
        rows = await db.fetchall("SELECT kind FROM tasks WHERE kind='delegate'")
        assert rows == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_only_triage_can_emit_kill_task(session_dir, marathon_env):
    """v0.4 MVP — KILL_TASK is allowed to be emitted only by triage.
    Even if executor would emit kill_task, PolicyGate denies via the
    role gate (KILL_TASK not in executor.allowed_intents).
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**marathon_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()

        outbox = agent_outbox_path(session_dir, "executor")
        write_envelope(outbox, Envelope.intent(
            from_agent="executor",
            intent_type="kill_task",
            payload={"task_id": "fake-task-id", "reason": "demo"},
        ))
        await ctx.multi_cli_router.drain_outbox_tick()

        denied = [
            m for m in await ctx.bus.tail(n=200)
            if m.topic == "observation"
            and m.payload.get("kind") == "policy_denied"
            and m.payload.get("agent") == "executor"
        ]
        assert denied, "expected a policy_denied observation for executor kill_task"
        # No kill events.
        kills = [m for m in await ctx.bus.tail(n=200) if m.topic == "kill"]
        assert kills == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_started_event_mirrors_into_every_inbox(session_dir, marathon_env):
    """The bootstrap `run_started` event addressed to '*' must end up
    in every active agent's inbox.jsonl after one mirror tick.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**marathon_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()
        await ctx.multi_cli_router.mirror_bus_tick()

        for agent in ("executor", "critic", "kernel", "triage"):
            inbox = agent_inbox_path(session_dir, agent)
            assert inbox.is_file(), f"missing inbox for {agent}"
            body = inbox.read_text(encoding="utf-8").strip()
            assert body, f"empty inbox for {agent}"
            assert "run_started" in body, f"run_started missing in {agent} inbox"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Bundled system prompts exist for every agent
# ---------------------------------------------------------------------------
def test_every_bundled_agent_card_has_system_prompt():
    from inference_optimizer.orchestrator.multi_cli.agent_card import (
        discover_agent_cards,
    )
    cards = discover_agent_cards(agents_root())
    for name, card in cards.items():
        assert card.system_prompt_path.is_file(), (
            f"agent {name} missing system prompt at {card.system_prompt_path}"
        )
        body = card.system_prompt_path.read_text(encoding="utf-8")
        # Each wrapper prompt must explain the inbox/outbox contract.
        assert "inbox.jsonl" in body
        assert "outbox.jsonl" in body
        # STOP signal — accept either the literal sentinel or the explicit
        # filename; v0.4 triage prompt mentions kill_task instead of
        # repeating the STOP literal, but executor/critic/kernel do.
        assert "STOP_AGENT_" in body or "kill_task" in body.lower(), (
            f"agent {name} prompt should mention STOP_AGENT_ or kill_task"
        )
