"""Phase 3 — every agent (executor / critic / watchdog / sage) running
as an independent CLI through the MultiCLIRouter.

These tests do *not* spawn real Claude/Codex CLIs. Instead they:

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
async def test_marathon_multi_cli_loads_all_four_agents(session_dir, marathon_env):
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
        # Every active marathon role should be lifted to a CLI.
        assert set(ctx.cli_agents) == {"executor", "critic", "watchdog", "sage"}
        assert ctx.in_proc_roles == []
        # Router should have all four agent cards.
        assert ctx.multi_cli_router is not None
        registered = set(ctx.multi_cli_router.agents.keys())
        assert registered == {"executor", "critic", "watchdog", "sage"}
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
            ("watchdog", "alert",
             {"severity": "low", "summary": "diagnostic ping",
              "detail": "no action needed"}),
            ("sage", "send_message",
             {"topic": "heartbeat", "body_md": "from sage CLI"}),
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
async def test_sage_cannot_delegate(session_dir, marathon_env):
    """Sage must be denied 'delegate' intents (PolicyGate). The Router
    + Conductor pipeline should record a policy_denied observation
    rather than queueing a task.
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

        outbox = agent_outbox_path(session_dir, "sage")
        write_envelope(outbox, Envelope.intent(
            from_agent="sage",
            intent_type="delegate",
            payload={"action_name": "baseline"},
        ))
        await ctx.multi_cli_router.drain_outbox_tick()

        denied = [
            m for m in await ctx.bus.tail(n=200)
            if m.topic == "observation"
            and m.payload.get("kind") == "policy_denied"
            and m.payload.get("agent") == "sage"
        ]
        assert denied, "expected a policy_denied observation for sage delegate"
        # And no delegate task got queued.
        rows = await db.fetchall("SELECT kind FROM tasks WHERE kind='delegate'")
        assert rows == []
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

        for agent in ("executor", "critic", "watchdog", "sage"):
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
        assert "STOP_AGENT_" in body
