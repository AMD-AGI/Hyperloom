"""Tests for Conductor ↔ MultiCLIRouter wiring (Phase 2 — executor first).

These tests focus on the Conductor's transport_mode dispatch logic. We
do *not* spawn actual CLI subprocesses; instead the test writes intent
envelopes directly to outbox.jsonl as if a CLI agent had emitted them,
then verifies the Router → PolicyGate → handle_intent pipeline picks
them up and lands them on the bus.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import (
    Conductor,
    ConductorContext,
    TransportMode,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
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
def quick_env(tmp_path):
    """Smallest viable env to drive Conductor.run() end-to-end."""
    return {
        "MODEL_PATH": str(tmp_path / "fake-model"),
        "MAX_HOURS": "0.001",  # 3.6s budget; clock will time out
        "TARGET_GAIN_PCT": "100",
    }


# ---------------------------------------------------------------------------
# transport_mode resolution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_proc_yields_no_cli_agents(session_dir, quick_env):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.SINGLE_PROC,
        )
        ctx = await conductor._bootstrap()
        assert ctx.cli_agents == ()
        assert ctx.in_proc_roles == ctx.roles
        assert ctx.multi_cli_router is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_multi_cli_lifts_every_role_to_cli(session_dir, quick_env, tmp_path):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            agents_root=tmp_path / "no-such-dir",  # forces stub cards
        )
        ctx = await conductor._bootstrap()
        # v0.4 — quick mode roster = [executor, triage] (triage always-on)
        assert ctx.cli_agents == ("executor", "triage")
        assert ctx.in_proc_roles == []
        assert ctx.multi_cli_router is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_hybrid_filters_to_named_agents(session_dir, quick_env, tmp_path):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.HYBRID,
            cli_agents=["executor", "ghost"],  # 'ghost' should be dropped
            agents_root=tmp_path / "no-such-dir",
        )
        ctx = await conductor._bootstrap()
        assert ctx.cli_agents == ("executor",)  # ghost dropped
        # v0.4 — quick mode also has triage active; in hybrid w/ only
        # executor in cli_agents, triage stays in-process.
        assert [r.name for r in ctx.in_proc_roles] == ["triage"]
        assert ctx.multi_cli_router is not None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Outbox → bus integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outbox_intent_routes_through_policy_to_bus(
    session_dir, quick_env, tmp_path
):
    """Write an intent envelope as if executor CLI had emitted it; verify
    the Router → PolicyGate → _handle_intent path persists it on the bus.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            agents_root=tmp_path / "no-such-dir",
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()
        assert ctx.multi_cli_router is not None

        outbox = agent_outbox_path(session_dir, "executor")
        write_envelope(outbox, Envelope.intent(
            from_agent="executor",
            intent_type="send_message",
            payload={"topic": "heartbeat", "body_md": "hello from CLI"},
        ))
        # One drain tick is enough since we only wrote one envelope.
        await ctx.multi_cli_router.drain_outbox_tick()

        bus_msgs = await ctx.bus.tail(n=50)
        # Should contain the heartbeat we just routed.
        assert any(
            m.from_agent == "executor"
            and m.topic == "heartbeat"
            and m.payload.get("body_md") == "hello from CLI"
            for m in bus_msgs
        ), [(m.from_agent, m.topic, m.payload) for m in bus_msgs]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_router_blocks_role_disallowed_intent(
    session_dir, quick_env, tmp_path
):
    """A critic CLI must not be able to emit a `delegate` — PolicyGate
    rejects it and the bus only sees a policy_denied observation.
    """
    # Start in guided mode so critic is active.
    env = {**quick_env, "MAX_HOURS": "3.0"}
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env=env,
            db=db,
            transport_mode=TransportMode.HYBRID,
            cli_agents=["critic"],
            agents_root=tmp_path / "no-such-dir",
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

        # The bus should NOT have any task entry / delegate event from
        # critic; only a policy_denied observation should land.
        bus_msgs = await ctx.bus.tail(n=50)
        denied = [
            m for m in bus_msgs
            if m.topic == "observation"
            and m.payload.get("kind") == "policy_denied"
            and m.payload.get("agent") == "critic"
        ]
        assert denied, [(m.topic, m.payload) for m in bus_msgs]
        # And there must be no proposal / delegate task created.
        task_rows = await db.fetchall("SELECT kind FROM tasks")
        assert task_rows == []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Bus → inbox mirror at run start (sanity)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bus_event_mirrors_into_executor_inbox(
    session_dir, quick_env, tmp_path
):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
            agents_root=tmp_path / "no-such-dir",
            router_tick_s=0.05,
        )
        ctx = await conductor._bootstrap()
        # Bootstrap emits a `run_started` event already; mirror it.
        wrote = await ctx.multi_cli_router.mirror_bus_tick()
        assert wrote >= 1
        inbox = agent_inbox_path(session_dir, "executor")
        assert inbox.is_file()
        body = inbox.read_text().strip().splitlines()
        first = json.loads(body[0])
        assert first["kind"] == "message"
        assert first["from_agent"] == "conductor"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Real card discovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_uses_bundled_agent_cards_by_default(session_dir, quick_env):
    """When no agents_root override is given, Conductor should pick up
    the bundled cards under src/inference_optimizer/agents/.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    try:
        conductor = Conductor(
            session_dir,
            backend=MockBackend(),
            env={**quick_env},
            db=db,
            transport_mode=TransportMode.MULTI_CLI,
        )
        ctx = await conductor._bootstrap()
        # Router should see executor's bundled card (real backend=claude).
        assert ctx.multi_cli_router is not None
        card = ctx.multi_cli_router.agents.get("executor")
        assert card is not None
        assert card.backend == "claude"
        assert card.role == "executor"
    finally:
        db.close()
