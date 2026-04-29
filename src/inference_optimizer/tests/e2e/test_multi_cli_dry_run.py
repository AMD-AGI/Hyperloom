"""End-to-end smoke for ``--transport multi-cli`` (Phase 3 validation).

What this proves
================

1. ``Conductor.run()`` boots cleanly when ``transport_mode`` is set to
   ``multi-cli`` (or ``hybrid``) — it spawns the MultiCLIRouter task in
   place of the legacy in-process reactors for the lifted agents.
2. Outbox intent envelopes (simulating what a real Claude/Codex CLI
   would write) flow through the Router → PolicyGate →
   ``Conductor._handle_intent`` pipeline to land on the SQLite bus.
3. The Router mirrors bus events into per-agent inboxes during the
   normal clock cadence, so a long-running CLI process picks them up on
   its next iteration.
4. Graceful shutdown still triggers ``time_exhausted`` and the Router
   tears down without dangling tasks.

This is the smoke test the plan calls "validation_marathon7h" — a real
24h run still has to happen on real hardware, but we now have a fast
correctness check that exercises the same Conductor code paths.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.conductor import (
    Conductor,
    StopReason,
    TransportMode,
)
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


TINY_MAX_HOURS = "0.0005"  # ~1.8s wall budget


@pytest.fixture(autouse=True)
def _reset_seq_allocator():
    _SEQ_ALLOCATOR.reset()
    yield
    _SEQ_ALLOCATOR.reset()


@pytest.mark.asyncio
async def test_multi_cli_quick_mode_completes(session_dir):
    """Boot in --transport multi-cli (quick mode → only executor lifted),
    let the clock fire a few times, and ensure graceful stop fires.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.05,
        clock_tick_s=0.1,
        router_tick_s=0.05,
        transport_mode=TransportMode.MULTI_CLI,
    )

    async def emit_a_heartbeat() -> None:
        """Simulate the executor CLI dropping one envelope into outbox."""
        # Wait long enough for bootstrap to create the outbox path.
        await asyncio.sleep(0.2)
        outbox = agent_outbox_path(session_dir, "executor")
        write_envelope(outbox, Envelope.intent(
            from_agent="executor",
            intent_type="send_message",
            payload={"topic": "heartbeat", "body_md": "hello from CLI"},
        ))

    emit_task = asyncio.create_task(emit_a_heartbeat())
    ctx = await asyncio.wait_for(conductor.run(), timeout=10.0)
    await emit_task

    assert ctx.state.stop_reason == StopReason.TIME_EXHAUSTED
    assert ctx.cli_agents == ("executor",)  # quick mode roster
    assert ctx.in_proc_roles == []
    assert ctx.multi_cli_router is not None

    # Re-open the bus to confirm the heartbeat made it through PolicyGate.
    bus = MessageBus(db)
    msgs = await bus.tail(n=200)
    assert any(
        m.from_agent == "executor"
        and m.topic == "heartbeat"
        and m.payload.get("body_md") == "hello from CLI"
        for m in msgs
    ), [(m.from_agent, m.topic, m.payload) for m in msgs]

    db.close()


@pytest.mark.asyncio
async def test_multi_cli_marathon_lifts_all_four_agents(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={
            "MODEL_PATH": "fake/model",
            "MAX_HOURS": "10",  # marathon
            "TARGET_GAIN_PCT": "100",
        },
        db=db,
        reactor_tick_s=0.05,
        clock_tick_s=0.1,
        router_tick_s=0.05,
        transport_mode=TransportMode.MULTI_CLI,
    )
    ctx = await conductor._bootstrap()
    assert set(ctx.cli_agents) == {"executor", "critic", "watchdog", "sage"}
    assert ctx.multi_cli_router is not None
    # Don't run() — that would actually spend 10h. Just confirm the
    # bootstrap produces the expected topology.
    db.close()


@pytest.mark.asyncio
async def test_hybrid_mode_keeps_other_roles_in_process(session_dir):
    """In hybrid mode, only the named agents become CLIs; the rest still
    run as in-process reactors so we don't lose coverage during migration.
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={
            "MODEL_PATH": "fake/model",
            "MAX_HOURS": "5",  # guided mode (executor + critic active)
            "TARGET_GAIN_PCT": "100",
        },
        db=db,
        reactor_tick_s=0.05,
        clock_tick_s=0.1,
        router_tick_s=0.05,
        transport_mode=TransportMode.HYBRID,
        cli_agents=["executor"],  # only executor is a CLI
    )
    ctx = await conductor._bootstrap()
    assert ctx.cli_agents == ("executor",)
    assert {r.name for r in ctx.in_proc_roles} == {"critic"}
    assert ctx.multi_cli_router is not None
    db.close()


@pytest.mark.asyncio
async def test_router_mirrors_bus_events_into_inbox_after_run(session_dir):
    """End-to-end: after a short run, every active agent should have at
    least one mirrored event (the boot ``run_started`` to ``*``).
    """
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_MAX_HOURS},
        db=db,
        reactor_tick_s=0.05,
        clock_tick_s=0.1,
        router_tick_s=0.05,
        transport_mode=TransportMode.MULTI_CLI,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    inbox = agent_inbox_path(session_dir, "executor")
    assert inbox.is_file()
    body = inbox.read_text(encoding="utf-8").strip()
    assert body, "executor inbox empty after multi-cli run"
    first = json.loads(body.splitlines()[0])
    assert first["kind"] == "message"
    assert first["from_agent"] == "conductor"
    db.close()
