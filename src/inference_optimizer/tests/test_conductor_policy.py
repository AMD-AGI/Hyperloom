"""Tests for Conductor → PolicyGate / multi-reactor wiring (F1c).

These tests prove:

    1. ``Conductor.run()`` spawns one reactor per role from
       ``roles_for_mode`` (quick=1, guided=2, marathon=4).
    2. Every parsed intent passes through ``PolicyGate.validate_intent``
       before ``_handle_intent``.
    3. A denied intent is logged on the bus as ``observation``
       payload.kind == ``policy_denied`` and never reaches the bus as a
       real send_message.
    4. Per-role allowed_tools are passed to the backend (Codex roles
       receive ``[]``, Claude roles receive ``["emit_intent"]``).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

import pytest

from inference_optimizer.orchestrator.backends import MockBackend, ScriptStep
from inference_optimizer.orchestrator.backends.base import Backend
from inference_optimizer.orchestrator.conductor import Conductor, StopReason
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.storage.connection import SqliteConnection


TINY_QUICK_HOURS = "0.0005"   # ~1.8s wall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _RecordingBackend(Backend):
    """Records every backend call and returns scripted intents per agent."""

    intents_per_agent: dict[str, list[Intent]] | None = None
    calls: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self):
        self.calls = []
        if self.intents_per_agent is None:
            self.intents_per_agent = {}

    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 10,
        extra: dict | None = None,
    ) -> list[Intent]:
        self.calls.append(
            {
                "agent": agent_name,
                "allowed_tools": tuple(allowed_tools),
                "extra": dict(extra or {}),
            }
        )
        # Return scripted intents (or default heartbeat).
        scripted = self.intents_per_agent.get(agent_name)
        if scripted is not None:
            # Pop and reuse final entry to keep the loop alive.
            if len(scripted) > 1:
                return [scripted.pop(0)]
            return [scripted[0]]
        # Default — empty so the reactor does nothing notable.
        return [
            Intent(
                type=IntentType.SEND_MESSAGE,
                payload={
                    "to": "*",
                    "topic": "heartbeat",
                    "body_md": f"alive {agent_name}",
                },
            )
        ]


# ---------------------------------------------------------------------------
# Multi-reactor spawn count
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_quick_mode_spawns_executor_and_triage(session_dir):
    """v0.4 — quick mode roster: executor + triage (always-on)."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = _RecordingBackend()
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_QUICK_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
        triage_tick_s=0.5,   # let triage fire at least once during the test
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    agents_called = {c["agent"] for c in backend.calls}
    assert "executor" in agents_called
    assert "triage" in agents_called
    db.close()


@pytest.mark.asyncio
async def test_guided_mode_spawns_executor_and_critic(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = _RecordingBackend()
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "3"},  # → guided
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def kick():
        await asyncio.sleep(1.0)
        if conductor.ctx is not None:
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=15.0
    )

    agents_called = {c["agent"] for c in backend.calls}
    assert "executor" in agents_called
    assert "critic" in agents_called
    db.close()


@pytest.mark.asyncio
async def test_marathon_mode_spawns_full_roster(session_dir):
    """v0.4 — guided/marathon roster: executor + critic + kernel + triage."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = _RecordingBackend()
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "12"},  # → marathon
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
        triage_tick_s=0.5,   # let triage fire at least once during the test
    )

    async def kick():
        await asyncio.sleep(2.0)
        if conductor.ctx is not None:
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=20.0
    )

    agents_called = {c["agent"] for c in backend.calls}
    # v0.4 — executor + critic + kernel + triage are the roster.
    assert "executor" in agents_called
    assert "critic" in agents_called
    assert "kernel" in agents_called
    assert "triage" in agents_called
    db.close()


# ---------------------------------------------------------------------------
# Per-role allowed_tools
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_codex_role_receives_no_tools(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = _RecordingBackend()
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "12"},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def kick():
        await asyncio.sleep(1.5)
        if conductor.ctx is not None:
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=20.0
    )

    by_agent = {c["agent"]: c for c in backend.calls}
    # v0.4 — all 4 roles are Claude-backed; each gets emit_intent.
    for name in ("executor", "critic", "triage", "kernel"):
        if name in by_agent:
            assert "emit_intent" in by_agent[name]["allowed_tools"], (
                f"{name} should receive emit_intent tool in v0.4"
            )
    db.close()


# ---------------------------------------------------------------------------
# PolicyGate denial flow
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disallowed_intent_logs_policy_denied_observation(session_dir):
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    # Have the executor try to update_state with a *core* field, which
    # PolicyGate denies (rule="state_field").
    bad_intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_best": "v2"}},
    )
    backend = MockBackend(
        script=[ScriptStep(intents=[bad_intent], only_if_agent="executor")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_QUICK_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    denials = [
        e for e in events
        if e.topic == "observation"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "policy_denied"
    ]
    assert denials, "expected at least one policy_denied observation"
    sample = denials[0]
    assert sample.payload.get("agent") == "executor"
    assert sample.payload.get("intent_type") == "update_state"
    assert sample.payload.get("rule") == "state_field"
    db.close()


@pytest.mark.asyncio
async def test_allowed_intent_reaches_bus_normally(session_dir):
    """A non-core update_state should pass policy and *not* produce a denial."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    good_intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_action": "kernel_opt"}},
    )
    backend = MockBackend(
        script=[ScriptStep(intents=[good_intent], only_if_agent="executor")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_QUICK_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    denials = [
        e for e in events
        if e.topic == "observation"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "policy_denied"
    ]
    assert not denials, f"unexpected denial: {[d.payload for d in denials]}"
    db.close()


@pytest.mark.asyncio
async def test_run_started_event_lists_roles(session_dir):
    """The ``run_started`` event should embed the resident roles."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    conductor = Conductor(
        session_dir,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_QUICK_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    started = [
        e for e in events
        if e.topic == "event"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "run_started"
    ]
    assert started
    payload = started[0].payload
    # v0.4 — quick mode roster is [executor, triage].
    assert payload.get("roles") == ["executor", "triage"]
    db.close()
