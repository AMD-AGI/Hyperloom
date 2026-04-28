"""Tests for the M7 Conductor wiring — IMPL-CHECKLIST §7.5/7.10/7.11/7.16/7.18/7.19/7.23.

Covers:

    - Persona file is loaded into the prompt header
    - IronRules block is rendered in the prompt
    - ``_dispatcher_loop`` actually drains queued ``delegate`` tasks via
      ``SubAgentRunner`` (subset of §7.10)
    - 30-min checkpoint cadence wires through ``Checkpoint.create``
    - ``ephemeral_rca_via_critic`` extracts the rca_finding payload
    - ``_open_parliament`` returns approved / rejected based on votes
    - ``TokenBudgetMeter.record`` / ``should_throttle`` rules
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.backends import MockBackend
from inference_optimizer.orchestrator.backends.mock import ScriptStep
from inference_optimizer.orchestrator.conductor import (
    Conductor,
    StopReason,
    TokenBudgetMeter,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.storage.connection import SqliteConnection


SKILL_ACTIONS_DIR = (
    Path(__file__).resolve().parents[3]
    / ".cursor" / "skills" / "inference-optimizer" / "actions"
)


# ---------------------------------------------------------------------------
# Persona index in compose_prompt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_compose_prompt_includes_persona_file(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    personas = tmp_path / "personas"
    personas.mkdir(parents=True, exist_ok=True)
    (personas / "executor.md").write_text(
        "I prefer vllm on dense models.", encoding="utf-8"
    )

    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt("executor", msgs=[])
    assert "I prefer vllm on dense models" in prompt
    assert "Persona (running notes)" in prompt
    assert "Iron Rules" in prompt
    db.close()


@pytest.mark.asyncio
async def test_compose_prompt_iron_rules_block_present(tmp_path: Path):
    """The dynamic IronRules block is always rendered — its content shifts
    with mode, but the universal IR-4 / IR-5 lines are mandatory.

    (The executor's *static* role prompt also references IR-1..IR-7 as a
    recap, so we cannot assert on rule absence in the full prompt; we
    only check that the dynamic block prepares the universal rules.)
    """
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt("executor", msgs=[])
    assert "IR-4" in prompt
    assert "IR-5" in prompt
    # The dynamic render uses an explicit "(block)" tag on each rule —
    # the executor.md static persona doesn't, so this lets us tell them
    # apart.
    assert "(block)" in prompt
    db.close()


@pytest.mark.asyncio
async def test_compose_prompt_sage_hint_block(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    prompt = conductor._compose_prompt(
        "executor",
        msgs=[],
        sage_hint="- prefer vllm for dense\n- watch out for kv-cache fp8",
    )
    assert "Sage hint" in prompt
    assert "kv-cache fp8" in prompt
    db.close()


# ---------------------------------------------------------------------------
# Persona index refresh after update_persona intent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_update_persona_refreshes_in_memory_index(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    intent = Intent(
        type=IntentType.UPDATE_PERSONA,
        payload={"body_md": "Critical lesson: always check GPU memory"},
    )
    await conductor._handle_intent("executor", intent)
    assert "always check GPU memory" in conductor.ctx.persona_index["executor"]
    db.close()


# ---------------------------------------------------------------------------
# Dispatcher loop drains queued delegate tasks
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_dispatcher_loop_drains_queued_delegates(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    registry = ActionRegistry(SKILL_ACTIONS_DIR).load()

    # Backend that emits one update_state intent — SubAgentRunner will
    # treat that as success metric extraction.
    backend = MockBackend(
        script=[
            ScriptStep(
                intents=[
                    Intent(
                        type=IntentType.UPDATE_STATE,
                        payload={"changes": {"current_tput": 5500.0}},
                    )
                ]
            )
        ]
    )
    conductor = Conductor(
        tmp_path,
        backend=backend,
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.005"},
        db=db,
        action_registry=registry,
        reactor_tick_s=0.1,
        clock_tick_s=0.1,
        enable_dispatcher=True,
    )

    async def kick_off():
        await conductor._bootstrap()
        # queue a delegate task directly
        ctx = conductor.ctx
        await ctx.tasks.create(
            kind="delegate",
            params={"action_name": "bench_runner",
                    "params": {}, "requested_by": "executor"},
            idempotency_key="dispatcher-test-1",
            requires_lanes=["benchmark_lane"],
            allowed_tools=["emit_intent", "Read", "Bash"],
            side_effects=["reads_server"],
            lease_ttl_sec=900,
        )

    await kick_off()
    # Now run the dispatcher one-shot.
    from inference_optimizer.orchestrator.sub_agent_runner import (
        dispatch_pending_delegates,
    )
    dispatched = await dispatch_pending_delegates(
        conductor.ctx.sub_agent_runner,
        db=db,
    )
    assert dispatched == 1
    finished = await conductor.ctx.tasks.list_by_state("succeeded")
    assert len(finished) == 1
    db.close()


# ---------------------------------------------------------------------------
# Ephemeral RCA via critic
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ephemeral_rca_returns_finding(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    rca_intent = Intent(
        type=IntentType.SEND_MESSAGE,
        payload={
            "topic": "rca_finding",
            "body_md": "kernel rebuild OOMed",
            "rca": {
                "root_cause": "OOM during rebuild",
                "evidence_event_seqs": [10, 11],
                "recommended_action": "abort",
            },
        },
    )
    backend = MockBackend(script=[ScriptStep(intents=[rca_intent])])
    conductor = Conductor(
        tmp_path,
        backend=backend,
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    finding = await conductor.ephemeral_rca_via_critic()
    assert finding is not None
    assert finding["rca"]["recommended_action"] == "abort"
    db.close()


@pytest.mark.asyncio
async def test_ephemeral_rca_returns_none_when_no_finding(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    # Backend emits an unrelated send_message
    backend = MockBackend(
        script=[
            ScriptStep(
                intents=[
                    Intent(
                        type=IntentType.SEND_MESSAGE,
                        payload={"topic": "event", "body_md": "noop"},
                    )
                ]
            )
        ]
    )
    conductor = Conductor(
        tmp_path,
        backend=backend,
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    finding = await conductor.ephemeral_rca_via_critic()
    assert finding is None
    db.close()


# ---------------------------------------------------------------------------
# _open_parliament + _record_proposal_for_self_review
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_record_proposal_for_self_review_writes_event(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    await conductor._record_proposal_for_self_review(
        {"action_name": "bench_runner"}
    )
    bus = MessageBus(db)
    events = await bus.tail(n=10)
    proposals = [
        e for e in events
        if e.topic == "proposal"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "self_review"
    ]
    assert proposals
    db.close()


@pytest.mark.asyncio
async def test_open_parliament_quick_mode_abstains(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        reactor_tick_s=0.05,
    )
    await conductor._bootstrap()
    verdict = await conductor._open_parliament({"action_name": "x"})
    assert verdict == "abstained"
    db.close()


@pytest.mark.asyncio
async def test_open_parliament_marathon_counts_votes(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "12"},
        db=db,
        enable_dispatcher=False,
        reactor_tick_s=0.05,
    )
    await conductor._bootstrap()

    async def cast_votes():
        await asyncio.sleep(0.05)
        ctx = conductor.ctx
        from inference_optimizer.orchestrator.message_bus import Message
        for v in ("approve", "approve", "reject"):
            await ctx.bus.append_and_seq(
                Message.new(
                    from_agent="critic",
                    to_agent="*",
                    topic="vote",
                    payload={"verdict": v},
                )
            )

    voter = asyncio.create_task(cast_votes())
    verdict = await conductor._open_parliament({"action_name": "x"})
    await voter
    assert verdict == "approved"
    db.close()


# ---------------------------------------------------------------------------
# TokenBudgetMeter
# ---------------------------------------------------------------------------
def test_token_meter_record_increments():
    m = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    assert m.tokens_used == 0
    m.record(prompt_tokens=1000, completion_tokens=500)
    assert m.tokens_used == 1500
    assert m.remaining() == m.budget - 1500


def test_token_meter_throttle_at_80pct():
    m = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    m.record(prompt_tokens=int(m.warn_at * 0.5))
    assert m.should_throttle() is False
    m.record(prompt_tokens=int(m.warn_at * 0.6))
    assert m.should_throttle() is True


def test_token_meter_reset():
    m = TokenBudgetMeter(ExecutionMode.GUIDED_KERNEL_OPT)
    m.record(prompt_tokens=m.warn_at + 1)
    assert m.should_throttle() is True
    m.reset()
    assert m.tokens_used == 0
    assert m.should_throttle() is False


def test_token_meter_marathon_budget_higher():
    m_q = TokenBudgetMeter(ExecutionMode.QUICK_PARAM_SWEEP)
    m_m = TokenBudgetMeter(ExecutionMode.MARATHON_MULTI_AGENT)
    assert m_m.budget > m_q.budget


# ---------------------------------------------------------------------------
# 30-min checkpoint cadence (only fires when enabled + interval passed)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_checkpoint_cadence_writes_on_first_call(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=True,
    )
    await conductor._bootstrap()
    handle = await conductor._maybe_checkpoint()
    assert handle is not None
    cps = list((tmp_path / "checkpoints").iterdir())
    assert cps
    db.close()


@pytest.mark.asyncio
async def test_checkpoint_cadence_throttles_within_30min(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=True,
    )
    await conductor._bootstrap()
    h1 = await conductor._maybe_checkpoint()
    h2 = await conductor._maybe_checkpoint()
    assert h1 is not None
    # second call should be throttled (within 30 min cadence)
    assert h2 is None
    db.close()


@pytest.mark.asyncio
async def test_checkpoint_cadence_disabled_returns_none(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
        enable_checkpointing=False,
    )
    await conductor._bootstrap()
    assert await conductor._maybe_checkpoint() is None
    db.close()


# ---------------------------------------------------------------------------
# resume_from_session_dir wiring
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_init_session_resume_loads_existing_state(tmp_path: Path):
    db = SqliteConnection(tmp_path / "storage" / "conductor.db")
    conductor = Conductor(
        tmp_path,
        backend=MockBackend(),
        env={"MODEL_PATH": "fake", "MAX_HOURS": "0.001"},
        db=db,
        enable_dispatcher=False,
    )
    await conductor._bootstrap()
    # seed a cursor row so resume has something to find
    from inference_optimizer.orchestrator.cursor_store import CursorStore
    cs = CursorStore(db)
    await cs.advance("executor", seq=99, msg_id="m99")
    state = await conductor._init_session_resume()
    assert state is not None
    assert state.cursors.get("executor") == 99
    db.close()
