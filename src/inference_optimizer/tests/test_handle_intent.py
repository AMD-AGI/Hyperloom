"""Tests for Conductor._handle_intent — F2 (10 intent type branches).

Each test scripts a single intent through MockBackend, runs the dry-run loop
briefly, then verifies the documented side-effect:

    send_message     -> bus(topic from payload)
    alert            -> bus(alert) + findings/alerts.jsonl
    propose_action   -> tasks(kind=proposal) + bus(proposal)
    delegate         -> tasks(kind=delegate)  + bus(proposal)
    update_state     -> state.changed + bus(decision)
    update_persona   -> personas/<agent>.md grew + bus(event:persona_update)
    ask_question     -> bus(question)
    answer           -> bus(answer)
    objection        -> bus(objection)
    vote             -> bus(vote)
"""
from __future__ import annotations

import asyncio
import json

import pytest

from inference_optimizer.orchestrator.backends import MockBackend, ScriptStep
from inference_optimizer.orchestrator.conductor import Conductor
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import MessageBus
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.storage.connection import SqliteConnection


TINY_QUICK_HOURS = "0.0005"


# ---------------------------------------------------------------------------
async def _run_with_intent(session_dir, intent: Intent, *, agent="executor"):
    """Boot conductor with one scripted intent for the given agent."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent=agent)]
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
    return db


def _topic_payloads(events, topic):
    return [
        e.payload for e in events
        if e.topic == topic and isinstance(e.payload, dict)
    ]


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_message_writes_topic_event(session_dir):
    intent = Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"to": "*", "topic": "event", "body_md": "hello"},
    )
    db = await _run_with_intent(session_dir, intent)
    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    bodies = [
        p.get("body_md")
        for p in _topic_payloads(events, "event")
        if isinstance(p, dict)
    ]
    assert "hello" in bodies
    db.close()


# ---------------------------------------------------------------------------
# alert
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_alert_writes_event_and_finding(session_dir):
    intent = Intent(
        type=IntentType.ALERT,
        payload={
            "severity": "high",
            "summary": "GPU OOM detected",
            "detail": "trace omitted",
        },
    )
    db = await _run_with_intent(session_dir, intent)
    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    alerts = _topic_payloads(events, "alert")
    assert any(p.get("summary") == "GPU OOM detected" for p in alerts)
    finding_path = session_dir / "findings" / "alerts.jsonl"
    assert finding_path.exists()
    rows = [
        json.loads(line)
        for line in finding_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(r["summary"] == "GPU OOM detected" for r in rows)
    db.close()


# ---------------------------------------------------------------------------
# propose_action
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_propose_action_creates_proposal_task_and_event(session_dir):
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "param_sweep_run",
            "predicted_gain_pct": 4.2,
            "params": {"chunked-prefill": True},
        },
    )
    db = await _run_with_intent(session_dir, intent)

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    proposals = _topic_payloads(events, "proposal")
    assert any(
        p.get("action_name") == "param_sweep_run" for p in proposals
    )

    rows = await db.fetchall(
        "SELECT * FROM tasks WHERE kind=?", ("proposal",)
    )
    assert rows
    payload = json.loads(rows[0]["params"])
    assert payload["action_name"] == "param_sweep_run"
    assert payload["predicted_gain_pct"] == 4.2
    assert payload["params"] == {"chunked-prefill": True}
    db.close()


@pytest.mark.asyncio
async def test_propose_action_idempotent(session_dir):
    """Re-emitting the same propose_action should not produce two task rows."""
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "bench_runner",
            "predicted_gain_pct": 1.0,
            "params": {},
        },
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    # Two scripted steps with identical intent.
    backend = MockBackend(
        script=[
            ScriptStep(intents=[intent], only_if_agent="executor"),
            ScriptStep(intents=[intent], only_if_agent="executor"),
        ]
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

    rows = await db.fetchall(
        "SELECT * FROM tasks WHERE kind=?", ("proposal",)
    )
    assert len(rows) == 1, f"expected 1 task, got {len(rows)}"
    db.close()


# ---------------------------------------------------------------------------
# delegate
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delegate_creates_delegate_task(session_dir):
    """Quick mode forbids delegate via PolicyGate (mode rule); we use guided."""
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "bench_runner", "params": {"warmup": 3}},
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="executor")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "3"},  # guided
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def kick():
        await asyncio.sleep(1.0)
        if conductor.ctx is not None:
            from inference_optimizer.orchestrator.conductor import StopReason
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=15.0
    )

    rows = await db.fetchall(
        "SELECT * FROM tasks WHERE kind=?", ("delegate",)
    )
    assert rows, "expected at least one delegate task"
    payload = json.loads(rows[0]["params"])
    assert payload["action_name"] == "bench_runner"
    assert payload["params"] == {"warmup": 3}
    db.close()


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_update_state_changes_shared_state(session_dir):
    intent = Intent(
        type=IntentType.UPDATE_STATE,
        payload={"changes": {"current_action": "kernel_opt"}, "rationale": "from sage"},
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="executor")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": TINY_QUICK_HOURS},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    ctx = await asyncio.wait_for(conductor.run(), timeout=10.0)
    assert ctx.state.current_action == "kernel_opt"

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    decisions = _topic_payloads(events, "decision")
    assert any(
        p.get("kind") == "state_updated"
        and p.get("changes", {}).get("current_action") == "kernel_opt"
        for p in decisions
    )
    db.close()


# ---------------------------------------------------------------------------
# update_persona
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_update_persona_appends_to_persona_file(session_dir):
    intent = Intent(
        type=IntentType.UPDATE_PERSONA,
        payload={"body_md": "I now believe chunked-prefill helps for this model."},
    )
    db = await _run_with_intent(session_dir, intent)

    persona_path = session_dir / "personas" / "executor.md"
    assert persona_path.exists()
    content = persona_path.read_text(encoding="utf-8")
    assert "chunked-prefill helps" in content

    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    persona_events = [
        e for e in events
        if e.topic == "event"
        and isinstance(e.payload, dict)
        and e.payload.get("kind") == "persona_update"
    ]
    assert persona_events
    assert persona_events[0].payload.get("agent") == "executor"
    db.close()


# ---------------------------------------------------------------------------
# ask_question / answer / objection / vote (simple-topic dispatcher)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_question_writes_question_event(session_dir):
    intent = Intent(
        type=IntentType.ASK_QUESTION,
        payload={"topic": "event", "question": "should we revert?"},
    )
    db = await _run_with_intent(session_dir, intent)
    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    qs = _topic_payloads(events, "question")
    assert any(p.get("question") == "should we revert?" for p in qs)
    db.close()


@pytest.mark.asyncio
async def test_answer_writes_answer_event(session_dir):
    intent = Intent(
        type=IntentType.ANSWER,
        payload={"in_reply_to": "abc-123", "answer": "yes"},
    )
    db = await _run_with_intent(session_dir, intent)
    bus = MessageBus(db)
    events = await bus.tail(n=1000)
    ans = _topic_payloads(events, "answer")
    assert any(p.get("answer") == "yes" for p in ans)
    db.close()


@pytest.mark.asyncio
async def test_objection_writes_objection_event(session_dir):
    intent = Intent(
        type=IntentType.OBJECTION,
        payload={
            "target_msg_id": "abc-123",
            "reason": "predicted gain is unrealistic",
        },
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    # Critic role required for OBJECTION (executor cannot emit it per policy).
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="critic")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "3"},  # guided -> +critic
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def kick():
        await asyncio.sleep(1.0)
        if conductor.ctx is not None:
            from inference_optimizer.orchestrator.conductor import StopReason
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=15.0
    )

    bus = MessageBus(db)
    # Filter by topic in SQL so the rare ``objection`` event isn't drowned
    # out by the high-frequency heartbeats both reactors emit during the
    # 1s test window. (Without ``topic=...`` we'd need n ≫ 1000 to surface
    # the seq=4 event in guided mode.)
    events = await bus.tail(n=200, topic="objection")
    objs = _topic_payloads(events, "objection")
    assert any(
        p.get("reason") == "predicted gain is unrealistic" for p in objs
    )
    db.close()


@pytest.mark.asyncio
async def test_vote_writes_vote_event(session_dir):
    intent = Intent(
        type=IntentType.VOTE,
        payload={"target_msg_id": "abc-123", "vote": "approve"},
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="critic")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "3"},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )

    async def kick():
        await asyncio.sleep(1.0)
        if conductor.ctx is not None:
            from inference_optimizer.orchestrator.conductor import StopReason
            conductor.ctx.state.set_stopping(StopReason.EMERGENCY)

    await asyncio.wait_for(
        asyncio.gather(conductor.run(), kick()), timeout=15.0
    )

    bus = MessageBus(db)
    # Same topic-overflow guard as the objection test above.
    events = await bus.tail(n=200, topic="vote")
    votes = _topic_payloads(events, "vote")
    assert any(p.get("vote") == "approve" for p in votes)
    db.close()


# ---------------------------------------------------------------------------
# task_idempotency_key helper
# ---------------------------------------------------------------------------
def test_task_idempotency_key_is_deterministic():
    a = Conductor._task_idempotency_key(
        kind="proposal",
        from_agent="executor",
        action_name="x",
        params={"foo": 1, "bar": 2},
    )
    b = Conductor._task_idempotency_key(
        kind="proposal",
        from_agent="executor",
        action_name="x",
        params={"bar": 2, "foo": 1},  # different key order
    )
    assert a == b
    c = Conductor._task_idempotency_key(
        kind="proposal",
        from_agent="executor",
        action_name="x",
        params={"foo": 1, "bar": 99},
    )
    assert a != c
