"""Tests for Conductor._handle_intent — v0.4 MVP intent branches.

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
    kill_task        -> tasks.transition(cancelled) + bus(kill)
                        + findings/kills.jsonl  (triage only)

OBJECTION / VOTE branches were removed in v0.4 (parliament gone).
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
    """Boot conductor with one scripted intent for the given agent.

    v0.4 — triage is always-on in quick mode but emits its own heartbeats
    that would drown the test's target event in a 1000-event tail. We
    pin ``triage_tick_s`` very high (effectively disabling triage during
    the sub-2s test window) so the executor's intent stays observable.
    """
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
        triage_tick_s=99999.0,   # disable triage flood in unit test
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
async def test_kill_task_writes_kill_event_and_finding(session_dir):
    """v0.4 — triage emits kill_task, conductor cancels the task and mirrors
    the kill onto bus(topic="kill") + findings/kills.jsonl.
    """
    import json
    from inference_optimizer.orchestrator.task_registry import TaskRegistry

    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    # Pre-seed a queued task so kill has something to cancel.
    tasks = TaskRegistry(db)
    seeded_task = await tasks.create(
        kind="delegate",
        params={"action_name": "baseline", "requested_by": "executor"},
        idempotency_key="seed-baseline-key",
        requires_lanes=["server"],
        lease_ttl_sec=60,
    )

    intent = Intent(
        type=IntentType.KILL_TASK,
        payload={
            "task_id": seeded_task.task_id,
            "reason": "stuck in queued >2x lease_ttl in test fixture",
            "scope": "task",
        },
    )
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="triage")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "0.0005"},  # quick
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
        triage_tick_s=0.1,   # speed triage up for the test
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
    events = await bus.tail(n=200, topic="kill")
    kills = _topic_payloads(events, "kill")
    assert any(
        p.get("task_id") == seeded_task.task_id
        and p.get("status") == "ok"
        for p in kills
    )
    # Task should now be cancelled.
    cancelled = await TaskRegistry(db).get(seeded_task.task_id)
    assert cancelled.state == "cancelled"
    # findings/kills.jsonl mirror.
    kills_file = session_dir / "findings" / "kills.jsonl"
    assert kills_file.is_file()
    rows = [
        json.loads(line) for line in kills_file.read_text().splitlines() if line.strip()
    ]
    assert any(r.get("task_id") == seeded_task.task_id for r in rows)
    db.close()


@pytest.mark.asyncio
async def test_kill_task_from_executor_is_policy_denied(session_dir):
    """v0.4 — only triage may kill_task; executor's attempt is rejected."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    intent = Intent(
        type=IntentType.KILL_TASK,
        payload={"task_id": "fake", "reason": "x", "scope": "task"},
    )
    backend = MockBackend(
        script=[ScriptStep(intents=[intent], only_if_agent="executor")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "0.0005"},
        db=db,
        reactor_tick_s=0.1,
        clock_tick_s=0.2,
    )
    await asyncio.wait_for(conductor.run(), timeout=10.0)

    # No kill events on the bus.
    bus = MessageBus(db)
    events = await bus.tail(n=200, topic="kill")
    assert events == []
    # PolicyGate should have written a policy_denied observation.
    obs = await bus.tail(n=1000, topic="observation")
    denied_kinds = [
        e.payload.get("kind") for e in obs if isinstance(e.payload, dict)
    ]
    assert "policy_denied" in denied_kinds
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


# ---------------------------------------------------------------------------
# Plan A — REQUEST / RESPONSE routing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_request_writes_request_event_addressed_to_target(session_dir):
    """REQUEST intent → ``topic="request"`` event with to_agent=<target>."""
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel",
            "kind": "select_kernels",
            "params": {"trace_path": "/tmp/x.json.gz"},
            "reason": "smoke",
        },
    )
    db = await _run_with_intent(session_dir, intent)
    bus = MessageBus(db)
    events = await bus.tail(n=200, topic="request")
    assert events, "no request events written"
    first = events[0]
    assert first.from_agent == "executor"
    assert first.to_agent == "kernel"
    assert first.payload["kind"] == "select_kernels"
    assert first.payload["target_agent"] == "kernel"
    assert first.payload["params"]["trace_path"] == "/tmp/x.json.gz"
    db.close()


@pytest.mark.asyncio
async def test_response_reverse_routes_to_request_sender(session_dir):
    """RESPONSE looks up the original request and addresses to_agent
    back to the request's from_agent (here: executor)."""
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    # Pre-seed: directly insert a request envelope so we have a known msg_id
    # to reply to. SqliteConnection ctor already ran ensure_schema.
    bus = MessageBus(db)
    from inference_optimizer.orchestrator.message_bus import Message
    request_msg = Message.new(
        from_agent="executor",
        to_agent="kernel",
        topic="request",
        payload={"kind": "select_kernels", "target_agent": "kernel"},
    )
    await bus.append_and_seq(request_msg)

    # Now script kernel to reply.
    response_intent = Intent(
        type=IntentType.RESPONSE,
        payload={
            "in_reply_to": request_msg.msg_id,
            "kind": "select_kernels_done",
            "status": "succeeded",
            "result": {"candidates": ["a.py", "b.py"]},
        },
    )
    backend = MockBackend(
        script=[ScriptStep(intents=[response_intent], only_if_agent="kernel")]
    )
    conductor = Conductor(
        session_dir,
        backend=backend,
        env={"MODEL_PATH": "fake/model", "MAX_HOURS": "3"},  # guided -> +kernel
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

    events = await bus.tail(n=200, topic="response")
    assert events, "no response events written"
    matching = [e for e in events if e.in_reply_to == request_msg.msg_id]
    assert matching, f"no response with in_reply_to={request_msg.msg_id!r}"
    resp = matching[0]
    # Reverse-routed back to the request's original sender (executor).
    assert resp.to_agent == "executor"
    assert resp.from_agent == "kernel"
    assert resp.payload["kind"] == "select_kernels_done"
    assert resp.payload["status"] == "succeeded"
    db.close()


@pytest.mark.asyncio
async def test_response_unknown_in_reply_to_falls_back_to_broadcast(session_dir):
    """If the original request can't be found, response is broadcast."""
    response_intent = Intent(
        type=IntentType.RESPONSE,
        payload={
            "in_reply_to": "nonexistent-msg-id-xxxxx",
            "kind": "ack",
            "status": "succeeded",
        },
    )
    db = SqliteConnection(session_dir / "storage" / "conductor.db")
    backend = MockBackend(
        script=[ScriptStep(intents=[response_intent], only_if_agent="kernel")]
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
    events = await bus.tail(n=200, topic="response")
    assert events
    resp = events[0]
    assert resp.to_agent == "*"  # broadcast fallback
    db.close()
