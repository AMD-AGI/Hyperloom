# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P0-1 protocol-layer tests (intent envelope, MessageBus, ResourceLockManager, TaskRegistry, CursorStore)."""

from __future__ import annotations

import time

import pytest

from inference_optimizer.orchestrator.cursor_store import CursorStore
from inference_optimizer.protocol.intent import (
    EMIT_INTENT_TOOL_SCHEMA,
    IntentType,
    IntentValidationError,
    validate_envelope,
)
from inference_optimizer.orchestrator.message_bus import (
    TOPIC_ALLOWLIST,
    Message,
    MessageBus,
)
from inference_optimizer.orchestrator.resource_lock import (
    KNOWN_LANES,
    LANE_CONFLICTS,
    LaneBusy,
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.task_registry import (
    IllegalTransition,
    TASK_STATES,
    TERMINAL_STATES,
    TaskRegistry,
)
from inference_optimizer.storage import SqliteConnection


@pytest.fixture
def db(tmp_path) -> SqliteConnection:
    sc = SqliteConnection(tmp_path / "p0_1.db")
    yield sc
    sc.close()


# intent_parser
def test_intent_type_v06_includes_review_verdict():
    assert IntentType.REVIEW_VERDICT.value == "review_verdict"


def test_intent_type_no_objection_or_vote():
    """v0.6: parliament removed entirely (DESIGN ADR-38)."""
    values = {t.value for t in IntentType}
    assert "objection" not in values
    assert "vote" not in values


def test_validate_envelope_minimal_ok():
    envelope = {
        "intents": [
            {"intent_type": "send_message", "payload": {"topic": "observation"}}
        ]
    }
    intents = validate_envelope(envelope)
    assert len(intents) == 1
    assert intents[0].type == IntentType.SEND_MESSAGE


def test_validate_envelope_review_verdict_payload():
    envelope = {
        "intents": [{
            "intent_type": "review_verdict",
            "payload": {
                "target_proposal_msg_id": "abc123",
                "verdict": "approve",
                "reasoning": "matches KB entry kb-7",
            },
        }]
    }
    intents = validate_envelope(envelope)
    assert intents[0].type == IntentType.REVIEW_VERDICT
    assert intents[0].payload["verdict"] == "approve"


def test_validate_envelope_review_verdict_missing_required_rejected():
    envelope = {"intents": [{"intent_type": "review_verdict", "payload": {"verdict": "approve"}}]}
    with pytest.raises(IntentValidationError, match="target_proposal_msg_id"):
        validate_envelope(envelope)


def test_validate_envelope_unknown_intent_rejected():
    envelope = {"intents": [{"intent_type": "objection", "payload": {}}]}
    with pytest.raises(IntentValidationError, match="not in allowed set"):
        validate_envelope(envelope)


def test_validate_envelope_propose_action_requires_predicted_gain():
    envelope = {"intents": [{"intent_type": "propose_action", "payload": {"action_name": "baseline"}}]}
    with pytest.raises(IntentValidationError, match="predicted_gain_pct"):
        validate_envelope(envelope)


def test_emit_intent_tool_schema_is_complete():
    assert EMIT_INTENT_TOOL_SCHEMA["name"] == "emit_intent"
    enum_values = set(EMIT_INTENT_TOOL_SCHEMA["input_schema"]["properties"]["intent_type"]["enum"])
    assert "review_verdict" in enum_values
    assert "objection" not in enum_values


# message_bus
def test_topic_allowlist_v06_changes():
    """v0.6: review_verdict added; objection/vote/parliament removed."""
    assert "review_verdict" in TOPIC_ALLOWLIST
    assert "advice" in TOPIC_ALLOWLIST
    assert "objection" not in TOPIC_ALLOWLIST
    assert "vote" not in TOPIC_ALLOWLIST
    assert "parliament_open" not in TOPIC_ALLOWLIST


@pytest.mark.asyncio
async def test_message_bus_append_assigns_seq(db):
    bus = MessageBus(db)
    msg = Message.new("Orchestration", "Kernel", "request",
                      {"target_agent": "kernel", "kind": "trace_analyze"})
    seq = await bus.append_and_seq(msg)
    assert seq >= 1
    assert msg.seq == seq


@pytest.mark.asyncio
async def test_message_bus_unknown_topic_rejected(db):
    bus = MessageBus(db)
    msg = Message.new("Orchestration", "Kernel", "not_a_real_topic", {})
    with pytest.raises(ValueError, match="unknown topic"):
        await bus.append_and_seq(msg)


@pytest.mark.asyncio
async def test_message_bus_priority_out_of_range(db):
    bus = MessageBus(db)
    msg = Message.new("Orchestration", "Kernel", "heartbeat", {}, priority=99)
    with pytest.raises(ValueError, match="priority must be 0..3"):
        await bus.append_and_seq(msg)


@pytest.mark.asyncio
async def test_message_bus_tail_filters_by_to_agent_with_broadcast(db):
    bus = MessageBus(db)
    await bus.append_and_seq(Message.new("Critic", "Orchestration", "review_verdict",
                                          {"target_proposal_msg_id": "p1", "verdict": "approve"}))
    await bus.append_and_seq(Message.new("Robustness", "*", "alert",
                                          {"severity": "high", "summary": "stall"}))
    await bus.append_and_seq(Message.new("Critic", "Kernel", "review_verdict",
                                          {"target_proposal_msg_id": "p2", "verdict": "reject"}))
    msgs = await bus.tail(to_agent="Orchestration")
    topics = {(m.from_agent, m.to_agent) for m in msgs}
    assert ("Critic", "Orchestration") in topics
    assert ("Robustness", "*") in topics
    assert ("Critic", "Kernel") not in topics


@pytest.mark.asyncio
async def test_message_bus_replay_for_returns_ascending(db):
    bus = MessageBus(db)
    seqs = []
    for topic in ("heartbeat", "observation", "decision"):
        s = await bus.append_and_seq(Message.new("Orchestration", "Kernel", topic, {}))
        seqs.append(s)
    msgs = await bus.replay_for("Kernel", after_seq=0)
    assert [m.seq for m in msgs] == sorted(seqs)


# resource_lock
def test_known_lanes_v08_includes_research_lane():
    """v0.8 M5 (KB_design §3.7) adds ``research_lane``; the four v0.6 serving lanes are unchanged."""
    assert set(KNOWN_LANES) == {
        "server_lifecycle",
        "workspace_mutation",
        "benchmark_lane",
        "profile_lane",
        "research_lane",
    }


def test_research_lane_has_no_conflicts():
    """Inv-7.2: research_lane is conflict-free vs. the serving lanes."""
    assert LANE_CONFLICTS["research_lane"] == frozenset()
    for lane, conflicts in LANE_CONFLICTS.items():
        assert "research_lane" not in conflicts, (
            f"lane={lane!r} unexpectedly lists research_lane as a "
            f"conflict (Inv-7.2 says research_lane is conflict-free)"
        )


def test_lane_conflicts_symmetric_for_bench_profile_server():
    assert "profile_lane" in LANE_CONFLICTS["benchmark_lane"]
    assert "benchmark_lane" in LANE_CONFLICTS["profile_lane"]
    assert "server_lifecycle" in LANE_CONFLICTS["benchmark_lane"]


@pytest.mark.asyncio
async def test_resource_lock_acquire_release(db):
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    lease = await locks.acquire_many(
        ["workspace_mutation"],
        holder_id="h1", task_id="t1", action="patch_applier", ttl_sec=60,
    )
    assert lease.lanes == ("workspace_mutation",)
    assert "workspace_mutation" in await locks.active_lanes()
    n = await locks.release(lease)
    assert n == 1
    assert await locks.active_lanes() == []


@pytest.mark.asyncio
async def test_resource_lock_lane_busy_rolls_back(db):
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    a = await locks.acquire_many(
        ["benchmark_lane"], holder_id="ha", task_id="ta", action="bench_runner", ttl_sec=60,
    )
    with pytest.raises(LaneBusy) as exc:
        await locks.acquire_many(
            ["benchmark_lane"], holder_id="hb", task_id="tb", action="bench_runner", ttl_sec=60,
        )
    assert "benchmark_lane" in exc.value.busy_lanes
    # Original lease should still be intact (atomic rollback)
    assert "benchmark_lane" in await locks.active_lanes()
    await locks.release(a)


@pytest.mark.asyncio
async def test_resource_lock_cross_lane_conflict_blocks_profile_during_bench(db):
    """benchmark_lane held → profile_lane acquire must fail (DESIGN §3.5.3)."""
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    bench = await locks.acquire_many(
        ["benchmark_lane"], holder_id="hb", task_id="tb", action="bench_runner", ttl_sec=60,
    )
    with pytest.raises(LaneBusy):
        await locks.acquire_many(
            ["profile_lane"], holder_id="hp", task_id="tp", action="profile_runner", ttl_sec=60,
        )
    await locks.release(bench)


@pytest.mark.asyncio
async def test_resource_lock_unknown_lane_rejected(db):
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    with pytest.raises(ValueError, match="unknown lane"):
        await locks.acquire_many(
            ["bogus_lane"], holder_id="h", task_id="t", action="x", ttl_sec=60,
        )


@pytest.mark.asyncio
async def test_resource_lock_heartbeat_and_stale_release(db):
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    lease = await locks.acquire_many(
        ["profile_lane"], holder_id="h1", task_id="t1", action="profile_runner", ttl_sec=60,
    )
    await locks.heartbeat(lease, ttl_sec=120)
    # Wrong holder release must not delete the row
    fake = type(lease)(
        holder_id="other", task_id="t1", action="profile_runner",
        lanes=lease.lanes, acquired_at=lease.acquired_at, expires_at=lease.expires_at,
    )
    n = await locks.release(fake)
    assert n == 0
    assert "profile_lane" in await locks.active_lanes()
    n = await locks.release(lease)
    assert n == len(lease.lanes)


@pytest.mark.asyncio
async def test_resource_lock_reap_expired_emits_event(db):
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    lease = await locks.acquire_many(
        ["workspace_mutation"], holder_id="h1", task_id="t1", action="patch_applier",
        ttl_sec=0,  # expires immediately
    )
    time.sleep(0.01)
    reaped = await locks.reap_expired()
    assert any(r["lane"] == "workspace_mutation" for r in reaped)
    bus = MessageBus(db)
    expired_events = await bus.tail(topic="lease_expired")
    assert len(expired_events) >= 1


# task_registry
def test_task_states_and_terminals():
    assert "queued" in TASK_STATES
    assert "succeeded" in TERMINAL_STATES
    assert "running" not in TERMINAL_STATES


@pytest.mark.asyncio
async def test_task_registry_create_idempotent(db):
    tr = TaskRegistry(db)
    t1 = await tr.create(kind="bench_runner", params={"x": 1}, idempotency_key="k1")
    t2 = await tr.create(kind="bench_runner", params={"x": 1}, idempotency_key="k1")
    assert t1.task_id == t2.task_id


@pytest.mark.asyncio
async def test_task_registry_legal_transitions(db):
    tr = TaskRegistry(db)
    t = await tr.create(kind="bench_runner", params={}, idempotency_key="kT1")
    t2 = await tr.transition(t.task_id, "running")
    assert t2.state == "running"
    assert t2.attempts == 1
    t3 = await tr.transition(t.task_id, "succeeded", evidence={"tput": 1840})
    assert t3.state == "succeeded"


@pytest.mark.asyncio
async def test_task_registry_illegal_transition_rejected(db):
    tr = TaskRegistry(db)
    t = await tr.create(kind="bench_runner", params={}, idempotency_key="kT2")
    with pytest.raises(IllegalTransition):
        await tr.transition(t.task_id, "succeeded")  # queued -> succeeded not allowed


@pytest.mark.asyncio
async def test_task_registry_cancel_family_bulk(db):
    tr = TaskRegistry(db)
    a = await tr.create(kind="deep_kernel_analysis", params={}, idempotency_key="ka")
    b = await tr.create(kind="deep_kernel_analysis", params={}, idempotency_key="kb")
    c = await tr.create(kind="bench_runner", params={}, idempotency_key="kc")
    cancelled = await tr.cancel_family(["deep_kernel_analysis"])
    assert set(cancelled) == {a.task_id, b.task_id}
    bench_after = await tr.get(c.task_id)
    assert bench_after.state == "queued"  # untouched


# cursor_store
@pytest.mark.asyncio
async def test_cursor_store_empty_default(db):
    cs = CursorStore(db)
    state = await cs.load("Orchestration")
    assert state.last_processed_seq == 0
    assert state.last_processed_msg_id == ""


@pytest.mark.asyncio
async def test_cursor_store_advance_forward(db):
    cs = CursorStore(db)
    s = await cs.advance("Critic", seq=5, msg_id="m5")
    assert s.last_processed_seq == 5
    s = await cs.advance("Critic", seq=10, msg_id="m10")
    assert s.last_processed_seq == 10


@pytest.mark.asyncio
async def test_cursor_store_never_moves_backwards(db):
    cs = CursorStore(db)
    await cs.advance("Critic", seq=10, msg_id="m10")
    s = await cs.advance("Critic", seq=3, msg_id="m3")
    assert s.last_processed_seq == 10
    assert await cs.is_already_processed("Critic", 3)
    assert not await cs.is_already_processed("Critic", 11)
