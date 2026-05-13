"""P0-3 Coordinator + MockBackend + SubAgentRunner tests.

Covers:

* MockBackend playback: scripted turns + exhaustion fallback heartbeat
* SubAgentRunner: lease acquired → task run → released; missing runner
  fails the task; exception in runner transitions to failed
* Coordinator.tick() exercises 4-agent reactor + dispatcher in single process
* PROPOSE_ACTION → bus 'proposal' + pending_proposals tracked
* REVIEW_VERDICT(approve) → task materialized; (reject) → no task
* DELEGATE → task queued + dispatcher runs registered runner
* REQUEST(orchestration→kernel) → bus 'request' to_agent=kernel
* RESPONSE → routed back to original requester
* KILL_TASK by Robustness cancels queued task
* PRUNE_BRANCH cancels queued family + future proposals soft-rejected
* PolicyDenied surfaces as 'observation' with rule= populated
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.sub_agent_runner import (
    RunnerContext,
    SubAgentResult,
    SubAgentRunner,
)
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.paths import db_path_for, make_session_dir
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    """Backend that always emits heartbeat — used for agents we don't care about."""
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends(scripts: dict[str, ScriptedPlan]) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for name in ("orchestration", "kernel", "critic", "robustness"):
        backends[name] = MockBackend(scripts.get(name, _silent_plan()), name=name)
    return backends


# ===========================================================================
# MockBackend
# ===========================================================================
@pytest.mark.asyncio
async def test_mock_backend_plays_scripted_turns():
    plan = ScriptedPlan(turns=[
        MockTurn(intents=[_heartbeat()], raw_text="t1"),
        MockTurn(intents=[Intent(IntentType.ALERT, payload={"severity": "low", "summary": "x"})], raw_text="t2"),
    ])
    backend = MockBackend(plan)
    r1 = await backend.run("p")
    r2 = await backend.run("p")
    assert r1.raw_text == "t1"
    assert r2.intents[0].type == IntentType.ALERT


@pytest.mark.asyncio
async def test_mock_backend_default_when_exhausted():
    plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backend = MockBackend(plan)
    r = await backend.run("p")
    assert r.intents[0].type == IntentType.SEND_MESSAGE
    assert r.intents[0].payload["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_mock_backend_records_calls():
    backend = MockBackend(_silent_plan())
    await backend.run("hello", system_prompt="sys", tools=["emit_intent"])
    assert backend.calls[0]["prompt"] == "hello"
    assert backend.calls[0]["tools"] == ["emit_intent"]


# ===========================================================================
# SubAgentRunner (standalone)
# ===========================================================================
@pytest.mark.asyncio
async def test_sub_agent_runner_succeeds(tmp_path):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("baseline", lambda ctx: _async_return({"tput": 1840}))
    task = await tr.create(kind="baseline", params={}, idempotency_key="k-baseline-1")
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result == {"tput": 1840}
    after = await tr.get(task.task_id)
    assert after.state == "succeeded"
    db.close()


async def _async_return(value):
    return value


@pytest.mark.asyncio
async def test_sub_agent_runner_no_executor_fails(tmp_path):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    task = await tr.create(kind="never_registered", params={}, idempotency_key="k-x")
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert "no runner" in res.error
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_acquires_lane(tmp_path):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    seen_lease = {}

    async def exe(ctx):
        seen_lease["lanes"] = ctx.lease.lanes if ctx.lease else None
        return {}

    sub.register_executor("bench_runner", exe)
    task = await tr.create(
        kind="bench_runner", params={}, idempotency_key="k-bench-1",
        requires_lanes=["benchmark_lane"], lease_ttl_sec=30,
    )
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert "benchmark_lane" in seen_lease["lanes"]
    # Lease released after run
    assert "benchmark_lane" not in await locks.active_lanes()
    db.close()


# ===========================================================================
# Coordinator — bounded ticks
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_starts_with_silent_backends(session_dir):
    backends = _build_backends({})
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        # 4 agents × 2 ticks × 1 heartbeat each = 8 send_message events
        msgs = await c.bus.tail(n=20, topic="heartbeat")
        assert len(msgs) == 8
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_stops_when_no_more_leverage(session_dir):
    backends = _build_backends({})
    c = Coordinator(session_dir, backends=backends)
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.current_best = {"action": "backends", "tput": 101.0}
        c.shared_state.params_no_promote_streak = 5
        c.shared_state.params_search = {
            "cursor": 29,
            "tested": {f"v{i}": {} for i in range(29)},
            "accepted": [],
            "rejected": [],
        }
        c.shared_state.last_select_kernels = {
            "reusable_native_kernel_ids": ["k003", "k006", "k007"],
        }
        c.shared_state.rejected_kernel_ids = ["k003"]
        c.shared_state.rejected_kernel_patches = [
            {"kernel_id": "k006", "reason": "max_e2e_attempts_3_without_keep"},
            {"kernel_id": "k007", "reason": "revert_decision"},
        ]
        c.shared_state.save(session_dir)

        stop_reason = await c.run(max_ticks=5, tick_interval_sec=0.0)

        assert stop_reason == "no_more_leverage"
        assert c.shared_state.stop_reason == "no_more_leverage"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_propose_action_creates_pending(session_dir):
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        assert len(c.state.pending_proposals) == 1
        prop = next(iter(c.state.pending_proposals.values()))
        assert prop.action_name == "baseline"
        assert prop.decided is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_review_verdict_approve_creates_task(session_dir):
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    plans = {
        "orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])]),
        # Critic will see proposal next tick, then approve
    }
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))

        # Inject Critic approval directly via _handle_intent
        verdict = Intent(type=IntentType.REVIEW_VERDICT, payload={
            "target_proposal_msg_id": proposal_id,
            "verdict": "approve",
            "reasoning": "matches kb-1",
        })
        await c._handle_intent("critic", verdict)

        approved = c.state.pending_proposals[proposal_id]
        assert approved.decided and approved.verdict == "approve"
        # task materialized
        decisions = await c.bus.tail(topic="decision")
        assert any(m.payload.get("kind") == "approved_proposal" for m in decisions)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_review_verdict_reject_no_task(session_dir):
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))
        await c._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT, payload={
                "target_proposal_msg_id": proposal_id,
                "verdict": "reject", "reasoning": "kb-2 says no",
                "kb_evidence": "kb-2",
            },
        ))
        decisions = await c.bus.tail(topic="decision")
        assert not any(m.payload.get("kind") == "approved_proposal" for m in decisions)
        # Verdict mirror message exists
        verdicts = await c.bus.tail(topic="review_verdict")
        assert any(m.payload.get("verdict") == "reject" for m in verdicts)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_delegate_task_run_via_dispatcher(session_dir):
    delegate = Intent(type=IntentType.DELEGATE, payload={
        "action_name": "baseline", "params": {"runs": 1},
        "idempotency_key": "k-deleg-1",
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[delegate])])}

    c = Coordinator(session_dir, backends=_build_backends(plans))
    # Register on Coordinator's built-in SubAgentRunner — sharing its db handle.
    c.sub.register_executor("baseline", lambda ctx: _async_return({"tput": 1840}))
    try:
        await c.tick(1)
        # The dispatcher inside tick() should have run the queued task
        dones = await c.bus.tail(topic="delegated_result")
        assert any(m.payload.get("state") == "succeeded" for m in dones)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_routes_to_kernel(session_dir):
    req = Intent(type=IntentType.REQUEST, payload={
        "target_agent": "kernel", "kind": "select_kernels",
        "params": {"top_k": 5},
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[req])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
        c.shared_state.save(session_dir)
        await c.tick(1)
        kernel_inbox = await c.bus.tail(to_agent="kernel", topic="request")
        assert any(m.payload.get("kind") == "select_kernels" for m in kernel_inbox)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_response_routes_back_to_requester(session_dir):
    req = Intent(type=IntentType.REQUEST, payload={
        "target_agent": "kernel", "kind": "select_kernels",
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[req])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
        c.shared_state.save(session_dir)
        await c.tick(1)
        # Find the request msg_id Coordinator inserted
        kernel_inbox = await c.bus.tail(to_agent="kernel", topic="request")
        assert kernel_inbox, "no request mirrored to kernel"
        request_msg_id = kernel_inbox[0].msg_id

        # Kernel emits a response in-reply-to that id
        await c._handle_intent("kernel", Intent(
            type=IntentType.RESPONSE, payload={
                "in_reply_to": request_msg_id,
                "kind": "select_kernels_done",
                "status": "ok",
                "result": {"chosen": ["k1", "k2"]},
            },
        ))
        responses = await c.bus.tail(topic="response", to_agent="orchestration")
        assert responses
        assert responses[0].payload["status"] == "ok"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_execution_order_denies_backends_before_profile(session_dir):
    """After baseline, profile is mandatory before backends/params/sweep."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "backends", "predicted_gain_pct": 5.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = ""
        c.shared_state.save(session_dir)

        await c.tick(1)

        assert not c.state.pending_proposals
        obs = await c.bus.tail(to_agent="orchestration", topic="observation")
        assert any(
            m.payload.get("kind") == "policy_denied"
            and m.payload.get("rule") == "execution_order"
            and "profile" in str(m.payload.get("hint"))
            for m in obs
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_execution_order_denies_backends_before_select_kernels(session_dir):
    """After profile, select_kernels is mandatory before backends/params/sweep."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "params", "predicted_gain_pct": 3.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace-a.json.gz"
        c.shared_state.last_select_kernels = {}
        c.shared_state.save(session_dir)

        await c.tick(1)

        assert not c.state.pending_proposals
        obs = await c.bus.tail(to_agent="orchestration", topic="observation")
        assert any(
            m.payload.get("kind") == "policy_denied"
            and m.payload.get("rule") == "execution_order"
            and "select_kernels" in str(m.payload.get("hint"))
            for m in obs
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_execution_checklist_is_in_orchestration_prompt(session_dir):
    c = Coordinator(session_dir, backends=_build_backends({}))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = ""
        c.shared_state.save(session_dir)

        prompt = await c._compose_prompt("orchestration")

        assert "Execution checklist" in prompt
        assert "profile is required now" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_kill_task_by_robustness(session_dir):
    delegate = Intent(type=IntentType.DELEGATE, payload={
        "action_name": "long_running", "params": {},
        "idempotency_key": "k-long-1",
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[delegate])])}

    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)  # delegate enqueued; dispatcher fails (no runner)
        all_tasks = await c.tasks.by_state("failed")
        assert all_tasks  # no runner → failed

        # Re-create a queued one for kill test (use a different idem key)
        new_task = await c.tasks.create(
            kind="long_running", params={}, idempotency_key="k-long-2",
        )
        await c._handle_intent("robustness", Intent(
            type=IntentType.KILL_TASK,
            payload={"task_id": new_task.task_id, "reason": "stalled", "scope": "task"},
        ))
        after = await c.tasks.get(new_task.task_id)
        assert after.state == "cancelled"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_prune_branch_cancels_family_and_future_proposals(session_dir):
    c = Coordinator(session_dir, backends=_build_backends({}))
    try:
        a = await c.tasks.create(kind="deep_kernel_analysis", params={}, idempotency_key="ka")
        b = await c.tasks.create(kind="deep_kernel_analysis", params={}, idempotency_key="kb")

        await c._handle_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "deep_kernel_analysis", "reason": "3 fails"},
        ))
        a_after = await c.tasks.get(a.task_id)
        b_after = await c.tasks.get(b.task_id)
        assert a_after.state == "cancelled"
        assert b_after.state == "cancelled"
        # P1-3: pruned_families now lives in persistent SharedState
        assert "deep_kernel_analysis" in c.shared_state.pruned_families

        # Future proposal of same family is soft-rejected
        await c._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "deep_kernel_analysis", "predicted_gain_pct": 5.0},
        ))
        assert not c.state.pending_proposals
        obs = await c.bus.tail(topic="observation")
        assert any(m.payload.get("kind") == "proposal_pruned" for m in obs)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_policy_denied_surfaces_as_observation(session_dir):
    # Critic tries to delegate (forbidden by role).
    bad = Intent(type=IntentType.DELEGATE, payload={"action_name": "baseline"})
    plans = {"critic": ScriptedPlan(turns=[MockTurn(intents=[bad])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        denied = await c.bus.tail(topic="observation")
        hits = [m for m in denied if m.payload.get("kind") == "policy_denied"]
        assert hits, "expected a policy_denied observation"
        assert hits[0].payload["rule"] == "role"
    finally:
        await c.stop()
