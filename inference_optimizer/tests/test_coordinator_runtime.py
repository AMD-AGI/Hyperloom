# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P0-3 Coordinator + MockBackend + SubAgentRunner tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator import coordinator
from inference_optimizer.orchestrator.action_executors import (
    _multi_node_server_lifecycle,
    report_executor,
)
from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import (
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    Coordinator,
)
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import (
    SubAgentRunner,
)
from inference_optimizer.orchestrator.task_registry import Task, TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import target_baseline_json
from inference_optimizer.storage import SqliteConnection


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


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


# MockBackend
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


# SubAgentRunner (standalone)
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
    assert "benchmark_lane" not in await locks.active_lanes()
    db.close()


# Coordinator — bounded ticks
@pytest.mark.asyncio
async def test_coordinator_starts_with_silent_backends(session_dir):
    backends = _build_backends({})
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        # 4 agents × 2 ticks × 1 heartbeat = 8 send_message events
        msgs = await c.bus.tail(n=20, topic="heartbeat")
        assert len(msgs) == 8
    finally:
        await c.stop()


# Backend-error streak (K4 — robustness/critic subprocess health)
class _AlwaysFailingBackend(Backend):
    """Backend that always raises BackendError."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> "BackendTurnResult":  # noqa: F821 — protocol return type, raises before returning
        from inference_optimizer.orchestrator.backends.base import BackendError
        self.calls += 1
        raise BackendError(f"simulated {self.name} subprocess crash #{self.calls}")


class _AlwaysCrashingBackend(Backend):
    """Backend that raises an unexpected exception from ``run``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> "BackendTurnResult":  # noqa: F821 — protocol return type, raises before returning
        self.calls += 1
        raise RuntimeError(f"simulated {self.name} unexpected crash #{self.calls}")


@pytest.mark.asyncio
async def test_backend_error_streak_fires_backend_unhealthy_once_at_threshold(
    session_dir, monkeypatch,
):
    """A consecutive BackendError streak promotes per-call ``backend_error``
    events into a single ``backend_unhealthy`` observation, fired once."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD", "3",
    )
    backends = _build_backends({})
    backends["robustness"] = _AlwaysFailingBackend("robustness")
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(4)
        observations = await c.bus.tail(n=50, topic="observation")
        backend_errors = [
            o for o in observations
            if (o.payload or {}).get("kind") == "backend_error"
            and (o.payload or {}).get("agent") == "robustness"
        ]
        backend_unhealthy = [
            o for o in observations
            if (o.payload or {}).get("kind") == "backend_unhealthy"
            and (o.payload or {}).get("agent") == "robustness"
        ]
        assert len(backend_errors) == 4
        assert len(backend_unhealthy) == 1
        promoted = backend_unhealthy[0].payload
        assert promoted["consecutive_errors"] == 3
        assert promoted["threshold"] == 3
        assert promoted["severity"] == "high"
        assert promoted["agent"] == "robustness"
        assert "subprocess backend has failed" in promoted["hint"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_unexpected_backend_exception_records_last_tick_exception(session_dir):
    backends = _build_backends({})
    backends["orchestration"] = _AlwaysCrashingBackend("orchestration")
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(1)
        assert c.shared_state.crash_count == 1
        assert c.shared_state.last_tick_exception["stage"] == "reactor_pass"
        assert c.shared_state.last_tick_exception["agent"] == "orchestration"
        assert c.shared_state.last_tick_exception["type"] == "RuntimeError"
        assert (
            "simulated orchestration unexpected crash"
            in c.shared_state.last_tick_exception["message"]
        )

        persisted = SharedState.load_or_init(session_dir)
        assert persisted.last_tick_exception == c.shared_state.last_tick_exception
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_backend_error_streak_resets_after_successful_turn(
    session_dir, monkeypatch,
):
    """A successful turn resets the streak counter and re-arms the alarm."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD", "2",
    )
    backends = _build_backends({})
    failing = _AlwaysFailingBackend("robustness")
    backends["robustness"] = failing
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        assert c._backend_error_streak["robustness"] == 2
        assert c._backend_error_alarm_armed["robustness"] is False

        # Healthy backend → reset.
        backends_silent = _build_backends({})
        c.backends["robustness"] = backends_silent["robustness"]
        await c.tick(1)
        assert c._backend_error_streak["robustness"] == 0
        assert c._backend_error_alarm_armed["robustness"] is True

        # Re-arm: failing backend back in → alarm fires again.
        c.backends["robustness"] = failing
        await c.tick(2)
        observations = await c.bus.tail(n=50, topic="observation")
        backend_unhealthy = [
            o for o in observations
            if (o.payload or {}).get("kind") == "backend_unhealthy"
        ]
        assert len(backend_unhealthy) == 2
        assert backend_unhealthy[-1].payload["consecutive_errors"] == 2
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
    }
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))

        verdict = Intent(type=IntentType.REVIEW_VERDICT, payload={
            "target_proposal_msg_id": proposal_id,
            "verdict": "approve",
            "reasoning": "matches kb-1",
        })
        await c._handle_intent("critic", verdict)

        approved = c.state.pending_proposals[proposal_id]
        assert approved.decided and approved.verdict == "approve"
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
    c.sub.register_executor("baseline", lambda ctx: _async_return({"tput": 1840}))
    try:
        await c.tick(1)
        dones = await c.bus.tail(topic="delegated_result")
        assert any(m.payload.get("state") == "succeeded" for m in dones)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_accepts_nested_params_idempotency_key(session_dir):
    """LLM sometimes puts idempotency_key under params; Coordinator must
    treat it as the delegate key and remove it from executor params."""
    delegate = Intent(type=IntentType.DELEGATE, payload={
        "action_name": "explore",
        "params": {
            "idempotency_key": "explore-round-2",
        },
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[delegate])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    captured: dict[str, object] = {}

    async def _runner(ctx):
        captured["params"] = dict(ctx.task.params)
        captured["idempotency_key"] = ctx.task.idempotency_key
        return {"status": "succeeded", "tput": 1.0}

    c.sub.register_executor("explore", _runner)
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.baseline_config_path = "/tmp/baseline.yaml"
        c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
        # Roofline-v2 N3: params delegate requires fresh roofline snapshot.
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/trace.json.gz",
            "analysis_md_text": "FAKE_REPORT",
        }
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert captured["idempotency_key"] == "explore-round-2"
        assert "idempotency_key" not in captured["params"]
        denied = await c.bus.tail(topic="observation")
        assert not any(
            m.payload.get("rule") == "duplicate_idempotency_key"
            for m in denied
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_routes_to_kernel(session_dir):
    req = Intent(type=IntentType.REQUEST, payload={
        "target_agent": "kernel", "kind": "trace_analyze",
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
        assert any(m.payload.get("kind") == "trace_analyze" for m in kernel_inbox)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_response_routes_back_to_requester(session_dir):
    req = Intent(type=IntentType.REQUEST, payload={
        "target_agent": "kernel", "kind": "trace_analyze",
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[req])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
        c.shared_state.save(session_dir)
        await c.tick(1)
        kernel_inbox = await c.bus.tail(to_agent="kernel", topic="request")
        assert kernel_inbox, "no request mirrored to kernel"
        request_msg_id = kernel_inbox[0].msg_id

        await c._handle_intent("kernel", Intent(
            type=IntentType.RESPONSE, payload={
                "in_reply_to": request_msg_id,
                "kind": "trace_analyze_done",
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
async def test_explore_not_denied_before_profile(session_dir):
    """P2_10: after baseline, ``explore`` is no longer blocked on empty ``last_profile_trace``."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "explore", "predicted_gain_pct": 5.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = ""
        c.shared_state.save(session_dir)

        await c.tick(1)

        obs = await c.bus.tail(to_agent="orchestration", topic="observation")
        for m in obs:
            if m.payload.get("kind") != "policy_denied":
                continue
            assert "profile must run" not in str(m.payload.get("hint") or ""), (
                f"profile-required gate fired despite removal: {m.payload!r}"
            )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_execution_order_does_not_deny_backends_when_trace_analyze_stale(
    session_dir,
):
    """Reverse regression: the action-layer ``trace_analyze`` hard-gate was
    removed; actions must NOT be denied when ``last_trace_analyze`` is stale."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "params", "predicted_gain_pct": 3.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace-a.json.gz"
        c.shared_state.last_trace_analyze = {}
        c.shared_state.save(session_dir)

        await c.tick(1)

        obs = await c.bus.tail(to_agent="orchestration", topic="observation")
        for m in obs:
            if m.payload.get("kind") != "policy_denied":
                continue
            assert "trace_analyze must run first" not in str(
                m.payload.get("hint") or m.payload.get("reason") or ""
            ), (
                "trace_analyze action-layer gate fired for params despite "
                f"removal: {m.payload!r}"
            )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_orchestration_prompt_has_no_execution_checklist(session_dir):
    """The Coordinator no longer injects an enforced next-step checklist."""
    c = Coordinator(session_dir, backends=_build_backends({}))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = ""
        c.shared_state.save(session_dir)

        prompt = await c._compose_prompt("orchestration")

        assert "Execution checklist" not in prompt
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
async def test_coordinator_prune_branch_cancels_family_and_records_advisory(session_dir):
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
        # Active queue still gets cancelled — the prune kills work in flight.
        assert a_after.state == "cancelled"
        assert b_after.state == "cancelled"
        assert "deep_kernel_analysis" in c.shared_state.pruned_families

        # Future propose_action carries an advisory observation but is not dropped.
        await c._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "deep_kernel_analysis", "predicted_gain_pct": 5.0},
        ))
        assert c.state.pending_proposals
        obs = await c.bus.tail(topic="observation")
        assert any(
            m.payload.get("kind") == "proposal_pruned_advisory" for m in obs
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_policy_denied_surfaces_as_observation(session_dir):
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


# (formerly test_coordinator_audit_wiring.py) — Coordinator audit-trail wiring.
def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


def _mute_action_scoring(coordinator: Coordinator) -> None:
    """v0.8 §3.9 Inv-9.1 — scoreboard retired; no-op kept for back-compat."""
    return None


def _mk_task(kind: str, task_id: str = "t-aud-1") -> Task:
    return Task(
        task_id=task_id,
        kind=kind,
        state="queued",
        params={},
        idempotency_key=f"idem-{task_id}",
    )


@pytest.mark.asyncio
async def test_promote_baseline_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("baseline", "t-base-1")
        result = {
            "output_throughput": 1500.0,
            "accuracy": 0.81,
            "materialized_config": "/tmp/baseline.with_envs.yaml",
            "workspace": "/runs/baseline/t-base-1",
        }
        await c._promote_to_shared_state("baseline", result, task=task)
        last = c.shared_state.last_baseline
        assert last
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["key_metric"] == pytest.approx(1500.0)
        assert last["key_metric_kind"] == "output_throughput"
        assert last["extras"]["accuracy"] == 0.81
        assert len(c.shared_state.baseline_attempts) == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_profile_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("profile", "t-prof-1")
        result = {
            "main_trace_path": "/tmp/trace.json",
            "output_throughput": 1234.5,
            "workspace": "/runs/profile/t-prof-1",
        }
        await c._promote_to_shared_state("profile", result, task=task)
        last = c.shared_state.last_profile
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["extras"]["trace_path"] == "/tmp/trace.json"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_explore_records_success_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        task = _mk_task("explore", "t-ex-1")
        result = {
            "status": "succeeded",
            "winners": [{
                "name": "v1",
                "extra_server_args": "--foo",
                "extra_envs": {"K": "1"},
            }],
            "best_variant": {
                "name": "v1",
                "extra_server_args": "--foo",
                "extra_envs": {"K": "1"},
            },
            "output_throughput": 900.0,
            "best_gain_pct": 12.5,
            "base_tput": 800.0,
            "round_id": "round-1",
        }
        await c._promote_to_shared_state("explore", result, task=task)
        last = c.shared_state.last_explore
        assert last["status"] == "succeeded"
        assert last["decision"] == "promoted"
        assert last["extras"]["best_variant_name"] == "v1"
        assert last["extras"]["winners_count"] == 1
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_sweep_records_discarded_attempt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("sweep", "t-sw-1")
        result = {
            "grid_size": 4,
            "best_overall": {"name": "c8_isl1k_osl1k", "tput": 1200.0},
            "pareto_front": [{"name": "a"}, {"name": "b"}],
        }
        await c._promote_to_shared_state("sweep", result, task=task)
        last = c.shared_state.last_sweep_attempt = c.shared_state.last_sweep
        assert c.shared_state.sweep_attempts[-1]["status"] == "succeeded"
        assert c.shared_state.sweep_attempts[-1]["decision"] == "discarded"
        assert c.shared_state.sweep_attempts[-1]["extras"]["grid_size"] == 4
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_explore_updates_validated_gain(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.current_best = {"action": "baseline", "tput": 1000.0}
        task = _mk_task("explore", "t-ex-rebench")
        result = {
            "status": "succeeded",
            "winners": [{
                "name": "kv_fp8",
                "extra_server_args": "--kv-cache-fp8",
                "extra_envs": {},
            }],
            "best_variant": {
                "name": "kv_fp8",
                "extra_server_args": "--kv-cache-fp8",
                "extra_envs": {},
            },
            "output_throughput": 1100.0,
            "best_gain_pct": 10.0,
            "round_id": "round-rebench",
        }
        await c._promote_to_shared_state("explore", result, task=task)
        assert c.shared_state.cumulative_gain_validated == pytest.approx(10.0)
        assert c.shared_state.cumulative_gain_validated_stack_len == \
            len(c.shared_state.optimization_stack)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_records_failure(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_task("baseline", "t-fail-1")
        result = {
            "status": "failed",
            "error_class": "no_report",
            "error": "benchmark_report.json missing under runs/...",
            "workspace": "/runs/baseline/t-fail-1/benchmark_sglang_xyz",
            "reported_success": False,
        }
        await c._handle_unpromotable_result(task, result)
        assert len(c.shared_state.baseline_attempts) == 1
        attempt = c.shared_state.baseline_attempts[-1]
        assert attempt["status"] == "failed"
        assert attempt["decision"] == "no_promote"
        assert attempt["error_class"] == "no_report"
        assert len(c.shared_state.last_action_failures) == 1
        fail = c.shared_state.last_action_failures[-1]
        assert fail["action"] == "baseline"
        assert fail["error_class"] == "no_report"
        assert c.shared_state.baseline_failure_streak == 1
        assert c.shared_state.stop_reason in ("", None)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_third_failure_sets_stop_reason(
    session_dir,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        for i in range(3):
            await c._handle_unpromotable_result(
                _mk_task("baseline", f"t-{i}"),
                {"status": "failed", "error_class": "no_report",
                 "error": "missing"},
            )
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason == "baseline_failed"
        assert len(c.shared_state.last_action_failures) == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_records_for_non_baseline_kinds(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._handle_unpromotable_result(
            _mk_task("explore", "t-ex-fail"),
            {"status": "failed", "error_class": "subprocess_nonzero",
             "error": "rc=1\nstderr blob"},
        )
        assert c.shared_state.baseline_failure_streak == 0
        assert c.shared_state.stop_reason in ("", None)
        assert len(c.shared_state.explore_attempts) == 1
        assert c.shared_state.explore_attempts[-1]["status"] == "failed"
        assert len(c.shared_state.last_action_failures) == 1
        fail = c.shared_state.last_action_failures[-1]
        assert fail["action"] == "explore"
        assert fail["stderr_tail"] is not None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_kernel_action_records_global_only(
    session_dir,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._handle_unpromotable_result(
            _mk_task("kernel_opt", "t-ko-fail"),
            {"status": "failed", "error_class": "timeout",
             "error": "wall-clock exceeded"},
        )
        assert not hasattr(c.shared_state, "kernel_opt_attempts_audit")
        assert len(c.shared_state.last_action_failures) == 1
        entry = c.shared_state.last_action_failures[-1]
        assert entry["action"] == "kernel_opt"
        assert entry["error_class"] == "timeout"
        assert entry["stderr_tail"] is not None
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_capture_failure_arms_eager_fallback(
    session_dir,
):
    """cuda_graph_capture_failed (no baseline yet) must arm the one-shot
    eager fallback flag through the real coordinator failure handler."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        assert c.shared_state.baseline_eager_fallback is False
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-cg-1"),
            {"status": "failed", "error_class": "cuda_graph_capture_failed",
             "error": "operation not permitted when stream is capturing"},
        )
        assert c.shared_state.baseline_eager_fallback is True
        # One-shot: a second capture failure must not re-arm (already set).
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-cg-2"),
            {"status": "failed", "error_class": "cuda_graph_capture_failed",
             "error": "operation not permitted when stream is capturing"},
        )
        assert c.shared_state.baseline_eager_fallback is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_capture_failure_no_arm_when_baseline_promoted(
    session_dir,
):
    """Resume case: with an existing baseline (tput > 0) the coordinator must
    NOT arm the eager fallback on a later cuda-graph capture failure."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1234.0
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-cg-resume"),
            {"status": "failed", "error_class": "cuda_graph_capture_failed",
             "error": "operation not permitted when stream is capturing"},
        )
        assert c.shared_state.baseline_eager_fallback is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_roofline_increments_failure_streak(
    session_dir, caplog,
):
    """Repro (session 20260529T104050Z task 42922ce4): watermark-roofline
    failure must increment roofline_failure_streak, eagerly clear
    auto_roofline_pending_task_id, and emit an Auto-roofline warning."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task_id = "t-roofline-fail-42922ce4"
        c.shared_state.auto_roofline_pending_task_id = task_id
        streak_before = c.shared_state.roofline_failure_streak
        result = {
            "status": "failed",
            "error_class": "profile_failed",
            "error": "profile sub-step failed",
            "phase": "profile",
            "sub_result": {"status": "failed", "error_class": "no_trace_files"},
        }
        import logging
        with caplog.at_level(logging.WARNING,
                             logger="inference_optimizer.orchestrator.coordinator"):
            await c._handle_unpromotable_result(
                _mk_task("roofline", task_id), result,
            )
        # (a) audit entry exists.
        assert len(c.shared_state.roofline_attempts) == 1
        attempt = c.shared_state.roofline_attempts[-1]
        assert attempt["status"] == "failed"
        assert attempt["decision"] == "no_promote"
        # (b) failure streak should be +1.
        assert c.shared_state.roofline_failure_streak == streak_before + 1, (
            "roofline_failure_streak silently stays at 0; LLM + operators have "
            "no way to know the watermark-driven analysis refresh failed."
        )
        # (c) pending gate should be cleared eagerly.
        assert c.shared_state.auto_roofline_pending_task_id == "", (
            "auto_roofline_pending_task_id still points at the failed task; "
            "subsequent dispatches stay blocked until denial-time lazy clear."
        )
        # (d) operator-visible warning must be logged.
        assert any(
            "Auto-roofline" in r.message and "failed" in r.message
            for r in caplog.records
        ), "no 'Auto-roofline ... failed' WARNING was logged"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_failed_initial_roofline_rearms_watermark_from_baseline(
    session_dir,
):
    """A failed initial roofline must not suppress later refresh attempts."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.cumulative_gain_validated = 25.0
        c.shared_state.last_roofline_tput = 0.0
        c.shared_state.roofline_failure_streak = 1
        c.shared_state.auto_roofline_pending_task_id = ""

        assert c._needs_roofline_for_watermark() is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_unattempted_initial_roofline_does_not_watermark_rearm(
    session_dir,
):
    """Before any failed/successful roofline, PRELUDE remains the only entry."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.cumulative_gain_validated = 25.0
        c.shared_state.last_roofline_tput = 0.0
        c.shared_state.roofline_failure_streak = 0
        c.shared_state.auto_roofline_pending_task_id = ""

        assert c._needs_roofline_for_watermark() is False
    finally:
        await c.stop()


# (formerly test_coordinator_baseline_fingerprint.py) — baseline-params fingerprint capture.
def _mk_baseline_task(params: dict, *, task_id: str = "t-fp-1") -> Task:
    return Task(
        task_id=task_id,
        kind="baseline",
        state="queued",
        params=params,
        idempotency_key=f"idem-{task_id}",
    )


def test_fingerprint_keys_covers_recovery_surface():
    expected = {
        "benchmark_script", "result_dir", "extra_server_args",
        "extra_envs", "model_path", "gpu_type", "config_path",
        "disable_run_eval",
    }
    assert set(_BASELINE_FINGERPRINT_KEYS) == expected


def test_fingerprint_normalizes_extra_envs_order():
    fp1 = _baseline_params_fingerprint({"extra_envs": {"A": "1", "B": "2"}})
    fp2 = _baseline_params_fingerprint({"extra_envs": {"B": "2", "A": "1"}})
    assert fp1 == fp2
    assert fp1["extra_envs"] == [["A", "1"], ["B", "2"]]


def test_fingerprint_missing_keys_become_none_or_empty():
    fp = _baseline_params_fingerprint({"benchmark_script": "sglang_mi300x.sh"})
    assert fp["benchmark_script"] == "sglang_mi300x.sh"
    assert fp["result_dir"] is None
    assert fp["extra_server_args"] is None
    assert fp["extra_envs"] == []
    assert fp["model_path"] is None
    fp_with_empty = _baseline_params_fingerprint({
        "benchmark_script": "sglang_mi300x.sh",
        "extra_envs": {},
    })
    assert fp == fp_with_empty


def test_fingerprint_stringifies_scalar_values():
    fp = _baseline_params_fingerprint({
        "benchmark_script": "sglang_mi300x.sh",
        "model_path": "/wekafs/models/DeepSeek-R1",
        "gpu_type": "mi300x",
    })
    assert all(isinstance(v, str) for k, v in fp.items() if v is not None and k != "extra_envs")


def test_fingerprint_different_overrides_produce_different_fingerprints():
    a = _baseline_params_fingerprint({"benchmark_script": "sglang_mi300x.sh"})
    b = _baseline_params_fingerprint({"benchmark_script": "dsr1_fp8_mi300x.sh"})
    c = _baseline_params_fingerprint({"result_dir": "/workspace"})
    d = _baseline_params_fingerprint({"extra_server_args": "--mem-fraction-static 0.9"})
    encoded = {json.dumps(x, sort_keys=True) for x in (a, b, c, d)}
    assert len(encoded) == 4


@pytest.mark.asyncio
async def test_promote_baseline_records_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_baseline_task({
            "benchmark_script": "sglang_mi300x.sh",
            "model_path": "/wekafs/models/DeepSeek-R1",
            "gpu_type": "mi300x",
        })
        result = {
            "output_throughput": 1500.0,
            "materialized_config": "/tmp/baseline.with_envs.yaml",
            "workspace": "/runs/baseline/t-fp-1",
        }
        await c._promote_to_shared_state("baseline", result, task=task)
        last = c.shared_state.last_baseline
        assert last["status"] == "succeeded"
        fp = last["extras"]["fingerprint"]
        assert fp["benchmark_script"] == "sglang_mi300x.sh"
        assert fp["model_path"] == "/wekafs/models/DeepSeek-R1"
        assert fp["gpu_type"] == "mi300x"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_records_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_baseline_task({
            "benchmark_script": "dsr1_fp8_mi300x.sh",
        })
        result = {
            "status": "failed",
            "error_class": "no_report",
            "error": "benchmark_report.json missing",
        }
        await c._handle_unpromotable_result(task, result)
        attempt = c.shared_state.baseline_attempts[-1]
        assert attempt["status"] == "failed"
        fp = attempt["extras"]["fingerprint"]
        assert fp["benchmark_script"] == "dsr1_fp8_mi300x.sh"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_non_baseline_omits_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = Task(
            task_id="t-ex-fail",
            kind="explore",
            state="queued",
            params={"benchmark_script": "sglang_mi300x.sh"},
            idempotency_key="idem-ex",
        )
        await c._handle_unpromotable_result(task, {"status": "failed"})
        attempt = c.shared_state.explore_attempts[-1]
        assert attempt["status"] == "failed"
        assert "fingerprint" not in attempt["extras"]
    finally:
        await c.stop()


# (formerly test_coordinator_failed_variants_audit.py) — ``_summarize_failed_variants`` + PR-3 timeout.
def test_summarize_failed_variants_returns_empty_when_input_not_list():
    assert coordinator._summarize_failed_variants(None) == []
    assert coordinator._summarize_failed_variants("not a list") == []
    assert coordinator._summarize_failed_variants(42) == []
    assert coordinator._summarize_failed_variants({}) == []


def test_summarize_failed_variants_returns_empty_when_no_failures():
    rows = [
        {"name": "v1", "status": "succeeded", "output_throughput": 640.0},
        {"name": "v2", "status": "succeeded", "output_throughput": 643.9},
    ]
    assert coordinator._summarize_failed_variants(rows) == []


def test_summarize_failed_variants_projects_expected_keys():
    rows = [
        {
            "name": "max_num_seqs_128",
            "status": "failed",
            "error_class": "mn_server_restart_failed",
            "error": (
                "server /health did not return 200 within 1800s "
                "(url=http://10.245.131.67:8888/health, "
                "last_err=ConnectError: All connection attempts failed)"
            ),
            "extra_server_args": "--max-num-seqs 128",
        },
        {
            "name": "max_num_seqs_512",
            "status": "succeeded",
            "output_throughput": 510.0,
        },
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 1
    assert out[0] == {
        "name": "max_num_seqs_128",
        "error_class": "mn_server_restart_failed",
        "error_excerpt": (
            "server /health did not return 200 within 1800s "
            "(url=http://10.245.131.67:8888/health, "
            "last_err=ConnectError: All connection attempts failed)"
        ),
        "extra_server_args": "--max-num-seqs 128",
    }


def test_summarize_failed_variants_truncates_error_excerpt_at_400_chars():
    huge_err = "x" * 5000
    rows = [
        {
            "name": "v",
            "status": "failed",
            "error_class": "ec",
            "error": huge_err,
            "extra_server_args": "",
        },
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert out[0]["error_excerpt"] is not None
    assert len(out[0]["error_excerpt"]) == 400


def test_summarize_failed_variants_caps_max_entries():
    rows = [
        {
            "name": f"v{i}",
            "status": "failed",
            "error_class": "ec",
            "error": "boom",
            "extra_server_args": f"--arg {i}",
        }
        for i in range(50)
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 10
    assert [e["name"] for e in out] == [f"v{i}" for i in range(10)]


def test_summarize_failed_variants_skips_non_dict_rows():
    rows = [
        None,
        "garbage",
        {"name": "real", "status": "failed", "error_class": "ec", "error": "msg"},
        42,
    ]
    out = coordinator._summarize_failed_variants(rows)
    assert len(out) == 1
    assert out[0]["name"] == "real"


def test_summarize_failed_variants_handles_missing_optional_fields():
    rows = [{"name": "v", "status": "failed"}]
    out = coordinator._summarize_failed_variants(rows)
    assert out == [{
        "name": "v",
        "error_class": None,
        "error_excerpt": None,
        "extra_server_args": "",
    }]


def test_default_health_timeout_is_900s_not_1800s():
    assert _multi_node_server_lifecycle.DEFAULT_HEALTH_TIMEOUT_S == 900


# (formerly test_n33_idle_closing_and_critic_archival.py) — N33: critic auto-approve
# archival actions + Coordinator silent-tick early-closing.
def _write_marker_target_baseline(session_dir: Path) -> None:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "no_target",
            "reason": "no_target_gpu_configured",
            "row_count": 0,
        }),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_idle_run_reaches_max_ticks_without_closing(session_dir):
    """An idle run keeps ticking until max_ticks rather than self-closing on silence."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        reason = await c.run(max_ticks=5, tick_interval_sec=0.0)
        assert reason == "max_ticks"
        assert c.shared_state.closing_phase is False
    finally:
        await c.stop()


def test_critic_md_carves_out_archival_actions():
    path = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "system_prompts" / "critic.md"
    )
    text = path.read_text(encoding="utf-8")
    assert "archival actions" in text.lower()
    assert "`report`" in text
    assert "`session_breakdown`" in text
    assert "`target_analysis`" in text
    assert "Always `approve` archival actions" in text


# (formerly test_n34_dispatcher_resilience_and_report_stop.py) — N34: dispatcher
# resilience to disappearing ``tasks`` rows + report-task run-loop exit.
@pytest.mark.asyncio
async def test_sub_agent_runner_swallows_tasknotfound_on_final_transition(
    tmp_path,
):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    payload = {
        "tput": 4632.8,
        "params_search_update": {"tested": {"fp1": {"name": "v1"}}},
    }

    async def long_runner(ctx):
        db.raw.execute("DELETE FROM tasks WHERE task_id=?", (ctx.task.task_id,))
        db.raw.commit()
        return payload

    sub.register_executor("params", long_runner)
    task = await tr.create(kind="params", params={}, idempotency_key="k-params-1")
    res = await sub.run_task(task)

    assert res.state == "succeeded"
    assert res.result == payload, (
        "executor result must survive the TaskNotFound -- the dispatcher "
        "uses it to update params_search ledger; losing it stalls N19c"
    )
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_swallows_tasknotfound_on_initial_transition(
    tmp_path, caplog,
):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    ran = {"called": False}

    async def runner(ctx):
        ran["called"] = True
        return {"tput": 1.0}

    sub.register_executor("baseline", runner)
    task = await tr.create(
        kind="baseline", params={}, idempotency_key="k-baseline-1",
    )
    db.raw.execute("DELETE FROM tasks WHERE task_id=?", (task.task_id,))
    db.raw.commit()

    with caplog.at_level("WARNING"):
        res = await sub.run_task(task)

    assert ran["called"] is True
    assert res.state == "succeeded"
    assert res.result == {"tput": 1.0}
    assert any(
        "vanished" in rec.message.lower()
        and "_transition_resilient" in rec.message
        for rec in caplog.records
    ), "expected the disappearing-row warning to fire"
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_normal_path_still_records_transitions(
    tmp_path,
):
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("baseline", lambda ctx: _async_return({"tput": 1.0}))
    task = await tr.create(kind="baseline", params={}, idempotency_key="k-ok")
    res = await sub.run_task(task)
    assert res.state == "succeeded"
    after = await tr.get(task.task_id)
    assert after.state == "succeeded"
    db.close()


async def _async_return(value: Any) -> Any:
    return value


@pytest.mark.asyncio
async def test_report_success_does_not_stop_run(session_dir):
    """A successful mid-run ``report`` task no longer sets ``stop_reason``; the run continues."""
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.save(session_dir)
    try:
        task = await c.tasks.create(
            kind="report",
            params={"session_dir": str(session_dir)},
            idempotency_key="k-report-1",
        )
        await c._pump_dispatcher_once()
        after = await c.tasks.get(task.task_id)
        assert after.state == "succeeded"
        assert not (c.shared_state.stop_reason or "").strip()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_report_success_does_not_overwrite_prior_stop_reason(session_dir):
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.stop_reason = "target_reached"
    c.shared_state.save(session_dir)
    try:
        task = await c.tasks.create(
            kind="report",
            params={"session_dir": str(session_dir)},
            idempotency_key="k-report-pre-set",
        )
        await c._pump_dispatcher_once()
        after = await c.tasks.get(task.task_id)
        assert after.state == "succeeded"
        assert c.shared_state.stop_reason == "target_reached"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_run_preserves_prior_stop_reason_when_loop_exits_without_new_reason(
    session_dir,
):
    """After a tick raises, the stop-condition path keeps the persisted terminal reason."""
    c = Coordinator(session_dir, backends=_silent_backends())
    c.shared_state.set_stop_reason("target_reached")
    c.shared_state.save(session_dir)

    # Raise inside the tick body before any stop-condition check.
    async def _boom():
        raise RuntimeError("tick exploded mid-run")

    c._advance_phase_if_needed = _boom  # type: ignore[assignment]

    try:
        reason = await c.run(max_ticks=5)
        assert reason == "target_reached"
        assert c.shared_state.stop_reason == "target_reached"
        assert c.shared_state.crash_count == 1
        assert c.shared_state.last_tick_exception["stage"] == "advance_phase"
        assert c.shared_state.last_tick_exception["type"] == "RuntimeError"
        persisted = SharedState.load_or_init(session_dir)
        assert persisted.stop_reason == "target_reached"
        assert persisted.last_tick_exception["stage"] == "advance_phase"
    finally:
        await c.stop()
