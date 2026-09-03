# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator + MockBackend + SubAgentRunner tests."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from hyperloom.orchestrator.actions.executors import (
    _multi_node_server_lifecycle,
    report_executor,
)
from hyperloom.orchestrator.roles import (
    Backend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
)
from hyperloom.orchestrator.loop.proposals import ProposalsCollaborator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.objective import TargetGainObjective
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.sub_agent_runner import (
    SubAgentRunner,
)
from hyperloom.orchestrator.state.task_registry import (
    IllegalTransition,
    Task,
    TaskRegistry,
)
from hyperloom.orchestrator.bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.dispatcher import DispatcherCollaborator
from hyperloom.inference_optimizer.session.session_paths import target_baseline_json
from hyperloom.orchestrator.bus.storage import SqliteConnection


async def _immediately(payload: dict) -> dict:
    """Executor stub that returns without yielding."""
    return payload


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    """Backend that always emits heartbeat — used for agents we don't care about."""
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends(scripts: dict[str, ScriptedPlan]) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for name in ("orchestration", "critic", "robustness"):
        backends[name] = MockBackend(scripts.get(name, _silent_plan()), name=name)
    return backends


# MockBackend
@pytest.mark.asyncio
async def test_mock_backend_plays_scripted_turns():
    plan = ScriptedPlan(
        turns=[
            MockTurn(intents=[_heartbeat()], raw_text="t1"),
            MockTurn(intents=[Intent(IntentType.ALERT, payload={"severity": "low", "summary": "x"})], raw_text="t2"),
        ]
    )
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
    assert res.error_class == "no_executor"
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_executor_exception_sets_error_class(tmp_path):
    """A raised executor exception must not collapse into the generic
    unknown_error gap bucket -- error_class carries the exception's own class name.
    """
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    async def exe(ctx):
        raise TimeoutError("benchmark server never came up")

    sub.register_executor("bench_runner", exe)
    task = await tr.create(kind="bench_runner", params={}, idempotency_key="k-y")
    res = await sub.run_task(task)
    assert res.state == "failed"
    assert res.error_class == "TimeoutError"
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_hands_its_lease_to_the_executor_and_gives_it_back(tmp_path):
    """Lanes are won by the caller and handed over; the runner returns them."""
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
        kind="bench_runner",
        params={},
        idempotency_key="k-bench-1",
        requires_lanes=["benchmark_lane"],
        lease_ttl_sec=30,
    )
    lease = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id=task.task_id,
        task_id=task.task_id,
        action="bench_runner",
        ttl_sec=30,
    )

    res = await sub.run_task(task, prebound_lease=lease)

    assert res.state == "succeeded"
    assert "benchmark_lane" in seen_lease["lanes"]
    assert "benchmark_lane" not in await locks.lane_holders()
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_refuses_a_lane_holding_task_with_no_lease(tmp_path):
    """Running it anyway would run it unserialised, which is the bug the lanes exist to prevent."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    ran = {"called": False}

    async def exe(_ctx):
        ran["called"] = True
        return {}

    sub.register_executor("bench_runner", exe)
    task = await tr.create(
        kind="bench_runner",
        params={},
        idempotency_key="k-bench-nolease",
        requires_lanes=["benchmark_lane"],
        lease_ttl_sec=30,
    )

    with pytest.raises(ValueError, match="without a lease"):
        await sub.run_task(task)

    assert ran["called"] is False
    assert (await tr.get(task.task_id)).state == "queued"
    db.close()


# Coordinator — bounded ticks
@pytest.mark.asyncio
async def test_coordinator_starts_with_silent_backends(session_dir):
    backends = _build_backends({})
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        # 3 agents × 2 ticks × 1 heartbeat = 6 send_message events
        msgs = await c.bus.tail(n=20, topic="heartbeat")
        assert len(msgs) == 6
    finally:
        await c.stop()


# Backend-error streak (robustness/critic subprocess health)
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
        from hyperloom.orchestrator.roles.base import BackendError

        self.calls += 1
        raise BackendError(f"simulated {self.name} subprocess crash #{self.calls}")


class _LLMFailingBackend(Backend):
    """Backend whose provider call fails (raises the ``LLMCallFailed`` marker)."""

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
        from hyperloom.orchestrator.roles.base import LLMCallFailed

        self.calls += 1
        raise LLMCallFailed(f"simulated {self.name} gateway 400 #{self.calls}")


def _llm_error_rows(session_dir: Path) -> list[dict]:
    from hyperloom.inference_optimizer.session.session_paths import llm_calls_path

    path = llm_calls_path(session_dir)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("status") == "error"]


@pytest.mark.asyncio
async def test_plain_backend_error_records_no_llm_error_row(session_dir):
    """A deterministic local fault must not be counted as a provider failure.

    ``BackendError`` covers unreadable ``emit.json``, a missing ``--review``
    path, an absent SDK — none of which touched the model. Recording those
    would make the Langfuse LLM error rate meaningless.
    """
    backends = _build_backends({})
    backends["robustness"] = _AlwaysFailingBackend("robustness")
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        assert _llm_error_rows(session_dir) == []
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_llm_call_failed_records_one_error_row_per_turn(session_dir):
    backends = _build_backends({})
    backends["robustness"] = _LLMFailingBackend("robustness")
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        rows = _llm_error_rows(session_dir)
        assert len(rows) == 2
        row = rows[0]
        assert row["component"] == "robustness"
        assert row["error_type"] == "LLMCallFailed"
        assert "gateway 400" in row["error_message"]
        assert row["input_tokens"] is None and row["output_tokens"] is None
    finally:
        await c.stop()


class _SelfTracingLLMFailingBackend(_LLMFailingBackend):
    """A self-tracing backend (critic-shaped): writes its own row, then raises.

    Having ``set_trace_context`` is the contract that marks a backend as owning
    its trace rows, so the Coordinator must not add one of its own.
    """

    def __init__(self, name: str, session_dir: Path) -> None:
        super().__init__(name)
        self._session_dir = session_dir
        self.trace_ctx_calls = 0
        self.trace_contexts: list[dict[str, object]] = []

    def set_trace_context(
        self,
        *,
        tick: int | None = None,
        phase: str | None = None,
        macro_cycle: int | None = None,
    ) -> None:
        self.trace_ctx_calls += 1
        self.trace_contexts.append({"tick": tick, "phase": phase, "macro_cycle": macro_cycle})

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> "BackendTurnResult":  # noqa: F821 — protocol return type, raises before returning
        from hyperloom.orchestrator.roles.base import LLMCallFailed
        from hyperloom.orchestrator.trace.llm_trace import LLMCallRecord, append_llm_call

        self.calls += 1
        error = LLMCallFailed(f"simulated {self.name} gateway 400 #{self.calls}")
        append_llm_call(
            session_dir=self._session_dir,
            record=LLMCallRecord.for_failure(
                session_id=self._session_dir.name,
                component="critic",
                role="critic",
                error=error,
                model="claude-opus-4-7",
            ),
        )
        raise error


@pytest.mark.asyncio
async def test_self_tracing_backend_failure_is_recorded_exactly_once(session_dir):
    """One provider failure must produce one row, not one per writer.

    The critic writes its own error row (carrying the review model) and then
    raises; if the Coordinator also wrote one, Langfuse would count a single
    critic failure twice, with disagreeing model/latency on the two rows.
    """
    backends = _build_backends({})
    backends["critic"] = _SelfTracingLLMFailingBackend("critic", session_dir)
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(2)
        rows = _llm_error_rows(session_dir)
        assert len(rows) == backends["critic"].calls
        # The surviving row is the backend's richer one (real review model).
        assert {r["model"] for r in rows} == {"claude-opus-4-7"}
        assert {r["component"] for r in rows} == {"critic"}
        assert backends["critic"].trace_contexts
        assert all("macro_cycle" in context for context in backends["critic"].trace_contexts)
    finally:
        await c.stop()


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
    session_dir,
    monkeypatch,
):
    """A consecutive BackendError streak promotes per-call ``backend_error``
    events into a single ``backend_unhealthy`` observation, fired once."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD",
        "3",
    )
    backends = _build_backends({})
    backends["robustness"] = _AlwaysFailingBackend("robustness")
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(4)
        observations = await c.bus.tail(n=50, topic="observation")
        backend_errors = [
            o
            for o in observations
            if (o.payload or {}).get("kind") == "backend_error" and (o.payload or {}).get("agent") == "robustness"
        ]
        backend_unhealthy = [
            o
            for o in observations
            if (o.payload or {}).get("kind") == "backend_unhealthy" and (o.payload or {}).get("agent") == "robustness"
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
        assert "simulated orchestration unexpected crash" in c.shared_state.last_tick_exception["message"]

        persisted = SharedState.load_or_init(session_dir)
        assert persisted.last_tick_exception == c.shared_state.last_tick_exception
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_backend_error_streak_resets_after_successful_turn(
    session_dir,
    monkeypatch,
):
    """A successful turn resets the streak counter and re-arms the alarm."""
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD",
        "2",
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
        backend_unhealthy = [o for o in observations if (o.payload or {}).get("kind") == "backend_unhealthy"]
        assert len(backend_unhealthy) == 2
        assert backend_unhealthy[-1].payload["consecutive_errors"] == 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_propose_action_creates_pending(session_dir):
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
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
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    plans = {
        "orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])]),
    }
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))

        verdict = Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={
                "target_proposal_msg_id": proposal_id,
                "verdict": "approve",
                "reasoning": "matches kb-1",
            },
        )
        await c._handle_intent("critic", verdict)

        approved = c.state.pending_proposals[proposal_id]
        assert approved.decided and approved.verdict == "approve"
        decisions = await c.bus.tail(topic="decision")
        assert any(m.payload.get("kind") == "approved_proposal" for m in decisions)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_review_verdict_reject_no_task(session_dir):
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))
        await c._handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={
                    "target_proposal_msg_id": proposal_id,
                    "verdict": "reject",
                    "reasoning": "kb-2 says no",
                    "kb_evidence": "kb-2",
                },
            ),
        )
        decisions = await c.bus.tail(topic="decision")
        assert not any(m.payload.get("kind") == "approved_proposal" for m in decisions)
        verdicts = await c.bus.tail(topic="review_verdict")
        assert any(m.payload.get("verdict") == "reject" for m in verdicts)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_delegate_task_run_via_dispatcher(session_dir):
    delegate = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "baseline",
            "params": {"runs": 1},
            "idempotency_key": "k-deleg-1",
        },
    )
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
    """When idempotency_key is under params, Coordinator treats it as the
    delegate key and removes it from executor params."""
    delegate = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "explore",
            "params": {
                "idempotency_key": "explore-round-2",
            },
        },
    )
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
        # params delegate requires a fresh roofline snapshot.
        c.shared_state.last_trace_analyze = {
            "trace_input": "/tmp/trace.json.gz",
            "analysis_md_text": "FAKE_REPORT",
        }
        c.shared_state.save(session_dir)
        await c.tick(1)
        assert captured["idempotency_key"] == "explore-round-2"
        assert "idempotency_key" not in captured["params"]
        denied = await c.bus.tail(topic="observation")
        assert not any(m.payload.get("rule") == "duplicate_idempotency_key" for m in denied)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_request_routes_to_kernel(session_dir):
    req = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "trace_analyze",
            "params": {"top_k": 5},
        },
    )
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[req])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    try:
        c.shared_state.baseline_tput = 100.0
        c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
        c.shared_state.save(session_dir)
        await c.tick(1)
        kernel_inbox = await c.bus.tail(to_agent="kernel_agent", topic="request")
        assert any(m.payload.get("kind") == "trace_analyze" for m in kernel_inbox)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_response_routes_back_to_requester(session_dir):
    """Programmatic handler emits RESPONSE to orchestration without an LLM turn."""
    from unittest.mock import patch
    from hyperloom.orchestrator.kernel import request_handlers as krh

    req = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "trace_analyze",
            "params": {"trace_input": "/tmp/trace.json.gz"},
        },
    )
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[req])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))

    async def _fake_handler(payload, *, session_dir, **_kw):
        return {"status": "ok", "hot_kernels": [], "candidates_path": ""}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS, {"trace_analyze": _fake_handler}):
        try:
            c.shared_state.baseline_tput = 100.0
            c.shared_state.last_profile_trace = "/tmp/trace.json.gz"
            c.shared_state.kernel_enabled = True
            c.shared_state.save(session_dir)
            await c.tick(1)
            kernel_inbox = await c.bus.tail(to_agent="kernel_agent", topic="request")
            assert kernel_inbox, "no request mirrored to kernel"
            responses = await c.bus.tail(topic="response", to_agent="orchestration")
            assert responses
            assert responses[0].payload["status"] == "ok"
            assert responses[0].payload["source"] == "programmatic_handler"
        finally:
            await c.stop()


@pytest.mark.asyncio
async def test_explore_not_denied_before_profile(session_dir):
    """After baseline, ``explore`` is no longer blocked on empty ``last_profile_trace``."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "explore",
            "predicted_gain_pct": 5.0,
        },
    )
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
    """Actions must NOT be denied when ``last_trace_analyze`` is stale (the
    action-layer ``trace_analyze`` hard-gate was removed)."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "params",
            "predicted_gain_pct": 3.0,
        },
    )
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
            assert "trace_analyze must run first" not in str(m.payload.get("hint") or m.payload.get("reason") or ""), (
                f"trace_analyze action-layer gate fired for params despite removal: {m.payload!r}"
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
async def test_coordinator_prune_branch_cancels_family_and_records_advisory(session_dir):
    c = Coordinator(session_dir, backends=_build_backends({}))
    try:
        # ``baseline`` flows through the normal Critic/pending-proposal path; the
        # prune-advisory mechanism under test is family-agnostic.
        a = await c.tasks.create(kind="baseline", params={}, idempotency_key="ka")
        b = await c.tasks.create(kind="baseline", params={}, idempotency_key="kb")

        await c._handle_intent(
            "robustness",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={"family": "baseline", "reason": "3 fails"},
            ),
        )
        a_after = await c.tasks.get(a.task_id)
        b_after = await c.tasks.get(b.task_id)
        # Active queue gets cancelled — the prune kills work in flight.
        assert a_after.state == "cancelled"
        assert b_after.state == "cancelled"
        assert "baseline" in c.shared_state.pruned_families

        # Future propose_action carries an advisory but is not dropped.
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "baseline", "predicted_gain_pct": 5.0},
            ),
        )
        assert c.state.pending_proposals
        obs = await c.bus.tail(topic="observation")
        assert any(m.payload.get("kind") == "proposal_pruned_advisory" for m in obs)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_prune_branch_queued_scope_drains_without_retiring(session_dir):
    """Orchestration can drain a backlog and still propose the family afterwards."""
    c = Coordinator(session_dir, backends=_build_backends({}))
    try:
        a = await c.tasks.create(kind="baseline", params={}, idempotency_key="qa")
        b = await c.tasks.create(kind="baseline", params={"tag": "x"}, idempotency_key="qb")

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.PRUNE_BRANCH,
                payload={
                    "family": "baseline",
                    "reason": "anchor already established",
                    "scope": "queued",
                },
            ),
        )

        assert (await c.tasks.get(a.task_id)).state == "cancelled"
        assert (await c.tasks.get(b.task_id)).state == "cancelled"
        assert "baseline" not in c.shared_state.pruned_families
        events = await c.bus.tail(topic="event")
        assert any(m.payload.get("kind") == "prune_branch" and m.payload.get("scope") == "queued" for m in events)
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


# Coordinator audit-trail wiring.
def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }


def _mute_action_scoring(coordinator: Coordinator) -> None:
    """No-op kept for back-compat."""
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
async def test_promote_baseline_keeps_higher_anchor(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 1500.0,
                "accuracy": 0.81,
                "materialized_config": "/tmp/first.with_envs.yaml",
                "subprocess_runtime_sec": 900.0,
            },
            task=_mk_task("baseline", "t-anchor-hi"),
        )
        await c._promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 1400.0,
                "accuracy": 0.44,
                "materialized_config": "/tmp/second.with_envs.yaml",
                "subprocess_runtime_sec": 100.0,
            },
            task=_mk_task("baseline", "t-anchor-lo"),
        )
        assert c.shared_state.baseline_tput == pytest.approx(1500.0)
        assert c.shared_state.baseline_accuracy == pytest.approx(0.81)
        assert c.shared_state.baseline_config_path == "/tmp/first.with_envs.yaml"
        assert c.shared_state.baseline_runtime_sec == pytest.approx(900.0)
        assert c.shared_state.current_best["tput"] == pytest.approx(1500.0)
        last = c.shared_state.last_baseline
        assert last["status"] == "succeeded"
        assert last["decision"] == "no_promote"
        assert last["extras"]["anchor_kept_tput"] == pytest.approx(1500.0)
    finally:
        await c.stop()


def test_inject_explore_runtime_params_includes_baseline_accuracy():
    class DummyCoordinator:
        shared_state = SharedState(baseline_accuracy=0.81)

    params: dict[str, Any] = {}
    ProposalsCollaborator(DummyCoordinator())._inject_explore_runtime_params(params)
    assert params["accuracy_baseline"] == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_promote_baseline_accepts_higher_rebaseline(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 1400.0,
                "accuracy": 0.44,
                "materialized_config": "/tmp/first.with_envs.yaml",
            },
            task=_mk_task("baseline", "t-anchor-lo"),
        )
        await c._promote_to_shared_state(
            "baseline",
            {
                "output_throughput": 1500.0,
                "accuracy": 0.81,
                "materialized_config": "/tmp/second.with_envs.yaml",
            },
            task=_mk_task("baseline", "t-anchor-hi"),
        )
        assert c.shared_state.baseline_tput == pytest.approx(1500.0)
        assert c.shared_state.baseline_accuracy == pytest.approx(0.81)
        assert c.shared_state.baseline_config_path == "/tmp/second.with_envs.yaml"
        assert c.shared_state.last_baseline["decision"] == "promoted"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_baseline_leaves_current_best_when_stack_non_empty(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1400.0
        c.shared_state.optimization_stack = [{"action": "replay_warm_recipe", "tput": 7500.0}]
        c.shared_state.current_best = {"action": "warm_replay", "tput": 7500.0}
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1500.0, "materialized_config": "/tmp/re.with_envs.yaml"},
            task=_mk_task("baseline", "t-anchor-stack"),
        )
        assert c.shared_state.baseline_tput == pytest.approx(1500.0)
        assert c.shared_state.current_best["action"] == "warm_replay"
        assert c.shared_state.current_best["tput"] == pytest.approx(7500.0)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_baseline_revalidation_reanchors_below_prior(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.accuracy_floor = 0.3
        task = Task(
            task_id="t-reval-1",
            kind="baseline",
            state="queued",
            params={"reason": "enablement_eval_revalidation"},
            idempotency_key="idem-t-reval-1",
        )
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1200.0, "accuracy": 0.72},
            task=task,
        )
        assert c.shared_state.baseline_tput == pytest.approx(1200.0)
        assert c.shared_state.baseline_accuracy == pytest.approx(0.72)
        assert c.shared_state.enablement.succeeded is True
        assert c.shared_state.last_baseline["decision"] == "promoted"
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
            "winners": [
                {
                    "name": "v1",
                    "extra_server_args": "--foo",
                    "extra_envs": {"K": "1"},
                    "tput": 900.0,
                }
            ],
            "best_variant": {
                "name": "v1",
                "extra_server_args": "--foo",
                "extra_envs": {"K": "1"},
                "tput": 900.0,
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
async def test_promote_explore_updates_validated_gain(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1000.0
        c.shared_state.current_best = {"action": "baseline", "tput": 1000.0}
        task = _mk_task("explore", "t-ex-rebench")
        result = {
            "status": "succeeded",
            "winners": [
                {
                    "name": "kv_fp8",
                    "extra_server_args": "--kv-cache-fp8",
                    "extra_envs": {},
                    "tput": 1100.0,
                }
            ],
            "best_variant": {
                "name": "kv_fp8",
                "extra_server_args": "--kv-cache-fp8",
                "extra_envs": {},
                "tput": 1100.0,
            },
            "output_throughput": 1100.0,
            "best_gain_pct": 10.0,
            "round_id": "round-rebench",
        }
        await c._promote_to_shared_state("explore", result, task=task)
        assert c.shared_state.cumulative_gain_validated == pytest.approx(10.0)
        assert c.shared_state.cumulative_gain_validated_stack_len == len(c.shared_state.optimization_stack)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_explore_multi_winner_watermark_stays_in_sync(session_dir):
    """cumulative_gain_validated_stack_len must equal len(optimization_stack) after N winners."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.baseline_tput = 1000.0
        task = _mk_task("explore", "t-multi-win")
        winner_a = {
            "name": "w-a",
            "fingerprint": "fp_a",
            "tput": 1100.0,
            "extra_server_args": "--a 1",
            "extra_envs": {},
        }
        winner_b = {
            "name": "w-b",
            "fingerprint": "fp_b",
            "tput": 1210.0,
            "extra_server_args": "--a 1 --b 2",
            "extra_envs": {},
        }
        result = {
            "status": "succeeded",
            "winners": [winner_a, winner_b],
            "best_variant": winner_a,
            "output_throughput": 1210.0,
            "best_gain_pct": 10.0,
            "round_id": "round-multi",
        }
        await c._promote_to_shared_state("explore", result, task=task)
        s = c.shared_state
        assert len(s.optimization_stack) == 2
        assert s.cumulative_gain_validated_stack_len == len(s.optimization_stack)
        assert not s.optimization_stack_has_unvalidated_keeps()
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
                {"status": "failed", "error_class": "no_report", "error": "missing"},
            )
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason == "baseline_failed"
        assert len(c.shared_state.last_action_failures) == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ["session_time_exhausted", "orchestrator_cancelled"])
async def test_baseline_rounds_the_run_stopped_do_not_charge_the_failure_streak(session_dir, error_class):
    """Three rounds the run stopped are not three baselines that failed.

    The executor refuses to grade a reaped round because it would put a verdict
    on a model the round never reached; the streak has to agree, or the session
    stops as ``baseline_failed`` on the evidence of its own clock.
    """
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        for i in range(3):
            await c._handle_unpromotable_result(
                _mk_task("baseline", f"t-stopped-{error_class}-{i}"),
                {
                    "status": "failed",
                    "error_class": error_class,
                    "error": "the run stopped this round before it measured anything",
                },
            )
        assert c.shared_state.baseline_failure_streak == 0
        assert c.shared_state.baseline_total_failures == 0
        assert c.shared_state.stop_reason in ("", None)
        # The rounds are still recorded: not charging them is not hiding them.
        assert len(c.shared_state.last_action_failures) == 3
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_stopped_baseline_round_does_not_clear_a_real_failure_streak(session_dir):
    """A stop the run chose neither charges the streak nor forgives what preceded it."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-real"),
            {"status": "failed", "error_class": "no_report", "error": "missing"},
        )
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-stopped"),
            {"status": "failed", "error_class": "session_time_exhausted", "error": "reaped"},
        )
        assert c.shared_state.baseline_failure_streak == 1
        assert c.shared_state.baseline_total_failures == 1
    finally:
        await c.stop()


def _eval_failed_result() -> dict:
    return {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": "ERROR: run_eval failed with exit code 1",
        "baseline_eval_failed": True,
        "baseline_eval_failure_kind": "eval_runtime_failure",
        "baseline_eval_accuracy_floor": 0.2,
        "baseline_eval_evidence": "run_eval failed with exit code 1",
        "baseline_eval_contract_fingerprint": "abc123",
        "materialized_config": "/runs/baseline/materialized.yaml",
    }


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_eval_pending_suppresses_stop_single_node(session_dir, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement_mode = "eval"
        for i in range(3):
            await c._handle_unpromotable_result(_mk_task("baseline", f"t-ev-{i}"), _eval_failed_result())
        assert c.shared_state.baseline_failure_streak == 3
        assert c.shared_state.stop_reason in ("", None)
        assert c.shared_state.enablement.origin == "eval"
        assert c.shared_state.enablement.pending is True
        assert c.shared_state.enablement.accuracy_floor == 0.2
        assert c.shared_state.enablement.probe_config_path == "/runs/baseline/materialized.yaml"
        assert c.shared_state.enablement.eval_contract_fingerprint == "abc123"
        assert c.shared_state.enablement.baseline_eval_kind == "eval_runtime_failure"
        assert c.shared_state.enablement.launch_log
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_eval_pending_multi_node_still_stops(session_dir, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement_mode = "eval"
        for i in range(3):
            await c._handle_unpromotable_result(_mk_task("baseline", f"t-mn-{i}"), _eval_failed_result())
        assert c.shared_state.stop_reason == "baseline_failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_eval_fails_fast_without_eval_lane(session_dir, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement_mode = "off"
        for i in range(3):
            await c._handle_unpromotable_result(_mk_task("baseline", f"t-noev-{i}"), _eval_failed_result())
        assert c.shared_state.stop_reason == "baseline_failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_fails_fast_when_enablement_off(session_dir, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        # An enablement round is on record, but the lane was never admitted, so
        # it must not hold the baseline_failed budget open.
        c.shared_state.enablement_mode = "off"
        c.shared_state.enablement.attempts = 2
        c.shared_state.enablement.inflight_task_id = "spec-off"
        for i in range(3):
            await c._handle_unpromotable_result(
                _mk_task("baseline", f"t-off-{i}"),
                {"status": "failed", "error_class": "server_init_dead"},
            )
        assert c.shared_state.stop_reason == "baseline_failed"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_baseline_finalizes_eval_origin_when_accuracy_meets_floor(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement.origin = "eval"
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.accuracy_floor = 0.3
        # Set tracked task_id so the gate recognizes this as the revalidation task.
        c.shared_state.enablement.revalidation_task_id = "t-reval-ok"
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1000.0, "completed_requests": 10, "accuracy": 0.42},
            task=_mk_task("baseline", "t-reval-ok"),
        )
        assert c.shared_state.baseline_tput == 1000.0
        assert c.shared_state.enablement.succeeded is True
        assert c.shared_state.enablement.validation_pending is False
        assert c.shared_state.enablement.origin == ""
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_baseline_unrelated_baseline_does_not_consume_pending(session_dir):
    """A baseline that is NOT the tracked revalidation task must not consume pending state."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement.origin = "eval"
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.accuracy_floor = 0.3
        c.shared_state.enablement.revalidation_task_id = "t-reval-tracked"
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1000.0, "completed_requests": 10, "accuracy": 0.42},
            task=_mk_task("baseline", "t-unrelated"),
        )
        assert c.shared_state.baseline_tput == 1000.0
        # Pending state must NOT be consumed by the unrelated baseline.
        assert c.shared_state.enablement.validation_pending is True
        assert c.shared_state.enablement.succeeded is False
        assert c.shared_state.enablement.revalidation_task_id == "t-reval-tracked"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_promote_baseline_sub_floor_accuracy_rearmes_stall(session_dir):
    """Tracked revalidation baseline with sub-floor accuracy should rearm, not succeed."""
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement.origin = "eval"
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.accuracy_floor = 0.8
        c.shared_state.enablement.revalidation_task_id = "t-reval-subflo"
        await c._promote_to_shared_state(
            "baseline",
            {"output_throughput": 1000.0, "completed_requests": 10, "accuracy": 0.5},
            task=_mk_task("baseline", "t-reval-subflo"),
        )
        # Baseline tput anchors normally, but enablement is NOT succeeded.
        assert c.shared_state.baseline_tput == 1000.0
        assert c.shared_state.enablement.succeeded is False
        assert c.shared_state.enablement.validation_pending is False
        assert c.shared_state.enablement.stall_streak == 1
        assert c.shared_state.enablement.revalidation_task_id == ""
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_persist_eval_failure_clears_pending_and_counts_stall(session_dir, monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.revalidation_task_id = "t-reval-fail"
        await c._handle_unpromotable_result(_mk_task("baseline", "t-reval-fail"), _eval_failed_result())
        assert c.shared_state.enablement.validation_pending is False
        assert c.shared_state.enablement.stall_streak == 1
    finally:
        await c.stop()


def _eval_unavailable_result() -> dict:
    """An eval-less baseline: succeeded, but measured no accuracy at all."""
    return {
        "status": "succeeded",
        "baseline_eval_failed": True,
        "baseline_eval_failure_kind": "accuracy_unavailable",
        "baseline_eval_observed_accuracy": None,
        "baseline_eval_accuracy_floor": 0.0,
        "baseline_eval_evidence": (
            "baseline accuracy did not meet floor: accuracy=None floor=0.0 task=None metric=None source=None"
        ),
        "baseline_eval_contract_fingerprint": "noeval-fp",
        "materialized_config": "/runs/baseline/noeval.yaml",
    }


@pytest.mark.asyncio
async def test_eval_less_baseline_does_not_downgrade_measured_trigger(session_dir, monkeypatch):
    """An eval-less re-baseline must not overwrite a measured enablement trigger.

    ``disable_run_eval`` re-baselines report ``accuracy_unavailable`` with no
    task/metric/source. They still count as a failed round, but the stored
    ``accuracy_below_floor`` evidence (the measurement enablement must
    reproduce) has to survive.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        st = c.shared_state
        st.enablement.baseline_eval_kind = "accuracy_below_floor"
        st.enablement.observed_accuracy = 0.0
        st.enablement.observed_task = "gsm8k"
        st.enablement.observed_metric = "exact_match,strict-match"
        st.enablement.baseline_eval_evidence = "measured: accuracy=0.0 task=gsm8k source=/runs/.../results.json"
        st.enablement.probe_config_path = "/runs/baseline/measured.yaml"
        st.enablement.eval_contract_fingerprint = "measured-fp"

        await c._handle_unpromotable_result(_mk_task("baseline", "t-noeval"), _eval_unavailable_result())

        # The measured characterization survives...
        assert st.enablement.baseline_eval_kind == "accuracy_below_floor"
        assert st.enablement.observed_task == "gsm8k"
        assert st.enablement.observed_metric == "exact_match,strict-match"
        assert "accuracy=0.0" in st.enablement.baseline_eval_evidence
        assert st.enablement.probe_config_path == "/runs/baseline/measured.yaml"
        assert st.enablement.eval_contract_fingerprint == "measured-fp"
        # ...while the round still registers as an eval-rooted failure.
        assert st.enablement.origin == "eval"
        assert st.enablement.pending is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_measured_trigger_overwrites_earlier_unavailable(session_dir, monkeypatch):
    """The guard is one-way: a real measurement still replaces an empty trigger."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        st = c.shared_state
        st.enablement.baseline_eval_kind = "accuracy_unavailable"
        st.enablement.baseline_eval_evidence = "accuracy=None"

        measured = _eval_unavailable_result()
        measured["baseline_eval_failure_kind"] = "accuracy_below_floor"
        measured["baseline_eval_observed_accuracy"] = 0.0
        measured["accuracy_task"] = "gsm8k"
        measured["baseline_eval_evidence"] = "measured: accuracy=0.0 task=gsm8k"

        await c._handle_unpromotable_result(_mk_task("baseline", "t-measured"), measured)

        assert st.enablement.baseline_eval_kind == "accuracy_below_floor"
        assert st.enablement.observed_task == "gsm8k"
        assert "measured" in st.enablement.baseline_eval_evidence
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_revalidation_boot_failure_clears_pending_and_rearmes(session_dir, monkeypatch):
    """Any revalidation failure (including plain boot failures) clears pending and increments stall."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        c.shared_state.enablement.validation_pending = True
        c.shared_state.enablement.revalidation_task_id = "t-reval-boot"
        c.shared_state.enablement.eval_contract_fingerprint = "frozen-fp"
        c.shared_state.enablement.accuracy_floor = 0.5
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-reval-boot"),
            {"status": "failed", "error_class": "oom"},
        )
        assert c.shared_state.enablement.validation_pending is False
        assert c.shared_state.enablement.stall_streak == 1
        assert c.shared_state.enablement.revalidation_task_id == ""
        # Frozen trigger identity must be preserved.
        assert c.shared_state.enablement.eval_contract_fingerprint == "frozen-fp"
        assert c.shared_state.enablement.accuracy_floor == 0.5
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", ["session_time_exhausted", "orchestrator_cancelled"])
async def test_a_revalidation_the_run_stopped_does_not_burn_the_stall_streak(
    session_dir,
    monkeypatch,
    error_class,
):
    """The same round cannot be exempt from one ledger and charged to the other.

    A reaped revalidation baseline is exempted from the baseline failure streak
    because nothing about the baseline was measured. Charging it to the
    enablement stall streak reaches the cap on the evidence of a clock, and the
    session's terminal reason becomes ``enablement_stalled`` for rounds nobody
    ever ran.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        st = c.shared_state
        st.enablement.validation_pending = True
        st.enablement.revalidation_task_id = "t-reval-stopped"
        st.enablement.stall_streak = 4
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-reval-stopped"),
            {"status": "failed", "error_class": error_class, "error": "reaped"},
        )
        assert st.enablement.stall_streak == 4
        assert st.stop_reason in ("", None)
        assert st.baseline_failure_streak == 0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_reaped_revalidation_leaves_the_window_open_for_a_resume(session_dir, monkeypatch):
    """Nothing else reopens the window, so the stop must not close it.

    ``validation_pending`` is set only by an eval-origin KEEP, and the
    revalidation enqueue is gated on it, so clearing it strands a KEEP'd patch
    that was never revalidated. The generation is bumped for the same reason
    opening the window bumps it: the idempotency key must not resolve to the row
    the run just stopped.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        st = c.shared_state
        st.enablement.validation_pending = True
        st.enablement.revalidation_task_id = "t-reval-open"
        st.enablement.revalidation_generation = 2
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-reval-open"),
            {"status": "failed", "error_class": "session_time_exhausted", "error": "reaped"},
        )
        assert st.enablement.validation_pending is True
        assert st.enablement.revalidation_task_id == ""
        assert st.enablement.revalidation_generation == 3
        assert st.enablement.inflight_task_id == ""
    finally:
        await c.stop()


async def _cancelled_revalidation_row(c: Coordinator, *, gen: int) -> Task:
    """A revalidation row for ``gen`` that the queue scan cancelled before dispatch."""
    task, _existing = await c.tasks.create_or_return_existing(
        kind="baseline",
        params={"reason": "enablement_eval_revalidation"},
        idempotency_key=f"enablement_revalidation:gen{gen}",
    )
    await c.tasks.transition(task.task_id, "cancelled", evidence={"reason": "time_budget"})
    return task


@pytest.mark.asyncio
async def test_a_revalidation_the_budget_cannot_fit_is_not_enqueued(session_dir, monkeypatch):
    """Opening a row the dispatcher would cancel on sight is what wedges the window.

    A revalidation is a full baseline, and the queue scan drops a queued one the
    wall-clock budget can no longer fit. That leaves a cancelled row owning this
    window's idempotency key -- and a row cancelled at dispatch never produces a
    result to route, so nothing advances the generation past it and every later
    tick resolves the window to a row that measured nothing.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        st = c.shared_state
        st.enablement.validation_pending = True
        st.enablement.revalidation_generation = 3
        st.max_minutes = 60
        st.elapsed_minutes = lambda **_kw: 60.0  # type: ignore[method-assign]

        assert await c._maybe_enqueue_enablement_baseline_revalidation() == ""

        # The window survives the stop: same generation, still pending, and no
        # row for the key a resume with budget left will need.
        assert st.enablement.validation_pending is True
        assert st.enablement.revalidation_generation == 3
        assert st.enablement.revalidation_task_id == ""
        assert await c.tasks.by_state("cancelled") == []
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_a_revalidation_key_spent_on_a_cancelled_row_opens_the_next_one(session_dir, monkeypatch):
    """A terminal row is a spent generation, not an enqueue.

    ``create_or_return_existing`` hands back the cancelled row for as long as the
    key names it, so without recognising that the window stays open resolving to
    it for the rest of the session.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        st = c.shared_state
        st.enablement.validation_pending = True
        st.enablement.revalidation_generation = 3
        spent = await _cancelled_revalidation_row(c, gen=3)

        tid = await c._maybe_enqueue_enablement_baseline_revalidation()

        assert tid and tid != spent.task_id, "the window resolved to the cancelled row"
        assert st.enablement.revalidation_generation == 4
        assert (await c.tasks.get(tid)).state == "queued"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_resume_does_not_charge_a_revalidation_the_run_cancelled(
    session_dir,
    monkeypatch,
):
    """The exemption the reap grants must not be charged back by the resume.

    The reap path leaves the window open without charging the stall streak,
    because a round the run stopped measured nothing. The resume-time recovery saw
    only "tracked row is terminal" and closed the window with the increment the
    reap went out of its way to avoid -- reaching the ``enablement_stalled`` cap on
    the evidence of a clock, one resume later.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        st = c.shared_state
        cancelled = await _cancelled_revalidation_row(c, gen=3)
        st.enablement.validation_pending = True
        st.enablement.revalidation_task_id = cancelled.task_id
        st.enablement.revalidation_generation = 3
        st.enablement.stall_streak = 4

        report: dict[str, Any] = {"fixes": []}
        await c.writeback._resume_recover_pending_revalidation(report)

        assert st.enablement.stall_streak == 4
        assert st.stop_reason in ("", None)
        # And the window is left usable rather than merely uncharged.
        assert st.enablement.validation_pending is True
        assert st.enablement.revalidation_task_id == ""
        assert st.enablement.revalidation_generation == 4
        assert [f["kind"] for f in report["fixes"]] == ["reopened_revalidation_the_run_cancelled"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_resume_still_closes_a_revalidation_window_that_had_its_chance(session_dir, monkeypatch):
    """A row that is terminal for any other reason is evidence, and still charged."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        st = c.shared_state
        task, _existing = await c.tasks.create_or_return_existing(
            kind="baseline",
            params={"reason": "enablement_eval_revalidation"},
            idempotency_key="enablement_revalidation:gen0",
        )
        await c.tasks.transition(task.task_id, "running")
        await c.tasks.transition(task.task_id, "succeeded")
        st.enablement.validation_pending = True
        st.enablement.revalidation_task_id = task.task_id
        st.enablement.stall_streak = 1

        report: dict[str, Any] = {"fixes": []}
        await c.writeback._resume_recover_pending_revalidation(report)

        assert st.enablement.validation_pending is False
        assert st.enablement.revalidation_task_id == ""
        assert st.enablement.stall_streak == 2
        assert [f["kind"] for f in report["fixes"]] == ["cleared_orphaned_revalidation_pending"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_records_for_non_baseline_kinds(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._handle_unpromotable_result(
            _mk_task("explore", "t-ex-fail"),
            {"status": "failed", "error_class": "subprocess_nonzero", "error": "rc=1\nstderr blob"},
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
            {"status": "failed", "error_class": "timeout", "error": "wall-clock exceeded"},
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
            {
                "status": "failed",
                "error_class": "cuda_graph_capture_failed",
                "error": "operation not permitted when stream is capturing",
            },
        )
        assert c.shared_state.baseline_eager_fallback is True
        # One-shot: a second capture failure must not re-arm (already set).
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-cg-2"),
            {
                "status": "failed",
                "error_class": "cuda_graph_capture_failed",
                "error": "operation not permitted when stream is capturing",
            },
        )
        assert c.shared_state.baseline_eager_fallback is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_baseline_eager_fallback_consume_updates_coordinator_live_state(
    session_dir,
):
    """Executor consumption must clear Coordinator's live state too, or a later
    coordinator save re-persists stale True and makes the one-shot eager
    fallback sticky for all later baseline retries."""
    from hyperloom.orchestrator.actions.executors.baseline import (
        BaselineExecutor,
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        await c._handle_unpromotable_result(
            _mk_task("baseline", "t-cg-arm"),
            {
                "status": "failed",
                "error_class": "cuda_graph_capture_failed",
                "error": "operation not permitted when stream is capturing",
            },
        )
        assert c.shared_state.baseline_eager_fallback is True

        executor = BaselineExecutor(
            session_dir=session_dir,
            shared_state=c.shared_state,
        )
        assert executor._consume_eager_fallback() is True
        assert c.shared_state.baseline_eager_fallback is False

        # A later coordinator path flushing the live SharedState.
        c.shared_state.save(session_dir)
        assert SharedState.load_or_init(session_dir).baseline_eager_fallback is False
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
            {
                "status": "failed",
                "error_class": "cuda_graph_capture_failed",
                "error": "operation not permitted when stream is capturing",
            },
        )
        assert c.shared_state.baseline_eager_fallback is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_roofline_increments_failure_streak(
    session_dir,
    caplog,
):
    """A watermark-roofline failure must increment roofline_failure_streak,
    eagerly clear auto_roofline_pending_task_id, and emit an Auto-roofline
    warning."""
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

        with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.loop.coordinator"):
            await c._handle_unpromotable_result(
                _mk_task("roofline", task_id),
                result,
            )
        # (a) audit entry exists.
        assert len(c.shared_state.roofline_attempts) == 1
        attempt = c.shared_state.roofline_attempts[-1]
        assert attempt["status"] == "failed"
        assert attempt["decision"] == "no_promote"
        # (b) failure streak +1.
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
        assert any("Auto-roofline" in r.message and "failed" in r.message for r in caplog.records), (
            "no 'Auto-roofline ... failed' WARNING was logged"
        )
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


# baseline-params fingerprint capture.
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
        "benchmark_script",
        "result_dir",
        "extra_server_args",
        "extra_envs",
        "model_path",
        "gpu_type",
        "config_path",
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
    fp_with_empty = _baseline_params_fingerprint(
        {
            "benchmark_script": "sglang_mi300x.sh",
            "extra_envs": {},
        }
    )
    assert fp == fp_with_empty


def test_fingerprint_stringifies_scalar_values():
    fp = _baseline_params_fingerprint(
        {
            "benchmark_script": "sglang_mi300x.sh",
            "model_path": "/path/models/DeepSeek-R1",
            "gpu_type": "mi300x",
        }
    )
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
        task = _mk_baseline_task(
            {
                "benchmark_script": "sglang_mi300x.sh",
                "model_path": "/path/models/DeepSeek-R1",
                "gpu_type": "mi300x",
            }
        )
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
        assert fp["model_path"] == "/path/models/DeepSeek-R1"
        assert fp["gpu_type"] == "mi300x"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_unpromotable_baseline_records_fingerprint(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    _mute_action_scoring(c)
    try:
        task = _mk_baseline_task(
            {
                "benchmark_script": "dsr1_fp8_mi300x.sh",
            }
        )
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


def test_default_health_timeout_is_900s_not_1800s():
    assert _multi_node_server_lifecycle.DEFAULT_HEALTH_TIMEOUT_S == 900


# Critic auto-approve archival actions + Coordinator silent-tick early-closing.
def _write_marker_target_baseline(session_dir: Path) -> None:
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "no_target",
                "reason": "no_target_gpu_configured",
                "row_count": 0,
            }
        ),
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
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

    path = asset_system_prompts_dir() / "critic.md"
    text = path.read_text(encoding="utf-8")
    assert "archival actions" in text.lower()
    assert "`report`" in text
    assert "`session_breakdown`" in text
    assert "`target_analysis`" in text
    assert "Always `approve` archival actions" in text


@pytest.mark.asyncio
async def test_sub_agent_runner_hands_back_lanes_when_the_claim_is_rejected(
    tmp_path,
):
    """A row cancelled between dispatch and the claim must not keep its lanes.

    The rejection itself is the double-spawn guard and still reaches the
    caller; what must not survive it is the lease, which would otherwise hold
    every conflicting lane until the TTL sweep noticed.
    """
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("explore", lambda _ctx: _immediately({}))

    task = await tr.create(
        kind="explore",
        params={},
        idempotency_key="k-claim-rejected",
        requires_lanes=["benchmark_lane"],
        lease_ttl_sec=600,
    )
    lease = await locks.acquire_many(
        ["benchmark_lane"],
        holder_id=task.task_id,
        task_id=task.task_id,
        action="explore",
        ttl_sec=600,
    )
    await tr.transition(task.task_id, "cancelled", evidence={"reason": "prune_branch"})

    with pytest.raises(IllegalTransition):
        await sub.run_task(task, prebound_lease=lease)

    assert await locks.lane_holders() == {}
    db.close()


@pytest.mark.asyncio
async def test_a_registered_run_leaves_a_row_queued_when_its_lanes_are_busy(
    tmp_path,
):
    """Losing the race for a lane is a retry, not a task that started.

    Claiming the row first would stamp it ``running`` for work that never
    began, and every ``tasks.running()`` reader -- the KERNEL idle guard, the
    CLOSE sequencer -- would hold a phase open for it until the lease TTL
    expired.
    """
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    ran = {"called": False}

    async def runner(_ctx):
        ran["called"] = True
        return {}

    sub.register_executor("profile", runner)
    await locks.acquire_many(
        ["profile_lane"],
        holder_id="someone-else",
        task_id="someone-else",
        action="profile",
        ttl_sec=600,
    )
    task = await tr.create(
        kind="profile",
        params={},
        idempotency_key="k-lane-busy",
        requires_lanes=["profile_lane"],
        lease_ttl_sec=600,
    )
    disp = DispatcherCollaborator(SimpleNamespace(locks=locks, sub=sub))

    assert await disp.run_task_registered(task) is None

    assert ran["called"] is False
    assert (await tr.get(task.task_id)).state == "queued"
    db.close()


@pytest.mark.asyncio
async def test_a_registered_run_is_reachable_by_the_wall_clock_defences(tmp_path):
    """The handle is the whole point: without it a cancel cannot find the action.

    Going straight to ``sub.run_task`` is what left the kernel-entry reprofile
    and the closing steps off ``_inflight_actions``, where a shutdown or a spent
    budget could not stop them.
    """
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    disp = DispatcherCollaborator(SimpleNamespace(locks=locks, sub=sub))
    registered: list[str] = []

    async def runner(ctx):
        registered.extend(disp._inflight_actions)
        return {}

    sub.register_executor("roofline", runner)
    task = await tr.create(kind="roofline", params={}, idempotency_key="k-registered")

    res = await disp.run_task_registered(task)

    assert res.state == "succeeded"
    assert registered == [task.task_id]
    assert disp._inflight_actions == {}, "the handle must not outlive the action"
    db.close()


@pytest.mark.asyncio
async def test_a_registered_run_labels_its_llm_calls_with_the_action(tmp_path):
    """This is the only place ``type`` is published for gateway attribution.

    A caller that reached ``sub.run_task`` directly would spend its tokens with
    no action label, which is exactly the hole this method exists to close.
    """
    from hyperloom.common.llm_attribution import current_action

    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    disp = DispatcherCollaborator(SimpleNamespace(locks=locks, sub=sub))
    seen: list[str] = []

    async def runner(ctx):
        seen.append(current_action())
        return {}

    sub.register_executor("conc_sweep", runner)
    task = await tr.create(kind="conc_sweep", params={}, idempotency_key="k-labelled")

    await disp.run_task_registered(task)

    assert seen == ["conc_sweep"]
    assert current_action() == "", "the label must not outlive the action"
    db.close()


@pytest.mark.asyncio
async def test_sub_agent_runner_fails_a_row_whose_workspace_cannot_be_made(
    tmp_path,
):
    """An ENOSPC while preparing the workspace is a failed task, not a live one."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)
    sub.register_executor("explore", lambda _ctx: _immediately({}))

    def _no_space(_task):
        raise OSError(28, "No space left on device")

    sub._pre_mkdir_workspace = _no_space
    task = await tr.create(kind="explore", params={}, idempotency_key="k-enospc")

    res = await sub.run_task(task)

    assert res.state == "failed"
    assert (await tr.get(task.task_id)).state == "failed"
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

    c.phase_machine._advance_phase_if_needed = _boom  # type: ignore[assignment]

    try:
        reason = await c.run(max_ticks=5)
        assert reason == "target_reached"
        assert c.shared_state.stop_reason == "target_reached"
        # Two advance_phase calls per run (pre-reactor hint consume + main) each
        # record a coordinator exception when the tick body raises.
        assert c.shared_state.crash_count == 2
        assert c.shared_state.last_tick_exception["stage"] == "advance_phase"
        assert c.shared_state.last_tick_exception["type"] == "RuntimeError"
        persisted = SharedState.load_or_init(session_dir)
        assert persisted.stop_reason == "target_reached"
        assert persisted.last_tick_exception["stage"] == "advance_phase"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_dispatch_audit_logs_task_without_executor(session_dir, caplog):
    # Defensive audit (log-only): a queued task whose kind has no registered
    # executor is flagged in the process log; dispatch itself is unchanged
    # (the task still fails on the missing runner).
    import logging

    delegate = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": "long_running",
            "params": {},
            "idempotency_key": "k-audit-dispatch-1",
        },
    )
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[delegate])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))

    async def _noop_executor(ctx):
        return {}

    # A fresh SubAgentRunner has an empty registry; the audit only fires once the
    # registry is populated, so register one unrelated executor first.
    c.sub.register_executor("report", _noop_executor)
    try:
        with caplog.at_level(logging.WARNING, logger="hyperloom.orchestrator.loop.dispatcher"):
            await c.tick(1)
        assert any("dispatch audit" in r.getMessage() and "long_running" in r.getMessage() for r in caplog.records)
        assert await c.tasks.by_state("failed")
    finally:
        await c.stop()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_target_reached_routes_through_close_phase(session_dir):
    """A met objective transitions to CLOSE instead of breaking out of the loop.

    ``machine_state`` registers ``target_reached`` as an "any phase -> CLOSE"
    transition reason, so a met target must reach the close sequencer rather
    than leaving the run with only the cli safety-net report.
    """
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.cumulative_gain_validated = 50.0
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=6,
        )
        assert reason == "target_reached"
        assert (c.shared_state.phase or "").upper() == "CLOSE"
        # The close sequencer actually ran; this is what separates a real close
        # from the cli safety-net path.
        assert c.shared_state.close_sequence_done is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_target_reached_at_session_bound_still_closes(session_dir):
    """A target met at/after the session bound must still reach the sequencer.

    Every ``_advance_phase_if_needed`` in the tick body is wrapped in
    ``_await_within_session_bound``, which skips the step once the bound has
    elapsed. Deferring the CLOSE transition to the next tick therefore never
    gets one, and the run falls back to the cli safety-net report -- the exact
    outcome this fix exists to prevent.
    """
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.cumulative_gain_validated = 50.0
    c.shared_state.save(session_dir)
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_minutes=0.0001,
            max_ticks=6,
        )
        assert reason == "target_reached"
        assert c.shared_state.close_sequence_done is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_target_reached_closes_despite_a_failed_state_save(session_dir):
    """Persisting the terminal is best-effort; it must not gate the close.

    The stop_reason is already set in memory, so a failed save leaves the run
    with a met target and no close sequence -- the safety-net path again.
    """
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.cumulative_gain_validated = 50.0
    c.shared_state.save(session_dir)

    real_save = c.shared_state.save
    state = {"tripped": False}

    def flaky_save(*args, **kwargs):
        if c.shared_state.stop_reason == "target_reached" and not state["tripped"]:
            state["tripped"] = True
            raise OSError("simulated transient state-save failure")
        return real_save(*args, **kwargs)

    c.shared_state.save = flaky_save  # type: ignore[method-assign]
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_ticks=6,
        )
        assert state["tripped"] is True
        assert reason == "target_reached"
        assert c.shared_state.close_sequence_done is True
    finally:
        c.shared_state.save = real_save  # type: ignore[method-assign]
        await c.stop()


@pytest.mark.asyncio
async def test_target_reached_close_still_runs_the_post_opt_roofline(session_dir):
    """A met target must not be treated as a wall-clock rescue.

    ``_maybe_run_close_post_opt_roofline`` returns early on ``closing_phase``,
    which means "the wall clock ran out, shed expensive work". Routing a success
    terminal through that flag would drop the post-opt snapshot the
    optimization-progress chart reads -- one of the artifacts skipping CLOSE
    loses in the first place.
    """
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.cumulative_gain_validated = 50.0
    # A kernel-level optimization landed, so the roofline step applies.
    c.shared_state.optimization_stack = [{"action": "integrate", "tput": 150.0}]
    c.shared_state.save(session_dir)

    ran: list[str] = []

    async def _record_roofline() -> None:
        ran.append(str(c.shared_state.closing_phase))

    c.phase_close._maybe_run_close_post_opt_roofline = _record_roofline  # type: ignore[method-assign]
    try:
        reason = await c.run(
            objective=TargetGainObjective(target_gain_pct=10.0),
            max_minutes=0.0001,
            max_ticks=6,
        )
        assert reason == "target_reached"
        assert ran == ["False"], f"post-opt roofline saw closing_phase={ran}"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_target_reached_close_is_not_cancelled_by_an_outer_bound(session_dir):
    """The sequencer's own per-step timeouts are the budget, not an outer one.

    ``CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC`` alone allows 600s, so any outer
    bound short enough to matter would cancel the step mid-flight and drop the
    run onto the safety net -- the outcome this routing exists to avoid. With
    the session bound already elapsed, an awaited step would be skipped outright
    unless the terminal close lifts it.
    """
    _write_marker_target_baseline(session_dir)
    c = Coordinator(session_dir, backends=_silent_backends())
    c.sub.register_executor("report", report_executor)
    c.shared_state.baseline_tput = 100.0
    c.shared_state.cumulative_gain_validated = 50.0
    c.shared_state.save(session_dir)

    # The bound is armed and elapsed; the terminal close must ignore it.
    c._run_deadline = time.monotonic() - 1.0
    assert c._seconds_until_session_bound() is not None
    c._terminal_closing = True
    assert c._seconds_until_session_bound() is None, "terminal close must be unbounded"
    c._terminal_closing = False

    real_advance = c.phase_machine._advance_phase_if_needed
    completed: list[bool] = []

    async def _slow_advance() -> None:
        # Any await at all is cancelled or skipped by an elapsed outer bound.
        await asyncio.sleep(0.3)
        await real_advance()
        completed.append(True)

    try:
        with mock.patch.object(c.phase_machine, "_advance_phase_if_needed", _slow_advance):
            reason = await c.run(
                objective=TargetGainObjective(target_gain_pct=10.0),
                max_minutes=0.0001,
                max_ticks=2,
            )
        assert reason == "target_reached"
        assert completed, "the close step was skipped or cancelled by an outer bound"
    finally:
        await c.stop()
