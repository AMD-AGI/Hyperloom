"""P0-5 end-to-end main-loop tests.

Covers the full P0 main path with all 4 agents wired to mock backends:

* Orchestration proposes → Critic mock approves → task materialized →
  dispatcher runs the registered baseline runner → succeeded.
* Orchestration emits REQUEST{target=kernel, kind=trace_analyze} →
  Coordinator mirrors to kernel inbox → Kernel mock auto-RESPONSEs →
  Coordinator routes response back to orchestration inbox.
* Robustness mock keeps ticking heartbeats throughout — no scheduling
  police intervention.
* The bundled demo script (``examples.p0_main_loop._run_demo``) finishes
  cleanly and reports non-zero counts for each milestone topic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.examples import p0_main_loop
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
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


# ===========================================================================
# Mock Kernel adapter unit tests
# ===========================================================================
@pytest.mark.asyncio
async def test_mock_kernel_responds_to_request():
    backend = MockKernelBackend()
    prompt = (
        "Inbox for kernel:\n"
        "  seq=4 msg_id=cafe1234 from=orchestration topic=request "
        "payload={'target_agent': 'kernel', 'kind': 'trace_analyze', 'params': {'top_k': 5}}"
    )
    res = await backend.run(prompt)
    assert len(res.intents) == 1
    intent = res.intents[0]
    assert intent.type == IntentType.RESPONSE
    assert intent.payload["in_reply_to"] == "cafe1234"
    assert intent.payload["kind"] == "trace_analyze_done"
    assert intent.payload["status"] == "ok"


@pytest.mark.asyncio
async def test_mock_kernel_dedups_same_request():
    backend = MockKernelBackend()
    prompt = (
        "Inbox for kernel:\n"
        "  seq=1 msg_id=ded00ad0 from=orchestration topic=request "
        "payload={'target_agent': 'kernel', 'kind': 'trace_analyze'}"
    )
    r1 = await backend.run(prompt)
    r2 = await backend.run(prompt)
    assert r1.intents[0].type == IntentType.RESPONSE
    # Second turn — same request should NOT be re-answered; expect heartbeat
    assert r2.intents[0].type == IntentType.SEND_MESSAGE
    assert r2.intents[0].payload["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_mock_kernel_heartbeat_when_no_request():
    backend = MockKernelBackend()
    res = await backend.run("(no new messages for kernel)")
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"


# ===========================================================================
# Full 4-agent main-loop e2e
# ===========================================================================
@pytest.mark.asyncio
async def test_e2e_propose_approve_dispatch_with_mock_executor(session_dir):
    """Orchestration → Critic (mock) → dispatcher → succeeded."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])],
                         default_intent=_heartbeat()),
            name="orchestration",
        ),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    c.sub.register_executor("baseline", lambda ctx: _async_value({"tput": 1840}))
    try:
        # tick 1: orchestration proposes
        # tick 2: critic sees + approves; coordinator materializes task
        # tick 3: dispatcher runs the runner → delegated_result succeeded
        await c.tick(3)

        proposals = await c.bus.tail(topic="proposal")
        verdicts = await c.bus.tail(topic="review_verdict")
        decisions = await c.bus.tail(topic="decision")
        results = await c.bus.tail(topic="delegated_result")

        assert len(proposals) >= 1
        assert any(v.payload.get("verdict") == "approve" for v in verdicts)
        assert any(d.payload.get("kind") == "approved_proposal" for d in decisions)
        assert any(
            r.payload.get("state") == "succeeded"
            and r.payload.get("result", {}).get("tput") == 1840
            for r in results
        )
    finally:
        await c.stop()


async def _async_value(v):
    return v


@pytest.mark.asyncio
async def test_e2e_request_response_round_trip(session_dir):
    """Plan A: orchestration REQUEST → kernel mock RESPONSE → routed back.

    Uses a kind (`explore_options`) that is NOT registered in
    KERNEL_REQUEST_HANDLERS so the request flows through the LLM-backed
    kernel agent instead of being intercepted by a programmatic handler.
    Production kinds (trace_analyze / run_optimization / integrate)
    have dedicated handler tests in test_p3_bugfixes.py and the
    integration suite.
    """
    req = Intent(type=IntentType.REQUEST, payload={
        "target_agent": "kernel",
        "kind": "explore_options",
        "params": {"top_k": 5},
    })
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[req])],
                         default_intent=_heartbeat()),
            name="orchestration",
        ),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        # tick 1: orchestration emits REQUEST → coordinator mirrors to kernel inbox
        # tick 2: kernel sees request → emits RESPONSE → coordinator routes back
        await c.tick(2)

        # Request mirrored to kernel
        kernel_inbox = await c.bus.tail(to_agent="kernel", topic="request")
        assert kernel_inbox

        # Response routed back to orchestration
        responses = await c.bus.tail(to_agent="orchestration", topic="response")
        assert responses
        r = responses[0]
        assert r.from_agent == "kernel"
        assert r.payload["status"] == "ok"
        assert r.payload["kind"] == "explore_options_done"
        assert r.payload["result"]["chosen"] == ["mock_kernel_1"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_e2e_robustness_heartbeats_throughout(session_dir):
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()),
                                      name="orchestration"),
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(4)
        beats = await c.bus.tail(topic="heartbeat", n=100)
        # Robustness alone contributes 4 heartbeats over 4 ticks.
        robustness_beats = [b for b in beats if b.from_agent == "robustness"]
        assert len(robustness_beats) == 4
    finally:
        await c.stop()


# ===========================================================================
# Demo script smoke
# ===========================================================================
@pytest.mark.asyncio
async def test_demo_script_runs_end_to_end(session_dir, capsys):
    """The packaged demo (`examples.p0_main_loop._run_demo`) completes."""
    summary = await p0_main_loop._run_demo(ticks=6)
    assert summary["proposals_seen"] >= 1
    assert summary["verdicts_seen"] >= 1
    assert summary["decisions_seen"] >= 1
    assert summary["delegated_results_seen"] >= 1
    assert summary["responses_seen"] >= 1
    captured = capsys.readouterr()
    assert "highlights" in captured.out
    assert "verdict" in captured.out
    assert "delegated_result" in captured.out
