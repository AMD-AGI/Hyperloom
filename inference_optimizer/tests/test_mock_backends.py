"""P0-4 mock Critic + mock Robustness adapter tests.

Covers:

* MockCriticBackend extracts msg_id from inbox prompt and emits one
  review_verdict(approve) per proposal seen
* MockCriticBackend deduplicates: same msg_id seen twice → single approval
* MockCriticBackend with no proposals → heartbeat only
* MockRobustnessBackend always emits heartbeat
* MockRobustnessBackend with alert_after_ticks emits alert on the Nth tick
* End-to-end: Coordinator with MockCriticBackend approves an Orchestration
  proposal and the dispatcher materializes the resulting task — without
  any hand-injected critic verdict (real Critic loop closes itself).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
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


def _backends_with_mock_critic_and_robustness(
    plans: dict[str, ScriptedPlan] | None = None,
) -> dict[str, object]:
    plans = plans or {}
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(plans.get("orchestration", silent), name="o"),
        "kernel":        MockBackend(plans.get("kernel", silent), name="k"),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


# ===========================================================================
# MockCriticBackend (unit)
# ===========================================================================
@pytest.mark.asyncio
async def test_mock_critic_extracts_msg_id_and_approves():
    backend = MockCriticBackend()
    prompt = (
        "Inbox for critic (newest last):\n"
        "  seq=3 msg_id=abcdef0123456789 from=orchestration topic=proposal "
        "payload={'action_name': 'baseline', 'predicted_gain_pct': 0.0}"
    )
    res = await backend.run(prompt)
    assert len(res.intents) == 1
    intent = res.intents[0]
    assert intent.type == IntentType.REVIEW_VERDICT
    assert intent.payload["target_proposal_msg_id"] == "abcdef0123456789"
    assert intent.payload["verdict"] == "approve"
    assert intent.payload["source"] == "mock"


@pytest.mark.asyncio
async def test_mock_critic_dedups_same_proposal():
    backend = MockCriticBackend()
    prompt = (
        "Inbox for critic:\n"
        "  seq=1 msg_id=deadbeef0001 from=orchestration topic=proposal payload={...}"
    )
    r1 = await backend.run(prompt)
    r2 = await backend.run(prompt)
    assert len(r1.intents) == 1 and r1.intents[0].type == IntentType.REVIEW_VERDICT
    # Second turn — same proposal shouldn't be approved again; expect heartbeat fallback.
    assert len(r2.intents) == 1
    assert r2.intents[0].type == IntentType.SEND_MESSAGE
    assert r2.intents[0].payload["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_mock_critic_emits_one_verdict_per_proposal():
    backend = MockCriticBackend()
    prompt = (
        "Inbox for critic:\n"
        "  seq=1 msg_id=aaa1 from=orchestration topic=proposal payload={...}\n"
        "  seq=2 msg_id=bbb2 from=orchestration topic=proposal payload={...}\n"
        "  seq=3 msg_id=ccc3 from=robustness topic=alert payload={...}"
    )
    res = await backend.run(prompt)
    assert len(res.intents) == 2
    targets = sorted(i.payload["target_proposal_msg_id"] for i in res.intents)
    assert targets == ["aaa1", "bbb2"]


@pytest.mark.asyncio
async def test_mock_critic_heartbeat_when_no_proposal():
    backend = MockCriticBackend()
    res = await backend.run("(no new messages for critic)")
    assert len(res.intents) == 1
    assert res.intents[0].type == IntentType.SEND_MESSAGE
    assert res.intents[0].payload["topic"] == "heartbeat"


# ===========================================================================
# MockRobustnessBackend (unit)
# ===========================================================================
@pytest.mark.asyncio
async def test_mock_robustness_always_heartbeat():
    backend = MockRobustnessBackend()
    for _ in range(3):
        res = await backend.run("anything")
        assert len(res.intents) == 1
        assert res.intents[0].payload["topic"] == "heartbeat"


@pytest.mark.asyncio
async def test_mock_robustness_alert_after_n_ticks():
    backend = MockRobustnessBackend(alert_after_ticks=2)
    r1 = await backend.run("p")
    r2 = await backend.run("p")
    r3 = await backend.run("p")
    types_per = [tuple(i.type for i in r.intents) for r in (r1, r2, r3)]
    assert types_per[0] == (IntentType.SEND_MESSAGE,)
    assert IntentType.ALERT in types_per[1]
    assert types_per[2] == (IntentType.SEND_MESSAGE,)


# ===========================================================================
# E2E with Coordinator — real Critic-loop closes itself (no manual verdict)
# ===========================================================================
@pytest.mark.asyncio
async def test_e2e_mock_critic_closes_proposal_loop(session_dir):
    """Orchestration proposes baseline; mock Critic auto-approves; task gets created."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    plans = {
        "orchestration": ScriptedPlan(turns=[
            MockTurn(intents=[propose]),  # tick 1: propose
            # tick 2+: silent (default heartbeat)
        ]),
    }
    backends = _backends_with_mock_critic_and_robustness(plans)

    c = Coordinator(session_dir, backends=backends)
    try:
        # Tick 1: orchestration proposes; critic doesn't see it yet (composes
        # prompt before propose appears — depends on iteration order).
        # Tick 2: critic sees proposal, approves; task materialized.
        await c.tick(2)

        approved_decisions = await c.bus.tail(topic="decision")
        approved = [m for m in approved_decisions if m.payload.get("kind") == "approved_proposal"]
        assert approved, "expected at least one approved_proposal decision"
        assert approved[0].payload["action_name"] == "baseline"

        # Pending proposal marked decided
        pending = list(c.state.pending_proposals.values())
        assert pending and pending[0].verdict == "approve"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_e2e_mock_robustness_keeps_emitting_heartbeats(session_dir):
    backends = _backends_with_mock_critic_and_robustness({})
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(3)
        beats = await c.bus.tail(topic="heartbeat", to_agent="*")
        # Each tick: orchestration / kernel / critic / robustness all heartbeat.
        # 3 ticks × 4 agents = 12.
        assert len(beats) >= 12
        assert any(m.from_agent == "robustness" for m in beats)
    finally:
        await c.stop()
