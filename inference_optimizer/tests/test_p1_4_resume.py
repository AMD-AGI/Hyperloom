"""P1-4 Coordinator resume tests.

Covers:

* Fresh session: is_resume=False, no replay needed
* Existing state.json triggers is_resume=True even with empty events
* Existing events trigger is_resume=True
* replay_for_resume reconstructs pending_proposals from undecided
  topic='proposal' events
* Approved proposals are NOT re-instantiated as pending after resume
* Rejected proposals are NOT re-instantiated as pending after resume
* Multiple proposals + mixed decisions reconstruct correct pending set
* Coordinator restart preserves pruned_families AND restores undecided
  pending_proposals AND keeps task lifecycle intact
* tick() lazily triggers replay_for_resume on first call (so callers
  don't have to remember to call it explicitly)
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


# ===========================================================================
# Resume detection
# ===========================================================================
@pytest.mark.asyncio
async def test_fresh_session_is_not_resume(session_dir):
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        info = c.resumed_from
        assert info["is_resume"] is False
        assert info["event_count"] == 0
        assert info["state_json_present"] is False
        assert info["rebuilt"] is False
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_existing_state_json_triggers_resume(session_dir):
    SharedState(session_id="resumed").save(session_dir)
    c = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c.resumed_from["is_resume"] is True
        assert c.resumed_from["state_json_present"] is True
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_existing_events_triggers_resume(session_dir):
    # First Coordinator: emit a heartbeat to populate the events table.
    c1 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c1.tick(1)
    finally:
        await c1.stop()
    # Second Coordinator on the same session_dir sees events ≥1.
    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        assert c2.resumed_from["is_resume"] is True
        assert c2.resumed_from["event_count"] >= 1
    finally:
        await c2.stop()


# ===========================================================================
# Resume rebuild — pending_proposals
# ===========================================================================
@pytest.mark.asyncio
async def test_replay_rebuilds_undecided_proposals(session_dir):
    """One propose, no verdict → resume restores it as pending."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    # Mock orchestration only — no Critic mock so no auto-approval.
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends_no_critic = {
        "orchestration": MockBackend(ScriptedPlan(turns=[MockTurn(intents=[propose])],
                                                    default_intent=_heartbeat()),
                                       name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends_no_critic)
    try:
        await c1.tick(1)
        assert len(c1.state.pending_proposals) == 1
        original_id = next(iter(c1.state.pending_proposals.keys()))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 1
        assert original_id in c2.state.pending_proposals
        restored = c2.state.pending_proposals[original_id]
        assert restored.action_name == "baseline"
        assert restored.from_agent == "orchestration"
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_approved_proposals(session_dir):
    """Approved proposal must not appear as pending after resume."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    backends = _backends_full()
    backends["orchestration"] = MockBackend(
        ScriptedPlan(turns=[MockTurn(intents=[propose])],
                     default_intent=_heartbeat()),
        name="orch",
    )
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(2)  # tick 1: propose, tick 2: critic auto-approves
        assert any(p.verdict == "approve" for p in c1.state.pending_proposals.values())
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        # The approved proposal should be filtered out.
        assert stats["pending_restored"] == 0
        assert c2.state.pending_proposals == {}
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_rejected_proposals(session_dir):
    """Rejected proposal also counted as decided → not pending after resume."""
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])],
                         default_intent=_heartbeat()),
            name="o",
        ),
        "kernel":     MockBackend(silent, name="k"),
        "critic":     MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1.tick(1)
        proposal_id = next(iter(c1.state.pending_proposals.keys()))
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_id, "verdict": "reject",
                     "reasoning": "violates kb-7", "kb_evidence": "kb-7"},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 0
        assert stats["verdicts_seen"] >= 1
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_mixed_pending_and_decided(session_dir):
    """3 proposals, 1 approved, 1 rejected, 1 undecided → 1 restored."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        # This test is about replay bookkeeping, not execution-order gating.
        # Seed prerequisites so arbitrary proposals are accepted.
        c1.shared_state.baseline_tput = 100.0
        c1.shared_state.last_profile_trace = "/tmp/profile.trace.json.gz"
        c1.shared_state.last_profile_pmc_summary = "/tmp/profile.pmc.json"
        c1.shared_state.last_select_kernels = {
            "trace_input": "/tmp/profile.trace.json.gz",
        }
        c1.shared_state.save(session_dir)
        proposal_ids = []
        for action in ("baseline", "profile", "backends"):
            await c1._handle_intent("orchestration", Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": action, "predicted_gain_pct": 0.0},
            ))
            # Grab the most recent proposal_msg_id
            tail = await c1.bus.tail(topic="proposal", n=1)
            proposal_ids.append(tail[0].msg_id)

        # Approve baseline (proposal_ids[0])
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_ids[0],
                     "verdict": "approve", "reasoning": "ok"},
        ))
        # Reject profile (proposal_ids[1])
        await c1._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT,
            payload={"target_proposal_msg_id": proposal_ids[1],
                     "verdict": "reject", "reasoning": "no", "kb_evidence": "kb-x"},
        ))
        # backends (proposal_ids[2]) left undecided
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        stats = await c2.replay_for_resume()
        assert stats["pending_restored"] == 1
        restored = next(iter(c2.state.pending_proposals.values()))
        assert restored.action_name == "backends"
    finally:
        await c2.stop()


# ===========================================================================
# Resume + SharedState combined
# ===========================================================================
@pytest.mark.asyncio
async def test_resume_preserves_pruned_and_restores_pending(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        # 1 prune + 1 undecided proposal
        await c1._handle_intent("robustness", Intent(
            type=IntentType.PRUNE_BRANCH,
            payload={"family": "deep_kernel", "reason": "x"},
        ))
        await c1._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        await c2.replay_for_resume()
        assert c2.shared_state.is_pruned("deep_kernel")
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_tick_lazily_runs_replay_on_resume(session_dir):
    """Resume callers shouldn't have to remember to call replay manually —
    the first tick() should do it."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c1 = Coordinator(session_dir, backends=backends)
    try:
        await c1._handle_intent("orchestration", Intent(
            type=IntentType.PROPOSE_ACTION,
            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends_full())
    try:
        # No explicit replay_for_resume — tick should trigger it
        assert c2.resumed_from["rebuilt"] is False
        await c2.tick(1)
        assert c2.resumed_from["rebuilt"] is True
        assert len(c2.state.pending_proposals) == 1
    finally:
        await c2.stop()
