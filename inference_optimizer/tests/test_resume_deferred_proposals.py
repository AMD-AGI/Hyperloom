"""Resume contract for Critic-approved proposals deferred by the
auto-roofline gate (post-PR-321 review Finding 1, P1).

When a watermark fires while a proposal is in front of the Critic,
``_materialize_approved_proposal`` cannot dispatch it until the
Coordinator-internal analysis task lands. The proposal is parked on
the in-memory ``_proposals_awaiting_roofline`` deque and a
``proposal_materialize_blocked`` observation is appended.

The in-memory deque does not survive a restart. Without an
explicit rebuild step the original Critic verdict marks the proposal
as decided, ``replay_for_resume`` drops it from ``pending_proposals``,
and the approved work is silently lost — there is no way for the LLM
to re-propose because PolicyGate denies the analysis lane.

This test pins the rebuild contract:

  * blocked-without-drained → re-queued onto ``_proposals_awaiting_roofline``
  * blocked-then-drained    → dropped (treated as already dispatched)
  * Critic-rejected         → never queued (defensive guard)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import Message
from inference_optimizer.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


async def _seed_deferred_proposal(
    c: Coordinator,
    *,
    action_name: str = "explore",
    drained: bool,
    rejected: bool = False,
    approved_variant_names: list[str] | None = None,
    kb_edge_ids: dict[str, str] | None = None,
) -> str:
    """Append a proposal + verdict + materialize-blocked observation,
    optionally followed by an ``approved_proposal`` decision. Returns
    the proposal_msg_id so the caller can cross-check the rebuild."""
    payload = {
        "action_name": action_name,
        "predicted_gain_pct": 5.0,
        "params": {"grid": [{"name": "v1"}, {"name": "v2"}]},
    }
    propose = Message.new("orchestration", "*", "proposal", payload)
    await c.bus.append_and_seq(propose)
    pid = propose.msg_id
    verdict_payload: dict
    if rejected:
        verdict_payload = {"target_proposal_msg_id": pid, "verdict": "reject"}
    else:
        verdict_payload = {"target_proposal_msg_id": pid, "verdict": "approve"}
    await c.bus.append_and_seq(
        Message.new("critic", "*", "review_verdict", verdict_payload),
    )
    await c.bus.append_and_seq(Message.new(
        "coordinator", "*", "observation",
        {
            "kind": "proposal_materialize_blocked",
            "reason": "wait_for_auto_roofline",
            "proposal_msg_id": pid,
            "action_name": action_name,
            "from_agent": "orchestration",
            "approved_variant_names": (
                list(approved_variant_names)
                if approved_variant_names is not None
                else None
            ),
            "kb_edge_ids": dict(kb_edge_ids or {}),
        },
    ))
    if drained:
        await c.bus.append_and_seq(Message.new(
            "coordinator", "*", "decision",
            {
                "kind": "approved_proposal",
                "task_id": f"task-{pid}",
                "action_name": action_name,
                "from_agent": "orchestration",
                "proposal_msg_id": pid,
            },
        ))
    return pid


@pytest.mark.asyncio
async def test_replay_rebuilds_deferred_queue_for_blocked_proposal(session_dir):
    """Blocked + Critic-approved + NO approved_proposal decision must
    re-populate ``_proposals_awaiting_roofline`` after restart."""
    c1 = Coordinator(session_dir, backends=_backends())
    try:
        pid = await _seed_deferred_proposal(
            c1, drained=False, approved_variant_names=["v1"],
            kb_edge_ids={"v1": "edge-1", "v2": "edge-2"},
        )
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends())
    try:
        stats = await c2.replay_for_resume()
        assert stats["deferred_restored"] == 1
        assert len(c2._proposals_awaiting_roofline) == 1
        pending, approved = c2._proposals_awaiting_roofline[0]
        assert pending.proposal_msg_id == pid
        assert pending.action_name == "explore"
        assert pending.from_agent == "orchestration"
        assert approved == {"v1"}
        assert pending.kb_edge_ids == {"v1": "edge-1", "v2": "edge-2"}
        # The verdict marks the proposal as decided → ``pending_proposals``
        # MUST NOT also contain it (otherwise a second Critic round
        # would fire).
        assert pid not in c2.state.pending_proposals
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_deferred_proposal_already_drained(session_dir):
    """Blocked + later ``approved_proposal`` decision → drain already
    fired before shutdown, do NOT re-queue."""
    c1 = Coordinator(session_dir, backends=_backends())
    try:
        await _seed_deferred_proposal(c1, drained=True)
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends())
    try:
        stats = await c2.replay_for_resume()
        assert stats["deferred_restored"] == 0
        assert c2._proposals_awaiting_roofline == []
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_skips_deferred_proposal_critic_rejected(session_dir):
    """Defensive: a blocked observation that pairs with a reject verdict
    should never re-queue (materialize only runs after approve, so this
    combination shouldn't occur in practice, but the rebuild must not
    re-dispatch a rejected proposal if it ever does)."""
    c1 = Coordinator(session_dir, backends=_backends())
    try:
        await _seed_deferred_proposal(c1, drained=False, rejected=True)
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends())
    try:
        stats = await c2.replay_for_resume()
        assert stats["deferred_restored"] == 0
        assert c2._proposals_awaiting_roofline == []
    finally:
        await c2.stop()


@pytest.mark.asyncio
async def test_replay_latest_blocked_payload_wins(session_dir):
    """If a proposal gets re-blocked (e.g. drain runs while a fresh
    watermark already fired), the latest blocked observation's
    ``approved_variant_names`` + ``kb_edge_ids`` should win — those
    are the freshest values the deferred queue ran with."""
    c1 = Coordinator(session_dir, backends=_backends())
    try:
        payload = {
            "action_name": "explore",
            "predicted_gain_pct": 5.0,
            "params": {"grid": [{"name": "v1"}, {"name": "v2"}]},
        }
        propose = Message.new("orchestration", "*", "proposal", payload)
        await c1.bus.append_and_seq(propose)
        pid = propose.msg_id
        await c1.bus.append_and_seq(Message.new(
            "critic", "*", "review_verdict",
            {"target_proposal_msg_id": pid, "verdict": "approve"},
        ))
        # Older blocked observation: approves both variants.
        await c1.bus.append_and_seq(Message.new(
            "coordinator", "*", "observation",
            {
                "kind": "proposal_materialize_blocked",
                "reason": "wait_for_auto_roofline",
                "proposal_msg_id": pid,
                "action_name": "explore",
                "from_agent": "orchestration",
                "approved_variant_names": ["v1", "v2"],
                "kb_edge_ids": {},
            },
        ))
        # Newer blocked observation: only v1 survived re-review.
        await c1.bus.append_and_seq(Message.new(
            "coordinator", "*", "observation",
            {
                "kind": "proposal_materialize_blocked",
                "reason": "wait_for_auto_roofline",
                "proposal_msg_id": pid,
                "action_name": "explore",
                "from_agent": "orchestration",
                "approved_variant_names": ["v1"],
                "kb_edge_ids": {"v1": "edge-new"},
            },
        ))
    finally:
        await c1.stop()

    c2 = Coordinator(session_dir, backends=_backends())
    try:
        await c2.replay_for_resume()
        assert len(c2._proposals_awaiting_roofline) == 1
        _, approved = c2._proposals_awaiting_roofline[0]
        assert approved == {"v1"}
        pending, _ = c2._proposals_awaiting_roofline[0]
        assert pending.kb_edge_ids == {"v1": "edge-new"}
    finally:
        await c2.stop()
