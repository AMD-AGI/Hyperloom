# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Critic turns are admitted only while a proposal awaits a verdict."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.proposals import record_proposal
from hyperloom.orchestrator.roles import MockBackend, MockTurn, ScriptedPlan
from hyperloom.orchestrator.roles.base import BackendTurnResult
from .conftest import silent_backends


async def _proposal(coordinator: Coordinator):
    return await record_proposal(
        coordinator,
        from_agent="orchestration",
        action_name="baseline",
        predicted_gain_pct=0.0,
        payload={"action_name": "baseline", "params": {}},
    )


async def test_admission_follows_pending_proposals_and_their_verdicts(session_dir):
    c = Coordinator(session_dir, backends=silent_backends())
    try:
        assert not await c.conversation.admit_turn("critic", time.time())
        pending = await _proposal(c)
        assert await c.conversation.admit_turn("critic", time.time())

        await c.router.handle_intent(
            "critic",
            Intent(
                type=IntentType.REVIEW_VERDICT,
                payload={"target_proposal_msg_id": pending.proposal_msg_id, "verdict": "reject"},
            ),
        )

        assert not await c.conversation.admit_turn("critic", time.time())
    finally:
        await c.stop()


async def test_idle_ticks_do_not_call_the_critic_or_update_its_activity(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=silent_backends())
    critic_run = AsyncMock(wraps=c.backends["critic"].run)
    monkeypatch.setattr(c.backends["critic"], "run", critic_run)
    c.shared_state.agent_last_active["critic"] = 123.0
    try:
        await c.tick(3)

        critic_run.assert_not_awaited()
        assert c.shared_state.agent_last_active["critic"] == 123.0
        assert not [message for message in await c.bus.tail(n=100) if message.from_agent == "critic"]
    finally:
        await c.stop()


async def test_a_proposal_is_reviewed_in_the_tick_that_routes_it(session_dir, monkeypatch):
    backends = silent_backends()
    backends["orchestration"] = MockBackend(
        ScriptedPlan(
            turns=[
                MockTurn(
                    intents=[
                        Intent(
                            type=IntentType.PROPOSE_ACTION,
                            payload={"action_name": "baseline", "predicted_gain_pct": 0.0},
                        )
                    ]
                )
            ]
        ),
        name="orchestration",
    )
    c = Coordinator(session_dir, backends=backends)

    async def _reject(prompt, **_kwargs):
        pending = next(iter(c.state.pending_proposals.values()))
        return BackendTurnResult(
            raw_text="reviewed",
            intents=[
                Intent(
                    type=IntentType.REVIEW_VERDICT,
                    payload={"target_proposal_msg_id": pending.proposal_msg_id, "verdict": "reject"},
                )
            ],
        )

    critic_run = AsyncMock(side_effect=_reject)
    monkeypatch.setattr(c.backends["critic"], "run", critic_run)
    try:
        await c.tick(1)
        assert critic_run.await_count == 1
        assert not c.state.pending_proposals
        last_active = c.shared_state.agent_last_active["critic"]

        await c.tick(1)

        assert critic_run.await_count == 1
        assert c.shared_state.agent_last_active["critic"] == last_active
    finally:
        await c.stop()


async def test_resume_restores_critic_admission_for_an_undecided_proposal(session_dir):
    first = Coordinator(session_dir, backends=silent_backends())
    try:
        pending = await _proposal(first)
    finally:
        await first.stop()

    resumed = Coordinator(session_dir, backends=silent_backends())
    try:
        await resumed.writeback.replay_for_resume()

        assert pending.proposal_msg_id in resumed.state.pending_proposals
        assert await resumed.conversation.admit_turn("critic", time.time())
    finally:
        await resumed.stop()
