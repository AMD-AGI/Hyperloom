# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validation + parity tests for :mod:`runtime.intent_envelope`."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import IntentEnvelopeValidationError
from hyperloom.agents.critic.runtime.intent_envelope import (
    ALLOWED_VERDICT_SOURCES,
    ReviewVerdict,
    build_advice_intent,
    build_envelope,
    build_idle_intent,
    build_review_verdict_intent,
)
from hyperloom.inference_optimizer.protocol.intent import ALLOWED_VERDICTS, DEFAULT_IDLE_BODY


def test_allowed_verdicts_match_design_v06():
    assert ALLOWED_VERDICTS == frozenset({"approve", "reject", "redirect", "advise", "needs_review"})


def test_allowed_verdict_sources_match_schema():
    assert ALLOWED_VERDICT_SOURCES == frozenset({"critic", "mock", "timeout", "critic_unavailable"})


def test_build_review_verdict_intent_minimal():
    intent = build_review_verdict_intent(ReviewVerdict(target_proposal_msg_id="abc1", verdict="approve"))
    assert intent.intent_type == "review_verdict"
    assert intent.payload["target_proposal_msg_id"] == "abc1"
    assert intent.payload["verdict"] == "approve"
    assert intent.payload["source"] == "critic"
    assert intent.payload["risks"] == []


def test_build_review_verdict_intent_full():
    intent = build_review_verdict_intent(
        ReviewVerdict(
            target_proposal_msg_id="abc1",
            verdict="redirect",
            reasoning="prefer dispatch fix first",
            confidence="high",
            predicted_gain_pct=2.5,
            kb_evidence=["kb_1"],
            packet_evidence=["benchmark.after.gain_pct"],
            risks=[{"type": "active_path_unproven", "severity": "blocker"}],
            required_evidence=["dispatch_evidence"],
            alternative_action="profile",
            notes=["see kb_1"],
        )
    )
    p = intent.payload
    assert p["confidence"] == "high"
    assert p["alternative_action"] == "profile"
    assert p["risks"][0]["type"] == "active_path_unproven"


def test_build_review_verdict_rejects_bad_source():
    with pytest.raises(IntentEnvelopeValidationError, match="source"):
        build_review_verdict_intent(ReviewVerdict(target_proposal_msg_id="abc1", verdict="approve", source="someone"))


def test_build_idle_intent_default():
    h = build_idle_intent()
    assert h.intent_type == "send_message"
    assert h.payload["topic"] == "observation"
    assert h.payload["body_md"] == DEFAULT_IDLE_BODY


def test_build_envelope_wraps_intents_in_order():
    env = build_envelope(
        [
            build_review_verdict_intent(ReviewVerdict(target_proposal_msg_id="m1", verdict="approve")),
            build_review_verdict_intent(ReviewVerdict(target_proposal_msg_id="m2", verdict="reject")),
        ]
    )
    d = env.to_dict()
    assert len(d["intents"]) == 2
    assert d["intents"][0]["payload"]["target_proposal_msg_id"] == "m1"


def test_advice_intent_carries_optional_target():
    intent = build_advice_intent("watch dispatch", target_proposal_msg_id="m1")
    assert intent.payload["topic"] == "advice"
    assert intent.payload["about_proposal_msg_id"] == "m1"
