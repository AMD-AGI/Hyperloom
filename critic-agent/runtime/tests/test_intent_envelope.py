"""Validation + parity tests for :mod:`runtime.intent_envelope`.

The parity assertions mirror the Coordinator-side schema in
``inference_optimizer/protocol/intent.py`` (REVIEW_VERDICT
required keys) and ``policy.REVIEW_VERDICTS`` so a future Coordinator
upgrade fails loudly here first.
"""

from __future__ import annotations

import pytest

from runtime.errors import IntentEnvelopeValidationError
from runtime.intent_envelope import (
    ALLOWED_VERDICTS,
    ALLOWED_VERDICT_SOURCES,
    DEFAULT_HEARTBEAT_TOPIC,
    Intent,
    IntentEnvelope,
    build_advice_intent,
    build_envelope,
    build_heartbeat_intent,
    build_review_verdict_intent,
    validate_envelope,
)


def test_allowed_verdicts_match_design_v06():
    assert ALLOWED_VERDICTS == frozenset(
        {"approve", "reject", "redirect", "advise", "needs_review"}
    )


def test_allowed_verdict_sources_match_schema():
    assert ALLOWED_VERDICT_SOURCES == frozenset(
        {"critic", "mock", "timeout", "critic_unavailable"}
    )


def test_build_review_verdict_intent_minimal():
    intent = build_review_verdict_intent(
        target_proposal_msg_id="abc1",
        verdict="approve",
    )
    assert intent.intent_type == "review_verdict"
    assert intent.payload["target_proposal_msg_id"] == "abc1"
    assert intent.payload["verdict"] == "approve"
    assert intent.payload["source"] == "critic"
    assert intent.payload["risks"] == []


def test_build_review_verdict_intent_full():
    intent = build_review_verdict_intent(
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
    p = intent.payload
    assert p["confidence"] == "high"
    assert p["alternative_action"] == "profile"
    assert p["risks"][0]["type"] == "active_path_unproven"


def test_build_review_verdict_rejects_bad_verdict():
    with pytest.raises(IntentEnvelopeValidationError, match="verdict"):
        build_review_verdict_intent(
            target_proposal_msg_id="abc1", verdict="lgtm"
        )


def test_build_review_verdict_rejects_bad_source():
    with pytest.raises(IntentEnvelopeValidationError, match="source"):
        build_review_verdict_intent(
            target_proposal_msg_id="abc1", verdict="approve", source="someone"
        )


def test_build_review_verdict_rejects_empty_target():
    with pytest.raises(IntentEnvelopeValidationError, match="target"):
        build_review_verdict_intent(
            target_proposal_msg_id="", verdict="approve"
        )


def test_build_heartbeat_intent_default():
    h = build_heartbeat_intent()
    assert h.intent_type == "send_message"
    assert h.payload["topic"] == DEFAULT_HEARTBEAT_TOPIC


def test_build_envelope_with_intents_validates():
    env = build_envelope([
        build_review_verdict_intent(target_proposal_msg_id="m1", verdict="approve"),
        build_review_verdict_intent(target_proposal_msg_id="m2", verdict="reject"),
    ])
    d = env.to_dict()
    assert len(d["intents"]) == 2
    assert d["intents"][0]["payload"]["target_proposal_msg_id"] == "m1"


def test_build_envelope_empty_falls_back_to_heartbeat():
    env = build_envelope([])
    d = env.to_dict()
    assert len(d["intents"]) == 1
    assert d["intents"][0]["intent_type"] == "send_message"
    assert d["intents"][0]["payload"]["topic"] == DEFAULT_HEARTBEAT_TOPIC


def test_validate_envelope_accepts_well_formed_dict():
    env = validate_envelope({
        "intents": [
            {
                "intent_type": "review_verdict",
                "payload": {
                    "target_proposal_msg_id": "abc",
                    "verdict": "approve",
                    "source": "critic",
                    "reasoning": "ok",
                },
            }
        ]
    })
    assert isinstance(env, IntentEnvelope)
    assert env.intents[0].payload["verdict"] == "approve"


def test_validate_envelope_rejects_disallowed_intent():
    with pytest.raises(IntentEnvelopeValidationError, match="not.*Critic"):
        validate_envelope({
            "intents": [
                {
                    "intent_type": "delegate",
                    "payload": {"action_name": "baseline"},
                }
            ]
        })


def test_validate_envelope_rejects_missing_payload_key():
    with pytest.raises(IntentEnvelopeValidationError, match="target"):
        validate_envelope({
            "intents": [
                {"intent_type": "review_verdict", "payload": {"verdict": "approve"}}
            ]
        })


def test_validate_envelope_rejects_empty_list():
    with pytest.raises(IntentEnvelopeValidationError, match="non-empty"):
        validate_envelope({"intents": []})


def test_validate_envelope_rejects_unknown_verdict():
    with pytest.raises(IntentEnvelopeValidationError, match="verdict"):
        validate_envelope({
            "intents": [
                {
                    "intent_type": "review_verdict",
                    "payload": {
                        "target_proposal_msg_id": "x",
                        "verdict": "yes",
                    },
                }
            ]
        })


def test_advice_intent_carries_optional_target():
    intent = build_advice_intent("watch dispatch", target_proposal_msg_id="m1")
    assert intent.payload["topic"] == "advice"
    assert intent.payload["about_proposal_msg_id"] == "m1"


def test_intent_to_dict_round_trip():
    a = Intent(intent_type="send_message", payload={"topic": "heartbeat"})
    out = validate_envelope({"intents": [a.to_dict()]})
    assert out.intents[0].intent_type == "send_message"
