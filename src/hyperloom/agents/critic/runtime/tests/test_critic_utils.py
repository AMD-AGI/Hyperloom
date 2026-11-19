# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for small pure-logic critic-runtime helpers."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import IntentEnvelopeValidationError
from hyperloom.agents.critic.runtime.importance_mapping import (
    cap_importance,
    importance_for_verdict,
)
from hyperloom.agents.critic.runtime.intent_envelope import validate_envelope


# importance_mapping tests
def test_importance_for_verdict() -> None:
    assert importance_for_verdict(verdict="advise") == 0.4
    assert importance_for_verdict(verdict="approve", confidence="high", has_measurement=True) == 0.7
    assert importance_for_verdict(verdict="approve", confidence="high", has_measurement=False) == 0.4
    assert importance_for_verdict(verdict="approve", confidence="low") == 0.4
    assert importance_for_verdict(verdict="approve", confidence="medium", has_measurement=True) == 0.5
    assert importance_for_verdict(verdict="approve", confidence=None) == 0.4


def test_cap_importance() -> None:
    assert cap_importance(2.0) == 0.84
    assert cap_importance(-1.0) == 0.0
    assert cap_importance(0.3) == pytest.approx(0.3)


# intent_envelope.validate_envelope
def test_validate_envelope_structural() -> None:
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope("nope")  # type: ignore[arg-type]
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({})
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": []})  # empty list
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": ["x"]})
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "alert"}]})
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "not_allowed", "payload": {}}]})


def test_validate_envelope_payload_rules() -> None:
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "alert", "payload": "x"}]})

    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "review_verdict", "payload": {"target_proposal_msg_id": "m1"}}]})

    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope(
            {
                "intents": [
                    {
                        "intent_type": "review_verdict",
                        "payload": {
                            "verdict": "approve",
                            "verdict_map": {"v1": {"verdict": "reject"}},
                            "target_proposal_msg_id": "m1",
                        },
                    }
                ]
            }
        )


def test_validate_envelope_happy() -> None:
    env = validate_envelope({"intents": [{"intent_type": "send_message", "payload": {"topic": "hi", "body_md": "x"}}]})
    assert env.intents[0].intent_type == "send_message"
