# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for small pure-logic critic-runtime helpers."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import (
    IntentEnvelopeValidationError,
    SlugifyError,
)
from hyperloom.agents.critic.runtime.importance_mapping import (
    cap_importance,
    importance_for_verdict,
)
from hyperloom.agents.critic.runtime.intent_envelope import validate_envelope
from hyperloom.agents.critic.runtime.slugify import slugify, slugify_safe


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


# slugify
def test_slugify_errors_and_success() -> None:
    assert slugify("Hello World Optimization") == "hello-world-optimization"
    with pytest.raises(SlugifyError):
        slugify(123)  # type: ignore[arg-type]
    with pytest.raises(SlugifyError):
        slugify("   ")
    with pytest.raises(SlugifyError):
        slugify("héllo-non-ascii-ünïcode")
    with pytest.raises(SlugifyError):
        slugify("short")  # below the minimum length


def test_slugify_safe_paths() -> None:
    assert slugify_safe("plain-ascii-topic") == "plain-ascii-topic"
    with pytest.raises(SlugifyError):
        slugify_safe("   ")

    # Non-ascii -> hash fallback.
    out2 = slugify_safe("Ελληνικά θέματα")
    assert out2.startswith("auto-")


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
