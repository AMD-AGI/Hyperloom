# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for small pure-logic critic-runtime helpers."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.category_mapping import filter_supported_categories, map_category_to_kind
from hyperloom.agents.critic.runtime.errors import (
    IntentEnvelopeValidationError,
    RuntimeAdapterError,
    SlugifyError,
)
from hyperloom.agents.critic.runtime.importance_mapping import (
    cap_importance,
    importance_for_kb_draft,
    importance_for_verdict,
)
from hyperloom.agents.critic.runtime.intent_envelope import validate_envelope
from hyperloom.agents.critic.runtime.slugify import slugify, slugify_safe


# --------------------------------------------------------------------------- #
# category_mapping                                                            #
# --------------------------------------------------------------------------- #
def test_map_category_to_kind() -> None:
    assert map_category_to_kind("pitfall") == "pitfall"
    assert map_category_to_kind("kernel_optimization") == "technique"
    with pytest.raises(RuntimeAdapterError):
        map_category_to_kind(123)  # type: ignore[arg-type]  # line 60
    with pytest.raises(RuntimeAdapterError):
        map_category_to_kind("no_such_category")


def test_filter_supported_categories() -> None:
    supported, rejected = filter_supported_categories(["pitfall", "bogus", "server_params"])
    assert supported == ["pitfall", "server_params"]
    assert rejected == ["bogus"]


# --------------------------------------------------------------------------- #
# importance_mapping                                                          #
# --------------------------------------------------------------------------- #
def test_importance_for_verdict() -> None:
    assert importance_for_verdict(verdict="advise") == 0.4
    assert importance_for_verdict(verdict="approve", confidence="high", has_measurement=True) == 0.7
    assert importance_for_verdict(verdict="approve", confidence="high", has_measurement=False) == 0.4
    # Low confidence branch (line 52).
    assert importance_for_verdict(verdict="approve", confidence="low") == 0.4
    assert importance_for_verdict(verdict="approve", confidence="medium", has_measurement=True) == 0.5
    assert importance_for_verdict(verdict="approve", confidence=None) == 0.4


def test_importance_for_kb_draft() -> None:
    assert importance_for_kb_draft(confidence=None) == 0.5
    # High-confidence draft (line 74).
    assert importance_for_kb_draft(confidence=0.9) == 0.6
    assert importance_for_kb_draft(confidence=0.5) == 0.5


def test_cap_importance() -> None:
    assert cap_importance(2.0) == 0.84
    assert cap_importance(-1.0) == 0.0
    assert cap_importance(0.3) == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# slugify                                                                     #
# --------------------------------------------------------------------------- #
def test_slugify_errors_and_success() -> None:
    assert slugify("Hello World Optimization") == "hello-world-optimization"
    with pytest.raises(SlugifyError):
        slugify(123)  # type: ignore[arg-type]  # line 63
    with pytest.raises(SlugifyError):
        slugify("   ")
    with pytest.raises(SlugifyError):
        slugify("héllo-non-ascii-ünïcode")
    with pytest.raises(SlugifyError):
        slugify("short")  # below the minimum length


def test_slugify_safe_paths() -> None:
    assert slugify_safe("plain-ascii-topic") == "plain-ascii-topic"
    with pytest.raises(SlugifyError):
        slugify_safe("   ")  # line 115

    # translate_fn that raises -> deterministic fallback (lines 127-128 via except).
    def _boom(_t: str) -> str:
        raise RuntimeError("translate failed")

    out = slugify_safe("日本語のトピック", _boom, fallback_prefix="auto")
    assert out.startswith("auto-")

    # No translate_fn for non-ascii -> hash fallback.
    out2 = slugify_safe("日本語のトピック")
    assert out2.startswith("auto-")

    # translate_fn returning too-short text -> SlugifyError caught -> fallback.
    out3 = slugify_safe("日本語", lambda _t: "x", fallback_prefix="p")
    assert out3.startswith("p-")


# --------------------------------------------------------------------------- #
# intent_envelope.validate_envelope                                           #
# --------------------------------------------------------------------------- #
def test_validate_envelope_structural() -> None:
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope("nope")  # type: ignore[arg-type]  # line 186
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({})  # line 188
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": []})  # empty list
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": ["x"]})  # line 195
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "alert"}]})  # line 197
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "not_allowed", "payload": {}}]})


def test_validate_envelope_payload_rules() -> None:
    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope({"intents": [{"intent_type": "alert", "payload": "x"}]})

    with pytest.raises(IntentEnvelopeValidationError):
        validate_envelope(
            {"intents": [{"intent_type": "review_verdict", "payload": {"target_proposal_msg_id": "m1"}}]}
        )

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
    env = validate_envelope(
        {"intents": [{"intent_type": "send_message", "payload": {"topic": "hi", "body_md": "x"}}]}
    )
    assert env.intents[0].intent_type == "send_message"
