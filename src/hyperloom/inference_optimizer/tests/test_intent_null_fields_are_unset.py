# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A payload field an agent wrote as JSON ``null`` reaches its readers as unset."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    validate_envelope,
)
from hyperloom.orchestrator.roles.mcp_emit_intent import (
    _emit_intent_handler,
    validate_emit_intent_input,
)


def _propose(params: dict) -> dict:
    return {
        "intents": [
            {
                "intent_type": "propose_action",
                "payload": {"action_name": "explore", "predicted_gain_pct": 3.0, "params": params},
            }
        ]
    }


def test_a_null_param_is_dropped_so_its_reader_resolves_a_default():
    """An explore round proposed ``keep_threshold_pct: null`` and died on ``float(None)``.

    ``params.get("keep_threshold_pct", fallback)`` returns the stored ``None``
    while the key is present, so the executor's own fallback never ran and the
    whole measured round was lost.
    """
    (intent,) = validate_envelope(_propose({"keep_threshold_pct": None, "grid": [{"extra_args": "--x"}]}))

    assert "keep_threshold_pct" not in intent.payload["params"]
    assert intent.payload["params"]["grid"] == [{"extra_args": "--x"}]


def test_a_null_removal_list_is_dropped_rather_than_read_as_an_empty_ablation():
    (intent,) = validate_envelope(_propose({"remove_args": None, "extra_args": "--y"}))

    assert "remove_args" not in intent.payload["params"]
    assert intent.payload["params"]["extra_args"] == "--y"


def test_a_null_top_level_field_is_dropped_too():
    (intent,) = validate_envelope(
        {
            "intents": [
                {
                    "intent_type": "propose_action",
                    "payload": {"action_name": "explore", "predicted_gain_pct": 3.0, "rationale": None},
                }
            ]
        }
    )

    assert "rationale" not in intent.payload


def test_a_null_required_field_is_refused_rather_than_carried():
    """Presence alone admitted a null required field; it is now the omission it spells."""
    with pytest.raises(IntentValidationError, match="action_name"):
        validate_envelope(
            {
                "intents": [
                    {"intent_type": "propose_action", "payload": {"action_name": None, "predicted_gain_pct": 1.0}}
                ]
            }
        )


def test_a_payload_that_sets_everything_is_untouched():
    params = {"keep_threshold_pct": 2.0, "remove_args": ["--disable-radix-cache"]}

    (intent,) = validate_envelope(_propose(params))

    assert intent.payload["params"] == params


def test_a_null_nested_deeper_than_params_is_left_alone():
    """Only the envelope's own fields are normalised; a grid entry carries whatever it carries."""
    (intent,) = validate_envelope(_propose({"grid": [{"extra_args": "--x", "note": None}]}))

    assert intent.payload["params"]["grid"] == [{"extra_args": "--x", "note": None}]


@pytest.mark.asyncio
async def test_the_tool_ack_agrees_with_the_harvest_on_a_null_required_field():
    """The ack is what the agent retries on, so it may not be more permissive than the harvest.

    ``_emit_intent_handler`` acked ``ok`` for ``action_name: null`` while
    ``validate_envelope`` dropped the intent, leaving the agent told it had
    succeeded with nothing to correct.
    """
    result = await _emit_intent_handler(
        {"intent_type": "propose_action", "payload": {"action_name": None, "predicted_gain_pct": 1.0}}
    )

    assert result["is_error"] is True
    assert "action_name" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_the_tool_ack_reaches_the_value_checks_the_harvest_applies():
    """The eager path stopped at presence, so a bad severity or verdict was acked and then lost."""
    result = await _emit_intent_handler(
        {"intent_type": "alert", "payload": {"severity": "catastrophic", "summary": "s"}}
    )

    assert result["is_error"] is True
    assert "severity" in result["content"][0]["text"]


def test_the_eager_validator_still_accepts_what_the_harvest_accepts():
    validate_emit_intent_input({"intent_type": "send_message", "payload": {"topic": "heartbeat"}})
