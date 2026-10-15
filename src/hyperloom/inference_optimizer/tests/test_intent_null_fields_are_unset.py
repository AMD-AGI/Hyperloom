# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A payload field an agent wrote as JSON ``null`` reaches its readers as unset."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    validate_envelope,
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
