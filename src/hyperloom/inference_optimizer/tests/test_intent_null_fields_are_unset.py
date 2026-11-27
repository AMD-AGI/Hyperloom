# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A payload field an agent wrote as JSON ``null`` reaches its readers as unset."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    validate_envelope,
)
from hyperloom.orchestrator.roles.mcp_emit_intent import build_emit_intent_server, validate_emit_intent_input


def _delegate(params: dict) -> dict:
    return {"intents": [{"intent_type": "delegate", "payload": {"action_name": "explore", "params": params}}]}


def test_a_null_param_is_dropped_so_its_reader_resolves_a_default():
    """An explore round asked for ``keep_threshold_pct: null`` and died on ``float(None)``.

    ``params.get("keep_threshold_pct", fallback)`` returns the stored ``None``
    while the key is present, so the executor's own fallback never ran and the
    whole measured round was lost.
    """
    (intent,) = validate_envelope(_delegate({"keep_threshold_pct": None, "grid": [{"extra_args": "--x"}]}))

    assert "keep_threshold_pct" not in intent.payload["params"]
    assert intent.payload["params"]["grid"] == [{"extra_args": "--x"}]


def test_a_null_removal_list_is_dropped_rather_than_read_as_an_empty_ablation():
    (intent,) = validate_envelope(_delegate({"remove_args": None, "extra_args": "--y"}))

    assert "remove_args" not in intent.payload["params"]
    assert intent.payload["params"]["extra_args"] == "--y"


def test_a_null_top_level_field_is_dropped_too():
    (intent,) = validate_envelope(
        {"intents": [{"intent_type": "delegate", "payload": {"action_name": "explore", "rationale": None}}]}
    )

    assert "rationale" not in intent.payload


def test_a_null_required_field_is_refused_rather_than_carried():
    """Presence alone admitted a null required field; it is now the omission it spells."""
    with pytest.raises(IntentValidationError, match="action_name"):
        validate_envelope({"intents": [{"intent_type": "delegate", "payload": {"action_name": None}}]})


def test_a_payload_that_sets_everything_is_untouched():
    params = {"keep_threshold_pct": 2.0, "remove_args": ["--disable-radix-cache"]}

    (intent,) = validate_envelope(_delegate(params))

    assert intent.payload["params"] == params


def test_a_null_nested_deeper_than_params_is_left_alone():
    """Only the envelope's own fields are normalised; a grid entry carries whatever it carries."""
    (intent,) = validate_envelope(_delegate({"grid": [{"extra_args": "--x", "note": None}]}))

    assert intent.payload["params"]["grid"] == [{"extra_args": "--x", "note": None}]


async def _emit(claude_sdk, tool_input: dict) -> tuple[dict, list]:
    """Answer one ``emit_intent`` call; the tool's reply and the intents it kept."""
    kept: list = []
    server = build_emit_intent_server(claude_sdk, kept)
    return await server["tools"][0].handler(tool_input), kept


@pytest.mark.asyncio
async def test_the_tool_refuses_a_null_required_field_and_keeps_no_intent(claude_sdk):
    """The ack is what the agent retries on: a null required field is an error it can correct, never an ``ok``."""
    result, kept = await _emit(claude_sdk, {"intent_type": "delegate", "payload": {"action_name": None}})

    assert result["is_error"] is True
    assert "action_name" in result["content"][0]["text"]
    assert kept == []


@pytest.mark.asyncio
async def test_the_tool_refuses_a_value_outside_its_set_and_keeps_no_intent(claude_sdk):
    result, kept = await _emit(
        claude_sdk, {"intent_type": "alert", "payload": {"severity": "catastrophic", "summary": "s"}}
    )

    assert result["is_error"] is True
    assert "severity" in result["content"][0]["text"]
    assert kept == []


def test_the_tool_validator_returns_the_intent_the_envelope_rule_builds():
    tool_input = {"intent_type": "send_message", "payload": {"topic": "heartbeat", "body_md": None}}

    assert validate_emit_intent_input(tool_input) == validate_envelope({"intents": [tool_input]})[0]
