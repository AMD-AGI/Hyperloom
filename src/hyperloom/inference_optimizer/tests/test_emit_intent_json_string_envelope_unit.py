# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the JSON-string ``emit_intent`` envelope.

OpenAI-compatible / litellm-style proxies deliver ``tool_use.input`` as a JSON
*string* rather than a native object, sometimes with ``payload`` stringified as
well. Before the fix these shapes decoded to an empty intent with
``decode_error is None``, so the orchestrator silently made no progress.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.orchestrator.roles.mcp_emit_intent import (
    decode_emit_intent_input,
)

_NATIVE = {"intent_type": "run_action", "payload": {"action": "baseline"}}


def test_json_string_envelope_is_decoded() -> None:
    decoded, error = decode_emit_intent_input(json.dumps(_NATIVE))
    assert error is None
    assert decoded == _NATIVE


def test_stringified_payload_is_decoded() -> None:
    envelope = {"intent_type": "run_action", "payload": json.dumps({"action": "baseline"})}
    decoded, error = decode_emit_intent_input(envelope)
    assert error is None
    assert decoded["payload"] == {"action": "baseline"}


def test_json_string_envelope_with_stringified_payload() -> None:
    envelope = json.dumps({"intent_type": "run_action", "payload": json.dumps({"action": "baseline"})})
    decoded, error = decode_emit_intent_input(envelope)
    assert error is None
    assert decoded == _NATIVE


def test_native_object_is_unchanged() -> None:
    decoded, error = decode_emit_intent_input(dict(_NATIVE))
    assert error is None
    assert decoded == _NATIVE


def test_unparsed_tool_input_wrapper_still_unwraps() -> None:
    wrapper = {"__unparsedToolInput": {"raw": json.dumps(_NATIVE)}}
    decoded, error = decode_emit_intent_input(wrapper)
    assert error is None
    assert decoded == _NATIVE


def test_json_string_wrapping_an_unparsed_wrapper() -> None:
    wrapper = json.dumps({"__unparsedToolInput": {"raw": json.dumps(_NATIVE)}})
    decoded, error = decode_emit_intent_input(wrapper)
    assert error is None
    assert decoded == _NATIVE


@pytest.mark.parametrize(
    "bad",
    ["not json at all", "{unbalanced", ""],
)
def test_malformed_string_reports_an_error(bad: str) -> None:
    decoded, error = decode_emit_intent_input(bad)
    assert decoded == {}
    assert error is not None
    assert "not valid JSON" in error


@pytest.mark.parametrize("scalar", ["[1, 2, 3]", '"just a string"', "42"])
def test_non_object_json_string_reports_an_error(scalar: str) -> None:
    decoded, error = decode_emit_intent_input(scalar)
    assert decoded == {}
    assert error is not None
    assert "did not decode to a JSON object" in error


@pytest.mark.parametrize("value", [None, 42, [1, 2, 3], object()])
def test_non_dict_non_str_reports_an_error(value: object) -> None:
    decoded, error = decode_emit_intent_input(value)
    assert decoded == {}
    assert error is not None
    assert "must be an object" in error


def test_unparseable_payload_string_is_left_alone() -> None:
    envelope = {"intent_type": "run_action", "payload": "not json"}
    decoded, error = decode_emit_intent_input(envelope)
    assert error is None
    assert decoded["payload"] == "not json"


def test_regression_empty_input_no_longer_masks_a_dropped_envelope() -> None:
    """The pre-fix caller coerced a string to ``{}``; that decoded silently.

    ``{}`` itself must still decode without error (it is a legitimate, if
    empty, native object), but the string envelope it used to replace must now
    survive intact -- which is what makes the drop observable rather than
    silent.
    """
    assert decode_emit_intent_input({}) == ({}, None)
    decoded, error = decode_emit_intent_input(json.dumps(_NATIVE))
    assert error is None
    assert decoded["intent_type"] == "run_action"
