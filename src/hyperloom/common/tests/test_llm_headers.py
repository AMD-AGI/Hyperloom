# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract for the ``*_CUSTOM_HEADERS`` wire format both protocol lines read."""

from __future__ import annotations

import logging

import pytest

from hyperloom.common.llm_headers import (
    expand_env_refs,
    format_custom_headers,
    parse_custom_headers,
)


def test_json_object_form() -> None:
    assert parse_custom_headers('{"Ocp-Apim-Subscription-Key": " abc123 ", "": "drop"}') == {
        "Ocp-Apim-Subscription-Key": "abc123"
    }


def test_line_form() -> None:
    assert parse_custom_headers("Ocp-Apim-Subscription-Key: abc123\nX-Team: hyperloom") == {
        "Ocp-Apim-Subscription-Key": "abc123",
        "X-Team": "hyperloom",
    }


def test_text_that_only_looks_like_json_falls_back_to_lines() -> None:
    assert parse_custom_headers("{not json}\nX-Fallback: yes") == {"X-Fallback": "yes"}


def test_a_json_array_names_no_header() -> None:
    """Well-formed JSON that cannot be a header map yields nothing, not punctuation."""
    assert parse_custom_headers('["X-Team: hyperloom"]') == {}


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\n"])
def test_nothing_configured_is_no_headers(raw: str | None) -> None:
    assert parse_custom_headers(raw) == {}


def test_env_refs_resolve_against_the_supplied_mapping() -> None:
    assert parse_custom_headers("Key: ${SUBKEY}", env={"SUBKEY": "xyz"}) == {"Key": "xyz"}


def test_env_refs_resolve_against_the_process_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HL_TEST_SUBKEY", "xyz")
    assert expand_env_refs("Key: ${HL_TEST_SUBKEY}") == "Key: xyz"
    monkeypatch.delenv("HL_TEST_SUBKEY")
    assert expand_env_refs("Key: ${HL_TEST_SUBKEY}") == "Key: "
    assert expand_env_refs("no refs here") == "no refs here"


def test_an_unresolved_reference_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_custom_headers("Key: ${NOPE}", env={}) == {"Key": ""}
    assert "empty value" in caplog.text


def test_headers_packed_onto_one_line_are_reported_not_split(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_custom_headers("user: alice, x-foo: bar") == {"user": "alice, x-foo: bar"}
    assert "one" in caplog.text


def test_a_comma_inside_one_value_is_left_alone(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_custom_headers("Accept: text/html, application/json") == {"Accept": "text/html, application/json"}
    assert not caplog.text


def test_a_line_without_a_colon_is_dropped_and_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_custom_headers("user: alice\nnonsense") == {"user": "alice"}
    assert "colon" in caplog.text


def test_format_round_trips_the_line_form() -> None:
    raw = "A: 1\nB: 2"
    assert format_custom_headers(parse_custom_headers(raw)) == raw
    assert format_custom_headers({}) == ""
