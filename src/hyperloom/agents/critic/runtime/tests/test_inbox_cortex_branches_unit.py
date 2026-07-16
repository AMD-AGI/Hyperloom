# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch coverage for inbox_parser helpers."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.errors import InboxParseError
from hyperloom.agents.critic.runtime.inbox_parser import (
    _agent_from_inbox_title,
    _parse_shared_state,
    _try_parse_payload,
    parse_inbox_prompt,
)


def test_parse_shared_state_empty_and_trailing_punct() -> None:
    assert _parse_shared_state("   \n  ") == {}
    parsed = _parse_shared_state("phase=EXPLORE, cycle=2;")
    assert parsed["phase"] == "EXPLORE"
    assert parsed["cycle"] == "2"


def test_try_parse_payload_paths() -> None:
    assert _try_parse_payload("") is None
    assert _try_parse_payload("{'a': 1}") == {"a": 1}
    assert _try_parse_payload('{"b": 2}') == {"b": 2}
    assert _try_parse_payload("not-a-dict") is None
    assert _try_parse_payload("[1, 2]") is None


def test_agent_from_inbox_title() -> None:
    assert _agent_from_inbox_title("Inbox for critic (newest last)") == "critic"
    assert _agent_from_inbox_title("Some Other Section") is None


def test_parse_inbox_prompt_type_error() -> None:
    with pytest.raises(InboxParseError):
        parse_inbox_prompt(123)  # type: ignore[arg-type]


def test_parse_inbox_prompt_preamble_and_sections() -> None:
    text = (
        "preamble line before any section\n"
        "=== Shared session state ===\n"
        "phase=EXPLORE cycle=1\n"
    )
    parsed = parse_inbox_prompt(text)
    d = parsed.to_dict()
    assert d["shared_state"].get("phase") == "EXPLORE"
