# Copyright Advanced Micro Devices, Inc. All rights reserved.

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


# --------------------------------------------------------------------------- #
# inbox_parser                                                                #
# --------------------------------------------------------------------------- #
def test_parse_shared_state_empty_and_trailing_punct() -> None:
    assert _parse_shared_state("   \n  ") == {}  # line 156
    parsed = _parse_shared_state("phase=EXPLORE, cycle=2;")
    assert parsed["phase"] == "EXPLORE"  # trailing comma trimmed (line 165)
    assert parsed["cycle"] == "2"


def test_try_parse_payload_paths() -> None:
    assert _try_parse_payload("") is None  # line 187
    assert _try_parse_payload("{'a': 1}") == {"a": 1}  # python repr form
    assert _try_parse_payload('{"b": 2}') == {"b": 2}  # JSON fallback (line 201)
    assert _try_parse_payload("not-a-dict") is None
    assert _try_parse_payload("[1, 2]") is None


def test_agent_from_inbox_title() -> None:
    assert _agent_from_inbox_title("Inbox for critic (newest last)") == "critic"
    assert _agent_from_inbox_title("Some Other Section") is None  # line 247


def test_parse_inbox_prompt_type_error() -> None:
    with pytest.raises(InboxParseError):
        parse_inbox_prompt(123)  # type: ignore[arg-type]  # line 272


def test_parse_inbox_prompt_preamble_and_sections() -> None:
    text = (
        "preamble line before any section\n"  # line 277 (title None -> continue)
        "=== Shared session state ===\n"
        "phase=EXPLORE cycle=1\n"
    )
    parsed = parse_inbox_prompt(text)
    # to_dict round-trips (lines 82/123).
    d = parsed.to_dict()
    assert d["shared_state"].get("phase") == "EXPLORE"
