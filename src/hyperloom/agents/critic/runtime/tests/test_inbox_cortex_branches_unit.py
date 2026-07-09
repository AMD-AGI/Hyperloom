# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for inbox_parser + cortex_kb_client helpers."""

from __future__ import annotations

import urllib.error

import pytest

from hyperloom.agents.critic.runtime.cortex_kb_client import CortexKBClient, _normalise_value
from hyperloom.agents.critic.runtime.errors import (
    InboxParseError,
    KBConflictError,
    KBTransportError,
    KBValidationError,
)
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


# --------------------------------------------------------------------------- #
# cortex_kb_client #
# --------------------------------------------------------------------------- #
def test_cortex_normalise_and_init() -> None:
    assert _normalise_value(None) == ""  # line 77
    with pytest.raises(ValueError):
        CortexKBClient("")  # line 122


def test_cortex_strip_kind() -> None:
    assert CortexKBClient._strip_kind("critic_pitfall") == "pitfall"  # lines 218-219
    assert CortexKBClient._strip_kind("plain") == "plain"
    assert CortexKBClient._strip_kind(None) == ""


def test_cortex_writes_rejected() -> None:
    client = CortexKBClient("http://kb.local")
    with pytest.raises(KBValidationError):
        client.upsert({})
    with pytest.raises(KBValidationError):
        client.batch_insert([])
    with pytest.raises(KBValidationError):
        client.add_edges([])


def test_cortex_list_with_metadata_filter(monkeypatch) -> None:
    client = CortexKBClient("http://kb.local")
    captured = {}

    def fake_post(path, body, *, endpoint_label):
        captured["body"] = body
        return {
            "points": [
                {
                    "id": "p1",
                    "attrs": {
                        "_critic_scope": {"model": "m"},
                        "_critic_kind": "technique",
                        "_critic_slug": "s",
                        "_critic_updated_at": 5.0,
                    },
                    "kind": "critic_technique",
                    "is_active": True,
                }
            ]
        }

    monkeypatch.setattr(client, "_post", fake_post)
    out = client.list(scope_filter={"model": "M"}, metadata_filter={"x": 1}, limit=10)
    assert out["count"] == 1
    assert "_critic_metadata" in captured["body"]["attrs_filter"]  # line 159


def test_cortex_post_409_and_urlerror(monkeypatch) -> None:
    import io

    client = CortexKBClient("http://kb.local", token="tok", retry_max=1, sleep_fn=lambda _s: None)

    def conflict(req, timeout=None):
        raise urllib.error.HTTPError("u", 409, "x", {}, io.BytesIO(b"dup"))  # type: ignore[arg-type]

    monkeypatch.setattr("hyperloom.agents.critic.runtime.cortex_kb_client.urllib.request.urlopen", conflict)
    with pytest.raises(KBConflictError):  # line 293
        client._post("/v1/points/query", {}, endpoint_label="list")

    def neterr(req, timeout=None):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("hyperloom.agents.critic.runtime.cortex_kb_client.urllib.request.urlopen", neterr)
    with pytest.raises(KBTransportError):  # lines 297-298 -> transport
        client._post("/v1/points/query", {}, endpoint_label="list")
