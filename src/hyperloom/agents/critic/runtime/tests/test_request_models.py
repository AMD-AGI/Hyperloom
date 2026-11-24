# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema validation for :mod:`runtime.request_models`."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.request_models import (
    CriticRequest,
    Proposal,
    _optional_str,
    _require_str,
    parse_request,
)
from hyperloom.agents.critic.runtime.errors import RequestValidationError


def test_parse_coordinator_inbox_with_raw_prompt():
    req = parse_request(
        {
            "session_id": "sess_1",
            "raw_prompt": "=== Inbox for critic ===\n  seq=1 msg_id=abc topic=proposal payload={}",
        }
    )
    assert isinstance(req, CriticRequest)
    assert req.session_id == "sess_1"
    assert req.raw_prompt is not None
    assert req.context == {}


def test_parse_coordinator_inbox_requires_raw_prompt():
    with pytest.raises(RequestValidationError, match="raw_prompt"):
        parse_request({"session_id": "sess_1"})


def test_parse_missing_session_rejected():
    with pytest.raises(RequestValidationError, match="session_id"):
        parse_request({"raw_prompt": "x"})


def test_proposal_from_payload_normalises_action_and_gain():
    p = Proposal.from_payload(
        msg_id="abc",
        from_agent="orchestration",
        payload={"action_name": "kernel_opt", "predicted_gain_pct": 4.2, "extra": "ok"},
        seq=12,
    )
    assert p.action_name == "kernel_opt"
    assert p.predicted_gain_pct == pytest.approx(4.2)
    assert p.payload["extra"] == "ok"
    assert p.seq == 12


def test_proposal_to_dict_carries_parse_defaults():
    d = Proposal.from_payload(msg_id="m1", from_agent="o", payload={"action_name": "baseline"}, seq=None).to_dict()
    assert d["msg_id"] == "m1"
    assert d["action_name"] == "baseline"
    assert d["predicted_gain_pct"] is None


def test_require_and_optional_validators() -> None:
    with pytest.raises(RequestValidationError):
        _require_str({"k": 5}, "k", where="t")
    with pytest.raises(RequestValidationError):
        _optional_str({"k": 5}, "k", where="t")
    assert _optional_str({}, "k", where="t") is None


def test_parse_request_errors() -> None:
    with pytest.raises(RequestValidationError):
        parse_request("not a dict")  # type: ignore[arg-type]
