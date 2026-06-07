# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Schema validation for :mod:`critic-agent.runtime.request_models`.

The test set is intentionally exhaustive on edge cases because the rest
of the runtime trusts these dataclasses.
"""

from __future__ import annotations

import pytest

from runtime.request_models import (
    COORDINATOR_INBOX,
    DECISION_REQUEST,
    CriticRequest,
    Proposal,
    parse_request,
)
from runtime.errors import RequestValidationError


def test_parse_coordinator_inbox_with_raw_prompt():
    req = parse_request({
        "kind": COORDINATOR_INBOX,
        "session_id": "sess_1",
        "raw_prompt": "=== Inbox for critic ===\n  seq=1 msg_id=abc topic=proposal payload={}",
    })
    assert isinstance(req, CriticRequest)
    assert req.kind == COORDINATOR_INBOX
    assert req.session_id == "sess_1"
    assert req.raw_prompt is not None
    assert req.proposals == []
    assert req.context == {}
    assert req.decision_id is None


def test_parse_coordinator_inbox_requires_raw_prompt():
    with pytest.raises(RequestValidationError, match="raw_prompt"):
        parse_request({
            "kind": COORDINATOR_INBOX,
            "session_id": "sess_1",
        })


def test_parse_decision_request_assigns_decision_id_when_missing():
    req = parse_request({
        "kind": DECISION_REQUEST,
        "session_id": "sess_1",
        "messages": [{"role": "coordinator", "content": "hi"}],
        "context": {"model": "Qwen3-14B", "framework": "sglang"},
        "decision": {"summary": "adopt patch x"},
    })
    assert req.kind == DECISION_REQUEST
    assert req.decision_id is not None
    assert req.decision_id.startswith("dec_")
    assert req.context == {"model": "Qwen3-14B", "framework": "sglang"}
    assert req.messages and req.messages[0]["role"] == "coordinator"


def test_parse_decision_request_keeps_explicit_decision_id():
    req = parse_request({
        "kind": DECISION_REQUEST,
        "session_id": "sess_1",
        "decision_id": "dec_known",
    })
    assert req.decision_id == "dec_known"


def test_parse_unknown_kind_rejected():
    with pytest.raises(RequestValidationError, match="kind"):
        parse_request({"kind": "wat", "session_id": "sess_1"})


def test_parse_missing_session_rejected():
    with pytest.raises(RequestValidationError, match="session_id"):
        parse_request({"kind": COORDINATOR_INBOX, "raw_prompt": "x"})


def test_parse_proposal_normalises_action_and_gain():
    req = parse_request({
        "kind": COORDINATOR_INBOX,
        "session_id": "s",
        "raw_prompt": "x",
        "proposals": [
            {
                "msg_id": "abc",
                "from_agent": "orchestration",
                "seq": 12,
                "payload": {
                    "action_name": "kernel_opt",
                    "predicted_gain_pct": 4.2,
                    "extra": "ok",
                },
            }
        ],
    })
    assert len(req.proposals) == 1
    p = req.proposals[0]
    assert isinstance(p, Proposal)
    assert p.msg_id == "abc"
    assert p.from_agent == "orchestration"
    assert p.action_name == "kernel_opt"
    assert p.predicted_gain_pct == pytest.approx(4.2)
    assert p.payload["extra"] == "ok"
    assert p.seq == 12


def test_parse_proposal_rejects_non_dict_payload():
    with pytest.raises(RequestValidationError, match="payload"):
        parse_request({
            "kind": COORDINATOR_INBOX,
            "session_id": "s",
            "raw_prompt": "x",
            "proposals": [{"msg_id": "abc", "from_agent": "o", "payload": "no"}],
        })


def test_parse_proposal_rejects_non_int_seq():
    with pytest.raises(RequestValidationError, match="seq"):
        parse_request({
            "kind": COORDINATOR_INBOX,
            "session_id": "s",
            "raw_prompt": "x",
            "proposals": [{"msg_id": "abc", "from_agent": "o", "seq": "12"}],
        })


def test_parse_proposal_missing_msg_id():
    with pytest.raises(RequestValidationError, match="msg_id"):
        parse_request({
            "kind": COORDINATOR_INBOX,
            "session_id": "s",
            "raw_prompt": "x",
            "proposals": [{"from_agent": "o"}],
        })


def test_parse_messages_must_be_dicts():
    with pytest.raises(RequestValidationError, match="messages"):
        parse_request({
            "kind": DECISION_REQUEST,
            "session_id": "s",
            "messages": ["not-a-dict"],
        })


def test_to_dict_round_trip_preserves_proposal_fields():
    req = parse_request({
        "kind": COORDINATOR_INBOX,
        "session_id": "s",
        "raw_prompt": "x",
        "proposals": [{
            "msg_id": "m1", "from_agent": "o",
            "payload": {"action_name": "baseline"},
        }],
    })
    d = req.to_dict()
    assert d["proposals"][0]["msg_id"] == "m1"
    assert d["proposals"][0]["action_name"] == "baseline"
    assert d["proposals"][0]["predicted_gain_pct"] is None
