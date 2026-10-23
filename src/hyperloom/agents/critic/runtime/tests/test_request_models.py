# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema validation for :mod:`runtime.request_models`."""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.request_models import (
    COORDINATOR_INBOX,
    CriticRequest,
    Proposal,
    parse_request,
)
from hyperloom.agents.critic.runtime.errors import RequestValidationError


def test_parse_coordinator_inbox_with_raw_prompt():
    req = parse_request(
        {
            "kind": COORDINATOR_INBOX,
            "session_id": "sess_1",
            "raw_prompt": "=== Inbox for critic ===\n  seq=1 msg_id=abc topic=proposal payload={}",
        }
    )
    assert isinstance(req, CriticRequest)
    assert req.kind == COORDINATOR_INBOX
    assert req.session_id == "sess_1"
    assert req.raw_prompt is not None
    assert req.proposals == []
    assert req.context == {}
    assert req.decision_id is None


def test_parse_coordinator_inbox_requires_raw_prompt():
    with pytest.raises(RequestValidationError, match="raw_prompt"):
        parse_request(
            {
                "kind": COORDINATOR_INBOX,
                "session_id": "sess_1",
            }
        )


@pytest.mark.parametrize(
    "kind", ["wat", "critic_decision_request", "kb_draft_request", "kb_hint_request", "objection_signal"]
)
def test_parse_non_inbox_kind_rejected(kind):
    with pytest.raises(RequestValidationError, match="kind"):
        parse_request({"kind": kind, "session_id": "sess_1", "raw_prompt": "x"})


def test_parse_missing_session_rejected():
    with pytest.raises(RequestValidationError, match="session_id"):
        parse_request({"kind": COORDINATOR_INBOX, "raw_prompt": "x"})


def test_parse_proposal_normalises_action_and_gain():
    req = parse_request(
        {
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
        }
    )
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
        parse_request(
            {
                "kind": COORDINATOR_INBOX,
                "session_id": "s",
                "raw_prompt": "x",
                "proposals": [{"msg_id": "abc", "from_agent": "o", "payload": "no"}],
            }
        )


def test_parse_proposal_rejects_non_int_seq():
    with pytest.raises(RequestValidationError, match="seq"):
        parse_request(
            {
                "kind": COORDINATOR_INBOX,
                "session_id": "s",
                "raw_prompt": "x",
                "proposals": [{"msg_id": "abc", "from_agent": "o", "seq": "12"}],
            }
        )


def test_parse_proposal_missing_msg_id():
    with pytest.raises(RequestValidationError, match="msg_id"):
        parse_request(
            {
                "kind": COORDINATOR_INBOX,
                "session_id": "s",
                "raw_prompt": "x",
                "proposals": [{"from_agent": "o"}],
            }
        )


def test_proposal_to_dict_carries_parse_defaults():
    req = parse_request(
        {
            "kind": COORDINATOR_INBOX,
            "session_id": "s",
            "raw_prompt": "x",
            "proposals": [
                {
                    "msg_id": "m1",
                    "from_agent": "o",
                    "payload": {"action_name": "baseline"},
                }
            ],
        }
    )
    d = req.proposals[0].to_dict()
    assert d["msg_id"] == "m1"
    assert d["action_name"] == "baseline"
    assert d["predicted_gain_pct"] is None
