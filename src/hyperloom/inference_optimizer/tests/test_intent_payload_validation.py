# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Value-level payload validation in ``validate_envelope``.

Required-key presence alone let bad values reach consumers that assume them:
``intent_router`` maps ALERT ``severity`` onto an interrupt priority and casts
EXTEND_LEASE ``extra_sec`` with a bare ``int()``, and an unrecognised
REVIEW_VERDICT ``verdict`` matched neither the approve nor the deny branch.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    validate_envelope,
)


def _envelope(intent_type: str, payload: dict) -> dict:
    return {"intents": [{"intent_type": intent_type, "payload": payload}]}


@pytest.mark.parametrize("severity", [None, "", "CRITICAL", 42, "info"])
def test_alert_rejects_invalid_severity(severity) -> None:
    with pytest.raises(IntentValidationError, match="severity"):
        validate_envelope(_envelope("alert", {"severity": severity, "summary": "x"}))


@pytest.mark.parametrize("summary", [None, "", "   "])
def test_alert_rejects_empty_summary(summary) -> None:
    with pytest.raises(IntentValidationError, match="summary"):
        validate_envelope(_envelope("alert", {"severity": "high", "summary": summary}))


def test_alert_accepts_valid_payload() -> None:
    intents = validate_envelope(_envelope("alert", {"severity": "medium", "summary": "disk full"}))
    assert intents[0].payload["severity"] == "medium"


@pytest.mark.parametrize("extra_sec", ["abc", -1, 0, None, True])
def test_extend_lease_rejects_invalid_extra_sec(extra_sec) -> None:
    with pytest.raises(IntentValidationError, match="extra_sec"):
        validate_envelope(_envelope("extend_lease", {"task_id": "t1", "extra_sec": extra_sec}))


def test_extend_lease_rejects_empty_task_id() -> None:
    with pytest.raises(IntentValidationError, match="task_id"):
        validate_envelope(_envelope("extend_lease", {"task_id": "", "extra_sec": 60}))


def test_extend_lease_accepts_valid_payload() -> None:
    intents = validate_envelope(_envelope("extend_lease", {"task_id": "t1", "extra_sec": 120}))
    assert intents[0].payload["extra_sec"] == 120


@pytest.mark.parametrize("verdict", [12345, None, "APPROVE", "maybe"])
def test_review_verdict_rejects_invalid_verdict(verdict) -> None:
    with pytest.raises(IntentValidationError, match="verdict"):
        validate_envelope(_envelope("review_verdict", {"target_proposal_msg_id": "m", "verdict": verdict}))


def test_review_verdict_rejects_invalid_verdict_in_map() -> None:
    with pytest.raises(IntentValidationError, match="verdict"):
        validate_envelope(
            _envelope(
                "review_verdict",
                {"target_proposal_msg_id": "m", "verdict_map": {"vA": {"verdict": "banana"}}},
            )
        )


def test_review_verdict_accepts_valid_payload() -> None:
    intents = validate_envelope(_envelope("review_verdict", {"target_proposal_msg_id": "m", "verdict": "approve"}))
    assert intents[0].payload["verdict"] == "approve"
