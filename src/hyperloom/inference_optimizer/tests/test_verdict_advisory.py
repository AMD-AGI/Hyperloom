# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory serialization of critic verdicts."""

from __future__ import annotations

from hyperloom.orchestrator.loop.verdicts import serialize_verdict_advisory


def test_serialize_verdict_advisory_full_fieldset():
    out = serialize_verdict_advisory(
        {
            "target_proposal_msg_id": "p1",
            "verdict": "advise",
            "reasoning": "ignored here",
            "required_evidence": ["bench at conc=64", "isolate decode"],
            "risks": [{"severity": "high", "text": "regress"}],
            "advice_text": "tune flag",
            "alternative_action": "explore",
            "notes": ["note a"],
            "kb_evidence": ["kb://1"],
            "packet_evidence": ["pkt://2"],
        }
    )
    assert out == {
        "required_evidence": ["bench at conc=64", "isolate decode"],
        "risks": [{"severity": "high", "text": "regress"}],
        "notes": ["note a"],
        "kb_evidence": ["kb://1"],
        "packet_evidence": ["pkt://2"],
        "advice_text": "tune flag",
        "alternative_action": "explore",
    }
    assert "verdict" not in out
    assert "reasoning" not in out


def test_serialize_verdict_advisory_drops_empty_and_blank():
    out = serialize_verdict_advisory(
        {
            "required_evidence": ["keep", "", None],
            "risks": [],
            "advice_text": "   ",
            "alternative_action": "",
            "notes": None,
        }
    )
    assert out == {"required_evidence": ["keep"]}


def test_serialize_verdict_advisory_coerces_scalar_to_list():
    out = serialize_verdict_advisory({"required_evidence": "single"})
    assert out == {"required_evidence": ["single"]}


def test_serialize_verdict_advisory_non_dict_is_empty():
    assert serialize_verdict_advisory(None) == {}  # type: ignore[arg-type]
    assert serialize_verdict_advisory("nope") == {}  # type: ignore[arg-type]
