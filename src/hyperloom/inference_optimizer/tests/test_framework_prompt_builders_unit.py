# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the FRAMEWORK prompt builder functions."""

from __future__ import annotations

import json


def test_audit_refine_prompt_structure():
    from hyperloom.agents.framework.audit import build_audit_refine_prompt

    static = {
        "semantic_status": "not_present",
        "applicability": "direct_apply",
        "confidence": 0.2,
        "evidence": [],
        "risks": [],
    }
    prompt = build_audit_refine_prompt(static, "diff content here")
    assert json.dumps(static, ensure_ascii=False) in prompt
    assert "diff content here" in prompt
    assert "STRICT JSON" in prompt
    assert "semantic_status" in prompt


def test_audit_refine_prompt_truncates_long_diff():
    from hyperloom.agents.framework.audit import build_audit_refine_prompt

    long_diff = "x" * 10000
    prompt = build_audit_refine_prompt({}, long_diff)
    assert long_diff[:6000] in prompt
    assert long_diff not in prompt  # full 10 000-char string must not appear
