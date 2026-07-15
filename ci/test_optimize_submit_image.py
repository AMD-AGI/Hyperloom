#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for CI optimize image selection."""

from __future__ import annotations

import optimize_submit as opt


def test_mimo_uses_default_sglang_image():
    image = opt.detect_image("sglang", "XiaomiMiMo/MiMo-V2-7B")

    assert image == opt._default_sglang_image()
    assert "sglang" in image


class _FakeSafe:
    def __init__(self, sid: str):
        self.sid = sid

    def _claw_session_id_for(self, task_id: str) -> str:
        assert task_id == "opt-123"
        return self.sid


def test_claw_session_resolver_falls_back_to_task_get():
    rec = opt.SubmissionRecord(model="org/model", status="submitted", task_id="opt-123")

    sid = opt._resolve_record_claw_session_id(_FakeSafe("claw-abc"), rec, {})

    assert sid == "claw-abc"


def test_claw_session_resolver_prefers_terminal_payload():
    rec = opt.SubmissionRecord(
        model="org/model",
        status="submitted",
        task_id="opt-123",
        claw_session_id="old-claw",
    )

    sid = opt._resolve_record_claw_session_id(
        _FakeSafe("fallback-claw"),
        rec,
        {"clawSessionId": "fresh-claw"},
    )

    assert sid == "fresh-claw"
