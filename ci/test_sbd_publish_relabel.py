#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the sbd publish relabel / leaderboard withholding logic."""

from __future__ import annotations

import pytest

import sbd_publish_relabel as srl


# ── resolve_stop_reason: v1-flat top level vs v2 nested ─────────────────────


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"status": "Failed"}, "failed"),
        ({"stop_reason": "Signal"}, "signal"),
        ({"session": {"stop_reason": "robustness_escalated"}}, "robustness_escalated"),
        # Top-level wins over nested when both present.
        ({"status": "baseline_failed", "session": {"stop_reason": "signal"}}, "baseline_failed"),
        ({}, ""),
        ({"session": "not-a-dict"}, ""),
    ],
)
def test_resolve_stop_reason(data, expected):
    assert srl.resolve_stop_reason(data) == expected


# ── resolve_gain: v2 final.cumulative_gain_pct_validated vs legacy fields ────


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"final": {"cumulative_gain_pct_validated": 0.0}}, 0.0),
        ({"final": {"cumulative_gain_pct_validated": 3.5}}, 3.5),
        # Legacy fallbacks when v2 final is absent.
        ({"gain_pct_sum": 2.0}, 2.0),
        ({"gain": 1.25}, 1.25),
        # v2 final takes precedence over legacy fields.
        ({"final": {"cumulative_gain_pct_validated": 0.0}, "gain_pct_sum": 9.9}, 0.0),
        ({}, 0.0),
        ({"final": "not-a-dict"}, 0.0),
    ],
)
def test_resolve_gain(data, expected):
    assert srl.resolve_gain(data) == expected


# ── apply_leaderboard_withholding: the actual rule ──────────────────────────


@pytest.mark.parametrize(
    "data,withheld",
    [
        # The exact v2 failure sample from the review: nested stop_reason + zero
        # validated gain -> must be withheld.
        (
            {"session": {"stop_reason": "robustness_escalated"}, "final": {"cumulative_gain_pct_validated": 0.0}},
            True,
        ),
        # robustness_escalated WITH real positive gain stays on the leaderboard.
        (
            {"session": {"stop_reason": "robustness_escalated"}, "final": {"cumulative_gain_pct_validated": 4.2}},
            False,
        ),
        # baseline_failed + zero gain -> withheld.
        (
            {"session": {"stop_reason": "baseline_failed"}, "final": {"cumulative_gain_pct_validated": 0.0}},
            True,
        ),
        # v1-flat failed + zero gain -> withheld.
        ({"status": "failed", "gain_pct_sum": 0.0}, True),
        # v1-flat failed but positive gain -> not withheld.
        ({"status": "failed", "gain_pct_sum": 1.5}, False),
        # Successful/non-withheld terminal state -> never withheld.
        ({"session": {"stop_reason": "delivered"}, "final": {"cumulative_gain_pct_validated": 5.0}}, False),
        # Unknown/empty status -> not withheld even at zero gain.
        ({"final": {"cumulative_gain_pct_validated": 0.0}}, False),
    ],
)
def test_apply_leaderboard_withholding(data, withheld):
    srl.apply_leaderboard_withholding(data)
    if withheld:
        assert data.get("show_on_leaderboard") is False
        assert data.get("leaderboard_withheld_reason")
    else:
        assert "show_on_leaderboard" not in data
        assert "leaderboard_withheld_reason" not in data


def test_relabel_stamps_schema_and_identifiers_and_withholds():
    data = {"session": {"stop_reason": "robustness_escalated"}, "final": {"cumulative_gain_pct_validated": 0.0}}

    out = srl.relabel(
        data,
        task_id="opt-123",
        claw_session_id="claw-abc",
        image="harbor/sglang:v0.5.12",
        isb=None,
    )

    assert out["schema_version"] == srl.SCHEMA_VERSION
    assert out["task_id"] == "opt-123"
    assert out["session"]["claw_session_id"] == "claw-abc"
    assert out["session"]["session_id"] == "claw-abc"
    assert out["session_meta"]["session_id"] == "claw-abc"
    assert out["session_meta"]["image"] == "harbor/sglang:v0.5.12"
    # Withholding still fires through relabel().
    assert out["show_on_leaderboard"] is False
    assert out["leaderboard_withheld_reason"] == "robustness_escalated"


def test_relabel_does_not_overwrite_existing_identifiers():
    data = {"task_id": "keep-me", "session": {"session_id": "keep-sid"}}

    out = srl.relabel(data, task_id="new-task", claw_session_id="new-claw", isb=None)

    assert out["task_id"] == "keep-me"
    assert out["session"]["session_id"] == "keep-sid"
