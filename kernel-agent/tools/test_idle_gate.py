###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the shared idle-gate helpers (_idle_gate).

Locks the single source of truth consumed by both trace-analysis routes: the
threshold resolution (default + env override + bad-value fallback) and the
high-idle warning shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _idle_gate as ig  # noqa: E402


def test_threshold_default(monkeypatch):
    monkeypatch.delenv(ig.HIGH_IDLE_PCT_THRESHOLD_ENV, raising=False)
    assert ig.resolve_idle_pct_threshold() == ig.HIGH_IDLE_PCT_THRESHOLD_DEFAULT == 80.0


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv(ig.HIGH_IDLE_PCT_THRESHOLD_ENV, "42.5")
    assert ig.resolve_idle_pct_threshold() == 42.5


def test_threshold_bad_value_falls_back(monkeypatch):
    monkeypatch.setenv(ig.HIGH_IDLE_PCT_THRESHOLD_ENV, "not-a-number")
    assert ig.resolve_idle_pct_threshold() == 80.0


def test_threshold_negative_falls_back(monkeypatch):
    monkeypatch.setenv(ig.HIGH_IDLE_PCT_THRESHOLD_ENV, "-5")
    assert ig.resolve_idle_pct_threshold() == 80.0


def test_high_idle_warning_shape():
    w = ig.build_high_idle_warning(
        idle_pct=91.234,
        threshold_pct=80.0,
        report_path=Path("/tmp/analysis.md"),
    )
    assert w["code"] == "high_gpu_idle_pct"
    assert w["severity"] == "warning"
    assert w["idle_pct"] == 91.23  # rounded to 2 dp
    assert w["threshold_pct"] == 80.0
    assert w["source"] == "/tmp/analysis.md"
    assert "GPU was idle 91.23%" in w["message"]
    assert "parameter optimization" in w["message"]
