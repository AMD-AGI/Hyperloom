# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""accuracy-gate KEEP enforcement helper."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    accuracy_keep_block,
    require_framework_accuracy_default,
)


def test_regression_always_blocks():
    blocked, reason, degraded = accuracy_keep_block(False, required=False, baseline_accuracy=0.0)
    assert blocked is True
    assert "regression" in reason
    assert degraded is False


def test_pass_never_blocks():
    blocked, _reason, degraded = accuracy_keep_block(True, required=True, baseline_accuracy=0.8)
    assert blocked is False
    assert degraded is False


def test_none_not_required_allows():
    blocked, _reason, degraded = accuracy_keep_block(None, required=False, baseline_accuracy=0.8)
    assert blocked is False
    assert degraded is False


def test_none_required_with_baseline_blocks():
    blocked, reason, degraded = accuracy_keep_block(None, required=True, baseline_accuracy=0.8)
    assert blocked is True
    assert "required" in reason
    assert degraded is False


def test_none_required_without_baseline_degrades():
    blocked, _reason, degraded = accuracy_keep_block(None, required=True, baseline_accuracy=0.0)
    assert blocked is False
    assert degraded is True


def test_none_required_with_non_numeric_baseline_degrades():
    blocked, _reason, degraded = accuracy_keep_block(None, required=True, baseline_accuracy=None)
    assert blocked is False
    assert degraded is True


def test_require_default_on(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY", raising=False)
    assert require_framework_accuracy_default() is True


def test_require_default_env_off(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY", "0")
    assert require_framework_accuracy_default() is False
