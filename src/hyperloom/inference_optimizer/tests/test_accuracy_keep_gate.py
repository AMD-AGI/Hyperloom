# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""accuracy-gate KEEP enforcement helper."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors._accuracy_gate import (
    BASELINE_ACCURACY_STOP_REASON,
    accuracy_keep_block,
    request_baseline_accuracy_stop,
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


class _StopRecorder:
    """Minimal SharedState stub capturing ``set_stop_reason`` calls."""

    def __init__(self) -> None:
        self.stop_reason = ""

    def set_stop_reason(self, value, **_kwargs):
        self.stop_reason = value
        return value


def test_request_baseline_accuracy_stop_records_reason():
    ss = _StopRecorder()
    assert request_baseline_accuracy_stop(ss, context="unit") is True
    assert ss.stop_reason == BASELINE_ACCURACY_STOP_REASON


def test_request_baseline_accuracy_stop_none_shared_state():
    assert request_baseline_accuracy_stop(None, context="unit") is False


def test_request_baseline_accuracy_stop_without_setter():
    class _NoSetter:
        pass

    assert request_baseline_accuracy_stop(_NoSetter(), context="unit") is False
