"""Tests for orchestrator/execution_mode.py."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.execution_mode import (
    ExecutionMode,
    choose_execution_mode,
)


@pytest.mark.parametrize(
    "max_hours, expected",
    [
        ("0.5", ExecutionMode.QUICK_PARAM_SWEEP),
        ("1.99", ExecutionMode.QUICK_PARAM_SWEEP),
        ("2", ExecutionMode.GUIDED_KERNEL_OPT),
        ("3", ExecutionMode.GUIDED_KERNEL_OPT),
        ("6", ExecutionMode.GUIDED_KERNEL_OPT),
        ("6.01", ExecutionMode.MARATHON_MULTI_AGENT),
        ("24", ExecutionMode.MARATHON_MULTI_AGENT),
        ("48", ExecutionMode.MARATHON_MULTI_AGENT),
    ],
)
def test_mode_thresholds(max_hours, expected):
    assert choose_execution_mode({"MAX_HOURS": max_hours}) is expected


def test_missing_max_hours():
    with pytest.raises(ValueError, match="required"):
        choose_execution_mode({})


def test_non_numeric_max_hours():
    with pytest.raises(ValueError, match="numeric"):
        choose_execution_mode({"MAX_HOURS": "lots"})


def test_non_positive_max_hours():
    with pytest.raises(ValueError, match="> 0"):
        choose_execution_mode({"MAX_HOURS": "0"})
    with pytest.raises(ValueError, match="> 0"):
        choose_execution_mode({"MAX_HOURS": "-3"})
