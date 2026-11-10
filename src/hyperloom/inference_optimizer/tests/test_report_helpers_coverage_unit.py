# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the report platform line and platform fingerprint."""

from __future__ import annotations

from hyperloom.common.perf_metric import GRADED_OUTPUT, VERDICT_KEEP
from hyperloom.orchestrator.actions.executors import report as rp

_OUTPUT_COMPARISON = dict(
    objective=GRADED_OUTPUT,
    reference=100.0,
    candidate=100.0,
    gain_pct=0.0,
    comparable=True,
    degrade_reason="",
    verdict=VERDICT_KEEP,
)


# ---- platform line: absence must be stated, not implied ----
def _summary_without_platform(**extra) -> dict:
    base = {
        "session_id": "s",
        "model_name": "m",
        "model_path": "/m",
        "stop_reason": "target_reached",
        "max_minutes": 60,
        "report_generated_at": "t0",
        "framework": "sglang",
        "current_best": {},
        "baseline_tput": 100.0,
        "cumulative_gain": 0.0,
        "cumulative_gain_validated": 0.0,
        "cumulative_gain_validated_stack_len": 0,
        "optimization_stack_len": 0,
        "crash_count": 0,
        "pruned_families": [],
        "event_counts_by_topic": {},
        "highlights": [],
        "performance_comparison": _OUTPUT_COMPARISON,
    }
    base.update(extra)
    return base


def test_report_says_platform_is_missing_when_the_summary_predates_the_field():
    """A summary with no platform key still gets a line."""
    md = rp._format_md(_summary_without_platform())
    assert "- platform       : not recorded" in md


def test_report_gives_the_reason_the_platform_is_missing_when_it_has_one():
    md = rp._format_md(_summary_without_platform(platform={"status": "unavailable", "reason": "no host CPU sysfs"}))
    assert "not recorded — no host CPU sysfs" in md


def test_platform_fingerprint_propagates_a_node_scope_failure(monkeypatch):
    import pytest

    def _boom():
        raise RuntimeError("node scope unreadable")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", _boom)

    with pytest.raises(RuntimeError, match="node scope unreadable"):
        rp._platform_fingerprint()
