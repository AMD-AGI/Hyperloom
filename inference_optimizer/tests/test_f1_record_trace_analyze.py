"""F1-1 — ``SharedState.record_trace_analyze`` writer.

Smoke-tests the canonical 11-field ``last_trace_analyze`` dict that
:class:`RooflineExecutor` relies on, plus the monotonic
``roofline_snapshot_id`` invariant.

Reference: ``plan_roofline_framework/F1_roofline_composite.MD`` §F1-1.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


def _trace_analyze_result(report_path: str, *, hot_kernels=None) -> dict:
    return {
        "status": "ok",
        "trace_report_path": report_path,
        "candidates_path": "/tmp/candidates.json",
        "hot_kernels": hot_kernels or [
            {
                "kernel_id": "k001",
                "name": "fused_rmsnorm",
                "gpu_pct": 28.78,
                "bottleneck": "memory",
                "arithmetic_intensity": 0.5,
                "source_file": "rmsnorm.py",
                "reusable_native_kernel": True,
                "recommended_backends": ["aiter"],
                "recommended_actions": ["run_optimization"],
            },
        ],
        "task_groups": [{"primary": "k001", "kids": ["k001"]}],
        "trace_health_warnings": [
            {"code": "high_gpu_idle_pct", "severity": "warn", "idle_pct": 31.2},
            "junk_should_be_filtered",  # filtered out: not a dict
            {"severity": "warn"},  # filtered out: no `code` field
        ],
    }


def test_record_trace_analyze_writes_canonical_fields(tmp_path: Path):
    md = tmp_path / "analysis.md"
    md.write_text("# TraceLens analysis\nbody.\n", encoding="utf-8")

    s = SharedState()
    s.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        _trace_analyze_result(str(md)),
    )

    cached = s.last_trace_analyze
    assert isinstance(cached, dict)
    assert cached["trace_input"] == "/tmp/trace.json"
    assert cached["candidates_path"] == "/tmp/candidates.json"
    assert cached["analysis_md_path"] == str(md)
    assert cached["analysis_md_text"].startswith("# TraceLens analysis")
    assert cached["hot_kernels_top15"][0]["kernel_id"] == "k001"
    assert cached["reusable_native_kernel_ids"] == ["k001"]
    # Filtering keeps only dict warnings with a ``code`` field.
    assert cached["trace_health_warnings"] == [
        {"code": "high_gpu_idle_pct", "severity": "warn", "idle_pct": 31.2},
    ]
    assert cached["task_groups"] == [{"primary": "k001", "kids": ["k001"]}]
    assert cached["roofline_snapshot_id"] == 1
    assert cached["roofline_baseline_gain_at_snapshot"] == pytest.approx(0.0)
    assert cached["ts"]


def test_record_trace_analyze_snapshot_id_is_monotonic(tmp_path: Path):
    md = tmp_path / "analysis.md"
    md.write_text("body", encoding="utf-8")

    s = SharedState()
    for expected_id in (1, 2, 3):
        s.record_trace_analyze(
            {"trace_input": f"/tmp/trace_{expected_id}.json"},
            _trace_analyze_result(str(md)),
        )
        assert s.last_trace_analyze["roofline_snapshot_id"] == expected_id
        assert s.roofline_snapshot_id == expected_id


def test_record_trace_analyze_captures_baseline_gain_at_write_time(
    tmp_path: Path,
):
    md = tmp_path / "analysis.md"
    md.write_text("body", encoding="utf-8")

    s = SharedState()
    s.cumulative_gain_validated = 7.5
    s.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        _trace_analyze_result(str(md)),
    )
    assert s.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == \
        pytest.approx(7.5)


def test_record_trace_analyze_missing_md_degrades_silently():
    s = SharedState()
    s.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        _trace_analyze_result("/nonexistent/analysis.md"),
    )
    assert s.last_trace_analyze["analysis_md_text"] == ""
    assert s.last_trace_analyze["analysis_md_path"] == "/nonexistent/analysis.md"
    assert s.last_trace_analyze["roofline_snapshot_id"] == 1


def test_record_trace_analyze_ignores_non_dict_result():
    s = SharedState()
    s.record_trace_analyze({"trace_input": "/tmp/trace.json"}, None)  # type: ignore[arg-type]
    assert s.last_trace_analyze == {}
    assert s.roofline_snapshot_id == 0


def test_record_trace_analyze_does_not_clobber_legacy_select_kernels(
    tmp_path: Path,
):
    """The two writers target distinct fields and must stay independent."""
    md = tmp_path / "analysis.md"
    md.write_text("body", encoding="utf-8")

    s = SharedState()
    s.last_select_kernels = {"trace_input": "legacy", "hot_kernels_top15": []}
    s.record_trace_analyze(
        {"trace_input": "/tmp/trace.json"},
        _trace_analyze_result(str(md)),
    )
    assert s.last_select_kernels == {
        "trace_input": "legacy", "hot_kernels_top15": [],
    }
    assert s.last_trace_analyze["trace_input"] == "/tmp/trace.json"
