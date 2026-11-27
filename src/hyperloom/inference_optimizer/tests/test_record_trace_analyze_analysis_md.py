# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``record_trace_analyze`` records the TraceLens report (its path, whether it held anything, monotonic snapshot id, point-in-time gain capture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.state.shared_state import SharedState


def _payload(trace: Path) -> dict:
    return {"trace_input": str(trace)}


def _result_with_report(analysis_md: Path) -> dict:
    return {
        "hot_kernels": [],
        "trace_report_path": str(analysis_md),
        "trace_health_warnings": [],
    }


def test_records_the_report_path_and_that_the_report_held_text(tmp_path: Path) -> None:
    """The report stays on disk: the state names it and says it was there, and holds none of its text."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text(
        "# Executive Summary\nCompute 51.6%, Idle 48.3%\n## Top Operations\n| aten::mm | 25% |\n",
        encoding="utf-8",
    )

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "trace.json"), _result_with_report(analysis_md))

    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(analysis_md)
    assert cached["has_analysis_md"] is True
    assert "Executive Summary" not in json.dumps(cached)


def test_records_the_fusion_findings_of_the_json_beside_the_md(tmp_path: Path) -> None:
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# Executive Summary\n", encoding="utf-8")
    fusion = {
        "operation": "Unfused Add + RMSNorm in `DecoderLayer`",
        "kind": "add_norm",
        "members": [{"kernel_name": ["add_kernel"], "kernel_launcher_path": "modeling.py(88): forward"}],
        "impact": {"mid": 2.4},
    }
    (tmp_path / "analysis.json").write_text(json.dumps({"fusion_optimizations": [fusion]}), encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "trace.json"), _result_with_report(analysis_md))

    assert state.last_trace_analyze["fusion_findings"] == [
        {
            "kind": "add_norm",
            "operation": "Unfused Add + RMSNorm in `DecoderLayer`",
            "projected_e2e_pct": 2.4,
            "members": [{"kernel_names": ["add_kernel"], "kernel_launcher_path": "modeling.py(88): forward"}],
        }
    ]


def test_missing_trace_report_path_records_no_report() -> None:
    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == ""
    assert cached["has_analysis_md"] is False
    assert cached["fusion_findings"] == []


@pytest.mark.parametrize("written", [None, ""], ids=["missing", "empty"])
def test_a_report_path_with_nothing_behind_it_records_no_report(tmp_path: Path, written: str | None) -> None:
    report = tmp_path / "analysis.md"
    if written is not None:
        report.write_text(written, encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {
            "hot_kernels": [],
            "trace_report_path": str(report),
            "trace_health_warnings": [],
        },
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(report)
    assert cached["has_analysis_md"] is False


def test_snapshot_id_monotonically_increases(tmp_path: Path) -> None:
    """Every successful call bumps ``roofline_snapshot_id`` by exactly 1."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("first", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "t1"), _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1

    analysis_md.write_text("second", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t2"), _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 2

    analysis_md.write_text("third", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t3"), _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 3


def test_snapshot_id_starts_at_one_after_empty_state() -> None:
    """Fresh SharedState (empty ``last_trace_analyze``) → snapshot 1."""
    state = SharedState()
    assert state.last_trace_analyze == {}
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1


def test_baseline_gain_captured_at_snapshot_time() -> None:
    """``roofline_baseline_gain_at_snapshot`` is a point-in-time capture of the validated gain."""
    state = SharedState()
    state.cumulative_gain_validated = 0.0
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 0.0

    state.cumulative_gain_validated = 3.2
    state.record_trace_analyze(
        {"trace_input": "x2"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 3.2

    state.cumulative_gain_validated = 7.5
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 3.2


def test_non_dict_result_is_ignored() -> None:
    """A non-dict ``result`` short-circuits without raising."""
    state = SharedState()
    state.record_trace_analyze({"trace_input": "x"}, None)  # type: ignore[arg-type]
    assert state.last_trace_analyze == {}


def test_existing_fields_still_populated(tmp_path: Path) -> None:
    """New fields are additive: pre-existing keys keep their values."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# hi", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "/some/trace.json"},
        {
            "hot_kernels": [
                {
                    "kernel_id": "k1",
                    "name": "aten::mm",
                    "gpu_pct": 25.0,
                    "reusable_native_kernel": False,
                },
                {
                    "kernel_id": "k2",
                    "name": "rmsnorm",
                    "gpu_pct": 10.0,
                    "reusable_native_kernel": True,
                },
            ],
            "candidates_path": "/some/kc.json",
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [
                {"code": "high_gpu_idle_pct", "severity": "warning", "message": "idle=48%", "idle_pct": 48.0},
            ],
        },
    )

    cached = state.last_trace_analyze
    assert cached["trace_input"] == "/some/trace.json"
    assert cached["candidates_path"] == "/some/kc.json"
    assert len(cached["hot_kernels_top15"]) == 2
    assert cached["reusable_native_kernel_ids"] == ["k2"]
    assert len(cached["trace_health_warnings"]) == 1
    assert cached["trace_health_warnings"][0]["code"] == "high_gpu_idle_pct"
    assert cached["analysis_md_path"] == str(analysis_md)
    assert cached["has_analysis_md"] is True
    assert cached["roofline_snapshot_id"] == 1
    assert cached["roofline_baseline_gain_at_snapshot"] == 0.0
    assert cached["ts"]


def test_record_trace_analyze_preserves_kernel_roofline_fields(
    tmp_path: Path,
) -> None:
    """Kernel-roofline fields from TraceLens candidates survive caching."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("# hi", encoding="utf-8")
    sidecar = tmp_path / "kernel_roofline.json"
    sidecar.write_text(json.dumps({"kernels": [{"kernel_id": "k1"}]}), encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "/some/trace.json"},
        {
            "hot_kernels": [
                {
                    "kernel_id": "k1",
                    "name": "rmsnorm_kernel",
                    "kernel_category": "LayerNorm",
                    "gpu_pct": 8.2,
                    "bottleneck": "memory",
                    "bound_type": "memory",
                    "flops_per_byte": 0.45,
                    "efficiency_percent": 31.2,
                    "compute_utilization_pct": 9.1,
                    "bandwidth_utilization_pct": 72.4,
                    "suggestion": "reduce memory traffic",
                    "roofline_name": "rmsnorm_kernel",
                    "source_file": "/tmp/rmsnorm.py",
                    "shapes": ["(16,64) bf16"],
                    "recommended_actions": ["fuse adjacent ops"],
                    "reusable_native_kernel": True,
                },
            ],
            "candidates_path": "/some/kc.json",
            "kernel_roofline_path": str(sidecar),
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [],
        },
    )

    cached = state.last_trace_analyze
    assert cached["kernel_roofline_path"] == str(sidecar)
    row = cached["hot_kernels_top15"][0]
    assert row["bound_type"] == "memory"
    assert row["arithmetic_intensity"] == 0.45
    assert row["flops_per_byte"] == 0.45
    assert row["efficiency_percent"] == 31.2
    assert row["compute_utilization_pct"] == 9.1
    assert row["bandwidth_utilization_pct"] == 72.4
    assert row["suggestion"] == "reduce memory traffic"
    assert row["shapes"] == ["(16,64) bf16"]
    # kernel_category propagates from TraceLens hot_kernels through to the cached row.
    assert row["kernel_category"] == "LayerNorm"
    assert cached["kernel_roofline_top15"][0] == row


# skipped_kernels projection: when all candidates are skipped, project them for the prompt.


def _result_with_skipped(skipped: list[dict]) -> dict:
    return {
        "hot_kernels": [],
        "skipped_kernels": skipped,
        "trace_health_warnings": [],
    }


def test_skipped_kernels_projected_when_hot_empty() -> None:
    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        _result_with_skipped(
            [
                {"kernel_id": "k001", "name": "aten::mm", "skip_reason": "source file not resolved", "gpu_pct": 3.3},
                {"kernel_id": "k003", "name": "aten::mm", "skip_reason": "source file not resolved", "gpu_pct": 17.1},
            ]
        ),
    )
    proj = state.last_trace_analyze["skipped_kernels_top"]
    # Sorted by gpu_pct desc.
    assert [p["kernel_id"] for p in proj] == ["k003", "k001"]
    assert proj[0]["name"] == "aten::mm"
    assert proj[0]["skip_reason"] == "source file not resolved"


def test_skipped_projection_truncates_to_15() -> None:
    state = SharedState()
    many = [
        {"kernel_id": f"k{i:03d}", "name": "aten::mm", "skip_reason": "source file not resolved", "gpu_pct": float(i)}
        for i in range(1, 26)
    ]
    state.record_trace_analyze({"trace_input": "x"}, _result_with_skipped(many))
    proj = state.last_trace_analyze["skipped_kernels_top"]
    assert len(proj) == 15
    assert proj[0]["kernel_id"] == "k025"  # highest gpu_pct first


def test_skipped_top_empty_when_no_skipped() -> None:
    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    assert state.last_trace_analyze["skipped_kernels_top"] == []


def test_blob_renders_skipped_when_no_routable_candidates() -> None:
    state = SharedState()
    blob = {
        "trace_input": "/t.json",
        "candidates_path": "/kc.json",
        "hot_kernels_top15": [],
        "reusable_native_kernel_ids": [],
        "skipped_kernels_top": [
            {"kernel_id": "k001", "name": "aten::mm", "skip_reason": "source file not resolved", "gpu_pct": 3.3},
        ],
        "trace_health_warnings": [],
    }
    rendered = state._format_trace_analyze_blob(blob)
    assert "skipped_kernels_top=[k001:aten::mm:source file not resolved]" in rendered


def test_blob_format_stable_when_candidates_present() -> None:
    state = SharedState()
    blob = {
        "trace_input": "/t.json",
        "candidates_path": "/kc.json",
        "hot_kernels_top15": [{"kernel_id": "k002", "name": "rmsnorm"}],
        "reusable_native_kernel_ids": ["k002"],
        "skipped_kernels_top": [
            {"kernel_id": "k001", "name": "aten::mm", "skip_reason": "x", "gpu_pct": 3.3},
        ],
        "trace_health_warnings": [],
    }
    rendered = state._format_trace_analyze_blob(blob)
    # Candidates present → legacy format, no skipped suffix injected.
    assert "skipped_kernels_top=" not in rendered
    assert "top=['k002']" in rendered
