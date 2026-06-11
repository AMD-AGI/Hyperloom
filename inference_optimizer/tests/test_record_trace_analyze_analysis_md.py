# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline-v2 C1: ``record_trace_analyze`` caches TraceLens analysis.md (full text, monotonic snapshot id, point-in-time gain capture, silent read failures)."""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator.shared_state import SharedState


def _payload(trace: Path) -> dict:
    return {"trace_input": str(trace)}


def _result_with_report(analysis_md: Path) -> dict:
    return {
        "hot_kernels": [],
        "trace_report_path": str(analysis_md),
        "trace_health_warnings": [],
    }


def test_caches_analysis_md_full_text(tmp_path: Path) -> None:
    """Path provided → file content cached verbatim."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text(
        "# Executive Summary\nCompute 51.6%, Idle 48.3%\n"
        "## Top Operations\n| aten::mm | 25% |\n",
        encoding="utf-8",
    )

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "trace.json"),
                                _result_with_report(analysis_md))

    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(analysis_md)
    assert "Executive Summary" in cached["analysis_md_text"]
    assert "Compute 51.6%" in cached["analysis_md_text"]
    assert "Top Operations" in cached["analysis_md_text"]


def test_missing_trace_report_path_yields_empty_text() -> None:
    """No ``trace_report_path`` field → ``analysis_md_text`` is ``""``."""
    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {"hot_kernels": [], "trace_health_warnings": []},
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == ""
    assert cached["analysis_md_text"] == ""


def test_unreadable_analysis_md_degrades_silently(tmp_path: Path) -> None:
    """Path points at a nonexistent file → empty text, no exception."""
    missing = tmp_path / "does_not_exist.md"

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {
            "hot_kernels": [],
            "trace_report_path": str(missing),
            "trace_health_warnings": [],
        },
    )
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(missing)
    assert cached["analysis_md_text"] == ""


def test_caches_large_analysis_md_without_truncation(tmp_path: Path) -> None:
    """Real-world 200 KB report round-trips intact (Decision A3: no truncation)."""
    analysis_md = tmp_path / "big_analysis.md"
    big_content = "# Analysis\n" + ("filler line\n" * 20000)
    analysis_md.write_text(big_content, encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(
        {"trace_input": "x"},
        {
            "hot_kernels": [],
            "trace_report_path": str(analysis_md),
            "trace_health_warnings": [],
        },
    )
    cached_text = state.last_trace_analyze["analysis_md_text"]
    assert len(cached_text) > 200_000
    assert cached_text.startswith("# Analysis\n")
    assert cached_text.endswith("filler line\n")


def test_snapshot_id_monotonically_increases(tmp_path: Path) -> None:
    """Every successful call bumps ``roofline_snapshot_id`` by exactly 1."""
    analysis_md = tmp_path / "analysis.md"
    analysis_md.write_text("first", encoding="utf-8")

    state = SharedState()
    state.record_trace_analyze(_payload(tmp_path / "t1"),
                                _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1

    analysis_md.write_text("second", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t2"),
                                _result_with_report(analysis_md))
    assert state.last_trace_analyze["roofline_snapshot_id"] == 2

    analysis_md.write_text("third", encoding="utf-8")
    state.record_trace_analyze(_payload(tmp_path / "t3"),
                                _result_with_report(analysis_md))
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
    """``roofline_baseline_gain_at_snapshot`` is a point-in-time capture of cumulative_gain; later mutations don't retroact."""
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
                {"code": "high_gpu_idle_pct", "severity": "warning",
                 "message": "idle=48%", "idle_pct": 48.0},
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
    assert cached["analysis_md_text"] == "# hi"
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
    sidecar.write_text(json.dumps({
        "kernels": [{
            "kernel_id": "k1",
            "rocprof_roofline": {
                "before_kernel_opt": {"status": "matched", "roofline_efficiency_pct": 31.2},
                "after_kernel_opt": {"status": "matched", "roofline_efficiency_pct": 44.0},
            },
        }]
    }), encoding="utf-8")

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
    assert row["rocprof_roofline"]["before_kernel_opt"]["roofline_efficiency_pct"] == 31.2
    assert row["rocprof_roofline"]["after_kernel_opt"]["roofline_efficiency_pct"] == 44.0
    # kernel_category propagates from TraceLens hot_kernels to kernel_attempt_summary.by_kernel[].kernel_category.
    assert row["kernel_category"] == "LayerNorm"
    assert cached["kernel_roofline_top15"][0] == row


# skipped_kernels projection + prompt rendering: when all candidates are skipped, project them so the prompt shows detected-but-unoptimizable kernels instead of an empty list the LLM hallucinates over.


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
        _result_with_skipped([
            {"kernel_id": "k001", "name": "aten::mm",
             "skip_reason": "source file not resolved", "gpu_pct": 3.3},
            {"kernel_id": "k003", "name": "aten::mm",
             "skip_reason": "source file not resolved", "gpu_pct": 17.1},
        ]),
    )
    proj = state.last_trace_analyze["skipped_kernels_top"]
    # Sorted by gpu_pct desc so the heaviest operator leads.
    assert [p["kernel_id"] for p in proj] == ["k003", "k001"]
    assert proj[0]["name"] == "aten::mm"
    assert proj[0]["skip_reason"] == "source file not resolved"


def test_skipped_projection_truncates_to_15() -> None:
    state = SharedState()
    many = [
        {"kernel_id": f"k{i:03d}", "name": "aten::mm",
         "skip_reason": "source file not resolved", "gpu_pct": float(i)}
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
            {"kernel_id": "k001", "name": "aten::mm",
             "skip_reason": "source file not resolved", "gpu_pct": 3.3},
        ],
        "trace_health_warnings": [],
    }
    rendered = state._format_trace_analyze_blob(blob)
    assert (
        "skipped_kernels_top=[k001:aten::mm:source file not resolved]"
        in rendered
    )


def test_blob_format_stable_when_candidates_present() -> None:
    state = SharedState()
    blob = {
        "trace_input": "/t.json",
        "candidates_path": "/kc.json",
        "hot_kernels_top15": [{"kernel_id": "k002", "name": "rmsnorm"}],
        "reusable_native_kernel_ids": ["k002"],
        "skipped_kernels_top": [
            {"kernel_id": "k001", "name": "aten::mm",
             "skip_reason": "x", "gpu_pct": 3.3},
        ],
        "trace_health_warnings": [],
    }
    rendered = state._format_trace_analyze_blob(blob)
    # Candidates present → legacy format, no skipped suffix injected.
    assert "skipped_kernels_top=" not in rendered
    assert "top=['k002']" in rendered


def test_non_routable_kernel_opt_skip_rejects_canonical_id() -> None:
    state = SharedState()
    state.record_kernel_opt({
        "status": "skipped",
        "decision": "REVERT",
        "error_class": "missing_native_source",
        "reason": "non_routable_candidate",
        "kernel_id": "k001",
        "requested_kernel_id": "kn001",
        "resolved_kernel_id": "k001",
        "kernel_name": "aten::mm",
        "verification": {"micro_speedup": 0.0, "best_artifact_path": ""},
        "proposal": {
            "decision": "REVERT",
            "reasons": ["source file not resolved"],
        },
    })

    assert "k001" in state.rejected_kernel_ids
    entry = state.kernel_opt_attempts["k001"]
    assert entry["last_decision"] == "REVERT"
    assert entry["last_status"] == "skipped"
    assert entry["rejected_reason"] == "revert_decision"
