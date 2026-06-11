# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Roofline Comparison pipeline tests — `report.py` ↔ `roofline_snapshots` (PR #321).

After PR #321 retired ``last_trace_analyze_baseline``, ``final.json``'s
``roofline_comparison`` block is built from the append-only
``SharedState.roofline_snapshots`` history, and the markdown section drops the
retired N31 trigger wording.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.report import (
    _build_summary_dict,
    _format_roofline_comparison_section,
)


# Test fixtures
def _snapshot(
    *,
    snapshot_id: int,
    analysis_md_path: str,
    ts: str,
    compute_pct: float | None = 67.26,
    idle_pct: float | None = 32.72,
    comm_pct: float | None = 0.0,
    top_bottleneck: str | None = "GEMM",
    top_kernel_name: str | None = "aten::mm",
    top_kernel_efficiency_pct: float | None = 65.16,
    top_kernel_gpu_pct: float | None = 17.64,
    top_kernel_bound_type: str | None = "compute",
    trace_input: str = "/tmp/trace",
    kernel_roofline_path: str = "",
) -> dict[str, Any]:
    """Build a roofline_snapshots entry matching the new schema."""
    return {
        "snapshot_id": snapshot_id,
        "analysis_md_path": analysis_md_path,
        "trace_input": trace_input,
        "ts": ts,
        "kernel_roofline_path": kernel_roofline_path,
        "compute_pct": compute_pct,
        "idle_pct": idle_pct,
        "comm_pct": comm_pct,
        "top_bottleneck": top_bottleneck,
        "top_kernel": (
            {
                "name": top_kernel_name,
                "gpu_pct": top_kernel_gpu_pct,
                "efficiency_pct": top_kernel_efficiency_pct,
                "bound_type": top_kernel_bound_type,
            }
            if top_kernel_name is not None
            else None
        ),
    }


def _mock_state(
    *,
    roofline_snapshots: list[dict[str, Any]] | None = None,
    last_trace_analyze: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Minimal state for `_build_summary_dict`; omits the removed `last_trace_analyze_baseline` field."""
    return SimpleNamespace(
        session_id="test-session",
        model_name="test-model",
        model_path="/tmp/model",
        model_class="dense",
        stop_reason="report_emitted",
        baseline_tput=4309.2,
        baseline_accuracy=0.0,
        last_remaining_gaps_assessment={},
        remaining_gaps_assessments=[],
        current_best={"action": "params", "tput": 4357.27},
        cumulative_gain=1.12,
        cumulative_gain_validated=0.33,
        cumulative_gain_validated_ts="2026-05-24T13:47:22+00:00",
        cumulative_gain_validated_stack_len=2,
        optimization_stack=[
            {"action": "params", "variant_name": "v1"},
            {"action": "params", "variant_name": "v2"},
        ],
        crash_count=0,
        pruned_families=[],
        max_minutes=150,
        last_trace_analyze=last_trace_analyze or {},
        roofline_snapshots=list(roofline_snapshots or []),
    )


@pytest.fixture
def analysis_md(tmp_path):
    """Materialise a minimal TraceLens-shaped analysis.md on disk."""
    def _make(name: str, marker_text: str = "from snapshot") -> str:
        path = tmp_path / name
        path.write_text(
            "# TraceLens Report\n\n"
            "## Executive Summary\n\n"
            f"This is the executive summary {marker_text}.\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            "| Compute % | 67.26 |\n",
            encoding="utf-8",
        )
        return str(path)
    return _make


# _build_summary_dict — wire-shape tests
def test_build_summary_zero_snapshots_omits_roofline_comparison():
    """With no snapshots, `final.json` carries no `roofline_comparison` key."""
    state = _mock_state(roofline_snapshots=[], last_trace_analyze={})
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert "roofline_comparison" not in summary


def test_build_summary_single_snapshot_emits_single_snapshot_mode(analysis_md):
    """One snapshot → mode='single_snapshot', baseline == latest from `roofline_snapshots[0]`."""
    path = analysis_md("analysis_1.md")
    snap1 = _snapshot(
        snapshot_id=1, analysis_md_path=path, ts="2026-05-24T13:00:02+00:00",
    )
    state = _mock_state(
        roofline_snapshots=[snap1],
        last_trace_analyze={
            "roofline_snapshot_id": 1,
            "analysis_md_path": path,
            "ts": "2026-05-24T13:00:02+00:00",
            "trace_input": "/tmp/trace",
        },
    )
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    cmp = summary.get("roofline_comparison")
    assert cmp is not None, "roofline_comparison must be emitted"
    assert cmp.get("mode") == "single_snapshot"
    base = cmp.get("baseline") or {}
    latest = cmp.get("latest") or {}
    assert base.get("snapshot_id") == 1
    assert base.get("analysis_md_path") == path
    assert base.get("compute_pct") == 67.26
    assert latest.get("snapshot_id") == 1
    assert latest.get("analysis_md_path") == path


def test_build_summary_two_snapshots_emits_before_after_with_delta(analysis_md):
    """Two snapshots → mode='before_after' with baseline=first, latest=last, and a populated delta."""
    p1 = analysis_md("analysis_1.md", "baseline snapshot")
    p2 = analysis_md("analysis_2.md", "post-optimization snapshot")
    snap1 = _snapshot(
        snapshot_id=1, analysis_md_path=p1, ts="2026-05-24T13:00:00+00:00",
        compute_pct=67.26, idle_pct=32.72,
        top_kernel_name="aten::mm", top_kernel_efficiency_pct=65.16,
    )
    snap2 = _snapshot(
        snapshot_id=2, analysis_md_path=p2, ts="2026-05-24T13:45:00+00:00",
        compute_pct=75.10, idle_pct=24.90,
        top_kernel_name="aten::addmm", top_kernel_efficiency_pct=72.40,
    )
    state = _mock_state(
        roofline_snapshots=[snap1, snap2],
        last_trace_analyze={
            "roofline_snapshot_id": 2,
            "analysis_md_path": p2,
            "ts": "2026-05-24T13:45:00+00:00",
        },
    )
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    cmp = summary.get("roofline_comparison")
    assert cmp is not None
    assert cmp.get("mode") == "before_after"
    base = cmp["baseline"]
    latest = cmp["latest"]
    assert base["snapshot_id"] == 1
    assert latest["snapshot_id"] == 2
    delta = cmp.get("delta") or {}
    assert pytest.approx(delta.get("compute_pct"), abs=0.01) == 7.84
    assert pytest.approx(delta.get("idle_pct"), abs=0.01) == -7.82
    assert pytest.approx(delta.get("top_kernel_efficiency_pct"), abs=0.01) == 7.24


# _format_roofline_comparison_section — markdown prose tests
def test_format_section_zero_snapshots_says_none_captured():
    """Rendering with no baseline/latest surfaces the explicit 'no snapshot captured' marker."""
    md = "\n".join(_format_roofline_comparison_section({}))
    assert "## Roofline Comparison" in md
    assert "No roofline snapshot was captured" in md


def test_format_section_single_snapshot_no_n31_wording(tmp_path):
    """A single-snapshot section drops the 'none captured' + N31 wording and explains the PR #321 watermark mechanism."""
    p = tmp_path / "analysis.md"
    p.write_text(
        "# TraceLens\n\n## Executive Summary\n\nbody\n",
        encoding="utf-8",
    )
    snap = _snapshot(
        snapshot_id=1, analysis_md_path=str(p),
        ts="2026-05-24T13:00:00+00:00",
    )
    cmp = {"mode": "single_snapshot", "baseline": snap, "latest": snap}
    md = "\n".join(_format_roofline_comparison_section(cmp))
    assert "## Roofline Comparison" in md
    assert "No roofline snapshot was captured" not in md
    assert "Per N31 design" not in md
    assert "N31" not in md
    assert "watermark" in md.lower() or "10% gain" in md.lower()


def test_format_section_before_after_renders_both_blocks(tmp_path):
    """Two-snapshot mode renders distinct baseline and post-optimization blocks."""
    p1 = tmp_path / "a1.md"
    p1.write_text("# TL\n\n## Executive Summary\n\nbody1\n", encoding="utf-8")
    p2 = tmp_path / "a2.md"
    p2.write_text("# TL\n\n## Executive Summary\n\nbody2\n", encoding="utf-8")
    snap1 = _snapshot(
        snapshot_id=1, analysis_md_path=str(p1),
        ts="2026-05-24T13:00:00+00:00",
    )
    snap2 = _snapshot(
        snapshot_id=2, analysis_md_path=str(p2),
        ts="2026-05-24T13:45:00+00:00",
        compute_pct=75.10,
    )
    cmp = {
        "mode": "before_after",
        "baseline": snap1,
        "latest": snap2,
        "delta": {"compute_pct": 7.84},
    }
    md = "\n".join(_format_roofline_comparison_section(cmp))
    assert "## Roofline Comparison" in md
    assert "Baseline snapshot #1" in md
    assert "Post-optimization snapshot #2" in md
    assert "N31" not in md


# kernel_roofline_path sidecar (9fe4609 contract)
def test_build_summary_propagates_kernel_roofline_path(analysis_md):
    """The ``kernel_roofline_path`` sidecar pointer survives into ``final.json`` (9fe4609)."""
    path = analysis_md("analysis_1.md")
    snap = _snapshot(
        snapshot_id=1,
        analysis_md_path=path,
        ts="2026-05-24T13:00:02+00:00",
        kernel_roofline_path="/tmp/session/reports/kernel_roofline.json",
    )
    state = _mock_state(roofline_snapshots=[snap])
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    cmp = summary.get("roofline_comparison")
    assert cmp is not None
    assert (
        cmp["baseline"].get("kernel_roofline_path")
        == "/tmp/session/reports/kernel_roofline.json"
    )
    assert (
        cmp["latest"].get("kernel_roofline_path")
        == "/tmp/session/reports/kernel_roofline.json"
    )


def test_build_roofline_snapshot_default_carries_empty_kernel_roofline_path(
    tmp_path,
):
    """``build_roofline_snapshot`` always exposes ``kernel_roofline_path`` (empty when not injected)."""
    from inference_optimizer.orchestrator.roofline_snapshot import (
        build_roofline_snapshot,
    )
    p = tmp_path / "analysis.md"
    p.write_text(
        "# TraceLens\n\n## Executive Summary\n\nbody\n", encoding="utf-8",
    )
    snap = build_roofline_snapshot(
        snapshot_id=1,
        ts="2026-05-24T13:00:00+00:00",
        analysis_md_path=str(p),
    )
    assert "kernel_roofline_path" in snap
    assert snap["kernel_roofline_path"] == ""


# Decode-roofline ceiling propagation (Step 2).
def _snapshot_with_ceiling(
    *,
    snapshot_id: int,
    analysis_md_path: str,
    ts: str,
    theoretical_peak_tok_per_sec: float | None,
    achieved_tok_per_sec: float | None,
    within_roofline_pct: float | None,
    gap_to_roofline_pct: float | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extend ``_snapshot`` with the new ceiling fields."""
    base = _snapshot(
        snapshot_id=snapshot_id,
        analysis_md_path=analysis_md_path,
        ts=ts,
        **kwargs,
    )
    base["theoretical_peak_tok_per_sec"] = theoretical_peak_tok_per_sec
    base["achieved_tok_per_sec"] = achieved_tok_per_sec
    base["within_roofline_pct"] = within_roofline_pct
    base["gap_to_roofline_pct"] = gap_to_roofline_pct
    return base


class TestBuildSnapshotCeilingFields:
    """``build_roofline_snapshot`` derives within/gap from peak + achieved."""

    def test_default_kwargs_yield_none_ceiling_fields(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        p = tmp_path / "analysis.md"
        p.write_text("# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8")
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="2026-05-24T13:00:00+00:00",
            analysis_md_path=str(p),
        )
        assert snap["theoretical_peak_tok_per_sec"] is None
        assert snap["achieved_tok_per_sec"] is None
        assert snap["within_roofline_pct"] is None
        assert snap["gap_to_roofline_pct"] is None

    def test_peak_plus_achieved_yields_within_and_gap(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        p = tmp_path / "analysis.md"
        p.write_text("# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8")
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="ts",
            analysis_md_path=str(p),
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
        )
        assert snap["theoretical_peak_tok_per_sec"] == 1000.0
        assert snap["achieved_tok_per_sec"] == 527.5
        assert snap["within_roofline_pct"] == 52.75
        assert snap["gap_to_roofline_pct"] == pytest.approx(47.25, abs=0.01)

    def test_zero_peak_keeps_within_gap_none(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        p = tmp_path / "analysis.md"
        p.write_text("# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8")
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="ts",
            analysis_md_path=str(p),
            theoretical_peak_tok_per_sec=0.0,
            achieved_tok_per_sec=527.5,
        )
        assert snap["theoretical_peak_tok_per_sec"] is None
        assert snap["within_roofline_pct"] is None
        assert snap["gap_to_roofline_pct"] is None


class TestComparisonDeltaIncludesWithinRoofline:
    """The comparison delta carries both within_roofline_pct and gap_to_roofline_pct."""

    def test_before_after_delta_gap_to_roofline_pct(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_comparison_from_history,
        )
        snap_base = _snapshot_with_ceiling(
            snapshot_id=1,
            analysis_md_path="/tmp/base.md",
            ts="2026-05-28T10:00:00+00:00",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        snap_latest = _snapshot_with_ceiling(
            snapshot_id=2,
            analysis_md_path="/tmp/latest.md",
            ts="2026-05-28T11:00:00+00:00",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=690.89,
            within_roofline_pct=69.09,
            gap_to_roofline_pct=30.91,
        )
        cmp = build_roofline_comparison_from_history([snap_base, snap_latest])
        delta = (cmp or {}).get("delta") or {}
        assert pytest.approx(delta.get("gap_to_roofline_pct"), abs=0.01) == -16.34

    def test_format_table_renders_gap_delta(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            format_roofline_metrics_table,
        )
        base = _snapshot_with_ceiling(
            snapshot_id=1,
            analysis_md_path="/tmp/base.md",
            ts="2026-05-28T10:00:00+00:00",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        latest = _snapshot_with_ceiling(
            snapshot_id=2,
            analysis_md_path="/tmp/latest.md",
            ts="2026-05-28T11:00:00+00:00",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=690.89,
            within_roofline_pct=69.09,
            gap_to_roofline_pct=30.91,
        )
        cmp = {
            "mode": "before_after",
            "baseline": base,
            "latest": latest,
            "delta": {
                "within_roofline_pct": 16.34,
                "gap_to_roofline_pct": -16.34,
            },
        }
        text = "\n".join(format_roofline_metrics_table(cmp))
        assert "Gap to roofline %" in text
        assert "-16.3" in text

    def test_before_after_delta_within_roofline_pct(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_comparison_from_history,
        )
        snap_base = _snapshot_with_ceiling(
            snapshot_id=1, analysis_md_path="/p1", ts="t1",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        snap_latest = _snapshot_with_ceiling(
            snapshot_id=2, analysis_md_path="/p2", ts="t2",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=690.89,
            within_roofline_pct=69.09,
            gap_to_roofline_pct=30.91,
        )
        cmp = build_roofline_comparison_from_history([snap_base, snap_latest])
        assert cmp is not None
        assert cmp["mode"] == "before_after"
        delta = cmp.get("delta") or {}
        assert pytest.approx(delta.get("within_roofline_pct"), abs=0.01) == 16.34


class TestSummaryDictCarriesCeilingThroughHistory:
    """End-to-end: a history snapshot's ceiling fields survive into summary['roofline_comparison']."""

    def test_single_snapshot_propagates_ceiling_to_summary(self, analysis_md):
        path = analysis_md("a.md")
        snap = _snapshot_with_ceiling(
            snapshot_id=1, analysis_md_path=path,
            ts="2026-05-24T13:00:00+00:00",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        state = _mock_state(roofline_snapshots=[snap])
        summary = _build_summary_dict(state, ev_counts={}, highlights=[])
        cmp = summary["roofline_comparison"]
        base = cmp["baseline"]
        assert base["theoretical_peak_tok_per_sec"] == 1000.0
        assert base["achieved_tok_per_sec"] == 527.5
        assert base["within_roofline_pct"] == 52.75
        assert base["gap_to_roofline_pct"] == 47.25


class TestFormatTableRendersCeiling:
    """``format_roofline_metrics_table`` renders the ceiling once plus achieved/within/gap rows."""

    def test_single_snapshot_renders_ceiling_and_within_rows(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            format_roofline_metrics_table,
        )
        snap = _snapshot_with_ceiling(
            snapshot_id=1, analysis_md_path="/p", ts="t",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        lines = format_roofline_metrics_table({
            "mode": "single_snapshot",
            "baseline": snap,
            "latest": snap,
        })
        text = "\n".join(lines)
        assert "Theoretical peak" in text
        assert "1000.0 tok/s" in text
        assert "Achieved output_throughput" in text
        assert "527.5 tok/s" in text
        assert "Within roofline %" in text
        assert "52.8%" in text or "52.75%" in text or "52.7%" in text
        assert "Gap to roofline %" in text

    def test_before_after_renders_ceiling_once_and_within_delta(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            format_roofline_metrics_table,
        )
        base = _snapshot_with_ceiling(
            snapshot_id=1, analysis_md_path="/p1", ts="t1",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=527.5,
            within_roofline_pct=52.75,
            gap_to_roofline_pct=47.25,
        )
        latest = _snapshot_with_ceiling(
            snapshot_id=2, analysis_md_path="/p2", ts="t2",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=690.89,
            within_roofline_pct=69.09,
            gap_to_roofline_pct=30.91,
        )
        cmp = {
            "mode": "before_after",
            "baseline": base,
            "latest": latest,
            "delta": {"within_roofline_pct": 16.34},
        }
        lines = format_roofline_metrics_table(cmp)
        text = "\n".join(lines)
        assert text.count("Theoretical peak") == 1
        assert "1000.0 tok/s" in text
        assert "527.5 tok/s" in text
        assert "690.9 tok/s" in text
        assert "Within roofline %" in text
        assert "+16.3" in text

    def test_table_without_ceiling_omits_ceiling_row(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            format_roofline_metrics_table,
        )
        snap = _snapshot(
            snapshot_id=1, analysis_md_path="/p", ts="t",
        )
        snap["theoretical_peak_tok_per_sec"] = None
        snap["achieved_tok_per_sec"] = None
        snap["within_roofline_pct"] = None
        snap["gap_to_roofline_pct"] = None
        lines = format_roofline_metrics_table({
            "mode": "single_snapshot",
            "baseline": snap,
            "latest": snap,
        })
        text = "\n".join(lines)
        assert "Theoretical peak" not in text


class TestRecordTraceAnalyzeStampsCeiling:
    """``record_trace_analyze`` stamps ceiling + achieved + within + gap onto every history snapshot."""

    @staticmethod
    def _mock_breakdown(monkeypatch, *, mem=0.0, cmp=0.0, peak=0.0, kind="unknown"):
        """Patch ``compute_roofline_breakdown_from_state`` to return a stub ``RooflineBreakdown``."""
        from inference_optimizer.orchestrator import roofline_ceiling
        from inference_optimizer.orchestrator.roofline_ceiling import (
            RooflineBreakdown,
        )
        br = RooflineBreakdown(mem, cmp, peak, kind)
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_roofline_breakdown_from_state",
            lambda _state, **_kw: br,
        )

    def test_stamps_ceiling_and_achieved_into_history(
        self, tmp_path, monkeypatch
    ):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(
            monkeypatch, mem=1000.0, cmp=5000.0, peak=1000.0, kind="memory",
        )
        md = tmp_path / "analysis.md"
        md.write_text(
            "# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8"
        )
        state = SharedState()
        state.baseline_tput = 527.5
        state.current_best = {"action": "params", "tput": 690.89}
        state.record_trace_analyze(
            {"trace_input": "/tmp/trace.json"},
            {
                "hot_kernels": [],
                "trace_report_path": str(md),
                "candidates_path": "/tmp/kc.json",
            },
        )
        assert len(state.roofline_snapshots) == 1
        snap = state.roofline_snapshots[0]
        assert snap["theoretical_peak_tok_per_sec"] == 1000.0
        assert snap["achieved_tok_per_sec"] == 690.89
        assert snap["within_roofline_pct"] == 69.09
        assert snap["gap_to_roofline_pct"] == pytest.approx(30.91, abs=0.01)

    def test_falls_back_to_baseline_tput_when_no_current_best(
        self, tmp_path, monkeypatch
    ):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(
            monkeypatch, mem=1000.0, cmp=5000.0, peak=1000.0, kind="memory",
        )
        md = tmp_path / "analysis.md"
        md.write_text(
            "# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8"
        )
        state = SharedState()
        state.baseline_tput = 527.5
        state.current_best = {}
        state.record_trace_analyze(
            {"trace_input": "/tmp/trace.json"},
            {
                "hot_kernels": [],
                "trace_report_path": str(md),
                "candidates_path": "/tmp/kc.json",
            },
        )
        snap = state.roofline_snapshots[0]
        assert snap["achieved_tok_per_sec"] == 527.5
        assert snap["within_roofline_pct"] == 52.75

    def test_zero_peak_keeps_within_gap_none(self, tmp_path, monkeypatch):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(monkeypatch)
        md = tmp_path / "analysis.md"
        md.write_text(
            "# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8"
        )
        state = SharedState()
        state.baseline_tput = 527.5
        state.record_trace_analyze(
            {"trace_input": "/tmp/trace.json"},
            {
                "hot_kernels": [],
                "trace_report_path": str(md),
                "candidates_path": "/tmp/kc.json",
            },
        )
        snap = state.roofline_snapshots[0]
        assert snap["theoretical_peak_tok_per_sec"] is None
        assert snap["within_roofline_pct"] is None
        assert snap["gap_to_roofline_pct"] is None


class TestBuildSnapshotTwoSidedRoofline:
    """``build_roofline_snapshot`` carries T_mem / T_cmp / bound_kind alongside the legacy peak field."""

    def test_two_sided_fields_default_to_unknown_for_legacy_callers(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="2026-05-30T07:00:00Z",
            analysis_md_path="",
            theoretical_peak_tok_per_sec=1000.0,
            achieved_tok_per_sec=500.0,
        )
        assert snap["theoretical_peak_tok_per_sec"] == 1000.0
        assert snap["roofline_mem_ceiling_tok_per_sec"] is None
        assert snap["roofline_cmp_ceiling_tok_per_sec"] is None
        assert snap["roofline_bound_kind"] == "unknown"

    def test_two_sided_fields_populated_when_provided(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="2026-05-30T07:00:00Z",
            analysis_md_path="",
            theoretical_peak_tok_per_sec=8000.0,
            achieved_tok_per_sec=6244.0,
            mem_ceiling_tok_per_sec=8000.0,
            cmp_ceiling_tok_per_sec=40_000.0,
            bound_kind="memory",
        )
        assert snap["roofline_mem_ceiling_tok_per_sec"] == 8000.0
        assert snap["roofline_cmp_ceiling_tok_per_sec"] == 40_000.0
        assert snap["roofline_bound_kind"] == "memory"
        # Caller-provided primary ceiling is persisted separately from sides.
        assert snap["theoretical_peak_tok_per_sec"] == 8000.0

    def test_zero_mem_cmp_serialize_as_none(self):
        from inference_optimizer.orchestrator.roofline_snapshot import (
            build_roofline_snapshot,
        )
        snap = build_roofline_snapshot(
            snapshot_id=1, ts="t", analysis_md_path="",
            theoretical_peak_tok_per_sec=0.0, achieved_tok_per_sec=0.0,
            mem_ceiling_tok_per_sec=0.0, cmp_ceiling_tok_per_sec=0.0,
            bound_kind="unknown",
        )
        assert snap["roofline_mem_ceiling_tok_per_sec"] is None
        assert snap["roofline_cmp_ceiling_tok_per_sec"] is None
        assert snap["roofline_bound_kind"] == "unknown"


class TestRecordTraceAnalyzeStampsTwoSidedRoofline:
    """``record_trace_analyze`` propagates the four breakdown fields into the history snapshot."""

    @staticmethod
    def _mock_breakdown(monkeypatch, *, mem, cmp, peak, kind):
        from inference_optimizer.orchestrator import roofline_ceiling
        from inference_optimizer.orchestrator.roofline_ceiling import (
            RooflineBreakdown,
        )
        br = RooflineBreakdown(mem, cmp, peak, kind)
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_roofline_breakdown_from_state",
            lambda _state, **_kw: br,
        )

    def _record(self, tmp_path, state):
        md = tmp_path / "analysis.md"
        md.write_text(
            "# TL\n\n## Executive Summary\n\nbody\n", encoding="utf-8"
        )
        state.record_trace_analyze(
            {"trace_input": "/tmp/trace.json"},
            {
                "hot_kernels": [],
                "trace_report_path": str(md),
                "candidates_path": "/tmp/kc.json",
            },
        )
        return state.roofline_snapshots[0]

    def test_memory_bound_breakdown_propagates_to_snapshot(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(
            monkeypatch, mem=8065.0, cmp=375_612.0, peak=8065.0,
            kind="memory",
        )
        state = SharedState()
        state.baseline_tput = 6244.3
        state.current_best = {}
        snap = self._record(tmp_path, state)
        assert snap["theoretical_peak_tok_per_sec"] == 8065.0
        assert snap["roofline_mem_ceiling_tok_per_sec"] == 8065.0
        assert snap["roofline_cmp_ceiling_tok_per_sec"] == 375_612.0
        assert snap["roofline_bound_kind"] == "memory"
        assert snap["within_roofline_pct"] == pytest.approx(77.43, abs=0.05)

    def test_compute_bound_breakdown_propagates_to_snapshot(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(
            monkeypatch, mem=8000.0, cmp=2000.0, peak=2000.0,
            kind="compute",
        )
        state = SharedState()
        state.baseline_tput = 1500.0
        state.current_best = {}
        snap = self._record(tmp_path, state)
        assert snap["theoretical_peak_tok_per_sec"] == 2000.0
        assert snap["roofline_bound_kind"] == "compute"

    def test_unknown_breakdown_keeps_all_ceiling_fields_none(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer.orchestrator.shared_state import SharedState
        self._mock_breakdown(
            monkeypatch, mem=0.0, cmp=0.0, peak=0.0, kind="unknown",
        )
        state = SharedState()
        state.baseline_tput = 1000.0
        state.current_best = {}
        snap = self._record(tmp_path, state)
        assert snap["theoretical_peak_tok_per_sec"] is None
        assert snap["roofline_mem_ceiling_tok_per_sec"] is None
        assert snap["roofline_cmp_ceiling_tok_per_sec"] is None
        assert snap["roofline_bound_kind"] == "unknown"
        assert snap["within_roofline_pct"] is None

    def test_perfmodel_breakdown_persists_decode_sides_and_bound(
        self, tmp_path, monkeypatch,
    ):
        from inference_optimizer.orchestrator import roofline_ceiling
        from inference_optimizer.orchestrator.shared_state import SharedState

        self._mock_breakdown(
            monkeypatch, mem=8000.0, cmp=40_000.0, peak=7900.0,
            kind="memory",
        )
        monkeypatch.setattr(roofline_ceiling, "load_model_meta", lambda *a, **kw: object())
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_roofline_from_perfmodel",
            lambda **kw: SimpleNamespace(
                decode_tok_per_s=7900.0,
                prefill_tok_per_s=123.0,
                decode_mem_tok_per_s=8000.0,
                decode_cmp_tok_per_s=40_000.0,
                bound_kind="memory",
                hbm_bw_gbps=5300.0,
                peak_achievable_tflops=708.0,
                ops=[],
            ),
        )
        state = SharedState()
        state.baseline_tput = 1000.0
        snap = self._record(tmp_path, state)
        perf = snap["perfmodel_breakdown"]

        assert perf["decode_tok_per_s"] == 7900.0
        assert perf["decode_mem_tok_per_s"] == 8000.0
        assert perf["decode_cmp_tok_per_s"] == 40_000.0
        assert perf["bound_kind"] == "memory"
