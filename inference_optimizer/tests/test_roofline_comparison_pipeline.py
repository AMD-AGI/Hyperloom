"""Roofline Comparison pipeline tests — `report.py` ↔ `roofline_snapshots`.

Covers the data path that produces ``final.json["roofline_comparison"]``
and renders ``## Roofline Comparison`` in ``final.md`` after PR #321
retired the legacy ``last_trace_analyze_baseline`` baseline-freeze
field. The new contract is:

* ``SharedState.roofline_snapshots`` accumulates one entry per
  successful ``record_trace_analyze``; entries are append-only so the
  first one survives every watermark-driven refresh of
  ``last_trace_analyze``.
* ``_build_summary_dict`` reads that history (not the removed
  ``last_trace_analyze_baseline``) to populate the
  ``roofline_comparison`` block on ``final.json``.
* ``_format_roofline_comparison_section`` renders the matching
  ``## Roofline Comparison`` section; its prose no longer references
  the retired N31 "final-roofline before report" trigger.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors.report import (
    _build_summary_dict,
    _format_roofline_comparison_section,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
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
        # 9fe4609 sidecar pointer to reports/kernel_roofline.json.
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
    """Minimal state for `_build_summary_dict`.

    Deliberately omits `last_trace_analyze_baseline` so the test
    reproduces the post-PR#321 production state (field removed from
    SharedState).
    """
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


# ---------------------------------------------------------------------------
# _build_summary_dict — wire-shape tests
# ---------------------------------------------------------------------------
def test_build_summary_zero_snapshots_omits_roofline_comparison():
    """With no captured snapshots, `final.json` must not carry a
    `roofline_comparison` key at all (vs. an empty stub)."""
    state = _mock_state(roofline_snapshots=[], last_trace_analyze={})
    summary = _build_summary_dict(state, ev_counts={}, highlights=[])
    assert "roofline_comparison" not in summary


def test_build_summary_single_snapshot_emits_single_snapshot_mode(analysis_md):
    """One captured snapshot → mode='single_snapshot', baseline == latest,
    both populated from `roofline_snapshots[0]`."""
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
    """Two captured snapshots → mode='before_after', baseline=first,
    latest=last, delta populated for the diffable fields."""
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


# ---------------------------------------------------------------------------
# _format_roofline_comparison_section — markdown prose tests
# ---------------------------------------------------------------------------
def test_format_section_zero_snapshots_says_none_captured():
    """Rendering with no baseline/latest must surface the explicit
    'no snapshot captured' marker (kept for back-compat)."""
    md = "\n".join(_format_roofline_comparison_section({}))
    assert "## Roofline Comparison" in md
    assert "No roofline snapshot was captured" in md


def test_format_section_single_snapshot_no_n31_wording(tmp_path):
    """With a single snapshot, the section must:
    (a) NOT claim 'no snapshot was captured';
    (b) NOT cite the retired N31 trigger contract;
    (c) explain the new (PR #321) gain/watermark mechanism.
    """
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
    """Two-snapshot mode renders distinct baseline and post-optimization
    blocks (with the corresponding snapshot ids)."""
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


# ---------------------------------------------------------------------------
# kernel_roofline_path sidecar (9fe4609 contract)
# ---------------------------------------------------------------------------
def test_build_summary_propagates_kernel_roofline_path(analysis_md):
    """The per-kernel roofline sidecar path written by
    ``tracelens_analysis.py`` (``reports/kernel_roofline.json``) must
    survive into ``final.json`` so dashboards have a stable pointer to
    the per-kernel arithmetic-intensity / efficiency table.

    9fe4609 (``feat: add kernel roofline sidecar evidence``) put the
    field on ``roofline_comparison.latest``; PR-321's history-driven
    rewrite must keep that surface non-empty whenever the underlying
    ``state.roofline_snapshots`` entry carries the path.
    """
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
    # single_snapshot mode → both sides carry the same sidecar pointer.
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
    """``roofline_snapshot.build_roofline_snapshot`` is the canonical
    factory; even when callers only have an analysis.md (no
    ``kernel_roofline_path`` to inject) the returned dict MUST still
    expose the key so downstream readers don't ``KeyError``."""
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
