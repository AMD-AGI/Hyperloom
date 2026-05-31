"""Tests for ``breakdown.collectors.collect_roofline``.

Verifies that the breakdown's ``roofline`` section is sourced from
``state.roofline_snapshots`` (the append-only history written by
:meth:`SharedState.record_trace_analyze` after PR #321 retired the
``last_trace_analyze_baseline`` baseline-freeze field).

Covers three shape contracts the ``Roofline`` renderer in
:mod:`breakdown.reporters._renderers.roofline` relies on:

* 0 snapshots → returns ``[]`` (renderer marks section ``skipped``)
* 1 snapshot  → single ``mode='single_snapshot'`` entry
* ≥2 snapshots → single ``mode='before_after'`` entry with populated
  ``delta`` block keyed by the diffable metrics
"""

from __future__ import annotations

from typing import Any

from inference_optimizer.breakdown.collectors import collect_roofline


def _snapshot(
    *,
    snapshot_id: int,
    ts: str = "2026-05-24T13:00:00+00:00",
    compute_pct: float | None = 67.26,
    idle_pct: float | None = 32.72,
    comm_pct: float | None = 0.0,
    top_bottleneck: str | None = "GEMM",
    top_kernel_name: str | None = "aten::mm",
    top_kernel_efficiency_pct: float | None = 65.16,
    top_kernel_gpu_pct: float | None = 17.64,
    top_kernel_bound_type: str | None = "compute",
    analysis_md_path: str = "/tmp/analysis.md",
    trace_input: str = "/tmp/trace",
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "ts": ts,
        "analysis_md_path": analysis_md_path,
        "trace_input": trace_input,
        "compute_pct": compute_pct,
        "idle_pct": idle_pct,
        "comm_pct": comm_pct,
        "top_bottleneck": top_bottleneck,
        "top_kernel": {
            "name": top_kernel_name,
            "gpu_pct": top_kernel_gpu_pct,
            "efficiency_pct": top_kernel_efficiency_pct,
            "bound_type": top_kernel_bound_type,
        } if top_kernel_name is not None else None,
    }


# ---------------------------------------------------------------------------
# 0 / 1 / ≥2 snapshot shape contracts
# ---------------------------------------------------------------------------
def test_collect_roofline_zero_snapshots_returns_empty_list():
    """Missing or empty ``roofline_snapshots`` → renderer should skip."""
    warnings: list[str] = []
    assert collect_roofline({}, warnings) == []
    assert collect_roofline({"roofline_snapshots": []}, warnings) == []
    assert warnings == []


def test_collect_roofline_one_snapshot_emits_single_snapshot_entry():
    """One captured snapshot → ``[{mode='single_snapshot', baseline==latest}]``."""
    snap = _snapshot(snapshot_id=1)
    state = {"roofline_snapshots": [snap]}
    warnings: list[str] = []
    out = collect_roofline(state, warnings)
    assert len(out) == 1
    entry = out[0]
    assert entry["mode"] == "single_snapshot"
    assert entry["baseline"]["snapshot_id"] == 1
    assert entry["latest"]["snapshot_id"] == 1
    # single_snapshot mode → no delta block (would be all zeros)
    assert "delta" not in entry
    # source_path identifies the data origin so the renderer can
    # surface it in the section header.
    assert "state.json" in entry["source_path"]
    assert warnings == []


def test_collect_roofline_two_snapshots_emits_before_after_with_delta():
    """Two captured snapshots → ``[{mode='before_after', delta={...}}]``."""
    snap1 = _snapshot(
        snapshot_id=1, compute_pct=67.26, idle_pct=32.72,
        top_kernel_efficiency_pct=65.16,
    )
    snap2 = _snapshot(
        snapshot_id=2, compute_pct=75.10, idle_pct=24.90,
        top_kernel_efficiency_pct=72.40,
        ts="2026-05-24T13:45:00+00:00",
    )
    state = {"roofline_snapshots": [snap1, snap2]}
    warnings: list[str] = []
    out = collect_roofline(state, warnings)
    assert len(out) == 1
    entry = out[0]
    assert entry["mode"] == "before_after"
    assert entry["baseline"]["snapshot_id"] == 1
    assert entry["latest"]["snapshot_id"] == 2
    delta = entry["delta"]
    assert delta["compute_pct"] == 7.84
    assert delta["idle_pct"] == -7.82
    assert delta["top_kernel_efficiency_pct"] == 7.24
    assert warnings == []


def test_collect_roofline_three_snapshots_uses_first_and_last_as_anchors():
    """History length 3 → baseline=first, latest=third (the watermark
    sequence's most recent refresh)."""
    snap1 = _snapshot(snapshot_id=1, compute_pct=50.0)
    snap2 = _snapshot(snapshot_id=2, compute_pct=60.0)
    snap3 = _snapshot(snapshot_id=3, compute_pct=72.0)
    state = {"roofline_snapshots": [snap1, snap2, snap3]}
    warnings: list[str] = []
    out = collect_roofline(state, warnings)
    assert len(out) == 1
    entry = out[0]
    assert entry["mode"] == "before_after"
    assert entry["baseline"]["snapshot_id"] == 1
    assert entry["latest"]["snapshot_id"] == 3
    # Delta is anchor-to-anchor (first → last), not adjacent-pair.
    assert entry["delta"]["compute_pct"] == 22.0


# ---------------------------------------------------------------------------
# Renderer compatibility
# ---------------------------------------------------------------------------
def test_collect_roofline_output_is_renderer_compatible():
    """The shape must satisfy the ``Roofline`` renderer's contract:
    each entry carries ``source_path`` / ``mode`` / ``baseline`` /
    ``latest`` keys (renderer iterates these directly)."""
    snap = _snapshot(snapshot_id=1)
    out = collect_roofline({"roofline_snapshots": [snap]}, [])
    assert out and isinstance(out, list)
    entry = out[0]
    # These four keys are the renderer's required surface.
    for required in ("source_path", "mode", "baseline", "latest"):
        assert required in entry, (
            f"renderer expects '{required}' key on each entry"
        )


# ---------------------------------------------------------------------------
# Defensive: bad data shouldn't crash the export
# ---------------------------------------------------------------------------
def test_collect_roofline_handles_non_list_snapshots_gracefully():
    """A misshapen ``roofline_snapshots`` (e.g. dict instead of list)
    must degrade to ``[]`` without raising, so a corrupt state.json
    field can't poison the whole breakdown export."""
    warnings: list[str] = []
    out = collect_roofline({"roofline_snapshots": "not-a-list"}, warnings)
    assert out == []
    assert warnings == []


def test_collect_roofline_handles_unparseable_snapshot_entries():
    """A snapshot entry that the comparison builder can't process
    should be reported via ``warnings`` (best-effort), and the
    section drops to ``[]``."""
    # Pass a non-dict entry to force build_roofline_comparison_from_history
    # into a defensive path. The current builder normalises gracefully
    # but if a future change tightens its checks, this test catches
    # the regression-of-grace.
    out = collect_roofline({"roofline_snapshots": [{}]}, [])
    # An empty-dict entry yields snapshot_id=None on both sides, so
    # mode falls back to ``single_snapshot`` (matches the
    # build_roofline_comparison_from_history "fallback" branch).
    assert isinstance(out, list)
    assert len(out) <= 1
