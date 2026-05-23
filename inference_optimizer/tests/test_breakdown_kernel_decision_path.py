"""Tests for P2-1: ``kernel_decision_path`` collector + renderer.

Covers:

* shape: ``build()`` returns ``kernel_decision_path: list`` and the
  field is always present (even when the session has zero kernel work).
* grouping: a synthetic state with select_kernels + kernel_opt history
  + integrate attempts ends up with one entry per kid, with steps
  ordered chronologically across step categories.
* enrichment: duration_seconds propagates from ``extras.duration_seconds``
  / per-attempt workspace ``benchmark_report.json``; ``backends_attempted``
  reflects the GEAK / OOB history; ``final_outcome`` is the terminal step.
* empty session: an essentially empty state yields ``[]`` without any
  new warnings.
* renderer: skips silently when the field is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.breakdown import build
from inference_optimizer.breakdown.collectors import collect_kernel_decision_path


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _full_state(tmp_path: Path) -> dict:
    """Synthetic state.json covering select + 2 backends of kernel_opt +
    one integrate attempt, plus one kid with no history."""
    return {
        "session_id": "kdp",
        "framework": "sglang",
        "last_select_kernels": {
            "ts": "2026-05-15T09:00:00+00:00",
            "hot_kernels_top15": [
                {
                    "kernel_id": "k001",
                    "name": "aten::mm",
                    "gpu_pct": 19.4,
                    "bottleneck": "compute",
                    "recommended_backends": ["geak", "claude"],
                    "recommended_actions": ["run_optimization"],
                    "reusable_native_kernel": False,
                },
                {
                    "kernel_id": "k002",
                    "name": "fused_rmsnorm",
                    "gpu_pct": 8.2,
                    "bottleneck": "memory",
                    "recommended_backends": ["claude"],
                    "recommended_actions": ["run_optimization"],
                    "reusable_native_kernel": True,
                },
            ],
        },
        "kernel_opt_attempts": {
            "k001": {
                "attempts": 2,
                "last_decision": "KEEP",
                "last_ts": "2026-05-15T10:30:00+00:00",
                "history": [
                    {
                        "decision": "PARTIAL",
                        "ts": "2026-05-15T10:00:00+00:00",
                        "gain_pct": 4.0,
                        "extras": {
                            "backend": "geak",
                            "duration_seconds": 90.0,
                            "task_id": "kopt-1",
                            "note": "compile_only",
                        },
                    },
                    {
                        "decision": "KEEP",
                        "ts": "2026-05-15T10:30:00+00:00",
                        "gain_pct": 12.5,
                        "extras": {
                            "backend": "claude",
                            "duration_seconds": 180.0,
                            "task_id": "kopt-2",
                            "note": "new_kernel_family",
                        },
                    },
                ],
            },
            "k002": {
                "attempts": 1,
                "last_decision": "REVERT",
                "last_ts": "2026-05-15T11:00:00+00:00",
                # No history list — exercise the terminal-decision
                # synthetic fallback.
            },
        },
        "kernel_integrate_attempts": {
            "k001|patches/k001/0001.patch|": {
                "key": "k001|patches/k001/0001.patch|",
                "kernel_id": "k001",
                "patch_path": "patches/k001/0001.patch",
                "target_file": "/path/to/mm.py",
                "attempts": [{
                    "decision": "KEEP",
                    "status": "succeeded",
                    "gain_pct": 11.0,
                    "ts": "2026-05-15T11:15:00+00:00",
                    "new_tput": 1110.0,
                }],
                "best_gain_pct": 11.0,
                "last_decision": "KEEP",
            },
        },
        "validate_stack_attempts": [{
            "ts": "2026-05-15T11:30:00+00:00",
            "task_id": "vs-1",
            "status": "succeeded",
            "decision": "promoted",
            "key_metric": 10.4,
            "extras": {"kernel_id": "k001", "note": "post-integrate"},
        }],
    }


def test_collect_kernel_decision_path_groups_and_orders(tmp_path: Path) -> None:
    state = _full_state(tmp_path)
    warnings: list[str] = []
    out = collect_kernel_decision_path(state, warnings, session_dir=None)
    by_kid = {e["kid"]: e for e in out}
    assert set(by_kid) == {"k001", "k002"}

    k1 = by_kid["k001"]
    assert k1["kernel_name"] == "aten::mm"
    steps_k1 = k1["steps"]
    # Ordered: select (09:00) → kernel_opt PARTIAL (10:00) →
    #          kernel_opt KEEP (10:30) → integrate KEEP (11:15) →
    #          validate promoted (11:30)
    assert [s["step"] for s in steps_k1] == [
        "select", "kernel_opt", "kernel_opt", "integrate", "validate",
    ]
    assert [s["ts"] for s in steps_k1] == [
        "2026-05-15T09:00:00+00:00",
        "2026-05-15T10:00:00+00:00",
        "2026-05-15T10:30:00+00:00",
        "2026-05-15T11:15:00+00:00",
        "2026-05-15T11:30:00+00:00",
    ]
    # duration_seconds populated from extras
    assert steps_k1[1]["duration_seconds"] == 90.0
    assert steps_k1[2]["duration_seconds"] == 180.0
    # ended_ts_utc derived
    assert steps_k1[1]["ended_ts_utc"] is not None
    assert steps_k1[1]["ended_ts_utc"].startswith("2026-05-15T10:01:30")
    # backends recovered from extras
    assert steps_k1[1]["backend"] == "geak"
    assert steps_k1[2]["backend"] == "claude"
    # outcome / gain_pct / decision_note plumbed through
    assert steps_k1[2]["outcome"] == "KEEP"
    assert steps_k1[2]["gain_pct"] == 12.5
    assert steps_k1[2]["decision_note"] == "new_kernel_family"
    assert steps_k1[3]["outcome"] == "KEEP"
    assert steps_k1[3]["gain_pct"] == 11.0
    # summary block
    summary_k1 = k1["summary"]
    assert summary_k1["total_steps"] == 5
    assert summary_k1["backends_attempted"] == ["geak", "claude"]
    assert summary_k1["final_outcome"] == "promoted"
    # k001 total_duration ≈ 90 + 180 (no duration on select / integrate / validate here)
    assert summary_k1["total_duration_seconds"] == 270.0

    # k002 — synthetic terminal-decision step from last_decision fallback
    k2 = by_kid["k002"]
    steps_k2 = k2["steps"]
    assert [s["step"] for s in steps_k2] == ["select", "kernel_opt"]
    assert steps_k2[1]["outcome"] == "REVERT"
    assert k2["summary"]["final_outcome"] == "REVERT"

    # No warnings emitted by this collector for synthetic, complete data.
    assert warnings == []


def test_collect_kernel_decision_path_empty_session() -> None:
    """An empty session (no select, no kernel_opt, no integrate) yields
    [] and no warnings."""
    warnings: list[str] = []
    out = collect_kernel_decision_path({}, warnings, session_dir=None)
    assert out == []
    assert warnings == []


def test_build_includes_kernel_decision_path_field(tmp_path: Path) -> None:
    """End-to-end via :func:`build` — top-level field is always present."""
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kdp-e2e"})
    _write_json(sd / "state.json", _full_state(tmp_path))
    bd = build(sd)
    assert "kernel_decision_path" in bd
    assert isinstance(bd["kernel_decision_path"], list)
    assert {e["kid"] for e in bd["kernel_decision_path"]} == {"k001", "k002"}


def test_build_includes_kernel_decision_path_when_empty(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "kdp-empty"})
    _write_json(sd / "state.json", {"session_id": "kdp-empty"})
    bd = build(sd)
    assert bd.get("kernel_decision_path") == []


def test_renderer_skips_when_field_absent(tmp_path: Path) -> None:
    from inference_optimizer.breakdown.reporters._renderers.kernel_decision_path import render
    # Field omitted entirely (legacy JSON shape) → skipped, no warnings.
    rs = render({})
    assert rs.skipped is True
    assert rs.warnings == []


def test_renderer_renders_table_when_populated(tmp_path: Path) -> None:
    from inference_optimizer.breakdown.reporters._renderers.kernel_decision_path import render
    state = _full_state(tmp_path)
    warnings: list[str] = []
    path = collect_kernel_decision_path(state, warnings, session_dir=None)
    rs = render({"kernel_decision_path": path})
    assert rs.skipped is False
    assert "k001" in rs.markdown_block
    assert "kernel_opt" in rs.markdown_block
    assert "integrate" in rs.markdown_block
    # funnel fact present
    assert any("Funnel:" in f for f in rs.key_facts)
