"""Tests for §15 ``roofline`` collector + renderer.

Covers the full path:

* ``collect_roofline`` discovers ``final.json`` files under common
  layouts (``reports/`` for the orchestrator final report, plus a few
  fall-back locations for the standalone roofline tool).
* The collector recognises both on-disk shapes — top-level
  ``mode``/``baseline``/``latest``/``delta`` AND wrapped under
  ``roofline_comparison``.
* Multiple ``final.json`` files are surfaced in mtime order
  (oldest first) so the list conveys an obvious timeline.
* Missing snapshots / missing delta land as None / {}; invalid JSON
  becomes a warning without crashing.
* The renderer skips silently when no roofline data is available, and
  produces a markdown block containing the key facts when it is.

The synthetic fixtures here mirror both the user-supplied schema
(``before_after`` mode, full ``delta``) and the real-session shape
observed on the test session (``single_snapshot`` mode, no ``delta``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from inference_optimizer.breakdown import build
from inference_optimizer.breakdown.collectors import collect_roofline
from inference_optimizer.breakdown.reporters.compose import render_session_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BEFORE_AFTER_BLOB = {
    "mode": "before_after",
    "baseline": {
        "snapshot_id": 1,
        "ts": "2026-05-22T11:00:00+00:00",
        "compute_pct": 86.2,
        "idle_pct": 5.26,
        "comm_pct": 8.51,
        "top_bottleneck": "MoE_fused",
        "top_kernel": {
            "name": "aiter::fmoe_fp8_blockscale_g1u1",
            "gpu_pct": 21.29,
            "efficiency_pct": 29.68,
            "bound_type": "compute",
        },
    },
    "latest": {
        "snapshot_id": 2,
        "ts": "2026-05-22T12:00:00+00:00",
        "compute_pct": 88.1,
        "idle_pct": 3.7,
        "comm_pct": 8.21,
        "top_bottleneck": "MoE_fused",
        "top_kernel": {
            "name": "aiter::fmoe_fp8_blockscale_g1u1",
            "gpu_pct": 19.5,
            "efficiency_pct": 34.88,
            "bound_type": "compute",
        },
    },
    "delta": {
        "compute_pct": 1.9,
        "idle_pct": -1.5,
        "comm_pct": -0.3,
        "top_kernel_efficiency_pct": 5.2,
    },
}


def _write_final_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# §15.1 collector — happy path (standalone tool shape, runs/roofline/)
# ---------------------------------------------------------------------------
def test_roofline_collector_reads_final_json(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write_final_json(
        sd / "runs" / "roofline" / "abcd1234" / "final.json",
        _BEFORE_AFTER_BLOB,
    )

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert warnings == []
    assert len(out) == 1
    e = out[0]
    assert e["source_path"] == "runs/roofline/abcd1234/final.json"
    assert e["mode"] == "before_after"

    bl = e["baseline"]
    assert bl["snapshot_id"] == 1
    assert bl["compute_pct"] == 86.2
    assert bl["idle_pct"] == 5.26
    assert bl["comm_pct"] == 8.51
    assert bl["top_bottleneck"] == "MoE_fused"
    assert bl["top_kernel"]["name"] == "aiter::fmoe_fp8_blockscale_g1u1"
    assert bl["top_kernel"]["gpu_pct"] == 21.29
    assert bl["top_kernel"]["efficiency_pct"] == 29.68
    assert bl["top_kernel"]["bound_type"] == "compute"

    lt = e["latest"]
    assert lt["snapshot_id"] == 2
    assert lt["compute_pct"] == 88.1

    assert e["delta"] == {
        "compute_pct": 1.9,
        "idle_pct": -1.5,
        "comm_pct": -0.3,
        "top_kernel_efficiency_pct": 5.2,
    }


# ---------------------------------------------------------------------------
# §15.2 collector — orchestrator-final-report shape (reports/final.json)
# ---------------------------------------------------------------------------
def test_roofline_collector_reads_orchestrator_wrapped_shape(tmp_path: Path) -> None:
    """The Hyperloom orchestrator emits ``reports/final.json`` with the
    roofline payload nested under ``roofline_comparison``. The collector
    must lift that sub-dict into the same wire shape as the standalone
    roofline tool's top-level output."""
    sd = tmp_path / "session"
    wrapped = {
        "session_id": "fake-1",
        "current_best": {"tput": 1234.0},
        "roofline_comparison": {
            "mode": "single_snapshot",
            "baseline": {
                "snapshot_id": 1,
                "ts": "2026-05-22T11:22:43+00:00",
                "compute_pct": 31.41,
                "idle_pct": 68.56,
                "comm_pct": 0.0,
                "top_bottleneck": "MoE_unfused",
                "top_kernel": {
                    "name": "aiter::ck_moe_stage1",
                    "gpu_pct": 9.17,
                    "efficiency_pct": 66.17,
                    "bound_type": "compute",
                },
            },
            "latest": {
                "snapshot_id": 2,
                "ts": "2026-05-22T12:45:15.211286+00:00",
                "compute_pct": 31.41,
                "idle_pct": 68.56,
                "comm_pct": 0.0,
                "top_bottleneck": "MoE_unfused",
                "top_kernel": {
                    "name": "aiter::ck_moe_stage1",
                    "gpu_pct": 9.17,
                    "efficiency_pct": 66.17,
                    "bound_type": "compute",
                },
            },
        },
    }
    _write_final_json(sd / "reports" / "final.json", wrapped)

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert warnings == []
    assert len(out) == 1
    e = out[0]
    assert e["source_path"] == "reports/final.json"
    assert e["mode"] == "single_snapshot"
    assert e["baseline"]["top_kernel"]["name"] == "aiter::ck_moe_stage1"
    assert e["latest"]["snapshot_id"] == 2
    # Real-session shape has no delta — collector should pass that
    # through as None (not {} — None preserves "delta literally absent").
    assert e["delta"] is None


# ---------------------------------------------------------------------------
# §15.3 collector — multiple files sorted oldest-first by mtime
# ---------------------------------------------------------------------------
def test_roofline_collector_handles_multiple_snapshots(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    older = sd / "runs" / "roofline" / "older" / "final.json"
    newer = sd / "runs" / "roofline" / "newer" / "final.json"

    _write_final_json(older, {**_BEFORE_AFTER_BLOB, "mode": "before_after"})
    # Force a clear mtime gap (some FSes round to the second).
    base = time.time()
    os.utime(older, (base - 100, base - 100))
    _write_final_json(
        newer,
        {
            **_BEFORE_AFTER_BLOB,
            "mode": "single",
            "baseline": {**_BEFORE_AFTER_BLOB["baseline"], "snapshot_id": 99},
        },
    )
    os.utime(newer, (base, base))

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert warnings == []
    assert len(out) == 2
    assert out[0]["source_path"] == "runs/roofline/older/final.json"
    assert out[0]["mode"] == "before_after"
    assert out[1]["source_path"] == "runs/roofline/newer/final.json"
    assert out[1]["mode"] == "single"
    assert out[1]["baseline"]["snapshot_id"] == 99


# ---------------------------------------------------------------------------
# §15.4 collector — missing latest / delta degrade to None
# ---------------------------------------------------------------------------
def test_roofline_collector_handles_missing_fields(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    minimal = {
        "mode": "single",
        "baseline": {
            "snapshot_id": 7,
            "compute_pct": 50.0,
            # Intentionally missing: ts, idle_pct, comm_pct, top_bottleneck,
            # top_kernel — collector should replace each with None.
        },
        # No `latest`, no `delta`.
    }
    _write_final_json(sd / "runs" / "roofline" / "min" / "final.json", minimal)

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert warnings == []
    assert len(out) == 1
    e = out[0]
    assert e["mode"] == "single"
    bl = e["baseline"]
    assert bl["snapshot_id"] == 7
    assert bl["compute_pct"] == 50.0
    assert bl["ts"] is None
    assert bl["idle_pct"] is None
    assert bl["comm_pct"] is None
    assert bl["top_bottleneck"] is None
    assert bl["top_kernel"] is None
    assert e["latest"] is None
    assert e["delta"] is None


# ---------------------------------------------------------------------------
# §15.5 collector — invalid JSON -> warning, not crash
# ---------------------------------------------------------------------------
def test_roofline_collector_handles_invalid_json(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    bad = sd / "runs" / "roofline" / "bad" / "final.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{ this is not valid json", encoding="utf-8")

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert out == []
    assert len(warnings) == 1
    assert "invalid JSON" in warnings[0]
    assert str(bad) in warnings[0] or "final.json" in warnings[0]


# ---------------------------------------------------------------------------
# §15.6 collector — empty session yields empty list, no warnings
# ---------------------------------------------------------------------------
def test_roofline_collector_empty_when_no_final_json(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "runs").mkdir()
    (sd / "reports").mkdir()

    warnings: list[str] = []
    out = collect_roofline(sd, warnings)

    assert out == []
    assert warnings == []


# ---------------------------------------------------------------------------
# §15.7 collector — non-roofline final.json is reported, not silently dropped
# ---------------------------------------------------------------------------
def test_roofline_collector_warns_on_unrecognised_shape(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write_final_json(
        sd / "reports" / "final.json",
        {"session_id": "x", "tput": 1234.0},   # neither shape we know about
    )
    warnings: list[str] = []
    out = collect_roofline(sd, warnings)
    assert out == []
    assert any("no recognisable roofline shape" in w for w in warnings)


# ---------------------------------------------------------------------------
# §15.8 build() integration — section is always present, list-typed
# ---------------------------------------------------------------------------
def test_build_includes_roofline_section(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir(parents=True, exist_ok=True)
    # Minimum viable session (manifest + state) so build() doesn't go
    # ``shell_only`` and suppress everything.
    (sd / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "session_id":     "roofline-test",
        "created_at_utc": "2026-05-22T11:00:00+00:00",
        "session_dir":    str(sd),
        "model_name":     "test",
        "model_path":     "/tmp/test",
        "framework":      "sglang",
    }), encoding="utf-8")
    (sd / "state.json").write_text(json.dumps({
        "session_id": "roofline-test",
        "framework":  "sglang",
    }), encoding="utf-8")

    _write_final_json(sd / "runs" / "roofline" / "abcd" / "final.json",
                      _BEFORE_AFTER_BLOB)

    breakdown = build(sd)
    assert "roofline" in breakdown
    assert isinstance(breakdown["roofline"], list)
    assert len(breakdown["roofline"]) == 1
    assert breakdown["roofline"][0]["mode"] == "before_after"
    assert breakdown["roofline"][0]["source_path"] == "runs/roofline/abcd/final.json"


def test_build_roofline_empty_when_no_files(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "session_id": "no-roofline", "framework": "sglang",
    }), encoding="utf-8")
    (sd / "state.json").write_text(json.dumps({"session_id": "no-roofline"}),
                                   encoding="utf-8")

    breakdown = build(sd)
    assert breakdown.get("roofline") == []


# ---------------------------------------------------------------------------
# §15.9 renderer — markdown contains the key fields; skipped on absence
# ---------------------------------------------------------------------------
def test_renderer_roofline_table_renders_keyfields() -> None:
    fixture = {
        "session": {"session_id": "rt-1"},
        "roofline": [{
            "source_path": "reports/final.json",
            "mode":        "before_after",
            "baseline":    _BEFORE_AFTER_BLOB["baseline"],
            "latest":      _BEFORE_AFTER_BLOB["latest"],
            "delta":       _BEFORE_AFTER_BLOB["delta"],
        }],
    }
    md = render_session_report(fixture).markdown
    assert "Roofline" in md
    assert "reports/final.json" in md
    assert "before_after" in md
    assert "aiter::fmoe_fp8_blockscale_g1u1" in md
    assert "MoE_fused" in md
    # Delta values should appear in the delta table.
    assert "compute_pct" in md
    assert "top_kernel_efficiency_pct" in md


def test_renderer_roofline_skipped_when_absent() -> None:
    fixture = {"session": {"session_id": "no-roofline-1"}}  # no key at all
    md = render_session_report(fixture).markdown
    assert "### Roofline" not in md


def test_renderer_roofline_skipped_when_empty_list() -> None:
    fixture = {"session": {"session_id": "no-roofline-2"}, "roofline": []}
    md = render_session_report(fixture).markdown
    assert "### Roofline" not in md
