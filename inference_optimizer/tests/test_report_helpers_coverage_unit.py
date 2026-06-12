# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for report.py pure formatting + file-reader helpers."""

from __future__ import annotations

import json

from inference_optimizer.orchestrator.action_executors import report as rp
from inference_optimizer.session_paths import (
    reports_dir,
    target_baseline_json,
)


# ---- _format_completeness_annotations ----
def test_completeness_annotations_empty():
    assert rp._format_completeness_annotations({}) == []


def test_completeness_annotations_full():
    out = rp._format_completeness_annotations({
        "has_unvalidated_keeps": True,
        "untried_hot_reusable_kernels": ["k1"],
        "pending_keep_kernels": ["k2"],
    })
    body = "\n".join(out)
    assert "unvalidated" in body
    assert "k1" in body
    assert "k2" in body


# ---- _format_steward_section ----
def test_steward_section_empty():
    assert rp._format_steward_section({}) == []


def test_steward_section_with_assessment():
    out = rp._format_steward_section({
        "remaining_gaps_assessment": {
            "recommendation": "stop", "ts": "t0",
            "remaining_potential_pct_estimate": 3.5,
            "rationale": "diminishing\nreturns",
            "next_gap_canonical_id": "gap.x",
        },
        "remaining_gaps_assessments_history": [1, 2, 3],
    })
    body = "\n".join(out)
    assert "stop" in body
    assert "3.50%" in body
    assert "gap.x" in body
    assert "prior assessments: 2" in body


# ---- _format_degraded_mode_section ----
def test_degraded_mode_section():
    out = rp._format_degraded_mode_section({
        "degraded_mode": True,
        "model_warnings": [
            {"model_name": "m", "architecture": "a", "signal": "img ignored"},
            "skip-non-dict",
        ],
    })
    body = "\n".join(out)
    assert "Degraded mode" in body
    assert "`m`" in body


def test_degraded_mode_section_empty():
    assert rp._format_degraded_mode_section({}) == []


# ---- _extract_executive_summary ----
def test_extract_exec_summary_no_path():
    assert "no analysis.md" in rp._extract_executive_summary("")


def test_extract_exec_summary_missing_file(tmp_path):
    out = rp._extract_executive_summary(str(tmp_path / "nope.md"))
    assert "could not read" in out


def test_extract_exec_summary_no_block(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# Title\nno exec block here\n", encoding="utf-8")
    assert "does not contain" in rp._extract_executive_summary(str(md))


def test_extract_exec_summary_present_and_image_stripped(tmp_path):
    md = tmp_path / "a.md"
    md.write_text(
        "## Executive Summary\n"
        "![chart](data:image/png;base64,AAAA)\n"
        "compute 70%\n"
        "## Next Section\nignored\n",
        encoding="utf-8",
    )
    out = rp._extract_executive_summary(str(md))
    assert "Executive Summary" in out
    assert "[image stripped]" in out
    assert "Next Section" not in out


def test_extract_exec_summary_truncates(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("## Executive Summary\n" + ("x" * 5000), encoding="utf-8")
    out = rp._extract_executive_summary(str(md))
    assert out.endswith("...")
    assert len(out) <= 2048


# ---- _load_external_baseline ----
def test_load_external_baseline_missing(tmp_path):
    assert rp._load_external_baseline(tmp_path) is None


def test_load_external_baseline_present(tmp_path):
    p = target_baseline_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    assert rp._load_external_baseline(tmp_path)["status"] == "ok"


def test_load_external_baseline_corrupt(tmp_path):
    p = target_baseline_json(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{bad", encoding="utf-8")
    assert rp._load_external_baseline(tmp_path) is None


# ---- _read_conc_sweep_pointer ----
def test_read_conc_sweep_pointer_missing(tmp_path):
    assert rp._read_conc_sweep_pointer(tmp_path) is None


def test_read_conc_sweep_pointer_present(tmp_path):
    rd = reports_dir(tmp_path)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "conc_sweep_summary.json").write_text(
        json.dumps({"status": "done", "summary": {"x": 1},
                    "budget_exhausted": True, "total_budget_sec": 10}),
        encoding="utf-8",
    )
    ptr = rp._read_conc_sweep_pointer(tmp_path)
    assert ptr["status"] == "done"
    assert ptr["budget_exhausted"] is True


def test_read_conc_sweep_pointer_corrupt(tmp_path):
    rd = reports_dir(tmp_path)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "conc_sweep_summary.json").write_text("{bad", encoding="utf-8")
    assert rp._read_conc_sweep_pointer(tmp_path) is None


# ---- _read_ko_summary_totals ----
def test_read_ko_summary_totals(tmp_path):
    p = tmp_path / "ko.json"
    p.write_text(json.dumps({"totals": {"a": 3, "b": 2.0, "c": "x"}}),
                 encoding="utf-8")
    totals = rp._read_ko_summary_totals(p)
    assert totals == {"a": 3, "b": 2}


def test_read_ko_summary_totals_missing(tmp_path):
    assert rp._read_ko_summary_totals(tmp_path / "nope.json") == {}


# ---- _highlight ----
def test_highlight_topics():
    assert "action_name" in rp._highlight({"action_name": "x"}, "proposal", "a")["summary"]
    assert "verdict" in rp._highlight({"verdict": "keep", "reasoning": "ok"}, "review_verdict", "a")["summary"]
    assert "kind" in rp._highlight({"kind": "k", "action_name": "n", "task_id": "12345678abc"}, "decision", "a")["summary"]
    dr = rp._highlight({"kind": "k", "state": "s", "result": {"output_throughput": 1, "decision": "keep"}}, "delegated_result", "a")
    assert "tput=1" in dr["summary"]
    assert "status" in rp._highlight({"kind": "k", "status": "ok"}, "response", "a")["summary"]
    assert "sev" in rp._highlight({"severity": "high", "summary": "boom"}, "alert", "a")["summary"]
    # fallback branch
    other = rp._highlight({"a": 1, "b": "two", "c": [1, 2]}, "weird_topic", "ag")
    assert "a" in other["summary"]
