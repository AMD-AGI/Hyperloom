# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for report.py pure formatting + file-reader helpers."""

from __future__ import annotations

import json

from hyperloom.orchestrator.actions.executors import report as rp
from hyperloom.inference_optimizer.session.session_paths import (
    reports_dir,
    target_baseline_json,
)


# ---- _format_completeness_annotations ----
def test_completeness_annotations_empty():
    assert rp._format_completeness_annotations({}) == []


def test_completeness_annotations_full():
    out = rp._format_completeness_annotations(
        {
            "has_unvalidated_keeps": True,
            "untried_hot_reusable_kernels": ["k1"],
            "pending_keep_kernels": ["k2"],
        }
    )
    body = "\n".join(out)
    assert "unvalidated" in body
    assert "k1" in body
    assert "k2" in body


# ---- _format_steward_section ----
def test_steward_section_empty():
    assert rp._format_steward_section({}) == []


def test_steward_section_with_assessment():
    out = rp._format_steward_section(
        {
            "remaining_gaps_assessment": {
                "recommendation": "stop",
                "ts": "t0",
                "remaining_potential_pct_estimate": 3.5,
                "rationale": "diminishing\nreturns",
                "next_gap_canonical_id": "gap.x",
            },
            "remaining_gaps_assessments_history": [1, 2, 3],
        }
    )
    body = "\n".join(out)
    assert "stop" in body
    assert "3.50%" in body
    assert "gap.x" in body
    assert "prior assessments: 2" in body


# ---- _format_degraded_mode_section ----
def test_degraded_mode_section():
    out = rp._format_degraded_mode_section(
        {
            "degraded_mode": True,
            "model_warnings": [
                {"model_name": "m", "architecture": "a", "signal": "img ignored"},
                "skip-non-dict",
            ],
        }
    )
    body = "\n".join(out)
    assert "Degraded mode" in body
    assert "`m`" in body


def test_degraded_mode_section_empty():
    assert rp._format_degraded_mode_section({}) == []


def test_format_md_shows_validated_gain_when_timestamp_missing():
    md = rp._format_md(
        {
            "session_id": "s1",
            "model_name": "m",
            "model_path": "/models/m",
            "stop_reason": "conc_sweep_done",
            "max_minutes": 360,
            "report_generated_at": "2026-06-23T00:00:00+00:00",
            "baseline_tput": 100.0,
            "current_best": {"action": "warm_replay", "tput": 136.146},
            "cumulative_gain": 36.146,
            "cumulative_gain_validated": 36.146,
            "cumulative_gain_validated_ts": "",
            "cumulative_gain_validated_stack_len": 1,
            "optimization_stack_len": 1,
            "crash_count": 0,
            "pruned_families": [],
            "event_counts_by_topic": {},
            "highlights": [],
        }
    )

    assert "cumulative_gain_val : `36.15%`" in md
    assert "ts=<missing>" in md
    assert "never validated" not in md


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
        "## Executive Summary\n![chart](data:image/png;base64,AAAA)\ncompute 70%\n## Next Section\nignored\n",
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
        json.dumps({"status": "done", "summary": {"x": 1}, "budget_exhausted": True, "total_budget_sec": 10}),
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
    p.write_text(json.dumps({"totals": {"a": 3, "b": 2.0, "c": "x"}}), encoding="utf-8")
    totals = rp._read_ko_summary_totals(p)
    assert totals == {"a": 3, "b": 2}


def test_read_ko_summary_totals_missing(tmp_path):
    assert rp._read_ko_summary_totals(tmp_path / "nope.json") == {}


# ---- _highlight ----
def test_highlight_topics():
    assert "action_name" in rp._highlight({"action_name": "x"}, "proposal", "a")["summary"]
    assert "verdict" in rp._highlight({"verdict": "keep", "reasoning": "ok"}, "review_verdict", "a")["summary"]
    assert (
        "kind" in rp._highlight({"kind": "k", "action_name": "n", "task_id": "12345678abc"}, "decision", "a")["summary"]
    )
    dr = rp._highlight(
        {"kind": "k", "state": "s", "result": {"output_throughput": 1, "decision": "keep"}}, "delegated_result", "a"
    )
    assert "tput=1" in dr["summary"]
    assert "status" in rp._highlight({"kind": "k", "status": "ok"}, "response", "a")["summary"]
    assert "sev" in rp._highlight({"severity": "high", "summary": "boom"}, "alert", "a")["summary"]
    # fallback branch
    other = rp._highlight({"a": 1, "b": "two", "c": [1, 2]}, "weird_topic", "ag")
    assert "a" in other["summary"]


# ---- _explain_stop_reason ----
def test_explain_stop_reason_robustness_escalated():
    msg = rp._explain_stop_reason("robustness_escalated")
    assert msg
    assert "robustness" in msg.lower()


def test_explain_stop_reason_target_reached():
    assert "target" in rp._explain_stop_reason("target_reached").lower()


def test_explain_stop_reason_unknown_is_empty():
    assert rp._explain_stop_reason("some_unmapped_reason") == ""
    assert rp._explain_stop_reason("") == ""


def test_format_md_renders_stop_explanation():
    md = rp._format_md(
        {
            "session_id": "s",
            "model_name": "m",
            "model_path": "/m",
            "stop_reason": "robustness_escalated",
            "stop_reason_explanation": "Robustness escalated: stopped early to protect validated gains.",
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
        }
    )
    assert "Why it stopped" in md
    assert "Robustness escalated" in md
