# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for report.py pure formatting + file-reader helpers."""

from __future__ import annotations

import json

from hyperloom.common.perf_metric import GRADED_OUTPUT, VERDICT_KEEP
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


_OUTPUT_COMPARISON = dict(
    objective=GRADED_OUTPUT,
    reference=100.0,
    candidate=100.0,
    gain_pct=0.0,
    comparable=True,
    degrade_reason="",
    verdict=VERDICT_KEEP,
)


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
def test_explain_stop_reason_target_reached():
    assert "target" in rp._explain_stop_reason("target_reached").lower()


def test_explain_stop_reason_unknown_is_empty():
    assert rp._explain_stop_reason("some_unmapped_reason") == ""
    assert rp._explain_stop_reason("") == ""


# ---- stop_reason explanation vocabulary coverage ----
def test_every_stop_reason_vocab_member_has_explanation():
    from hyperloom.inference_optimizer.breakdown.stop_reasons import STOP_REASON_VOCAB

    missing = sorted(r for r in STOP_REASON_VOCAB if not rp._explain_stop_reason(r))
    assert missing == [], f"stop reasons without an explanation: {missing}"


def test_classify_root_cause_prefers_kv_cache_oom_over_generic_oom():
    assert (
        rp.classify_root_cause_type(
            "kv_cache_oom",
            "CUDA out of memory; no GPU memory for the KV cache",
        )
        == "kv_cache_oom"
    )


# ---- platform line: absence must be stated, not implied ----
def _summary_without_platform(**extra) -> dict:
    base = {
        "session_id": "s",
        "model_name": "m",
        "model_path": "/m",
        "stop_reason": "target_reached",
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
        "performance_comparison": _OUTPUT_COMPARISON,
    }
    base.update(extra)
    return base


def test_report_says_platform_is_missing_when_the_summary_predates_the_field():
    """A summary with no platform key still gets a line."""
    md = rp._format_md(_summary_without_platform())
    assert "- platform       : not recorded" in md


def test_report_gives_the_reason_the_platform_is_missing_when_it_has_one():
    md = rp._format_md(_summary_without_platform(platform={"status": "unavailable", "reason": "no host CPU sysfs"}))
    assert "not recorded — no host CPU sysfs" in md


def test_platform_fingerprint_propagates_a_node_scope_failure(monkeypatch):
    import pytest

    def _boom():
        raise RuntimeError("node scope unreadable")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", _boom)

    with pytest.raises(RuntimeError, match="node scope unreadable"):
        rp._platform_fingerprint()
