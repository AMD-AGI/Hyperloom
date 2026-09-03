# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
            "stop_reason": "sweep_done",
            "max_minutes": 360,
            "report_generated_at": "2026-06-23T00:00:00+00:00",
            "baseline_tput": 100.0,
            "current_best": {"action": "warm_replay", "tput": 136.146},
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
def test_the_chart_precheck_agrees_with_the_renderer(tmp_path, monkeypatch):
    """A point the renderer would drop must not let the pre-check admit the chart.

    The guard used to probe one field, so an agentic rung carrying throughput
    but no interactivity passed it and then produced nothing to plot.
    """
    from hyperloom.inference_optimizer.session.session_paths import reports_dir

    rdir = reports_dir(tmp_path)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "conc_sweep_summary.json").write_text(
        json.dumps(
            {
                "benchmark_mode": "agentx",
                "tp": 8,
                "baseline": {"points": [{"conc": 8, "total_token_throughput": 25984.8}]},
                "optimized": {"points": []},
            }
        ),
        encoding="utf-8",
    )

    calls: list[object] = []
    monkeypatch.setattr(rp, "render_conc_sweep_curve", lambda *a, **kw: calls.append(a), raising=False)

    class _S:
        tp = 8
        model_name = "m"
        gpu_type = "mi300x"

    assert rp._render_conc_sweep_curve_for_report(tmp_path, tmp_path, _S()) is None
    assert calls == []


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


# ---- _count_server_boot_failures ----
def test_count_server_boot_failures_missing(tmp_path):
    assert rp._count_server_boot_failures(tmp_path) == 0
    assert rp._count_server_boot_failures(None) == 0


def test_count_server_boot_failures_counts_warmup_failed(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "optimization_journal.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"reason": "warmup_failed"},
                    {"reason": "gain_below_threshold"},
                    {"reason": "warmup_failed"},
                    {"outcome": "KEEP"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert rp._count_server_boot_failures(tmp_path) == 2


# ---- stop_reason fallback during closing_phase ----
def test_build_summary_stop_reason_falls_back_to_time_exhausted_in_closing():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.stop_reason = ""
    state.closing_phase = True
    summary = rp._build_summary_dict(state, {}, [])
    assert summary["stop_reason"] == "time_exhausted"
    assert summary["stop_reason_explanation"]


def test_build_summary_keeps_explicit_stop_reason_over_closing_fallback():
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState()
    state.stop_reason = "target_reached"
    state.closing_phase = True
    summary = rp._build_summary_dict(state, {}, [])
    assert summary["stop_reason"] == "target_reached"


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


class _SweepState:
    def __init__(self, last_conc_sweep):
        self.last_conc_sweep = last_conc_sweep


def test_a_skipped_sweep_is_not_described_as_a_finished_one():
    """``sweep_done`` is also the exit for a sweep that declined to run."""
    state = _SweepState({"status": "succeeded", "was_skipped": True, "skip_reason": "no_optimization_to_compare"})
    msg = rp._explain_stop_reason("sweep_done", state)
    assert "did not run" in msg
    assert "no_optimization_to_compare" in msg


def test_a_sweep_that_ran_keeps_the_plain_explanation():
    state = _SweepState({"status": "succeeded", "was_skipped": False})
    assert rp._explain_stop_reason("sweep_done", state) == rp._explain_stop_reason("sweep_done")


def test_a_skip_with_no_recorded_reason_still_says_it_was_skipped():
    state = _SweepState({"status": "succeeded", "was_skipped": True, "skip_reason": ""})
    assert "did not run" in rp._explain_stop_reason("sweep_done", state)


def test_a_session_budget_skip_is_described_as_a_sweep_that_did_not_run():
    state = _SweepState({"status": "skipped", "was_skipped": True, "skip_reason": "session_time_budget"})
    msg = rp._explain_stop_reason("sweep_done", state)
    assert "did not run" in msg
    assert "session_time_budget" in msg


def test_a_sweep_that_spent_its_budget_is_not_reported_as_one_that_never_ran(tmp_path):
    """The budget path records was_skipped for a sweep that ran its whole ladder."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    live = SharedState()
    live.record_conc_sweep(
        {
            "status": "skipped",
            "was_skipped": True,
            "budget_exhausted": True,
            "skip_reason": "budget_exhausted_no_successful_pairs",
        }
    )
    live.save(tmp_path)
    # The report is written from a reloaded state, so the flag that separates
    # the two skips has to survive the round trip to be readable at all.
    state = SharedState.load_or_init(tmp_path)

    msg = rp._explain_stop_reason("sweep_done", state)
    assert "did not run" not in msg
    assert "budget" in msg
    assert "budget_exhausted_no_successful_pairs" in msg


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


# ---- stop_reason explanation vocabulary coverage ----
def test_every_stop_reason_vocab_member_has_explanation():
    from hyperloom.orchestrator.phases.machine_state import STOP_REASON_VOCAB

    missing = sorted(r for r in STOP_REASON_VOCAB if not rp._explain_stop_reason(r))
    assert missing == [], f"stop reasons without an explanation: {missing}"


def test_classify_root_cause_prefers_kv_cache_oom_over_generic_oom():
    assert (
        rp._classify_root_cause_type(
            "kv_cache_oom",
            "CUDA out of memory; no GPU memory for the KV cache",
        )
        == "kv_cache_oom"
    )
