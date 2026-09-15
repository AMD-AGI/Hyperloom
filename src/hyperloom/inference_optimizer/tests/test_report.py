# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for report.py pure formatting + file-reader helpers."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_OUTPUT,
    INTVTY_V1,
    VERDICT_KEEP,
    VERDICT_RECORDED,
    VERDICT_REVERT,
)
from hyperloom.orchestrator.actions.executors import report as rp
from hyperloom.inference_optimizer.session.session_paths import (
    reports_dir,
    target_baseline_json,
)


def _agentx_reference():
    return {
        "status": "ok",
        "reason": "ok",
        "source": "https://reference.test/api/v1",
        "query": {"benchmark_mode": "agentx", "model": "GLM-5.2", "gpu": "b300", "precision": "fp4"},
        "best": {"conc": 8, "tput_per_gpu": 9999.0, "e2e_norm_intvty_p90": 99.0, "benchmark_id": "other-conc"},
        "all_concurrencies": [
            {"conc": 4, "decode_tp": 8, "tput_per_gpu": 800.0, "e2e_norm_intvty_p90": 20.0, "benchmark_id": "42"},
            {
                "conc": 8,
                "decode_tp": 8,
                "tput_per_gpu": 9999.0,
                "e2e_norm_intvty_p90": 99.0,
                "benchmark_id": "other-conc",
            },
        ],
    }


@pytest.mark.parametrize("advisory_enabled", [True, False])
def test_report_agentx_comparison_reads_persisted_target(report_performance_state, tmp_path, advisory_enabled):
    from hyperloom.orchestrator.knowledge import research_hints

    state = report_performance_state
    state.benchmark_mode = "agentx"
    state.tp, state.conc = 2, 4
    state.current_best.update(total_throughput=800.0, e2e_norm_intvty_p90=5.0)
    state.target_advisory_enabled = advisory_enabled
    target = {
        "benchmark_mode": "agentx",
        "throughput_basis": "total_token_throughput_per_gpu",
        "per_conc": [{"conc": 4, "tput_per_gpu": 800.0, "e2e_norm_intvty_p90": 20.0, "source": "measured"}],
    }
    assert research_hints.write_competitor_target(tmp_path, target)
    reference = _agentx_reference()
    reference["all_concurrencies"] = []
    before = deepcopy(reference)
    summary = rp._build_summary_dict(state, {}, [], external_baseline=reference, session_dir=tmp_path)
    comparison = summary["external_baseline"]["comparison"]
    assert comparison == research_hints.gap_for_state(research_hints.load_competitor_target(tmp_path), state)
    assert comparison["throughput_gap_pct"] == 50.0
    assert comparison["interactivity_gap_pct"] == 75.0
    assert comparison["primary_gap"] == "latency"
    assert reference == before
    md = "\n".join(rp._format_external_baseline_section(summary["external_baseline"]))
    assert "+75.0%" in md
    assert "9999" not in md


def test_report_does_not_rebuild_missing_competitor_target(report_performance_state, tmp_path, caplog):
    summary = rp._build_summary_dict(
        report_performance_state, {}, [], external_baseline=_agentx_reference(), session_dir=tmp_path
    )
    comparison = summary["external_baseline"]["comparison"]
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "target_unavailable"
    assert "gap vs target" not in "\n".join(rp._format_external_baseline_section(summary["external_baseline"]))
    assert "target_unavailable" in caplog.text
    assert str(tmp_path) in caplog.text


def test_agentx_report_keeps_no_data_reason_without_computing_gap(report_performance_state):
    reference = {
        **_agentx_reference(),
        "status": "no_match",
        "reason": "fetch_error",
        "best": None,
        "all_concurrencies": [],
    }
    summary = rp._build_summary_dict(report_performance_state, {}, [], external_baseline=reference)
    md = "\n".join(rp._format_external_baseline_section(summary["external_baseline"]))
    assert "fetch_error" in md
    assert "gap vs target" not in md


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


@pytest.fixture
def report_performance_state(monkeypatch):
    from hyperloom.orchestrator.state.shared_state import SharedState

    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_NOISE_PCT", raising=False)
    monkeypatch.setattr(rp, "_platform_fingerprint", lambda *_: {"status": "not_recorded"})
    state = SharedState()
    state.framework = "vllm"
    state.benchmark_mode = "agentx"
    state.baseline_tput = 100.0
    state.baseline_perf = {"output_throughput": 100.0, "total_throughput": 1000.0, GRADED_INTVTY: 100.0}
    state.current_best = {
        "action": "integrate",
        "tput": 150.0,
        "output_throughput": 150.0,
        "total_throughput": 1200.0,
        # The graded axis moves by the same 20% as the guard axis, so a degrade to the output axis is visible as a
        # different figure (50%) rather than hiding behind a coincidence.
        GRADED_INTVTY: 120.0,
        "extra_envs": {"RECIPE": "measured"},
    }
    state.optimization_stack = [{"action": "integrate"}]
    state.cumulative_gain_validated = 50.0
    state.cumulative_gain_validated_ts = "2026-06-23T00:00:00+00:00"
    state.cumulative_gain_validated_stack_len = 1
    return state


@pytest.mark.parametrize(
    "mode,missing_side,missing_axis",
    [
        pytest.param(INTVTY_V1, None, None, id="actual-intvty"),
        pytest.param(GRADED_OUTPUT, None, None, id="explicit-output"),
        pytest.param(INTVTY_V1, "baseline_perf", "total_throughput", id="reference-total-missing"),
        pytest.param(INTVTY_V1, "baseline_perf", GRADED_INTVTY, id="reference-intvty-missing"),
        pytest.param(INTVTY_V1, "current_best", "total_throughput", id="candidate-total-missing"),
        pytest.param(INTVTY_V1, "current_best", GRADED_INTVTY, id="candidate-intvty-missing"),
    ],
)
def test_report_performance_comparison_snapshots_effective_axes(
    report_performance_state, monkeypatch, mode, missing_side, missing_axis
):
    state = report_performance_state
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", mode)
    if missing_side:
        getattr(state, missing_side).pop(missing_axis)
    graded = mode == INTVTY_V1 and missing_side is None
    reason = {"baseline_perf": "baseline_axes_missing", "current_best": "candidate_axes_missing"}.get(missing_side, "")

    summary = rp._build_summary_dict(state, {}, [])
    record = summary["performance_comparison"]

    expected = {
        "objective": GRADED_INTVTY if graded else GRADED_OUTPUT,
        "reference": 100.0,
        "candidate": 120.0 if graded else 150.0,
        "gain_pct": pytest.approx(20.0 if graded else 50.0),
        "comparable": missing_side is None,
        "degrade_reason": reason,
        "verdict": VERDICT_KEEP,
        # The guard axis is snapshotted only when the objective actually applied.
        "tput_reference": 1000.0 if graded else 0.0,
        "tput_candidate": 1200.0 if graded else 0.0,
    }
    assert {key: record[key] for key in expected} == expected
    assert summary["cumulative_gain_validated"] == 50.0
    assert summary["cumulative_gain_validated_ts"] == state.cumulative_gain_validated_ts
    assert summary["cumulative_gain_validated_stack_len"] == 1


@pytest.mark.parametrize("mode", [INTVTY_V1, GRADED_OUTPUT])
def test_report_performance_render_survives_json_reload_and_environment_change(
    report_performance_state, monkeypatch, mode
):
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", mode)
    summary = json.loads(json.dumps(rp._build_summary_dict(report_performance_state, {}, [])))
    rendered = rp._format_md(summary)

    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", GRADED_OUTPUT if mode == INTVTY_V1 else INTVTY_V1)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "99")
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")

    assert rp._format_md(summary) == rendered


@pytest.mark.parametrize("missing_axis", ["total_throughput", GRADED_INTVTY])
def test_report_performance_missing_candidate_axes_does_not_claim_intvty_grading(
    report_performance_state, missing_axis
):
    report_performance_state.current_best.pop(missing_axis)
    rendered = rp._format_md(rp._build_summary_dict(report_performance_state, {}, []))

    assert f"grading mode        : `{INTVTY_V1}`" not in rendered
    assert GRADED_OUTPUT in rendered
    assert "candidate_axes_missing" in rendered


@pytest.mark.parametrize(
    "stack_len,stored_gain,status",
    [
        pytest.param(1, 20.0, "current", id="current"),
        pytest.param(2, 50.0, "stale", id="stale"),
        pytest.param(1, 50.0, "inconsistent", id="inconsistent"),
    ],
)
def test_report_performance_diagnostic_gain_does_not_overwrite_validation_stamp(
    report_performance_state, stack_len, stored_gain, status
):
    state = report_performance_state
    state.optimization_stack = [{"action": "integrate"}] * stack_len
    state.cumulative_gain_validated = stored_gain
    summary = rp._build_summary_dict(state, {}, [])
    rendered = rp._format_md(summary)

    assert summary["performance_comparison"]["gain_pct"] == pytest.approx(20.0)
    assert summary["cumulative_gain_validated"] == stored_gain
    assert summary["cumulative_validation_status"] == status
    assert f"cumulative_gain_val : `{stored_gain:.2f}%`" in rendered
    assert "+20.00%" in rendered
    assert ("stack changed since validation" in rendered) is (stack_len > 1)
    if status == "inconsistent":
        assert "inconsistent" in rendered.lower()
    assert state.cumulative_gain_validated == stored_gain


def test_report_performance_without_validation_stamp_is_diagnostic_only(report_performance_state):
    state = report_performance_state
    state.cumulative_gain_validated = 0.0
    state.cumulative_gain_validated_ts = ""
    state.cumulative_gain_validated_stack_len = 0
    summary = rp._build_summary_dict(state, {}, [])

    assert summary["performance_comparison"]["gain_pct"] == pytest.approx(20.0)
    assert summary["cumulative_validation_status"] == "unavailable"
    assert summary["cumulative_gain_validated"] == 0.0
    assert "never validated" in rp._format_md(summary)


@pytest.mark.parametrize(
    "intvty,total,verdict,gain",
    [
        pytest.param(120.0, 1200.0, VERDICT_KEEP, 20.0, id="keep"),
        # Neither axis dominates: interactivity is flat but inside the band, so the point is stored, not promoted.
        pytest.param(100.0, 1200.0, VERDICT_RECORDED, 0.0, id="recorded"),
        pytest.param(90.0, 800.0, VERDICT_REVERT, -10.0, id="revert"),
    ],
)
def test_report_performance_comparison_records_two_dimensional_verdict(
    report_performance_state, intvty, total, verdict, gain
):
    report_performance_state.current_best[GRADED_INTVTY] = intvty
    report_performance_state.current_best["total_throughput"] = total
    record = rp._build_summary_dict(report_performance_state, {}, [])["performance_comparison"]

    assert record["comparable"] is True
    assert record["verdict"] == verdict
    assert record["gain_pct"] == pytest.approx(gain)
    assert (record["tput_reference"], record["tput_candidate"]) == (1000.0, total)


def test_report_performance_summary_does_not_alias_live_measurements(report_performance_state):
    state = report_performance_state
    summary = rp._build_summary_dict(state, {}, [])
    snapshot = deepcopy(summary)

    state.current_best["total_throughput"] = 9999.0
    state.current_best["extra_envs"]["RECIPE"] = "unmeasured"
    state.baseline_perf["total_throughput"] = 8888.0

    assert summary == snapshot


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
    # The report is written from a reloaded state, so the flag that separates the two skips has to survive the round
    # trip to be readable at all.
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
