# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SBD V6 measurement / kernel stage projections and the ``close`` key.

Companion to ``test_sbd_v6_initial.py``, which covers the durable ``install``
/ ``model_gate`` events and the Framework Agent projection. Everything here is
projected at export time from V5 sections, so most tests call
``collect_v6_timeline`` / ``collect_v6_close`` directly with the sections a
real exporter run would hand them; the additivity and ordering tests go
through ``exporter.build`` because that is where the isolation actually has to
hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import exporter
from hyperloom.inference_optimizer.breakdown.collectors import v6 as v6_collectors
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.session.sbd_v6 import write_timeline_event


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _event(timeline: list[dict], event_type: str) -> dict | None:
    return next((event for event in timeline if event["type"] == event_type), None)


def _events(timeline: list[dict], event_type: str) -> list[dict]:
    return [event for event in timeline if event["type"] == event_type]


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def test_baseline_projects_measurement_and_its_phase_window(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={},
        baseline={
            "throughput_tok_s_per_gpu": 102.5,
            "ttft_mean_ms": 41.0,
            "e2el_mean_ms": 980.0,
            "benchmark_report_path": "runs/baseline/benchmark_report.json",
            "attempts_history": [{"status": "ok", "ts": "2026-08-27T01:05:00+00:00"}],
        },
        phase_timeline=[
            {"action": "baseline", "ts": "2026-08-27T01:00:00+00:00", "task_id": "t-base-1", "status": "failed"},
            {"action": "baseline", "ts": "2026-08-27T01:05:00+00:00", "task_id": "t-base-2", "status": "ok"},
            {"action": "sweep", "ts": "2026-08-27T02:00:00+00:00", "task_id": "t-sweep"},
        ],
    )

    event = _event(timeline, "baseline")
    assert event["kind"] == "baseline"
    assert event["status"] == "succeeded"
    assert event["start_time"] == "2026-08-27T01:00:00+00:00"
    assert event["end_time"] == "2026-08-27T01:05:00+00:00"
    assert event["ext"]["task_id"] == "t-base-2"
    assert event["ext"]["throughput_tok_s_per_gpu"] == 102.5
    assert event["ext"]["benchmark_report_path"] == "runs/baseline/benchmark_report.json"


def test_baseline_reports_failure_when_no_attempt_produced_throughput(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={},
        baseline={
            "throughput_tok_s_per_gpu": 0.0,
            "attempts_history": [
                {"status": "failed", "ts": "2026-08-27T01:00:00+00:00", "error_class": "ServerLaunchError",
                 "error_excerpt": "port already bound"},
            ],
        },
        phase_timeline=[{"action": "baseline", "ts": "2026-08-27T01:00:00+00:00", "status": "failed"}],
    )

    event = _event(timeline, "baseline")
    assert event["status"] == "failed"
    # 0.0 is the V5 "never measured" sentinel and must not read as a measurement.
    assert event["ext"]["throughput_tok_s_per_gpu"] is None
    assert event["ext"]["failure"] == {"error_class": "ServerLaunchError", "error": "port already bound"}


def test_baseline_without_any_evidence_produces_no_event(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={},
        baseline={"throughput_tok_s_per_gpu": 0.0, "attempts_history": []},
        phase_timeline=[],
    )

    assert _event(timeline, "baseline") is None


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _sweep_section() -> dict:
    return {
        "all_variants": [
            {
                "variant_name": "variant_c64_i1024_o1024",
                "conc": 64,
                "isl": 1024,
                "osl": 1024,
                "status": "ok",
                "output_throughput_tok_s": 130.0,
                "benchmark_report_path": "runs/sweep/v1/benchmark_report.json",
            },
            {
                "variant_name": "variant_c128_i1024_o512",
                "conc": 128,
                "isl": 1024,
                "osl": 512,
                "status": "failed",
                "output_throughput_tok_s": None,
                "error": "server crashed",
            },
        ],
        "best_overall": {"variant_name": "variant_c64_i1024_o1024"},
    }


def test_sweep_reads_the_grid_back_off_the_points_it_measured(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={
            "last_sweep": {"workspace": "runs/sweep"},
            "sweep_attempts": [{"ts": "2026-08-27T02:30:00+00:00", "status": "ok", "task_id": "t-sweep"}],
        },
        sweep=_sweep_section(),
        phase_timeline=[{"action": "sweep", "ts": "2026-08-27T02:00:00+00:00", "task_id": "t-sweep"}],
    )

    event = _event(timeline, "sweep")
    # One point measured, one lost: the grid ran but did not run whole.
    assert event["status"] == "degraded"
    assert event["ext"]["plan"]["conc_grid"] == [64, 128]
    assert event["ext"]["plan"]["isl_grid"] == [1024]
    assert event["ext"]["plan"]["osl_grid"] == [512, 1024]
    # No producer records where the grid came from.
    assert event["ext"]["plan"]["grid_source"] is None
    assert event["ext"]["artifacts"]["sweep_dir"] == "runs/sweep"
    assert event["ext"]["artifacts"]["sweep_report_paths"] == ["runs/sweep/v1/benchmark_report.json"]
    assert [variant["variant_id"] for variant in event["ext"]["sweep"]["all_variants"]] == [
        "variant_c64_i1024_o1024",
        "variant_c128_i1024_o512",
    ]


def test_sweep_input_anchor_does_not_borrow_the_end_of_session_throughput(tmp_path):
    """``current_best.tput`` is the final figure, not the sweep's entry point."""
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"last_sweep": {"workspace": "runs/sweep"}, "current_best": {"tput": 175.0}},
        sweep=_sweep_section(),
        baseline={"attempts_history": [{"task_id": "t-base", "status": "ok"}]},
    )

    anchor = _event(timeline, "sweep")["ext"]["input_anchor"]
    assert anchor["baseline_task_id"] == "t-base"
    assert anchor["input_throughput_tok_s_per_gpu"] is None
    assert anchor["current_best_task_id"] is None


def test_sweep_without_any_evidence_produces_no_event(tmp_path):
    timeline = collect_v6_timeline(tmp_path, [], state={}, sweep={"all_variants": []})

    assert _event(timeline, "sweep") is None


# ---------------------------------------------------------------------------
# conc_sweep
# ---------------------------------------------------------------------------
def _conc_sweep_section() -> dict:
    return {
        "status": "ok",
        "budget_exhausted": False,
        "concs_requested": [8, 16],
        "total_budget_sec": 600,
        "baseline": {
            "extra_server_args": "",
            "points": [{"conc": 8, "status": "ok", "output_throughput": 90.0}],
        },
        "optimized": {
            "extra_server_args": "--enable-torch-compile",
            "points": [{"conc": 8, "status": "ok", "output_throughput": 99.0}],
        },
        "comparison": [
            {"conc": 8, "baseline_tput": 90.0, "optimized_tput": 99.0, "speedup": 1.1},
            {"conc": 16, "baseline_tput": 120.0, "optimized_tput": None, "speedup": None,
             "optimized_status": "failed"},
        ],
        "summary": {"best_conc": 8, "best_speedup": 1.1},
        "workspace": "runs/conc_sweep",
        "elapsed_sec": 300.0,
    }


def test_conc_sweep_renames_the_comparison_columns_and_keeps_the_arms(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"last_conc_sweep": {"ts": "2026-08-27T04:00:00+00:00", "status": "succeeded"}},
        conc_sweep_summary=_conc_sweep_section(),
    )

    event = _event(timeline, "conc_sweep")
    assert event["status"] == "succeeded"
    # Only the completion time is recorded; the window closes rather than
    # collapsing onto a single instant.
    assert event["start_time"] == ""
    assert event["end_time"] == "2026-08-27T04:00:00+00:00"
    assert event["ext"]["comparison"][0] == {
        "conc": 8,
        "baseline_output_throughput": 90.0,
        "optimized_output_throughput": 99.0,
        "speedup": 1.1,
        "error": None,
    }
    # An unpaired point says which arm failed; the arm's own row says why.
    assert event["ext"]["comparison"][1]["speedup"] is None
    assert event["ext"]["comparison"][1]["error"] == "failed"
    # "" is the baseline arm's defining value, not a missing one.
    assert event["ext"]["arms"]["baseline"]["extra_server_args"] == ""
    assert event["ext"]["arms"]["optimized"]["extra_server_args"] == "--enable-torch-compile"
    assert event["ext"]["result"]["best_conc"] == 8
    assert event["ext"]["runtime"]["elapsed_sec"] == 300.0


def test_conc_sweep_cut_short_by_its_budget_is_degraded_not_succeeded(tmp_path):
    summary = _conc_sweep_section() | {"budget_exhausted": True}
    timeline = collect_v6_timeline(tmp_path, [], state={}, conc_sweep_summary=summary)

    event = _event(timeline, "conc_sweep")
    assert event["status"] == "degraded"
    assert event["ext"]["result"]["budget_exhausted"] is True


def test_conc_sweep_without_any_evidence_produces_no_event(tmp_path):
    assert _event(collect_v6_timeline(tmp_path, [], state={}), "conc_sweep") is None


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------
def _kernel_state(cycles: tuple[int, ...] = (0,)) -> dict:
    history = []
    for cycle in cycles:
        hour = 1 + cycle
        history.extend(
            [
                {"from_phase": "BASELINE", "to_phase": "KERNEL_AGENT", "cycle": cycle,
                 "ts": f"2026-08-27T0{hour}:00:00+00:00"},
                {"from_phase": "KERNEL_AGENT", "to_phase": "SWEEP", "cycle": cycle,
                 "ts": f"2026-08-27T0{hour}:50:00+00:00", "reason": "kernel_no_more_leverage"},
            ]
        )
    return {"phase": "SWEEP", "macro_cycle": cycles[-1], "phase_history": history, "kernel_optimizer": "forge"}


def _kernel_attempt(attempt_id: str, cycle: int, **overrides) -> dict:
    return {
        "attempt_id": attempt_id,
        "kind": "kernel_optimization",
        "macro_cycle": cycle,
        "backend": "triton",
        "status": "succeeded",
        "decision": "KEEP",
        "throughput_before": 100.0,
        "throughput_after": 112.0,
        "local_gain_pct": 12.0,
        "keep_threshold_pct": 3.0,
        "kernel_id": "rmsnorm",
        "ended_at": f"2026-08-27T0{1 + cycle}:30:00+00:00",
        "artifacts": [{"kind": "benchmark_report", "path": "runs/k/benchmark_report.json"}],
        "gates": [{"kind": "accuracy", "status": "ok", "decision": "PASS", "reason": ""}],
    } | overrides


def test_kernel_projects_one_event_per_visit_and_keeps_attempts_on_their_cycle(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state((0, 1)),
        optimizations={
            "attempts": [
                _kernel_attempt("att-c0", 0),
                _kernel_attempt("att-c1", 1),
                # Framework work shares the attempt model and must not leak in.
                {"attempt_id": "att-fw", "kind": "framework_authoring", "macro_cycle": 0, "status": "succeeded"},
            ]
        },
    )

    events = _events(timeline, "kernel")
    assert [event["ext"]["macro_cycle"] for event in events] == [0, 1]
    assert [[attempt["attempt_id"] for attempt in event["ext"]["attempts"]] for event in events] == [
        ["att-c0"],
        ["att-c1"],
    ]
    assert events[0]["kind"] == "agent"
    assert events[0]["start_time"] == "2026-08-27T01:00:00+00:00"
    assert events[0]["end_time"] == "2026-08-27T01:50:00+00:00"
    assert events[0]["ext"]["exit_reason"] == "kernel_no_more_leverage"
    assert events[0]["ext"]["entry"]["from_phase"] == "BASELINE"
    assert events[0]["ext"]["entry"]["route"] == "forge"
    # The first attempt's own comparison base is what the visit started from.
    assert events[0]["ext"]["entry"]["input_throughput"] == 100.0


def test_kernel_lanes_stay_empty_when_only_one_produced_candidates(tmp_path):
    state = _kernel_state() | {
        "last_fusion": {
            "run_id": "fus-1",
            "status": "ok",
            "kept": True,
            "best_pattern": "rmsnorm_add",
            "serving_speedup": 1.07,
            "micro_decision": "candidate",
            "ts": "2026-08-27T01:20:00+00:00",
        },
        "last_fusion_integrate": {"decision": "KEEP", "reason": "gain above threshold"},
    }

    event = _event(collect_v6_timeline(tmp_path, [], state=state), "kernel")

    assert [run["run_id"] for run in event["ext"]["fusion_runs"]] == ["fus-1"]
    assert event["ext"]["fusion_runs"][0]["candidate_speedup_basis"] == "serving_ab"
    assert event["ext"]["fusion_runs"][0]["outcome"] == "KEEP"
    assert event["ext"]["kernel_rewrites"] == []
    assert event["ext"]["gemm_tuning_runs"] == []
    assert event["ext"]["collective_runs"] == []
    assert event["ext"]["geak_runs"] == []
    # Candidates but no final rebench: the visit did work that never landed.
    assert event["status"] == "degraded"


def test_kernel_links_each_lane_row_to_the_rebench_that_validated_it(tmp_path):
    state = _kernel_state() | {
        "last_fusion": {"run_id": "fus-1", "status": "ok", "kept": True, "ts": "2026-08-27T01:20:00+00:00"},
    }
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=state,
        optimizations={
            "attempts": [
                _kernel_attempt("att-rewrite", 0),
                _kernel_attempt("att-fusion", 0, producer="forge_fusion", name="fusion"),
                _kernel_attempt("att-gemm", 0, kind="gemm_tuning"),
            ]
        },
        kernel_journey={
            "kernels": [
                {
                    "kernel_id": "rmsnorm",
                    "name": "rmsnorm",
                    "backend_attempts": [
                        {"attempt_id": "bk-1", "backend": "triton", "status": "ok", "decision": "candidate",
                         "compile_passed": True, "correctness_passed": True, "micro_speedup": 1.4,
                         "ts": "2026-08-27T01:10:00+00:00"}
                    ],
                    "e2e": {"decision": "KEEP", "target_file": "layers/rmsnorm.py"},
                }
            ]
        },
    )

    ext = _event(timeline, "kernel")["ext"]
    assert ext["kernel_rewrites"][0]["final_rebench_attempt_ids"] == ["att-rewrite"]
    assert ext["fusion_runs"][0]["final_rebench_attempt_ids"] == ["att-fusion"]
    assert [attempt["source_kind"] for attempt in ext["attempts"]] == [
        "kernel_rewrite",
        "fusion",
        "gemm_tuning",
    ]
    assert ext["kernel_rewrites"][0]["verification"] == {
        "compile_passed": True,
        "correctness_passed": True,
        # Which reference the check ran against is decided inside the backend.
        "correctness_source": None,
        "micro_speedup": 1.4,
    }


def test_kernel_failure_is_read_off_the_phase_exit_evidence(tmp_path):
    state = _kernel_state()
    state["phase_history"][-1] = {
        "from_phase": "KERNEL_AGENT",
        "to_phase": "CLOSE",
        "cycle": 0,
        "ts": "2026-08-27T01:50:00+00:00",
        "reason": "kernel_agent_failed",
        "evidence": {"error_class": "CompilerError", "error": "ptxas died", "failed_task_id": "t-k9"},
    }

    event = _event(collect_v6_timeline(tmp_path, [], state=state), "kernel")

    assert event["status"] == "failed"
    assert event["ext"]["failure"] == {
        "failed_task_id": "t-k9",
        "error_class": "CompilerError",
        "error": "ptxas died",
    }


def test_kernel_evidence_without_phase_history_is_kept_under_one_warned_window(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state={"macro_cycle": 2},
        optimizations={"attempts": [_kernel_attempt("att-orphan", 2)]},
    )

    events = _events(timeline, "kernel")
    assert len(events) == 1
    assert events[0]["ext"]["macro_cycle"] == 2
    assert [attempt["attempt_id"] for attempt in events[0]["ext"]["attempts"]] == ["att-orphan"]
    assert any("no KERNEL_AGENT phase history" in warning for warning in warnings)


def test_kernel_warns_when_the_session_scoped_geak_record_spans_several_visits(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state((0, 1)),
        geak={"engaged": True, "status": "ok", "gain_pct": 8.0, "final_throughput_tok_s": 120.0},
    )

    events = _events(timeline, "kernel")
    assert [len(event["ext"]["geak_runs"]) for event in events] == [1, 0]
    assert any("session-scoped across 2 kernel visits" in warning for warning in warnings)


def test_kernel_without_a_visit_or_evidence_produces_no_event(tmp_path):
    assert _events(collect_v6_timeline(tmp_path, [], state={}), "kernel") == []


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------
def _close_state(steps: list[dict], **overrides) -> dict:
    return {
        "phase": "CLOSE",
        "phase_history": [
            {"from_phase": "SWEEP", "to_phase": "CLOSE", "ts": "2026-08-27T05:00:00+00:00",
             "evidence": {"close_steps": steps}},
        ],
    } | overrides


def test_close_is_degraded_while_the_breakdown_predates_the_rest_of_the_sequence(tmp_path):
    """Pins the known CLOSE write-ordering limitation.

    ``session_breakdown`` is step 2, so the four steps after it are never on
    disk and ``close_sequence_done`` is always false. If the write order is
    ever fixed this test goes red, which is the point.
    """
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "geak_rebench_drain", "status": "skipped", "ts": "2026-08-27T05:00:02+00:00"},
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00", "task_id": "t-report"},
            {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00", "task_id": "t-bd"},
        ]
    )

    close = collect_v6_close(tmp_path, state, {}, [])

    assert close["status"] == "degraded"
    assert close["close_sequence_done"] is False
    assert [step["step"] for step in close["steps"]] == [
        "sequencer_started",
        "geak_rebench_drain",
        "report",
        "session_breakdown",
    ]
    assert close["start_time"] == "2026-08-27T05:00:01+00:00"
    assert close["end_time"] == "2026-08-27T05:00:45+00:00"
    assert close["steps"][2]["task_id"] == "t-report"


def test_close_is_succeeded_only_when_every_step_settled_and_the_sequence_finished(tmp_path):
    state = _close_state(
        [
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    assert collect_v6_close(tmp_path, state, {}, [])["status"] == "succeeded"


def test_close_reports_degraded_when_a_step_failed(tmp_path):
    state = _close_state(
        [
            {"step": "report", "status": "failed", "ts": "2026-08-27T05:00:30+00:00", "detail": "task_state='failed'"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    close = collect_v6_close(tmp_path, state, {}, [])
    assert close["status"] == "degraded"
    assert close["steps"][0]["detail"] == "task_state='failed'"


def test_close_without_any_step_reports_failed(tmp_path):
    close = collect_v6_close(tmp_path, {"phase_history": []}, {}, [])

    assert close["status"] == "failed"
    assert close["steps"] == []
    assert close["start_time"] == ""


def test_close_falls_back_to_the_phase_entry_when_no_step_was_recorded(tmp_path):
    state = {"phase_history": [{"from_phase": "SWEEP", "to_phase": "CLOSE", "ts": "2026-08-27T05:00:00+00:00"}]}

    assert collect_v6_close(tmp_path, state, {}, [])["start_time"] == "2026-08-27T05:00:00+00:00"


def test_close_collects_steps_split_across_phase_history_rows(tmp_path):
    state = {
        "phase_history": [
            {"to_phase": "CLOSE", "ts": "2026-08-27T05:00:00+00:00",
             "evidence": {"close_steps": [{"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"}]}},
            {"to_phase": "CLOSE", "ts": "2026-08-27T05:01:00+00:00",
             "evidence": {"close_steps": [{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}]}},
        ],
        "close_sequence_done": True,
    }

    close = collect_v6_close(tmp_path, state, {}, [])
    assert [step["step"] for step in close["steps"]] == ["report", "done"]
    assert close["status"] == "succeeded"


def test_close_surfaces_robustness_escalation_and_its_signals(tmp_path):
    state = _close_state([{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}])
    state["stop_reason"] = "robustness_escalated"
    signals = [{"ts": "2026-08-27T04:00:00+00:00", "signal": "crash", "action": "restart",
                "workdir": "robustness-workdir/0"}]

    close = collect_v6_close(tmp_path, state, {"robustness_signals": signals}, [])

    assert close["robustness"]["escalated"] is True
    assert close["robustness"]["signals"] == signals


def test_close_artifacts_point_only_at_files_that_exist(tmp_path):
    _write_json(tmp_path / "reports" / "final.json", {"ok": True})
    state = _close_state(
        [{"step": "artifact_package", "status": "done", "ts": "2026-08-27T05:02:00+00:00",
          "detail": str(tmp_path / "bundle.zip")}]
    )

    artifacts = collect_v6_close(tmp_path, state, {}, [])["artifacts"]

    assert artifacts["final_json_path"] == "reports/final.json"
    assert artifacts["final_md_path"] is None
    assert artifacts["session_breakdown_path"] == "session_breakdown.json"
    assert artifacts["artifact_package_path"] == "bundle.zip"


def test_close_ignores_a_skipped_artifact_package_detail(tmp_path):
    """``detail`` doubles as the skip reason; only a ``done`` row holds a path."""
    state = _close_state(
        [{"step": "artifact_package", "status": "skipped", "ts": "2026-08-27T05:02:00+00:00",
          "detail": "no artifacts matched or dest unwritable"}]
    )

    assert collect_v6_close(tmp_path, state, {}, [])["artifacts"]["artifact_package_path"] is None


# ---------------------------------------------------------------------------
# cross-cutting: ordering and additivity
# ---------------------------------------------------------------------------
def test_projected_stages_interleave_with_durable_events_by_time(tmp_path):
    write_timeline_event(
        tmp_path,
        {
            "type": "install",
            "kind": "install",
            "status": "succeeded",
            "start_time": "2026-08-27T00:58:00+00:00",
            "end_time": "2026-08-27T00:59:00+00:00",
            "ext": {"run_kind": "fresh", "hard_fail_step_id": None, "runtime_snapshot": {}, "steps": []},
        },
    )

    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        baseline={
            "throughput_tok_s_per_gpu": 100.0,
            "attempts_history": [{"status": "ok", "ts": "2026-08-27T00:30:00+00:00"}],
        },
        conc_sweep_summary=_conc_sweep_section(),
    )

    # The conc sweep has no recorded time at all, so it sorts last rather than
    # to the epoch.
    assert [event["type"] for event in timeline] == ["baseline", "install", "kernel", "conc_sweep"]


@pytest.mark.parametrize(
    "projector",
    ["project_baseline_event", "project_sweep_event", "project_conc_sweep_event", "project_kernel_events"],
)
def test_a_raising_stage_projector_cannot_disturb_the_v5_payload(tmp_path, monkeypatch, projector):
    _write_json(
        tmp_path / "state.json",
        {"session_id": "s1", "model_name": "M", "framework": "sglang", "baseline_tput": 100.0, "phase": "CLOSE"},
    )
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M", "framework": "sglang"})
    before = exporter.build(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError(f"{projector} exploded")

    monkeypatch.setattr(v6_collectors, projector, _boom)
    after = exporter.build(tmp_path)

    v6_keys = {"exported_at_utc", "metadata", "outcome", "timeline", "close"}
    assert {key: value for key, value in after.items() if key not in v6_keys} == {
        key: value for key, value in before.items() if key not in v6_keys
    }
    assert after["warnings"] == before["warnings"]
    assert after["timeline"] == []
    assert any(projector in warning or "timeline" in warning for warning in after["metadata"]["warnings"])


def test_a_raising_close_collector_cannot_disturb_the_v5_payload(tmp_path, monkeypatch):
    _write_json(tmp_path / "state.json", {"session_id": "s1", "model_name": "M", "phase": "CLOSE"})
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M"})
    before = exporter.build(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("close exploded")

    monkeypatch.setattr(exporter.collectors, "collect_v6_close", _boom)
    after = exporter.build(tmp_path)

    assert after["warnings"] == before["warnings"]
    assert after["close"] == {}
    assert any("close" in warning for warning in after["metadata"]["warnings"])
