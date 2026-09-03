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
import zipfile
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown import exporter, session_package
from hyperloom.inference_optimizer.breakdown.collectors import v6 as v6_collectors
from hyperloom.inference_optimizer.breakdown.collectors.v6 import collect_v6_timeline
from hyperloom.inference_optimizer.breakdown.collectors.v6_close import collect_v6_close
from hyperloom.inference_optimizer.session.sbd_v6 import write_timeline_event
from hyperloom.orchestrator.phases.machine_state import GEAK_TERMINAL_STATUSES


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
                {
                    "status": "failed",
                    "ts": "2026-08-27T01:00:00+00:00",
                    "error_class": "ServerLaunchError",
                    "error_excerpt": "port already bound",
                },
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
            "points": [
                {"conc": 8, "status": "ok", "output_throughput": 90.0},
                {"conc": 16, "status": "ok", "output_throughput": 120.0},
            ],
        },
        "optimized": {
            "extra_server_args": "--enable-torch-compile",
            "points": [
                {"conc": 8, "status": "ok", "output_throughput": 99.0},
                {"conc": 16, "status": "failed", "output_throughput": None, "error": "server OOM at conc=16"},
            ],
        },
        # ``conc_pair_comparison`` stamps both arm statuses on every row, and
        # ``None`` where an arm has no point at that concurrency at all.
        "comparison": [
            {
                "conc": 8,
                "baseline_tput": 90.0,
                "optimized_tput": 99.0,
                "speedup": 1.1,
                "baseline_status": "ok",
                "optimized_status": "ok",
            },
            {
                "conc": 16,
                "baseline_tput": 120.0,
                "optimized_tput": None,
                "speedup": None,
                "baseline_status": "ok",
                "optimized_status": "failed",
            },
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
        "baseline_throughput": 90.0,
        "optimized_throughput": 99.0,
        "speedup": 1.1,
        "error": None,
    }
    # An unpaired point names the arm that broke and quotes that arm's own
    # error. Reporting the first of the two statuses would have said
    # "succeeded" here, since it is the baseline arm that came through.
    assert event["ext"]["comparison"][1]["speedup"] is None
    assert event["ext"]["comparison"][1]["error"] == "optimized: server OOM at conc=16"
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
                {
                    "from_phase": "BASELINE",
                    "to_phase": "KERNEL_AGENT",
                    "cycle": cycle,
                    "ts": f"2026-08-27T0{hour}:00:00+00:00",
                },
                {
                    "from_phase": "KERNEL_AGENT",
                    "to_phase": "SWEEP",
                    "cycle": cycle,
                    "ts": f"2026-08-27T0{hour}:50:00+00:00",
                    "reason": "kernel_no_more_leverage",
                },
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


def test_kernel_ignores_an_integrate_patch_the_framework_agent_owns(tmp_path):
    """``integrate_patch`` lands every patch source, so its kind proves nothing.

    Filtering the ledger on kind alone files a Framework Agent enablement patch
    under the Kernel timeline. Ownership lives on ``agent`` / ``phase``, which
    ``_attempt_agent`` has already resolved through ``patch_author``.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-kernel", 0, agent="kernel_agent", phase="KERNEL_AGENT"),
                _kernel_attempt(
                    "att-framework",
                    0,
                    kind="integrate_patch",
                    agent="framework_agent",
                    phase="FRAMEWORK_AGENT",
                    kernel_id=None,
                ),
            ]
        },
    )

    ext = _event(timeline, "kernel")["ext"]
    assert [attempt["attempt_id"] for attempt in ext["attempts"]] == ["att-kernel"]


def test_a_framework_owned_integrate_patch_alone_does_not_conjure_a_kernel_visit(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state={"phase": "CLOSE", "phase_history": []},
        optimizations={
            "attempts": [
                {
                    "attempt_id": "att-framework",
                    "kind": "integrate_patch",
                    "agent": "framework_agent",
                    "phase": "FRAMEWORK_AGENT",
                    "macro_cycle": 0,
                    "status": "succeeded",
                    "decision": "KEEP",
                }
            ]
        },
    )

    assert _events(timeline, "kernel") == []


def test_kernel_falls_back_to_kind_for_a_row_recorded_before_ownership_existed(tmp_path):
    """Old recordings carry no ``agent``/``phase``; a kernel-only kind still counts.

    ``integrate_patch`` is deliberately not in that fallback set — it is the
    one kind more than one agent produces.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-legacy-kernel", 0),
                {"attempt_id": "att-legacy-patch", "kind": "integrate_patch", "macro_cycle": 0, "status": "succeeded"},
            ]
        },
    )

    ext = _event(timeline, "kernel")["ext"]
    assert [attempt["attempt_id"] for attempt in ext["attempts"]] == ["att-legacy-kernel"]


def test_kernel_is_failed_when_every_final_rebench_faulted(tmp_path):
    """A visit whose rebenches all errored measured nothing.

    Counting attempts as evidence of success reported ``succeeded`` for a visit
    where no candidate was ever weighed.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-1", 0, status="failed", decision="FAILED", throughput_after=None),
                _kernel_attempt("att-2", 0, status="error", decision="FAILED", throughput_after=None),
            ]
        },
    )

    event = _event(timeline, "kernel")
    assert event["status"] == "failed"
    assert [attempt["status"] for attempt in event["ext"]["attempts"]] == ["failed", "failed"]


def test_kernel_is_succeeded_when_a_rebench_concluded_even_against_the_candidate(tmp_path):
    """A REVERT is a verdict. The visit worked; the candidate did not."""
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={"attempts": [_kernel_attempt("att-1", 0, decision="REVERT", local_gain_pct=0.4)]},
    )

    assert _event(timeline, "kernel")["status"] == "succeeded"


def test_kernel_lanes_stay_empty_when_only_one_produced_candidates(tmp_path):
    state = _kernel_state() | {
        "last_fusion": {
            "fusion_run_id": "fus-1",
            "status": "ok",
            "kept": True,
            "best_pattern": "rmsnorm_add",
            "serving_speedup": 1.07,
            "micro_decision": "candidate",
            "ts": "2026-08-27T01:20:00+00:00",
        },
        "last_fusion_integrate": {
            "fusion_run_id": "fus-1",
            "decision": "KEEP",
            "reason": "gain above threshold",
        },
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
                        {
                            "attempt_id": "bk-1",
                            "backend": "triton",
                            "status": "ok",
                            "decision": "KEEP",
                            "compile_passed": True,
                            "correctness_passed": True,
                            "micro_speedup": 1.4,
                            "ts": "2026-08-27T01:10:00+00:00",
                        }
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


def _journey_kernel(kernel_id: str, ts: str) -> dict:
    return {
        "kernel_id": kernel_id,
        "name": kernel_id,
        "backend_attempts": [
            {
                "attempt_id": f"bk-{kernel_id}",
                "backend": "triton",
                "status": "ok",
                "decision": "KEEP",
                "micro_speedup": 1.3,
                "ts": ts,
            }
        ],
        "e2e": {"decision": "KEEP", "target_file": f"layers/{kernel_id}.py"},
    }


def test_two_rewrites_in_one_visit_each_link_only_to_their_own_rebench(tmp_path):
    """Grouping by ``source_kind`` alone gave every rewrite every attempt.

    With one row per lane the cross-product is invisible; with two it claims
    each candidate was validated by a rebench of the other one.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-rmsnorm", 0, kernel_id="rmsnorm"),
                _kernel_attempt("att-softmax", 0, kernel_id="softmax"),
            ]
        },
        kernel_journey={
            "kernels": [
                _journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00"),
                _journey_kernel("softmax", "2026-08-27T01:20:00+00:00"),
            ]
        },
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert {row["kernel_id"]: row["final_rebench_attempt_ids"] for row in rewrites} == {
        "rmsnorm": ["att-rmsnorm"],
        "softmax": ["att-softmax"],
    }


def test_an_unidentifiable_rebench_is_left_unlinked_and_warned_about(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-rmsnorm", 0, kernel_id="rmsnorm"),
                # Names a kernel neither recorded rewrite claims.
                _kernel_attempt("att-orphan", 0, kernel_id="layernorm"),
            ]
        },
        kernel_journey={
            "kernels": [
                _journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00"),
                _journey_kernel("softmax", "2026-08-27T01:20:00+00:00"),
            ]
        },
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert {row["kernel_id"]: row["final_rebench_attempt_ids"] for row in rewrites} == {
        "rmsnorm": ["att-rmsnorm"],
        "softmax": [],
    }
    # The attempt still appears in the ledger; only the candidate edge is
    # withheld, because guessing which of the two it belongs to would read as
    # a candidate having been validated when it was not.
    assert "att-orphan" in [attempt["attempt_id"] for attempt in _event(timeline, "kernel")["ext"]["attempts"]]
    assert any("match none of the 2 recorded candidates" in warning for warning in warnings)


def test_a_lone_candidate_does_not_absorb_a_rebench_of_a_different_kernel(tmp_path):
    """Being the only candidate is not evidence the record meant this one.

    The attempt says it rebenched ``layernorm``; the visit's only rewrite is
    ``rmsnorm``. Absorbing it would report that ``rmsnorm`` was validated
    end-to-end on the strength of a measurement of something else.
    """
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        optimizations={"attempts": [_kernel_attempt("att-1", 0, kernel_id="layernorm")]},
        kernel_journey={"kernels": [_journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00")]},
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert rewrites[0]["final_rebench_attempt_ids"] == []
    assert any("name kernels (layernorm)" in warning for warning in warnings)


def test_a_lone_candidate_absorbs_a_rebench_that_names_no_subject(tmp_path):
    """The fallback survives for what it was for: an attempt with no subject.

    ``source_id`` degrades to the attempt's own id when the producer recorded
    no kernel, which is a record of *what kind* of thing was rebenched and
    nothing more. With one candidate of that kind in the visit, it can only be
    that one.
    """
    warnings: list[str] = []
    attempt = _kernel_attempt("att-1", 0)
    attempt.pop("kernel_id", None)
    attempt.pop("name", None)
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        optimizations={"attempts": [attempt]},
        kernel_journey={"kernels": [_journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00")]},
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert rewrites[0]["final_rebench_attempt_ids"] == ["att-1"]
    assert warnings == []


def test_a_fusion_rebench_naming_its_kernel_is_not_a_conflict(tmp_path):
    """Kernel ids and fusion run ids are different vocabularies.

    A fusion rebench records the kernel it patched; the fusion lane is keyed by
    run id. Reading those as rival claims would withhold a link the record
    fully supports, so the conflict rule only fires on a lane that is itself
    keyed on kernel identity.
    """
    warnings: list[str] = []
    state = _kernel_state() | {
        "last_fusion": {"run_id": "fus-1", "status": "ok", "kept": True, "ts": "2026-08-27T01:20:00+00:00"},
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        optimizations={
            "attempts": [_kernel_attempt("att-fusion", 0, producer="forge_fusion", kernel_id="rmsnorm")],
        },
    )

    fusion_runs = _event(timeline, "kernel")["ext"]["fusion_runs"]
    assert fusion_runs[0]["final_rebench_attempt_ids"] == ["att-fusion"]
    assert warnings == []


def _two_backend_kernel() -> dict:
    """One kernel, two backends, one winner — the shape adoption is about."""
    return {
        "kernels": [
            {
                "kernel_id": "rmsnorm",
                "name": "rmsnorm",
                "micro_speedup": 1.4,
                "backend_attempts": [
                    {
                        "attempt_id": "backend-good",
                        "backend": "triton",
                        "status": "completed",
                        "decision": "KEEP",
                        "micro_speedup": 1.4,
                        "best_artifact_path": "kernels/rmsnorm.py",
                        "ts": "2026-08-27T01:10:00+00:00",
                    },
                    {
                        "attempt_id": "backend-bad",
                        "backend": "hip",
                        "status": "failed",
                        "decision": "FAILED",
                        "error": "compile error",
                        "ts": "2026-08-27T01:12:00+00:00",
                    },
                ],
                "e2e": {"decision": "KEEP", "target_file": "layers/rmsnorm.py"},
            }
        ]
    }


def test_only_the_adopted_backend_attempt_carries_the_kernels_e2e_outcome(tmp_path):
    """``e2e`` describes the attempt that was integrated, not every attempt.

    V5 already stamps the kernel decision onto the adopted attempt alone. A
    losing backend inheriting ``KEEP`` reads as a rewrite that shipped, next to
    a ``micro_decision`` of ``FAILED`` saying it never compiled.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        kernel_journey=_two_backend_kernel(),
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert {row["rewrite_id"]: row["outcome"] for row in rewrites} == {
        "backend-good": "KEEP",
        "backend-bad": "FAILED",
    }
    assert {row["rewrite_id"]: row["micro_decision"] for row in rewrites} == {
        "backend-good": "KEEP",
        "backend-bad": "FAILED",
    }
    # The kernel's best micro speedup belongs to the attempt that achieved it.
    assert {row["rewrite_id"]: row["verification"]["micro_speedup"] for row in rewrites} == {
        "backend-good": 1.4,
        "backend-bad": None,
    }


def _forge_result() -> dict:
    """A KernelForge result as ``run_attempt`` + ``build_verification`` write it.

    The attempt rows carry no verdict and no gates — the backend records how
    the run went and nothing else — and ``optimized_path`` is the attempt's
    stdout log, not the rewrite. The verdict, the gates and the source
    artifact are computed once for the kernel and sit beside ``attempts``.
    """
    return {
        "kernel_id": "rmsnorm",
        "run_id": "run-1",
        "attempts": [
            {
                "attempt_id": "a-good",
                "backend": "forge",
                "status": "completed",
                "optimized_path": "optimized/a-good_stdout.log",
                "created_at": "2026-08-27T01:10:00+00:00",
                "elapsed_s": 42.0,
            },
            {
                "attempt_id": "a-bad",
                "backend": "claude",
                "status": "failed",
                "error_type": "compile_error",
                "error": "no kernel emitted",
                "optimized_path": "optimized/a-bad_stdout.log",
                "created_at": "2026-08-27T01:12:00+00:00",
            },
        ],
        "verification": {
            "compile_passed": True,
            "correctness_passed": True,
            "correctness_source": "forge_rewrite_reference",
            "micro_speedup": 1.2,
            "best_attempt_id": "a-good",
            "best_backend": "forge",
            "best_artifact_path": "kernels/rmsnorm.py",
            "artifact_valid": True,
        },
        "proposal": {"decision": "KEEP", "reasons": ["kernel artifact ready"]},
    }


def test_a_recorded_forge_run_reaches_v6_with_its_verdict_on_the_winner(tmp_path):
    """recorder -> assembler -> V6, on the shape a real backend writes.

    Hand-built journey fixtures put ``decision`` / ``compile_passed`` /
    ``best_artifact_path`` straight onto the attempt, which no producer does.
    Read off the attempt alone, a successful adopted rewrite came out as
    ``micro_decision: SKIPPED`` with null gates and an artifact pointing at the
    backend's stdout log. This drives the real chain instead.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import assembler, record_kernel_backend_result

    record_kernel_backend_result(tmp_path, _forge_result())
    kernel_journey = assembler.assemble_parts(tmp_path).get("kernel_journey")

    warnings: list[str] = []
    timeline = collect_v6_timeline(tmp_path, warnings, state=_kernel_state(), kernel_journey=kernel_journey)

    rewrites = {row["rewrite_id"]: row for row in _event(timeline, "kernel")["ext"]["kernel_rewrites"]}
    winner = rewrites["a-good"]
    assert winner["micro_decision"] == "KEEP"
    assert winner["execution_status"] == "succeeded"
    assert winner["verification"] == {
        "compile_passed": True,
        "correctness_passed": True,
        "correctness_source": "forge_rewrite_reference",
        "micro_speedup": 1.2,
    }
    # The rewritten source, not the attempt's stdout log.
    assert winner["artifact"]["artifact_path"] == "kernels/rmsnorm.py"

    loser = rewrites["a-bad"]
    # A failed attempt with no verdict of its own is a failure, not a skip.
    assert loser["micro_decision"] == "FAILED"
    assert loser["outcome"] == "FAILED"
    # The winner's gates and artifact are the winner's.
    assert loser["verification"]["compile_passed"] is None
    assert loser["verification"]["correctness_passed"] is None
    assert loser["artifact"]["artifact_path"] == "optimized/a-bad_stdout.log"
    assert warnings == []


def test_the_kernel_verdict_stays_unstamped_when_nothing_was_adopted(tmp_path):
    """No ``best_attempt_id`` means no attempt was adopted.

    ``_best_attempt_id``'s speedup fallback would still name one, which would
    hand a kernel-wide REVERT to whichever row sorted first.
    """
    from hyperloom.inference_optimizer.breakdown.recorder import assembler, record_kernel_backend_result

    result = _forge_result()
    result["verification"] |= {
        "compile_passed": False,
        "correctness_passed": False,
        "best_attempt_id": "",
        "best_artifact_path": "",
    }
    result["proposal"] = {"decision": "REVERT", "reasons": ["compile failed"]}
    result["attempts"][0]["status"] = "failed"
    record_kernel_backend_result(tmp_path, result)

    attempts = assembler.assemble_parts(tmp_path)["kernel_journey"]["kernels"][0]["backend_attempts"]
    assert {row["attempt_id"]: row["decision"] for row in attempts} == {"a-good": "FAILED", "a-bad": "FAILED"}
    assert {row["attempt_id"]: row["compile_passed"] for row in attempts} == {"a-good": None, "a-bad": None}
    assert {row["attempt_id"]: row["best_artifact_path"] for row in attempts} == {"a-good": "", "a-bad": ""}


def test_a_losing_backend_attempt_does_not_claim_the_kernels_rebench(tmp_path):
    """Every attempt on a kernel shares its ``kernel_id``.

    Offering that id as an identity key for all of them lets a failed backend
    claim the rebench the adopted one triggered.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={"attempts": [_kernel_attempt("att-rmsnorm", 0, kernel_id="rmsnorm")]},
        kernel_journey=_two_backend_kernel(),
    )

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert {row["rewrite_id"]: row["final_rebench_attempt_ids"] for row in rewrites} == {
        "backend-good": ["att-rmsnorm"],
        "backend-bad": [],
    }


def test_a_kernel_with_one_backend_attempt_is_unchanged_by_adoption(tmp_path):
    """The common shape: nothing to lose to, so nothing is withheld."""
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={"attempts": [_kernel_attempt("att-rmsnorm", 0, kernel_id="rmsnorm")]},
        kernel_journey={"kernels": [_journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00")]},
    )

    row = _event(timeline, "kernel")["ext"]["kernel_rewrites"][0]
    assert row["outcome"] == "KEEP"
    assert row["final_rebench_attempt_ids"] == ["att-rmsnorm"]


def test_producer_status_words_are_mapped_onto_the_documented_enums(tmp_path):
    """``completed`` and ``ok`` are ordinary successes, not schema violations."""
    warnings: list[str] = []
    state = _kernel_state() | {
        "last_fusion": {"run_id": "fus-1", "status": "ok", "kept": True, "ts": "2026-08-27T01:20:00+00:00"},
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        kernel_journey=_two_backend_kernel(),
    )

    ext = _event(timeline, "kernel")["ext"]
    assert {row["rewrite_id"]: row["execution_status"] for row in ext["kernel_rewrites"]} == {
        "backend-good": "succeeded",
        "backend-bad": "failed",
    }
    assert ext["fusion_runs"][0]["status"] == "succeeded"
    # Neither spelling is drift; both are the normal success path.
    assert warnings == []


def test_an_unknown_status_word_is_reported_as_failed_and_warned_about(tmp_path):
    warnings: list[str] = []
    journey = _two_backend_kernel()
    journey["kernels"][0]["backend_attempts"][0]["status"] = "quiesced"
    timeline = collect_v6_timeline(tmp_path, warnings, state=_kernel_state(), kernel_journey=journey)

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert rewrites[0]["execution_status"] == "failed"
    assert any("unrecognized kernel_rewrites status 'quiesced'" in warning for warning in warnings)


def test_the_journey_rollup_vocabulary_is_translated_not_uppercased(tmp_path):
    """``outcome`` is a closed enum; ``kernel_journey.outcome`` is not in it."""
    warnings: list[str] = []
    journey = _two_backend_kernel()
    journey["kernels"][0]["e2e"] = {}
    journey["kernels"][0]["outcome"] = "adopted"
    timeline = collect_v6_timeline(tmp_path, warnings, state=_kernel_state(), kernel_journey=journey)

    rewrites = _event(timeline, "kernel")["ext"]["kernel_rewrites"]
    assert rewrites[0]["outcome"] == "KEEP"
    assert warnings == []


def test_a_visit_whose_every_rebench_was_skipped_is_not_succeeded(tmp_path):
    """A skip measured nothing, so it concludes nothing.

    The candidate that was built and never validated leaves the visit
    ``degraded``, not ``succeeded``.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-1", 0, kernel_id="rmsnorm", status="skipped", decision="FAILED"),
            ]
        },
        kernel_journey={"kernels": [_journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00")]},
    )

    assert _event(timeline, "kernel")["status"] == "degraded"


def test_a_visit_holding_only_skipped_rebenches_and_no_candidates_is_skipped(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-1", 0, kernel_id="rmsnorm", status="skipped", decision="FAILED"),
            ]
        },
    )

    assert _event(timeline, "kernel")["status"] == "skipped"


def test_a_measured_rebench_still_makes_the_visit_succeeded(tmp_path):
    """The ladder's ordinary path, pinned against the skipped-attempt fix."""
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt("att-1", 0, kernel_id="rmsnorm", status="succeeded", decision="REVERT"),
            ]
        },
        kernel_journey={"kernels": [_journey_kernel("rmsnorm", "2026-08-27T01:10:00+00:00")]},
    )

    assert _event(timeline, "kernel")["status"] == "succeeded"


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


_GEAK_OK = {"engaged": True, "status": "ok", "gain_pct": 8.0, "final_throughput_tok_s": 120.0}


def _assembled_optimizations(tmp_path):
    """Return the canonical optimization projection of recorder output."""
    from hyperloom.inference_optimizer.breakdown.collectors.optimizations import collect_recorded_optimizations
    from hyperloom.inference_optimizer.breakdown.recorder import assembler

    assembled = assembler.assemble_parts(tmp_path)
    warnings: list[str] = []
    optimizations = collect_recorded_optimizations(
        "s1",
        assembled.get("operations") or [],
        assembled.get("measurements") or [],
        assembled.get("adoptions") or [],
        assembled.get("artifacts") or [],
        [],
        [],
        warnings,
    )
    return assembled, optimizations, warnings


def test_a_recorded_geak_final_validation_reaches_v6_as_a_keep(tmp_path):
    """GEAK's real final verdict lives on ``kernel_optimizer_run``.

    That route-level operation is intentionally outside V5
    ``optimizations.attempts``. V6 must recover its final-validation substep
    rather than relying on a hand-built ``kernel_optimization`` attempt no
    producer writes.
    """
    from hyperloom.inference_optimizer.breakdown.collectors.geak import collect_geak
    from hyperloom.inference_optimizer.breakdown.recorder import record_geak_operation

    result = {
        "status": "ok",
        "baseline_throughput_tok_s": 100.0,
        "final_throughput_tok_s": 120.0,
        "throughput_speedup": 1.2,
        "report_path": "geak/report.json",
        "accepted_config": {"extra_server_args": "--enable-geak"},
    }
    record_geak_operation(tmp_path, stage="runner_result", result=result, status="ok", macro_cycle=0)
    record_geak_operation(
        tmp_path,
        stage="final_validation",
        result=result,
        status="succeeded",
        validated=True,
        measured_tput=120.0,
        validation_source="geak_orch_harness",
        macro_cycle=0,
    )
    assembled, optimizations, warnings = _assembled_optimizations(tmp_path)
    # This is the production mismatch the V6 projector has to bridge.
    assert optimizations["attempts"] == []

    state = _kernel_state() | {"kernel_optimizer": "geak", "geak_result": result}
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        geak=collect_geak(tmp_path, state, warnings),
        optimizations=optimizations,
        recorded_operations=assembled.get("operations") or [],
    )

    event = _event(timeline, "kernel")
    attempt = event["ext"]["attempts"][0]
    run = event["ext"]["geak_runs"][0]
    assert attempt["source_kind"] == "geak_e2e"
    assert attempt["status"] == "succeeded"
    assert attempt["decision"] == "KEEP"
    assert attempt["output_throughput"] == 120.0
    assert attempt["validation_source"] == "geak_same_harness"
    assert run["final_rebench_attempt_ids"] == [attempt["attempt_id"]]
    assert run["outcome"] == "KEEP"


def test_canonical_geak_attempt_outranks_route_fallback_and_carries_both_gains(tmp_path):
    """PR #1340's attempt is the verdict and ledger identity for new sessions."""
    warnings: list[str] = []
    route = {
        "operation_id": "geak-route-0",
        "kind": "kernel_optimizer_run",
        "name": "geak",
        "strategy": "geak",
        "agent": "kernel_agent",
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "status": "succeeded",
        "substeps": [
            {
                "kind": "final_validation",
                "status": "succeeded",
                "ended_at": "2026-08-27T01:29:00+00:00",
                "metadata": {"final_validation": True},
            }
        ],
        "gates": [
            {
                "kind": "final_validation",
                "status": "passed",
                "evidence": {"source": "geak_orch_harness", "measured_tput": 120.0},
            }
        ],
    }
    canonical = _kernel_attempt(
        "geak-e2e-0",
        0,
        # PR #1340 deliberately allows a GEAK route win to use the GEMM kind
        # when an AITER_CONFIG_* table is the lever. Its name still makes it a
        # GEAK final rebench rather than a standalone GEMM-tuning lane.
        kind="gemm_tuning",
        name="geak_e2e",
        backend="geak",
        producer="coordinator",
        kernel_id="geak",
        local_gain_pct=4.0,
        throughput_before=115.0,
        throughput_after=120.0,
        validation_basis="e2e_validation",
    )
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state() | {"kernel_optimizer": "geak"},
        geak=_GEAK_OK,
        optimizations={
            "attempts": [canonical],
            "entries": [{"adopted_attempt_id": "geak-e2e-0", "gain_pct": 5.0}],
        },
        recorded_operations=[route],
    )

    event = _event(timeline, "kernel")
    assert len(event["ext"]["attempts"]) == 1
    attempt = event["ext"]["attempts"][0]
    run = event["ext"]["geak_runs"][0]
    assert attempt["attempt_id"] == "geak-e2e-0"
    assert attempt["source_kind"] == "geak_e2e"
    assert attempt["local_gain_pct"] == 4.0
    assert attempt["attributed_gain_pct"] == 5.0
    assert run["final_rebench_attempt_ids"] == ["geak-e2e-0"]
    assert warnings == []


def test_a_recorded_geak_measured_rejection_is_a_successful_rebench(tmp_path):
    """A GEAK candidate can measure cleanly and still lose to current_best."""
    from hyperloom.inference_optimizer.breakdown.collectors.geak import collect_geak
    from hyperloom.inference_optimizer.breakdown.recorder import record_geak_operation

    result = {
        "status": "ok",
        "baseline_throughput_tok_s": 100.0,
        "final_throughput_tok_s": 118.0,
        "throughput_speedup": 1.18,
        "report_path": "geak/report.json",
        "revalidation_status": "no_promote",
        "final_validation": {
            "decision": "REJECTED",
            "reason": "rebench_did_not_beat_current_best",
            "current_best_tput": 125.0,
        },
    }
    record_geak_operation(
        tmp_path,
        stage="final_validation_failed",
        result=result,
        status="failed",
        validated=False,
        measured_tput=118.0,
        validation_source="geak_promote_rejected",
        macro_cycle=0,
    )
    assembled, optimizations, warnings = _assembled_optimizations(tmp_path)
    state = _kernel_state() | {"kernel_optimizer": "geak", "geak_result": result}
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        geak=collect_geak(tmp_path, state, warnings),
        optimizations=optimizations,
        recorded_operations=assembled.get("operations") or [],
    )

    event = _event(timeline, "kernel")
    attempt = event["ext"]["attempts"][0]
    run = event["ext"]["geak_runs"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["decision"] == "REVERT"
    assert attempt["base_tput"] == 125.0
    assert attempt["output_throughput"] == 118.0
    assert attempt["local_gain_pct"] == pytest.approx(-5.6)
    assert attempt["attributed_gain_pct"] is None
    assert run["final_rebench_attempt_ids"] == [attempt["attempt_id"]]
    assert run["outcome"] == "REVERT"


def test_a_recorded_geak_failure_without_measurement_is_a_failed_rebench(tmp_path):
    """A failed launch/benchmark does not adjudicate the pending candidate."""
    from hyperloom.inference_optimizer.breakdown.collectors.geak import collect_geak
    from hyperloom.inference_optimizer.breakdown.recorder import record_geak_operation

    result = {
        "status": "ok",
        "baseline_throughput_tok_s": 100.0,
        "final_throughput_tok_s": 118.0,
        "throughput_speedup": 1.18,
        "report_path": "geak/report.json",
        "error": "server failed to launch for final validation",
    }
    record_geak_operation(
        tmp_path,
        stage="final_validation_failed",
        result=result,
        status="failed",
        validated=False,
        measured_tput=None,
        validation_source="geak_orch_harness",
        macro_cycle=0,
    )
    assembled, optimizations, warnings = _assembled_optimizations(tmp_path)
    state = _kernel_state() | {"kernel_optimizer": "geak", "geak_result": result}
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        geak=collect_geak(tmp_path, state, warnings),
        optimizations=optimizations,
        recorded_operations=assembled.get("operations") or [],
    )

    event = _event(timeline, "kernel")
    attempt = event["ext"]["attempts"][0]
    run = event["ext"]["geak_runs"][0]
    assert attempt["status"] == "failed"
    assert attempt["decision"] == "FAILED"
    assert attempt["is_fault"] is True
    assert attempt["output_throughput"] is None
    assert run["final_rebench_attempt_ids"] == [attempt["attempt_id"]]
    assert run["outcome"] == "NEEDS_REVIEW"


def test_a_failed_geak_final_step_outranks_a_stale_passed_gate(tmp_path):
    """Explicit terminal failure cannot be promoted by contradictory old evidence."""
    warnings: list[str] = []
    operation = {
        "operation_id": "geak-route-1",
        "kind": "kernel_optimizer_run",
        "name": "geak",
        "strategy": "geak",
        "agent": "kernel_agent",
        "phase": "KERNEL_AGENT",
        "macro_cycle": 0,
        "status": "failed",
        "substeps": [
            {
                "substep_id": "final-failed",
                "kind": "final_validation_failed",
                "status": "failed",
                "ended_at": "2026-08-27T01:30:00+00:00",
                "metadata": {"final_validation": False},
            }
        ],
        "gates": [
            {
                "gate_id": "stale-final-gate",
                "kind": "final_validation",
                "status": "passed",
                "evidence": {
                    "source": "geak_orch_harness",
                    "details": {"measured_tput": 98.0, "current_best_tput": 100.0},
                },
            }
        ],
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state() | {"kernel_optimizer": "geak"},
        geak=_GEAK_OK,
        recorded_operations=[operation],
    )

    event = _event(timeline, "kernel")
    attempt = event["ext"]["attempts"][0]
    run = event["ext"]["geak_runs"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["decision"] == "REVERT"
    assert attempt["output_throughput"] == 98.0
    assert run["outcome"] == "REVERT"


def test_a_recorded_geak_final_validation_places_the_run_on_its_kernel_visit(tmp_path):
    """The route cycle resolves the otherwise session-scoped GEAK ambiguity."""
    warnings: list[str] = []
    operation = {
        "operation_id": "geak-route-cycle-1",
        "kind": "kernel_optimizer_run",
        "name": "geak",
        "strategy": "geak",
        "agent": "kernel_agent",
        "phase": "KERNEL_AGENT",
        "macro_cycle": 1,
        "status": "succeeded",
        "substeps": [
            {
                "substep_id": "final-cycle-1",
                "kind": "final_validation",
                "status": "succeeded",
                "ended_at": "2026-08-27T02:30:00+00:00",
                "metadata": {"final_validation": True},
            }
        ],
        "gates": [
            {
                "gate_id": "final-gate-cycle-1",
                "kind": "final_validation",
                "status": "passed",
                "evidence": {"source": "geak_orch_harness", "measured_tput": 120.0},
            }
        ],
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state((0, 1)) | {"kernel_optimizer": "geak"},
        geak=_GEAK_OK,
        recorded_operations=[operation],
    )

    events = _events(timeline, "kernel")
    assert [len(event["ext"]["geak_runs"]) for event in events] == [0, 1]
    assert [len(event["ext"]["attempts"]) for event in events] == [0, 1]
    run = events[1]["ext"]["geak_runs"][0]
    attempt = events[1]["ext"]["attempts"][0]
    assert run["final_rebench_attempt_ids"] == [attempt["attempt_id"]]
    assert run["outcome"] == "KEEP"
    assert not any("session-scoped across" in warning for warning in warnings)


def test_a_recorded_measured_revert_is_a_successful_rebench(tmp_path):
    """``reverted`` is a business result, not a failed measurement."""
    from hyperloom.inference_optimizer.breakdown.recorder import record_kernel_e2e

    record_kernel_e2e(
        tmp_path,
        kernel_id="rmsnorm",
        integrated=False,
        e2e_gain_pct=-2.0,
        validated=False,
        decision="REVERT",
        patch_path="kernels/rmsnorm.patch",
        target_file="layers/rmsnorm.py",
        result={"base_tput": 100.0, "new_tput": 98.0, "decision_reason": "regressed"},
    )
    assembled, optimizations, warnings = _assembled_optimizations(tmp_path)
    assert optimizations["attempts"][0]["status"] == "reverted"

    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        kernel_journey=assembled.get("kernel_journey"),
        optimizations=optimizations,
        recorded_operations=assembled.get("operations") or [],
    )

    attempt = _event(timeline, "kernel")["ext"]["attempts"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["decision"] == "REVERT"
    assert attempt["is_fault"] is False
    assert attempt["base_tput"] == 100.0
    assert attempt["output_throughput"] == 98.0
    assert attempt["failure"] == {"error_class": None, "error": None}


def test_an_unmeasured_revert_is_a_failed_fault(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        optimizations={
            "attempts": [
                _kernel_attempt(
                    "att-unmeasured-revert",
                    0,
                    status="reverted",
                    decision="REVERT",
                    throughput_after=None,
                    decision_source="benchmark",
                    decision_reason="benchmark produced no usable throughput",
                )
            ]
        },
    )

    attempt = _event(timeline, "kernel")["ext"]["attempts"][0]
    assert attempt["status"] == "failed"
    assert attempt["decision"] == "REVERT"
    assert attempt["is_fault"] is True
    assert attempt["failure"] == {
        "error_class": "benchmark",
        "error": "benchmark produced no usable throughput",
    }


def test_a_geak_run_the_rebench_reverted_does_not_report_keep(tmp_path):
    """The runner's ``status`` is GEAK's claim, not the session's verdict.

    Reading ``ok`` as ``KEEP`` puts an adopted GEAK run next to the final
    rebench that rejected it — two contradictory records of the same event.
    """
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        geak=_GEAK_OK,
        optimizations={
            "attempts": [
                _kernel_attempt("geak-rb", 0, backend="geak", producer="geak", decision="REVERT", local_gain_pct=-4.5)
            ]
        },
    )

    run = _event(timeline, "kernel")["ext"]["geak_runs"][0]
    assert run["final_rebench_attempt_ids"] == ["geak-rb"]
    assert run["outcome"] == "REVERT"


def test_a_geak_run_the_rebench_kept_reports_keep(tmp_path):
    timeline = collect_v6_timeline(
        tmp_path,
        [],
        state=_kernel_state(),
        geak=_GEAK_OK,
        optimizations={"attempts": [_kernel_attempt("geak-rb", 0, backend="geak", producer="geak")]},
    )

    run = _event(timeline, "kernel")["ext"]["geak_runs"][0]
    assert run["final_rebench_attempt_ids"] == ["geak-rb"]
    assert run["outcome"] == "KEEP"


@pytest.mark.parametrize(
    ("revalidation_status", "outcome"),
    [("no_promote", "REVERT"), ("no_material", "REVERT"), ("fallback_failed", "FAILED")],
)
def test_a_closed_out_geak_candidate_reports_the_writeback_verdict(tmp_path, revalidation_status, outcome):
    """The verdict is stamped on ``geak_result``, not on the run record."""
    state = _kernel_state() | {"geak_result": {"status": "ok", "revalidation_status": revalidation_status}}
    timeline = collect_v6_timeline(tmp_path, [], state=state, geak=_GEAK_OK)

    assert _event(timeline, "kernel")["ext"]["geak_runs"][0]["outcome"] == outcome


def test_an_unadjudicated_geak_run_is_pending_rather_than_kept(tmp_path):
    """Nothing measured it. That is not a rejection, and not an adoption."""
    timeline = collect_v6_timeline(tmp_path, [], state=_kernel_state(), geak=_GEAK_OK)

    run = _event(timeline, "kernel")["ext"]["geak_runs"][0]
    assert run["final_rebench_attempt_ids"] == []
    assert run["outcome"] == "NEEDS_REVIEW"


@pytest.mark.parametrize("decisions", [("KEEP", "REVERT"), ("REVERT", "KEEP")])
def test_conflicting_geak_rebenches_remain_pending(tmp_path, decisions):
    """Contradictory final verdicts must not silently collapse to KEEP."""
    warnings: list[str] = []
    attempts = [
        _kernel_attempt(
            "geak-rb-1",
            0,
            backend="geak",
            producer="geak",
            decision=decisions[0],
            ended_at="2026-08-27T01:20:00+00:00",
        ),
        _kernel_attempt(
            "geak-rb-2",
            0,
            backend="geak",
            producer="geak",
            decision=decisions[1],
            ended_at="2026-08-27T01:30:00+00:00",
        ),
    ]
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        geak=_GEAK_OK,
        optimizations={"attempts": attempts},
    )

    run = _event(timeline, "kernel")["ext"]["geak_runs"][0]
    assert run["final_rebench_attempt_ids"] == ["geak-rb-1", "geak-rb-2"]
    assert run["outcome"] == "NEEDS_REVIEW"
    assert any("linked final rebenches disagree" in warning for warning in warnings)


def _fusion_state(**integrate) -> dict:
    return _kernel_state() | {
        "last_fusion": {
            "fusion_run_id": "fus-new",
            "status": "ok",
            "kept": True,
            "micro_decision": "candidate",
            "patch": "fusion/new.patch",
            "ts": "2026-08-27T01:20:00+00:00",
        },
        "last_fusion_integrate": integrate,
    }


def test_a_fusion_run_does_not_inherit_the_previous_rounds_integrate_verdict(tmp_path):
    """``last_fusion`` is rewritten every run; ``last_fusion_integrate`` is not.

    A fusion that is not kept never reaches integrate, and one missing its
    patch or target file returns before it, so the verdict left in state can
    belong to an earlier round.
    """
    warnings: list[str] = []
    state = _fusion_state(fusion_run_id="fus-old", decision="KEEP", reason="old run passed")
    timeline = collect_v6_timeline(tmp_path, warnings, state=state)

    run = _event(timeline, "kernel")["ext"]["fusion_runs"][0]
    assert run["run_id"] == "fus-new"
    assert run["outcome"] == "NEEDS_REVIEW"
    assert run["reason"] != "old run passed"
    assert any("does not name this fusion run" in warning for warning in warnings)


def test_a_fusion_run_reports_the_verdict_that_names_it(tmp_path):
    warnings: list[str] = []
    state = _fusion_state(fusion_run_id="fus-new", decision="KEEP", reason="gain above threshold")
    run = _event(collect_v6_timeline(tmp_path, warnings, state=state), "kernel")["ext"]["fusion_runs"][0]

    assert run["outcome"] == "KEEP"
    assert run["reason"] == "gain above threshold"
    assert warnings == []


def test_a_pre_id_fusion_pairs_on_the_patch_the_verdict_names(tmp_path):
    """Sessions recorded before ``fusion_run_id`` existed still pair."""
    state = _fusion_state(decision="KEEP", patch_path="fusion/new.patch")
    state["last_fusion"].pop("fusion_run_id")
    timeline = collect_v6_timeline(tmp_path, [], state=state)

    assert _event(timeline, "kernel")["ext"]["fusion_runs"][0]["outcome"] == "KEEP"


def test_a_micro_kept_fusion_with_no_integrate_verdict_is_not_a_final_keep(tmp_path):
    """KernelForge keeping a fusion is a microbenchmark result.

    Adoption is the e2e re-baseline's call, and it never ran here.
    """
    run = _event(collect_v6_timeline(tmp_path, [], state=_fusion_state()), "kernel")["ext"]["fusion_runs"][0]
    assert run["micro_decision"] == "candidate"
    assert run["outcome"] == "NEEDS_REVIEW"


def test_a_collective_campaign_pending_integration_is_not_a_final_keep(tmp_path):
    """The campaign row is written when the microbenchmark decides.

    Its integration fields are merged on later, so a KEPT campaign that never
    reached the gate — or crashed between the two writes — sits in state
    reading ``KEEP`` with no ``integration_decision``.
    """
    collective = {
        "attempts": [
            {
                "collective_attempt_id": "coll-1",
                "kernel_id": "all_reduce",
                "status": "ok",
                "decision": "KEEP",
                "kept": True,
                "kernel_speedup": 1.3,
                "ts": "2026-08-27T01:20:00+00:00",
            }
        ]
    }
    timeline = collect_v6_timeline(tmp_path, [], state=_kernel_state(), collective=collective)

    run = _event(timeline, "kernel")["ext"]["collective_runs"][0]
    assert run["micro_decision"] == "candidate"
    assert run["outcome"] == "NEEDS_REVIEW"

    collective["attempts"][0]["integration_decision"] = "REVERT"
    timeline = collect_v6_timeline(tmp_path, [], state=_kernel_state(), collective=collective)
    assert _event(timeline, "kernel")["ext"]["collective_runs"][0]["outcome"] == "REVERT"


def test_kernel_without_a_visit_or_evidence_produces_no_event(tmp_path):
    assert _events(collect_v6_timeline(tmp_path, [], state={}), "kernel") == []


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------
def _close_state(steps: list[dict], **overrides) -> dict:
    return {
        "phase": "CLOSE",
        "phase_history": [
            {
                "from_phase": "SWEEP",
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:00:00+00:00",
                "evidence": {"close_steps": steps},
            },
        ],
    } | overrides


def test_close_is_degraded_while_the_breakdown_predates_the_rest_of_the_sequence(tmp_path):
    """The step-2 snapshot describes the close-out only as far as itself.

    ``session_breakdown`` is step 2, so at the moment it runs the four steps
    after it do not exist yet and ``close_sequence_done`` is false. Reporting
    ``degraded`` for that is correct — the record really is incomplete. The
    sequencer's final ``patch_breakdown_close`` is what supersedes it; see
    ``test_close_patch_*``.
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


def test_close_succeeds_even_though_sequencer_started_never_leaves_running(tmp_path):
    """``sequencer_started`` is a marker, not a unit of work.

    The sequencer records it once on entry and never revisits it, so treating
    ``running`` as unsettled made ``succeeded`` unreachable no matter how
    cleanly the session closed.
    """
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "fact_finalize", "status": "done", "ts": "2026-08-27T05:00:10+00:00"},
            {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00"},
            {"step": "artifact_package", "status": "skipped", "ts": "2026-08-27T05:00:55+00:00"},
            {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )
    warnings: list[str] = []

    close = collect_v6_close(tmp_path, state, {}, warnings)

    assert close["status"] == "succeeded"
    # ``fact_finalize`` is emitted by the sequencer but was missing from the V6
    # field design's enum. It is a known step, not drift.
    assert "fact_finalize" in [step["step"] for step in close["steps"]]
    assert warnings == []


def test_close_still_waits_on_a_step_that_really_is_running(tmp_path):
    state = _close_state(
        [
            {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
            {"step": "report", "status": "running", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )

    assert collect_v6_close(tmp_path, state, {}, [])["status"] == "degraded"


def test_close_passes_through_an_unknown_step_and_warns(tmp_path):
    state = _close_state(
        [
            {"step": "teleport_to_s3", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
            {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
        ],
        close_sequence_done=True,
    )
    warnings: list[str] = []

    close = collect_v6_close(tmp_path, state, {}, warnings)

    # Dropping it would lose a step the producer really recorded.
    assert [step["step"] for step in close["steps"]] == ["teleport_to_s3", "done"]
    assert close["status"] == "succeeded"
    assert any("teleport_to_s3" in warning for warning in warnings)


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
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:00:00+00:00",
                "evidence": {"close_steps": [{"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"}]},
            },
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T05:01:00+00:00",
                "evidence": {"close_steps": [{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}]},
            },
        ],
        "close_sequence_done": True,
    }

    close = collect_v6_close(tmp_path, state, {}, [])
    assert [step["step"] for step in close["steps"]] == ["report", "done"]
    assert close["status"] == "succeeded"


def test_close_surfaces_robustness_escalation_and_its_signals(tmp_path):
    state = _close_state([{"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"}])
    state["stop_reason"] = "robustness_escalated"
    signals = [
        {"ts": "2026-08-27T04:00:00+00:00", "signal": "crash", "action": "restart", "workdir": "robustness-workdir/0"}
    ]

    close = collect_v6_close(tmp_path, state, {"robustness_signals": signals}, [])

    assert close["robustness"]["escalated"] is True
    assert close["robustness"]["signals"] == signals


def test_close_artifacts_point_only_at_files_that_exist(tmp_path):
    _write_json(tmp_path / "reports" / "final.json", {"ok": True})
    state = _close_state(
        [
            {
                "step": "artifact_package",
                "status": "done",
                "ts": "2026-08-27T05:02:00+00:00",
                "detail": str(tmp_path / "bundle.zip"),
            }
        ]
    )

    artifacts = collect_v6_close(tmp_path, state, {}, [])["artifacts"]

    assert artifacts["final_json_path"] == "reports/final.json"
    assert artifacts["final_md_path"] is None
    assert artifacts["session_breakdown_path"] == "session_breakdown.json"
    assert artifacts["artifact_package_path"] == "bundle.zip"


def test_close_ignores_a_skipped_artifact_package_detail(tmp_path):
    """``detail`` doubles as the skip reason; only a ``done`` row holds a path."""
    state = _close_state(
        [
            {
                "step": "artifact_package",
                "status": "skipped",
                "ts": "2026-08-27T05:02:00+00:00",
                "detail": "no artifacts matched or dest unwritable",
            }
        ]
    )

    assert collect_v6_close(tmp_path, state, {}, [])["artifacts"]["artifact_package_path"] is None


# ---------------------------------------------------------------------------
# close: the end-of-sequence refresh
# ---------------------------------------------------------------------------
_FULL_CLOSE_STEPS = [
    {"step": "sequencer_started", "status": "running", "ts": "2026-08-27T05:00:01+00:00"},
    {"step": "fact_finalize", "status": "done", "ts": "2026-08-27T05:00:05+00:00"},
    {"step": "report", "status": "done", "ts": "2026-08-27T05:00:30+00:00"},
    {"step": "session_breakdown", "status": "done", "ts": "2026-08-27T05:00:45+00:00"},
    {"step": "artifact_package", "status": "done", "ts": "2026-08-27T05:00:55+00:00", "detail": "/workspace/s.zip"},
    {"step": "ndjson_drain", "status": "skipped", "ts": "2026-08-27T05:01:00+00:00"},
    {"step": "done", "status": "done", "ts": "2026-08-27T05:01:05+00:00"},
]


def _session_with_step_two_breakdown(tmp_path: Path) -> Path:
    """Build a session whose breakdown was written mid-CLOSE, as step 2 does."""
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS[:4]))
    exporter.write_breakdown_json(tmp_path)
    # The sequencer then finishes, persisting the remaining steps.
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))
    return tmp_path / exporter.BREAKDOWN_FILENAME


def test_close_patch_replaces_the_step_two_snapshot_with_the_finished_sequence(tmp_path):
    target = _session_with_step_two_breakdown(tmp_path)
    before = json.loads(target.read_text(encoding="utf-8"))
    assert before["close"]["status"] == "degraded"
    assert [step["step"] for step in before["close"]["steps"]] == [
        "sequencer_started",
        "fact_finalize",
        "report",
        "session_breakdown",
    ]

    assert exporter.patch_breakdown_close(tmp_path) is True

    after = json.loads(target.read_text(encoding="utf-8"))
    assert after["close"]["status"] == "succeeded"
    assert after["close"]["close_sequence_done"] is True
    assert [step["step"] for step in after["close"]["steps"]] == [step["step"] for step in _FULL_CLOSE_STEPS]
    assert after["close"]["end_time"] == "2026-08-27T05:01:05+00:00"


def test_close_patch_touches_nothing_but_the_close_key(tmp_path):
    """The whole point of a patch over a rebuild: every other key is frozen."""
    target = _session_with_step_two_breakdown(tmp_path)
    before = json.loads(target.read_text(encoding="utf-8"))

    exporter.patch_breakdown_close(tmp_path)

    after = json.loads(target.read_text(encoding="utf-8"))
    assert set(after) == set(before)
    assert {key: value for key, value in after.items() if key != "close"} == {
        key: value for key, value in before.items() if key != "close"
    }


def test_close_patch_is_idempotent(tmp_path):
    tmp_path_target = _session_with_step_two_breakdown(tmp_path)
    assert exporter.patch_breakdown_close(tmp_path) is True
    # Nothing changed the second time, so nothing is rewritten.
    assert exporter.patch_breakdown_close(tmp_path) is False
    assert json.loads(tmp_path_target.read_text(encoding="utf-8"))["close"]["status"] == "succeeded"


def test_close_patch_is_a_no_op_without_a_breakdown(tmp_path):
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))

    assert exporter.patch_breakdown_close(tmp_path) is False


def test_close_patch_leaves_a_payload_that_never_carried_close_alone(tmp_path):
    """A V5-only breakdown has no ``close`` key, and gaining one is a surface change."""
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    _write_json(target, {"schema_version": "hyperloom.session_breakdown.v5.0", "baseline": {}})
    _write_json(tmp_path / "state.json", _close_state(_FULL_CLOSE_STEPS, close_sequence_done=True))

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert "close" not in json.loads(target.read_text(encoding="utf-8"))


def test_close_patch_swallows_a_corrupt_breakdown(tmp_path):
    """It runs at shutdown and must never mask the session's stop_reason."""
    target = tmp_path / exporter.BREAKDOWN_FILENAME
    target.write_text("{not json", encoding="utf-8")

    assert exporter.patch_breakdown_close(tmp_path) is False
    assert target.read_text(encoding="utf-8") == "{not json"


# ---------------------------------------------------------------------------
# what the consumer actually receives
# ---------------------------------------------------------------------------
def _packaged_close(session_dir: Path, dest_root: Path) -> tuple[dict, dict]:
    """Return the ``close`` key as delivered, from inside the zip and loose.

    External sync ships the package, not the session directory, so these two
    copies — not the one under ``session_dir`` — are what a consumer reads.
    """
    zip_path = dest_root / session_package.PACKAGE_SUBDIR / "sess-1.zip"
    with zipfile.ZipFile(zip_path) as bundle:
        zipped = json.loads(bundle.read(exporter.BREAKDOWN_FILENAME))
    loose = json.loads((dest_root / exporter.BREAKDOWN_FILENAME).read_text(encoding="utf-8"))
    return zipped["close"], loose["close"]


def test_the_delivered_package_carries_the_finished_close_section(tmp_path):
    """Patching the session copy is not delivery; the package has to be rebuilt.

    Mirrors the sequencer's order: package (CLOSE step 5), then patch the close
    section, then rebuild the bundle so the copies that ship agree with it.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    assert exporter.patch_breakdown_close(session_dir) is True
    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)

    zipped, loose = _packaged_close(session_dir, dest_root)
    for delivered in (zipped, loose):
        assert delivered["status"] == "succeeded"
        assert delivered["close_sequence_done"] is True
        assert [step["step"] for step in delivered["steps"]] == [step["step"] for step in _FULL_CLOSE_STEPS]


def test_a_package_built_before_the_patch_ships_the_step_two_snapshot(tmp_path):
    """The regression this guards: the fix reaching the session dir only.

    Without the rebuild the session copy reads ``succeeded`` while both
    delivered copies still say ``degraded`` and stop four steps in — the state
    that made the previous round's fix invisible to its consumers.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    target = _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    exporter.patch_breakdown_close(session_dir)

    assert json.loads(target.read_text(encoding="utf-8"))["close"]["status"] == "succeeded"
    zipped, loose = _packaged_close(session_dir, dest_root)
    for stale in (zipped, loose):
        assert stale["status"] == "degraded"
        assert "artifact_package" not in {step["step"] for step in stale["steps"]}


def test_the_delivered_manifest_describes_the_rebuilt_bundle(tmp_path):
    """A surgical member swap would leave the manifest describing the old file.

    Hence a full repackage: the manifest is rebuilt from the members that were
    actually written, so its digest of ``session_breakdown.json`` matches what
    the consumer unzips.
    """
    session_dir = tmp_path / "session"
    dest_root = tmp_path / "dest"
    _session_with_step_two_breakdown(session_dir)

    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)
    exporter.patch_breakdown_close(session_dir)
    session_package.package_session_artifacts(session_dir, session_id="sess-1", dest_root=dest_root)

    zip_path = dest_root / session_package.PACKAGE_SUBDIR / "sess-1.zip"
    with zipfile.ZipFile(zip_path) as bundle:
        manifest = json.loads(bundle.read(session_package.MANIFEST_JSON_NAME))
        member = bundle.getinfo(exporter.BREAKDOWN_FILENAME)
    entry = next(row for row in manifest["included_files"] if row["path"] == exporter.BREAKDOWN_FILENAME)
    assert entry["bytes"] == member.file_size


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------
def _v6_outcome(optimizations: dict) -> dict:
    return v6_collectors.collect_v6_outcome(
        session={"stop_reason": "target_reached"},
        baseline={},
        final={},
        optimizations=optimizations,
        state={"phase": "CLOSE"},
        timeline=[],
    )


def test_outcome_projects_authoritative_gain_by_v6_source_and_kernel_backend():
    outcome = _v6_outcome(
        {
            "available": True,
            "summary_by_source": {
                "warm_replay": {"keeps": 1, "total_gain_pct": 1.25},
                "explore": {"keeps": 2, "total_gain_pct": 2.0},
                "framework_agent": {"keeps": 1, "total_gain_pct": 0.75},
                "kernel_agent": {
                    "keeps": 4,
                    "total_gain_pct": 5.5,
                    "by_backend": {
                        "geak": {"keeps": 2, "total_gain_pct": 4.25, "non_attributable_keeps": 1},
                        "forge": {"keeps": 1, "total_gain_pct": 1.25, "non_attributable_keeps": 0},
                    },
                },
            },
            "validation": {
                "attributed_total_gain_pct": 9.5,
                "unattributed_gain_pct": 0.5,
                "reconciliation_gap_pct": 0.5,
            },
        }
    )

    attribution = outcome["validation"]["attribution"]
    assert attribution == {
        "available": True,
        "by_source": {
            "warm_replay": {"total_gain_pct": 1.25, "keep_count": 1},
            "framework_agent": {"total_gain_pct": 2.75, "keep_count": 3},
            "kernel": {
                "total_gain_pct": 5.5,
                "keep_count": 4,
                "by_backend": {
                    "geak": {
                        "total_gain_pct": 4.25,
                        "keep_count": 2,
                        "non_attributable_keep_count": 1,
                    },
                    "forge": {
                        "total_gain_pct": 1.25,
                        "keep_count": 1,
                        "non_attributable_keep_count": 0,
                    },
                },
            },
        },
    }


def test_outcome_marks_gain_totals_unknown_when_the_canonical_ledger_is_unavailable():
    attribution = _v6_outcome({"available": False})["validation"]["attribution"]

    assert attribution["available"] is False
    assert attribution["by_source"]["warm_replay"]["total_gain_pct"] is None
    assert attribution["by_source"]["framework_agent"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["by_backend"]["geak"]["total_gain_pct"] is None
    assert attribution["by_source"]["kernel"]["by_backend"]["forge"]["total_gain_pct"] is None


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
    ["project_baseline_event", "project_conc_sweep_event", "project_kernel_events"],
)
def test_a_raising_stage_projector_costs_only_its_own_stage(tmp_path, monkeypatch, projector):
    """One stage blowing up must not take the durable events or its peers down.

    The exporter wraps the whole timeline collector, so without per-projector
    isolation a kernel-stage bug discards the ``install`` event a session read
    off disk before the Coordinator existed -- the one record a run that never
    reached KERNEL actually has.
    """
    _write_json(
        tmp_path / "state.json",
        {"session_id": "s1", "model_name": "M", "framework": "sglang", "baseline_tput": 100.0, "phase": "CLOSE"},
    )
    _write_json(tmp_path / "manifest.json", {"session_id": "s1", "model_name": "M", "framework": "sglang"})
    write_timeline_event(
        tmp_path,
        {"type": "install", "kind": "install", "status": "succeeded", "start_time": "", "end_time": ""},
    )
    before = exporter.build(tmp_path)
    stage = {
        "project_baseline_event": "baseline",
        "project_conc_sweep_event": "conc_sweep",
        "project_kernel_events": "kernel",
    }[projector]
    assert "install" in {event["type"] for event in before["timeline"]}

    def _boom(*args, **kwargs):
        raise RuntimeError(f"{projector} exploded")

    monkeypatch.setattr(v6_collectors, projector, _boom)
    after = exporter.build(tmp_path)

    v6_keys = {"exported_at_utc", "metadata", "outcome", "timeline", "close"}
    assert {key: value for key, value in after.items() if key not in v6_keys} == {
        key: value for key, value in before.items() if key not in v6_keys
    }
    assert after["warnings"] == before["warnings"]

    types_after = [event["type"] for event in after["timeline"]]
    # The durable event survives, and so does every stage that projected.
    assert "install" in types_after
    assert stage not in types_after
    assert types_after == [event["type"] for event in before["timeline"] if event["type"] != stage]
    assert any(f"v6.timeline.{stage}" in warning for warning in after["metadata"]["warnings"])


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


# ---------------------------------------------------------------------------
# fabrication, settlement, identity and vocabulary
# ---------------------------------------------------------------------------
def _events(timeline: list[dict], event_type: str) -> list[dict]:
    return [event for event in timeline if event["type"] == event_type]


_GEAK_ECHO = {"engaged": True, "status": "missing", "error_class": "no_result", "accepted_kernels": []}


def test_geak_config_echo_does_not_fabricate_a_kernel_visit(tmp_path):
    """``kernel_optimizer=geak`` on its own is a launch flag, not a Kernel visit.

    GEAK is the default backend, so ``collect_geak`` answers ``status:
    missing`` for every session that selected it and ended before KERNEL --
    baseline failures and enablement stalls included. Counting that as evidence
    invented a degraded ``kernel`` event, carrying a FAILED ``geak_run``, for
    runs that never dispatched a kernel at all.
    """
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state={"phase": "PRELUDE", "macro_cycle": 0, "phase_history": [], "kernel_optimizer": "geak"},
        geak=_GEAK_ECHO,
    )

    assert _events(timeline, "kernel") == []
    assert not any("kernel evidence exists" in warning for warning in warnings)


def test_the_geak_echo_is_still_reported_inside_a_real_visit(tmp_path):
    """The echo is ignored as *evidence*; a visit that happened still shows it."""
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state() | {"kernel_optimizer": "geak"},
        geak=_GEAK_ECHO,
    )

    run = _events(timeline, "kernel")[0]["ext"]["geak_runs"][0]
    # One normalizer feeds both fields, so they cannot contradict each other.
    assert (run["status"], run["outcome"]) == ("failed", "FAILED")


def _geak_run_row(tmp_path, geak_status: str) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state() | {"kernel_optimizer": "geak"},
        geak={"engaged": True, "status": geak_status, "accepted_kernels": []},
    )
    return _events(timeline, "kernel")[0]["ext"]["geak_runs"][0], warnings


@pytest.mark.parametrize("geak_status", sorted(GEAK_TERMINAL_STATUSES))
def test_no_terminal_geak_status_reaches_the_normalizer_as_drift(tmp_path, geak_status):
    """Every status the runtime declares terminal must already be known here.

    Read off the producer's own set rather than restated, so a status added to
    ``GEAK_TERMINAL_STATUSES`` and to nothing else fails here instead of
    surfacing months later as a drift warning on a run that worked. That is how
    ``no_gain`` and ``baseline_reproduction_failed`` were both missed: the first
    vocabulary was assembled from what production had already written, which
    cannot contain a status nothing has hit yet.
    """
    run, warnings = _geak_run_row(tmp_path, geak_status)

    assert run["status"] in {"succeeded", "failed", "skipped"}
    assert not [w for w in warnings if "unrecognized geak_runs status" in w]


@pytest.mark.parametrize(
    ("geak_status", "expected_status", "expected_outcome"),
    [
        ("ok", "succeeded", "NEEDS_REVIEW"),
        # A completed run that found nothing, and the third most common status
        # in production -- 212 sessions against 115 ``ok``. Reported as a GEAK
        # fault until the alias existed.
        ("no_gain", "succeeded", "NEEDS_REVIEW"),
        # The runner's baseline reference did not reproduce the orchestrator's,
        # so the run's gain is non-comparable and ``phases/kernel.py`` refuses
        # to promote it. A real failure, and one the runtime names.
        ("baseline_reproduction_failed", "failed", "FAILED"),
        ("failed", "failed", "FAILED"),
        ("error", "failed", "FAILED"),
        ("missing", "failed", "FAILED"),
        ("no_result_recovered_from_disk", "failed", "FAILED"),
        # Reaches the field from the runner without being terminal: 95 sessions.
        ("timeout", "failed", "FAILED"),
        ("skipped", "skipped", "SKIPPED"),
    ],
)
def test_every_geak_status_maps_to_its_own_classification(tmp_path, geak_status, expected_status, expected_outcome):
    """The classification each status lands on, not merely that it is known.

    Union of two sources, because neither is complete on its own:
    ``GEAK_TERMINAL_STATUSES`` is what the runtime declares, and the corpus of
    2404 sessions carrying a geak block is what it has actually written --
    ``missing`` (1503), ``timeout`` (95) and the collector's disk-recovery word
    appear only in the second.
    """
    run, _ = _geak_run_row(tmp_path, geak_status)

    assert (run["status"], run["outcome"]) == (expected_status, expected_outcome)


def test_the_pinned_vocabulary_covers_the_producer_set(tmp_path):
    """The parametrization above must not fall behind the producer's set."""
    pinned = {
        "ok",
        "no_gain",
        "baseline_reproduction_failed",
        "failed",
        "error",
        "missing",
        "no_result_recovered_from_disk",
        "timeout",
        "skipped",
    }
    assert GEAK_TERMINAL_STATUSES <= pinned


def test_a_skipped_geak_spelling_does_not_contradict_itself(tmp_path):
    """``not_run`` used to emit ``status: failed`` beside ``outcome: SKIPPED``."""
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state() | {"kernel_optimizer": "geak"},
        geak={"engaged": True, "status": "not_run", "accepted_kernels": []},
    )

    run = _events(timeline, "kernel")[0]["ext"]["geak_runs"][0]
    assert (run["status"], run["outcome"]) == ("skipped", "SKIPPED")


_GEMM_RUN = {
    "engine": "forge",
    "status": "succeeded",
    "tuned_file": "runs/gemm/aiter.csv",
    "best_speedup": 1.4,
    "ts": "2026-08-27T01:20:00+00:00",
}


def test_a_gemm_candidate_is_settled_by_the_rebench_that_kept_it(tmp_path):
    """A tuned table nothing adjudicated is pending, not skipped.

    ``SKIPPED`` is the one outcome ``_settle_pending_outcomes`` will not
    revisit, so a GEMM run the final rebench went on to KEEP used to sit on the
    timeline as work that never happened.
    """
    warnings: list[str] = []
    attempt = _kernel_attempt(
        "gemm-rebench-0",
        0,
        kind="gemm_tuning",
        name="runs/gemm/aiter.csv",
        kernel_id="",
        backend="forge",
    )
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        optimizations={"gemm_tuning_runs": [_GEMM_RUN], "attempts": [attempt]},
    )

    lane = _events(timeline, "kernel")[0]["ext"]["gemm_tuning_runs"][0]
    assert lane["micro_decision"] == "candidate"
    assert lane["final_rebench_attempt_ids"] == ["gemm-rebench-0"]
    assert lane["outcome"] == "KEEP"


def test_a_gemm_run_with_no_rebench_stays_pending(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        optimizations={"gemm_tuning_runs": [_GEMM_RUN]},
    )

    assert _events(timeline, "kernel")[0]["ext"]["gemm_tuning_runs"][0]["outcome"] == "NEEDS_REVIEW"


def test_unidentified_backend_attempts_do_not_share_one_rewrite_id(tmp_path):
    """Two backends on one kernel must stay two rows.

    The fallback used to be the ``kernel_id`` every backend attempt on a kernel
    carries, so both rows took one id, both passed the adopted-set test, and
    the losing backend claimed the winner's final rebench.
    """
    warnings: list[str] = []
    journey = {
        "kernels": [
            {
                "kernel_id": "rmsnorm",
                "name": "rmsnorm",
                "e2e": {"decision": "KEEP", "ts": "2026-08-27T01:30:00+00:00"},
                "backend_attempts": [
                    {"backend": "geak", "status": "succeeded", "micro_speedup": 1.1},
                    {
                        "backend": "forge",
                        "status": "succeeded",
                        "micro_speedup": 1.9,
                        "best_artifact_path": "runs/k/forge.py",
                    },
                ],
            }
        ]
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        kernel_journey=journey,
        optimizations={"attempts": [_kernel_attempt("rebench-0", 0, kernel_id="rmsnorm")]},
    )

    rewrites = _events(timeline, "kernel")[0]["ext"]["kernel_rewrites"]
    assert len({row["rewrite_id"] for row in rewrites}) == 2
    # Only the adopted backend answers for the kernel's rebench.
    assert {row["backend"]: row["final_rebench_attempt_ids"] for row in rewrites} == {
        "forge": ["rebench-0"],
        "geak": [],
    }


def test_a_rebench_two_candidates_answer_to_is_left_unlinked(tmp_path):
    """Ambiguity is ambiguity whether the attempt matched none or several."""
    warnings: list[str] = []
    collective = {
        "attempts": [
            {
                "collective_attempt_id": "c1",
                "kernel_id": "all_reduce",
                "status": "succeeded",
                "kept": True,
                "ts": "2026-08-27T01:10:00+00:00",
            },
            {
                "collective_attempt_id": "c2",
                "kernel_id": "all_reduce",
                "status": "succeeded",
                "kept": True,
                "ts": "2026-08-27T01:20:00+00:00",
            },
        ]
    }
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=_kernel_state(),
        collective=collective,
        optimizations={
            "attempts": [_kernel_attempt("coll-rebench-0", 0, kind="kernel_collective", kernel_id="all_reduce")]
        },
    )

    runs = _events(timeline, "kernel")[0]["ext"]["collective_runs"]
    assert [row["final_rebench_attempt_ids"] for row in runs] == [[], []]
    assert any("more than one recorded candidate" in warning for warning in warnings)


def test_a_kept_visit_that_then_errored_is_degraded_not_failed(tmp_path):
    state = _kernel_state()
    state["phase_history"][-1]["evidence"] = {"error_class": "TimeoutError", "error": "sweep handoff timed out"}
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        state=state,
        optimizations={"attempts": [_kernel_attempt("rebench-0", 0)]},
    )

    event = _events(timeline, "kernel")[0]
    assert event["status"] == "degraded"
    # The error is not lost; it is just not the whole verdict.
    assert event["ext"]["failure"]["error_class"] == "TimeoutError"


def test_a_malformed_conc_grid_does_not_raise_out_of_the_projector(tmp_path):
    warnings: list[str] = []
    timeline = collect_v6_timeline(
        tmp_path,
        warnings,
        conc_sweep_summary={"status": "ok", "concs_requested": 64},
    )

    assert _events(timeline, "conc_sweep")[0]["ext"]["plan"]["concs_requested"] == [64]


def test_an_unknown_close_step_status_is_reported(tmp_path):
    warnings: list[str] = []
    state = {
        "close_sequence_done": True,
        "phase_history": [
            {
                "to_phase": "CLOSE",
                "ts": "2026-08-27T02:00:00+00:00",
                "evidence": {
                    "close_steps": [{"step": "report", "status": "completed", "ts": "2026-08-27T02:00:01+00:00"}]
                },
            }
        ],
    }
    section = collect_v6_close(tmp_path, state, {}, warnings)

    # Passed through unchanged -- inventing ``done`` is the one thing this key
    # cannot afford -- but no longer silent about it.
    assert section["steps"][0]["status"] == "completed"
    assert any("unrecognized close step status" in warning for warning in warnings)
