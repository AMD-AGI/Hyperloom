# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``roofline`` action, standalone and inline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.event_sink import make_sink
from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import (
    ANALYSIS_ATTEMPT_INITIAL,
    ANALYSIS_ATTEMPT_N26_RETRY,
    PRODUCER,
    PROFILE_ATTEMPT_AFTER_ZERO_OPS,
    PROFILE_ATTEMPT_INITIAL,
    _summarize_trace_files,
    make_roofline_recorder,
    roofline_event_id,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.actions.executors.profile import (
    CHECK_GRAPH_LAUNCH_COVERAGE,
    CHECK_RANK_SHAPE,
    CHECK_TRACE_HAS_OPS,
    _build_trace_validate,
)
from hyperloom.orchestrator.kernel.request_handlers import _analysis_steady_state


@pytest.fixture(autouse=True)
def _bound_session(tmp_path):
    """Bind the session the way startup does, so no call below takes a path."""
    with session_scope(tmp_path):
        yield tmp_path


def _roofline_events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "roofline"]


def _recorder(
    *,
    task_id: str = "t-1",
    task_kind: str = "",
    reason: str = "",
    framework: str = "sglang",
    phase: str = "prelude",
    macro_cycle: int = 0,
):
    """Build a standalone recorder, the way a dispatched roofline gets one."""
    recorder = make_roofline_recorder(
        make_sink(roofline_event_id(phase, macro_cycle), producer=PRODUCER),
        task_id=task_id,
        task_kind=task_kind,
        reason=reason,
        framework=framework,
    )
    assert recorder is not None
    return recorder


def _actions(session_dir: Path, index: int = 0) -> list[dict[str, Any]]:
    return _roofline_events(session_dir)[index]["ext"]["actions"]


def _profile_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "succeeded",
        "framework": "sglang",
        "workspace": "/w",
        "trace_dir": "/w/traces",
        "main_trace_path": "/w/traces/merged-a.pt.trace.json.gz",
        "trace_files": ["/w/traces/merged-a.pt.trace.json.gz"],
        "profile_trace_selection_reason": "merged_trace_preferred",
        "trace_health": {"issues": [], "zero_ops": False, "checks": []},
    }
    result.update(overrides)
    return result


def _ta_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok",
        "run_id": "tl-1",
        "orchestrator_mode": "claude_agent_sdk",
        "hot_kernels": [{"name": "gemm", "gpu_time_us": 10.0, "gpu_pct": 50.0, "count": 2}],
        "trace_health_warnings": [],
        "analysis_meta": {
            "route": "agent",
            "tool": "tracelens",
            "steady_state": {"requested_mode": "decode_only", "source": "split_chunk"},
            "preflight": {"trace_input_type": "file", "trace_file_count": 1},
            "split": {"chunks_extracted": 3},
            "selection": {"selected_chunk": "decode_only_steady_state_0.json.gz"},
            "steps": [{"step_id": "discover_inputs", "order": 1, "status": "ok"}],
            "route_ext": {},
        },
    }
    result.update(overrides)
    return result


def test_begin_puts_the_event_on_the_timeline(tmp_path: Path) -> None:
    recorder = _recorder(reason="prelude_initial")
    recorder.begin(max_profile_attempts=3)

    events = _roofline_events(tmp_path)
    assert len(events) == 1
    assert events[0]["status"] == "running"
    assert events[0]["id"] == roofline_event_id("prelude", 0)
    assert "end_time" not in events[0]
    assert events[0]["ext"]["in_flight_substep"] == "profile"
    assert recorder.event_id == roofline_event_id("prelude", 0)


def test_the_action_carries_its_own_request_and_budget(tmp_path: Path) -> None:
    recorder = _recorder(reason="prelude_initial")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="never booted")

    action = _actions(tmp_path)[0]
    assert action["request"]["arm"] == "baseline"
    assert action["profile"]["max_attempts"] == 3


def test_preflight_conditions_survive_onto_the_action(tmp_path: Path) -> None:
    """What the action found before profiling is carried whether or not the run went on to succeed."""
    recorder = _recorder(reason="prelude_initial")
    recorder.begin(max_profile_attempts=3)
    recorder.record_preflight(
        {
            "orphans_reaped": [1234],
            "orphans_reaped_count": 1,
            "disk": {"free_bytes": 1024, "free_pct": 3.5},
            "stale_traces": [{"path": "runs/roofline/t-1/old.pt.trace.json.gz"}],
            "stale_trace_count": 1,
        }
    )
    recorder.finish_failed(phase="profile", message="never booted")

    preflight = _actions(tmp_path)[0]["preflight"]
    assert preflight["orphans_reaped_count"] == 1
    assert preflight["disk"]["free_pct"] == 3.5
    assert preflight["stale_trace_count"] == 1


def test_a_run_that_succeeded_still_reports_its_engine_dying(tmp_path: Path) -> None:
    """The process outcome rides on the attempt row, because the result dict cannot show it."""
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="succeeded",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=30.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(),
        server_liveness={"pidfiles": 1, "alive": 0, "dead_with_pidfile": 1},
    )
    recorder.finish_failed(phase="trace_analyze", message="stopped")

    run = _actions(tmp_path)[0]["profile"]["runs"][0]
    assert run["status"] == "succeeded"
    assert run["server_liveness"]["dead_with_pidfile"] == 1


def test_shape_capture_dispatch_does_not_claim_a_measured_arm(tmp_path: Path) -> None:
    recorder = _recorder(task_id="t-capture-1", task_kind="gemm_shape_capture", framework="vllm")
    recorder.begin(max_profile_attempts=1)
    recorder.finish_failed(phase="profile", message="stopped")

    request = _actions(tmp_path)[0]["request"]
    assert request["task_kind"] == "gemm_shape_capture"
    assert request["arm"] == ""


def test_roofline_dispatch_without_a_reason_still_names_its_arm(tmp_path: Path) -> None:
    recorder = _recorder(task_id="t-2", task_kind="roofline")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="stopped")

    request = _actions(tmp_path)[0]["request"]
    assert request["task_kind"] == "roofline"
    assert request["arm"] == "current_best"


def _validate(**overrides: Any) -> dict[str, Any]:
    """A certificate shaped like the one selfcert produces, with the tables populated."""
    validate: dict[str, Any] = {
        "schema_version": 1,
        "probe_version": "selfcert-1.0.0",
        "probe_status": "ok",
        "verdict": {
            "usable_by": ["bypass", "tracelens"],
            "decode_conclusions_valid": True,
            "silently_wrong": False,
            "severity": "ok",
            "blocking_reasons": [],
            "warnings": [],
            "recommended_splitter_mode": "decode_only",
            "measures": {"attributed_pct": 91.0, "graph_launch_coverage": 0.9},
            "thresholds_effective": {"graph_launch_coverage_max": 0.2},
        },
        "trace_dir_level": {
            "selected_role": "source",
            "production_selected_role": "source",
            "production_would_analyze_split_chunk": False,
            "file_count_by_role": {"source": 1, "capture_sidecar": 3, "split_chunk": 0},
            "candidates": [{"path": f"/w/t/{i}.json", "bytes": 10, "role": "source"} for i in range(40)],
            "capture_sidecar_probe": {
                "files_present": 3,
                "files_scanned": 3,
                "truncated_scan": False,
                "op_meta_coverage": 0.75,
                "cpu_op_total": 40,
                "kernel_count": 9,
                "files": [{"path": f"/w/t/capture_traces/bs_{i}.json"} for i in range(3)],
            },
        },
        "rank_level": [
            {
                "rank": 0,
                "parse": {
                    "event_total": 5000,
                    "aggregation_scope": "full_trace",
                    "truncated": False,
                    "truncation_reason": "",
                },
                "attribution": {
                    "attributed_pct": 91.0,
                    "attributed_gpu_ms": 45.5,
                    "gpu_kernel_sum_ms": 50.0,
                    "attributed_kernels": 91,
                    "unlinked_kernels": 9,
                    "graph_attributed_kernels": 80,
                    "cuda_runtime_links": 100,
                    "op_meta_coverage": 0.8,
                    "op_meta_basis": {"cpu_op_total": 100, "cpu_op_with_meta": 80},
                },
                "density": {
                    "kernel_count": 100,
                    "graph_mode": True,
                    "graph_launch_count": 20,
                    "graph_launch_coverage": 0.9,
                    "graph_under_recorded": False,
                    "graph_under_recorded_threshold": 0.2,
                    "busy_fraction": 0.7,
                    "kernel_per_launch": 5.0,
                },
                "time_structure": {"idle_pct_full_trace": 30.0},
                "annotations": {"step_root_count": 12},
            }
        ],
        "chunk_level": [{"path": "/w/t/chunk0.json"}, {"path": "/w/t/chunk1.json"}],
        "checks": [],
    }
    validate.update(overrides)
    return validate


def test_selfcert_scalars_reach_the_event_and_the_tables_do_not(tmp_path: Path) -> None:
    """The numbers a reader needs come inline; the per-rank and per-file tables stay behind a path."""
    recorder = _recorder()
    recorder.begin(max_profile_attempts=1)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="succeeded",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=30.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(
            trace_validate=_validate(),
            trace_validate_path="/w/traces/selfcert.json",
        ),
    )
    recorder.finish_failed(phase="analysis", message="stop")

    validate = _actions(tmp_path)[0]["profile"]["runs"][0]["validate"]
    quality = validate["trace_quality"]

    assert validate["certificate_path"] == "/w/traces/selfcert.json"
    assert validate["severity"] == "ok"
    assert validate["schema_version"] == 1
    # Attribution with the terms of its ratio, so 91% of 50ms cannot be mistaken for 91% of an hour.
    assert quality["attributed_pct"] == 91.0
    assert quality["attributed_gpu_ms"] == 45.5
    assert quality["gpu_kernel_sum_ms"] == 50.0
    assert quality["op_meta_basis"] == {"cpu_op_total": 100, "cpu_op_with_meta": 80}
    # The scope the ratios were computed over.
    assert quality["aggregation_scope"] == "full_trace"
    assert quality["truncated"] is False
    # The boolean and the threshold it was compared against.
    assert quality["graph_under_recorded"] is False
    assert quality["graph_under_recorded_threshold"] == 0.2
    # Capture sidecars, previously never measured anywhere.
    assert quality["capture_op_meta_coverage"] == 0.75
    assert quality["capture_sidecar_files_present"] == 3
    assert quality["analyzed_rank"] == 0
    assert quality["rank_count_certified"] == 1
    assert quality["chunk_count_certified"] == 2
    assert quality["file_count_by_role"]["capture_sidecar"] == 3

    # The unbounded tables must not have been copied in.
    flat = json.dumps(validate)
    assert "candidates" not in flat
    assert "rank_level" not in flat
    assert "chunk_level" not in flat
    assert "capture_traces/bs_" not in flat


def test_an_attempt_that_produced_no_trace_still_carries_its_patch_state(tmp_path: Path) -> None:
    """Patch facts cannot live in the certificate: the attempts whose patching is in question produce none."""
    recorder = _recorder()
    recorder.begin(max_profile_attempts=3)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=1.0,
        disable_cuda_graph=False,
        profile_result=None,
        failure={"phase": "profile", "error_class": "RuntimeError", "message": "boom"},
        instrumentation={
            "check_id": "instrumentation_preflight",
            "status": "failed",
            "detail": {
                "tracelens_patch_status": "unavailable",
                "degraded_reason": "tracelens_runtime_patch_unavailable",
                "patchers": {"benchmark_lib": True, "benchmark_serving": False},
                "failed_patchers": ["benchmark_serving"],
            },
        },
    )
    recorder.finish_failed(phase="profile", message="boom")

    run = _actions(tmp_path)[0]["profile"]["runs"][0]
    assert run["validate"] == {}, "no trace was produced, so there is no certificate"
    assert run["instrumentation"]["status"] == "failed"
    assert run["instrumentation"]["detail"]["failed_patchers"] == ["benchmark_serving"]
    assert run["instrumentation"]["detail"]["tracelens_patch_status"] == "unavailable"


def test_a_successful_attempt_records_the_patchers_that_worked(tmp_path: Path) -> None:
    """A patch that succeeded used to write nothing, making "fine" and "nobody looked" the same record."""
    recorder = _recorder()
    recorder.begin(max_profile_attempts=1)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="succeeded",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=30.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(trace_validate=_validate()),
        instrumentation={
            "check_id": "instrumentation_preflight",
            "status": "passed",
            "detail": {
                "tracelens_patch_status": "ok",
                "degraded_reason": "",
                "patchers": {"benchmark_lib": True, "benchmark_serving": True},
                "failed_patchers": [],
            },
        },
    )
    recorder.finish_failed(phase="analysis", message="stop")

    detail = _actions(tmp_path)[0]["profile"]["runs"][0]["instrumentation"]["detail"]
    assert detail["tracelens_patch_status"] == "ok"
    assert detail["patchers"] == {"benchmark_lib": True, "benchmark_serving": True}
    assert detail["failed_patchers"] == []


def test_profile_retries_collapse_into_one_action(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=12.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(trace_health={"zero_ops": True, "issues": ["[7] no ops"], "checks": []}),
        failure={"phase": "profile_zero_ops", "error_class": "zero_ops", "message": "metadata-only trace"},
    )
    recorder.record_profile_run(
        run_index=2,
        attempt_reason=PROFILE_ATTEMPT_AFTER_ZERO_OPS,
        status="succeeded",
        started_at="2026-01-01T00:01:00+00:00",
        duration_sec=30.0,
        disable_cuda_graph=True,
        profile_result=_profile_result(),
    )
    recorder.adopt_profile_run(run_index=2, profile_result=_profile_result(), params={"reason": "kernel_followup"})
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=3,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1},
        trace_path="/w/traces/merged-a.pt.trace.json.gz",
    )

    actions = _actions(tmp_path)
    assert len(actions) == 1
    profile = actions[0]["profile"]
    assert profile["attempt_count"] == 2
    assert [row["attempt_reason"] for row in profile["runs"]] == [
        PROFILE_ATTEMPT_INITIAL,
        PROFILE_ATTEMPT_AFTER_ZERO_OPS,
    ]
    assert [row["effective"] for row in profile["runs"]] == [False, True]
    assert profile["effective_run_index"] == 2
    assert profile["graph_capture_disabled"] is True
    assert profile["effective_run"]["trace"]["main_path"].endswith("merged-a.pt.trace.json.gz")


def test_analysis_retry_keeps_both_runs_and_one_conclusion(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.record_analysis_run(
        run_index=1,
        attempt_reason=ANALYSIS_ATTEMPT_INITIAL,
        status="failed",
        started_at="2026-01-01T00:02:00+00:00",
        duration_sec=5.0,
        trace_input="/w/traces/a.gz",
        ta_result=_ta_result(status="failed"),
        failure={"phase": "trace_analyze", "error_class": "", "message": "steady_state_chunk_low_quality"},
    )
    retried = _ta_result(n26_auto_retry={"applied": True, "from_mode": "mixed", "to_mode": "decode_only"})
    recorder.record_analysis_run(
        run_index=2,
        attempt_reason=ANALYSIS_ATTEMPT_N26_RETRY,
        status="succeeded",
        started_at="2026-01-01T00:03:00+00:00",
        duration_sec=6.0,
        trace_input="/w/traces/a.gz",
        requested_steady_state_mode="decode_only",
        ta_result=retried,
    )
    recorder.adopt_analysis_run(run_index=2, ta_result=retried, trace_input="/w/traces/a.gz")
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=1,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1},
        trace_path="/w/traces/a.gz",
    )

    analysis = _actions(tmp_path)[0]["analysis"]
    assert analysis["attempt_count"] == 2
    assert analysis["effective_run_index"] == 2
    assert [row["effective"] for row in analysis["runs"]] == [False, True]
    assert analysis["n26_auto_retry"]["to_mode"] == "decode_only"
    effective = analysis["effective_run"]
    assert effective["route"] == "agent"
    assert effective["tool"] == "tracelens"
    assert effective["hot_kernels"]["count"] == 1
    assert effective["preflight"]["trace_file_count"] == 1


def test_failed_action_names_the_failing_substep(tmp_path: Path) -> None:
    recorder = _recorder(reason="prelude_initial")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="trace_analyze", message="splitter produced no steady-state chunks")

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert event["end_time"]
    action = event["ext"]["actions"][0]
    assert action["failed_substep"] == "analysis"
    assert action["failure"]["phase"] == "trace_analyze"
    assert action["in_flight_substep"] is None


def _succeed(recorder, *, snapshot_id: int = 1) -> None:
    recorder.finish_succeeded(
        snapshot_id=snapshot_id,
        hot_kernel_count=3,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": snapshot_id},
        trace_path="/w/traces/a.gz",
    )


def test_rooflines_of_one_phase_and_cycle_are_actions_of_one_event(tmp_path: Path) -> None:
    for index, reason in enumerate(("kernel_followup", "close_post_opt")):
        recorder = _recorder(task_id=f"t-{index}", reason=reason, phase="sweep", macro_cycle=2)
        recorder.begin(max_profile_attempts=3)
        _succeed(recorder, snapshot_id=index + 1)

    events = _roofline_events(tmp_path)
    assert len(events) == 1
    assert events[0]["id"] == roofline_event_id("sweep", 2)
    assert [action["request"]["reason"] for action in events[0]["ext"]["actions"]] == [
        "kernel_followup",
        "close_post_opt",
    ]
    assert events[0]["status"] == "succeeded"


def test_a_different_phase_or_cycle_is_a_different_event(tmp_path: Path) -> None:
    for phase, cycle in (("prelude", 0), ("sweep", 1), ("sweep", 2)):
        recorder = _recorder(task_id=f"t-{phase}-{cycle}", phase=phase, macro_cycle=cycle)
        recorder.begin(max_profile_attempts=3)
        _succeed(recorder)

    assert sorted(event["id"] for event in _roofline_events(tmp_path)) == [
        roofline_event_id("prelude", 0),
        roofline_event_id("sweep", 1),
        roofline_event_id("sweep", 2),
    ]


def test_one_failed_action_is_not_hidden_by_a_later_success(tmp_path: Path) -> None:
    first = _recorder(task_id="t-0", phase="sweep", macro_cycle=2)
    first.begin(max_profile_attempts=3)
    first.finish_failed(phase="profile", message="server never booted")
    second = _recorder(task_id="t-1", phase="sweep", macro_cycle=2)
    second.begin(max_profile_attempts=3)
    _succeed(second)

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert [action["status"] for action in event["ext"]["actions"]] == ["failed", "succeeded"]


def test_degraded_when_attribution_folded(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=2,
        hot_kernel_count=0,
        kernel_attribution_degraded=True,
        cached={"roofline_snapshot_id": 2, "analysis_md_path": "/w/reports/analysis.md"},
        trace_path="/w/traces/a.gz",
    )

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "degraded"
    assert event["ext"]["actions"][0]["outcome"]["kernel_attribution_degraded"] is True


def _write_sidecar(session_dir: Path, **overrides: Any) -> str:
    """Write a kernel-roofline sidecar the way the analyzer subprocess does."""
    import json

    payload: dict[str, Any] = {
        "schema_version": "1",
        "source": "tracelens_analysis",
        "trace_input": "/w/traces/a.gz",
        "trace_input_type": "file",
        "analysis_md_path": "/w/reports/analysis.md",
        "kernel_candidates_path": "/w/reports/candidates.json",
        "kernels": [
            {
                "kernel_id": "k-cheap",
                "name": "rms_norm",
                "gpu_pct": 4.0,
                "duration_us": 12.5,
                "call_count": 900,
                "kernel_category": "norm",
                "bound_type": "memory",
                "arithmetic_intensity": 0.5,
                "efficiency_percent": 3.0,
                "bandwidth_utilization_pct": 41.0,
                "recommended_actions": ["fuse"],
                "roofline_source": "analytical",
            },
            {
                "kernel_id": "k-hot",
                "name": "gemm",
                "gpu_pct": 61.0,
                "duration_us": 900.0,
                "call_count": 120,
                "kernel_category": "gemm",
                "bound_type": "compute",
                "arithmetic_intensity": 180.0,
                "efficiency_percent": 78.0,
                "compute_utilization_pct": 77.5,
                "reusable_native_kernel": True,
                "roofline_source": "analytical",
            },
        ],
    }
    payload.update(overrides)
    path = session_dir / "kernel_roofline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_the_action_carries_the_per_kernel_roofline_table(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=2,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1, "kernel_roofline_path": _write_sidecar(tmp_path)},
        trace_path="/w/traces/a.gz",
    )

    table = _actions(tmp_path)[0]["kernel_roofline"]
    assert table["source"] == "tracelens_analysis"
    assert table["kernel_count"] == 2
    assert table["truncated"] is False
    # Ranked by GPU share, so the reader meets the expensive kernel first.
    assert [row["kernel_id"] for row in table["kernels"]] == ["k-hot", "k-cheap"]
    hot, cheap = table["kernels"]
    assert hot["bound_type"] == "compute"
    assert hot["efficiency_percent"] == 78.0
    assert hot["arithmetic_intensity"] == 180.0
    assert hot["duration_us"] == 900.0
    assert hot["call_count"] == 120
    assert hot["reusable_native_kernel"] is True
    # The cheap kernel is exactly what a top-N-by-cost cut would have dropped
    # and exactly what a reader is hunting: 4% of the GPU at 3% efficiency.
    assert cheap["bound_type"] == "memory"
    assert cheap["efficiency_percent"] == 3.0
    assert cheap["recommended_actions"] == ["fuse"]


def test_a_missing_sidecar_does_not_fail_a_run_that_succeeded(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=2,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1, "kernel_roofline_path": str(tmp_path / "absent.json")},
        trace_path="/w/traces/a.gz",
    )

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["actions"][0]["kernel_roofline"] is None


def test_the_outcome_carries_the_snapshot_and_not_only_its_id(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=7,
        hot_kernel_count=2,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 7},
        trace_path="/w/traces/a.gz",
        snapshot={
            "snapshot_id": 7,
            "ts": "2026-09-06T10:00:00Z",
            "framework": "sglang",
            "achieved_tok_per_sec": 1200.0,
            "theoretical_peak_tok_per_sec": 4000.0,
            "roofline_mem_ceiling_tok_per_sec": 4000.0,
            "roofline_cmp_ceiling_tok_per_sec": 9000.0,
            "roofline_bound_kind": "memory",
            "within_roofline_pct": 30.0,
            "gap_to_roofline_pct": 70.0,
            "compute_pct": 55.0,
            "idle_pct": 30.0,
            "comm_pct": 15.0,
            "top_bottleneck": "memory",
            "top_kernel": {"name": "gemm", "gpu_pct": 61.0, "efficiency_pct": 78.0, "bound_type": "compute"},
            "roofline_provenance": {"formula": "decode_mem"},
            "perfmodel_breakdown": {"bound_kind": "memory", "ops": [{"name": "attn", "time_s": 0.001}]},
        },
    )

    snapshot = _actions(tmp_path)[0]["outcome"]["snapshot"]
    assert snapshot["snapshot_id"] == 7
    assert snapshot["achieved_tok_per_sec"] == 1200.0
    assert snapshot["theoretical_peak_tok_per_sec"] == 4000.0
    assert snapshot["roofline_bound_kind"] == "memory"
    assert snapshot["gap_to_roofline_pct"] == 70.0
    assert snapshot["top_kernel"]["bound_type"] == "compute"
    assert snapshot["roofline_provenance"] == {"formula": "decode_mem"}
    assert snapshot["perfmodel_breakdown"]["op_count"] == 1


def test_an_analysis_with_no_snapshot_records_none_rather_than_an_empty_one(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    _succeed(recorder)

    assert _actions(tmp_path)[0]["outcome"]["snapshot"] is None


def test_the_table_is_recorded_once_however_the_action_unwinds(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=2,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1, "kernel_roofline_path": _write_sidecar(tmp_path)},
        trace_path="/w/traces/a.gz",
    )
    recorder.finish_crashed(RuntimeError("late"))

    table = _actions(tmp_path)[0]["kernel_roofline"]
    assert [row["kernel_id"] for row in table["kernels"]] == ["k-hot", "k-cheap"]
    assert table["kernel_count"] == 2


def test_an_inline_action_leaves_no_roofline_event(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.breakdown.recorder.assembler import roofline_event_parts
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id
    from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import assemble_roofline_action

    event = kernel_event_id(3)
    recorder = make_roofline_recorder(
        make_sink(event, producer=PRODUCER),
        task_id="t-reprofile",
        task_kind="roofline",
        reason="kernel_entry_g0_abc",
        framework="sglang",
        owns_event=False,
    )
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    _succeed(recorder)

    assert _roofline_events(tmp_path) == []
    action = assemble_roofline_action(roofline_event_parts(), event=event, task_id="t-reprofile")
    assert action is not None
    assert action["status"] == "succeeded"
    assert action["request"]["reason"] == "kernel_entry_g0_abc"


def test_trace_file_summary_stays_bounded_on_multi_rank() -> None:
    files = [f"/w/traces/xdit_rank{rank}_{index}.pt.trace.json.gz" for rank in range(8) for index in range(53)]
    summary = _summarize_trace_files({"trace_files": files, "main_trace_path": files[0]})

    assert summary["file_count"] == 424
    assert summary["rank_count"] == 8
    assert len(summary["sample_files"]) == 4
    assert summary["files_by_rank"]["3"] == 53


def test_steady_state_normalizes_across_tools() -> None:
    bypass = _analysis_steady_state(
        {
            "steady_window": {"step_name": "denoise", "step_count": 20},
            "estimated": False,
            "aggregation_scope": "steady_state",
        },
        requested_mode="decode_only",
        tool="bypass",
    )
    assert bypass["source"] == "in_reader_window"
    assert bypass["fell_back_to_full_trace"] is False
    assert bypass["selected"]["step_count"] == 20

    fallback = _analysis_steady_state(
        {"steady_window": None, "estimated": True, "aggregation_scope": "full_trace"},
        requested_mode="decode_only",
        tool="bypass",
    )
    assert fallback["fell_back_to_full_trace"] is True

    tracelens = _analysis_steady_state(
        {"run_meta": {"selection": {"selected_chunk": "decode_only_steady_state_0.json.gz"}}},
        requested_mode="decode_only",
        tool="tracelens",
    )
    # Same question, different mechanism: one selects a chunk file, the other a window in memory, and a consumer must
    # not have to branch on which ran.
    assert tracelens["source"] == "split_chunk"
    assert tracelens["selected"]["selected_chunk"].startswith("decode_only")
    assert set(bypass) == set(tracelens)


def _certificate(*, density: dict[str, Any], verdict: dict[str, Any], rank_count: int = 1) -> dict[str, Any]:
    """A probe record reduced to the fields the validation block reads."""
    return {
        "schema_version": 1,
        "probe_version": "selfcert-1.0.0",
        "trace_dir_level": {"rank_count": rank_count},
        "rank_level": [
            {
                "rank": 0,
                "density": density,
                "split_forecast": {"viable_modes": ["mixed"], "viable_consumer_modes": ["mixed"]},
            }
        ],
        "verdict": {"thresholds_effective": {"graph_launch_coverage_max": 0.5}, **verdict},
    }


def test_trace_validate_keeps_the_two_verdict_axes_apart() -> None:
    out = _build_trace_validate(
        {"checks": [{"check_id": CHECK_TRACE_HAS_OPS, "status": "passed"}]},
        trace_dir=Path("/w"),
        framework="sglang",
        certificate=_certificate(
            density={
                "graph_mode": True,
                "graph_launch_count": 128,
                "graph_launches_with_kernels": 1,
                "graph_launch_coverage": 0.0078,
                "graph_under_recorded": True,
            },
            verdict={
                "usable_by": ["bypass", "tracelens"],
                "decode_conclusions_valid": False,
                "silently_wrong": True,
            },
        ),
    )
    assert out["verdict"]["usable_by"] == ["bypass", "tracelens"]
    assert out["verdict"]["decode_conclusions_valid"] is False
    assert out["verdict"]["silently_wrong"] is True
    assert out["probe_status"] == "ok"
    # Chunks do not exist until the splitter runs, so the profile stage records the forecast the analysis stage will
    # later be measured against instead.
    assert out["chunk_level"] == []
    assert out["steady_state_forecast"]["viable_modes"] == ["mixed"]

    coverage = next(row for row in out["checks"] if row["check_id"] == CHECK_GRAPH_LAUNCH_COVERAGE)
    assert coverage["status"] == "failed"
    # Numerator and denominator are kept apart: a coverage ratio alone cannot say whether the capture recorded two
    # launches or two hundred.
    assert coverage["detail"]["graph_launch_count"] == 128
    assert coverage["detail"]["graph_launches_with_kernels"] == 1
    assert coverage["detail"]["coverage_max"] == 0.5


def test_eager_capture_skips_coverage_instead_of_failing_it() -> None:
    out = _build_trace_validate(
        {"checks": []},
        trace_dir=Path("/w"),
        framework="sglang",
        certificate=_certificate(
            density={
                "graph_mode": False,
                "graph_launch_count": 0,
                "graph_launches_with_kernels": 0,
                "graph_launch_coverage": None,
                "graph_under_recorded": False,
            },
            verdict={"usable_by": ["bypass", "tracelens"], "decode_conclusions_valid": True, "silently_wrong": False},
        ),
    )
    coverage = next(row for row in out["checks"] if row["check_id"] == CHECK_GRAPH_LAUNCH_COVERAGE)
    assert coverage["status"] == "skipped"
    assert "no CUDA graph launches" in coverage["skip_reason"]


def test_tensor_parallel_capture_reports_the_uncertified_ranks() -> None:
    out = _build_trace_validate(
        {"checks": []},
        trace_dir=Path("/w"),
        framework="sglang",
        certificate=_certificate(
            density={"graph_mode": True, "graph_under_recorded": False},
            verdict={"usable_by": ["bypass"], "decode_conclusions_valid": True, "silently_wrong": False},
            rank_count=8,
        ),
    )
    shape = next(row for row in out["checks"] if row["check_id"] == CHECK_RANK_SHAPE)
    assert shape["status"] == "skipped"
    assert shape["detail"]["rank_count"] == 8
    assert shape["detail"]["certified_rank_count"] == 1


def test_probe_failure_is_recorded_rather_than_read_as_a_verdict() -> None:
    out = _build_trace_validate(
        {"checks": [{"check_id": CHECK_TRACE_HAS_OPS, "status": "passed"}]},
        trace_dir=Path("/w"),
        framework="sglang",
        probe_error="OSError: trace unreadable",
    )
    assert out["probe_status"] == "failed"
    assert out["probe_error"] == "OSError: trace unreadable"
    assert out["verdict"] == {}
    assert [row["check_id"] for row in out["checks"]] == [CHECK_TRACE_HAS_OPS]


def test_validate_lands_per_profile_attempt(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.record_profile_run(
        run_index=1,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="failed",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=1.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(
            trace_validate={
                "verdict": {"usable_by": [], "decode_conclusions_valid": None, "silently_wrong": False},
                "probe_status": "ok",
                "checked_at": "2026-01-01T00:00:01+00:00",
                "checks": [{"check_id": CHECK_TRACE_HAS_OPS, "status": "failed"}],
            }
        ),
        failure={"phase": "profile_zero_ops", "error_class": "zero_ops", "message": "no ops"},
    )
    recorder.record_profile_run(
        run_index=2,
        attempt_reason=PROFILE_ATTEMPT_AFTER_ZERO_OPS,
        status="succeeded",
        started_at="2026-01-01T00:01:00+00:00",
        duration_sec=2.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(
            trace_validate={
                "verdict": {
                    "usable_by": ["bypass", "tracelens"],
                    "decode_conclusions_valid": True,
                    "silently_wrong": False,
                },
                "probe_status": "ok",
                "checked_at": "2026-01-01T00:01:01+00:00",
                "checks": [{"check_id": CHECK_TRACE_HAS_OPS, "status": "passed"}],
            }
        ),
    )
    _succeed(recorder)

    runs = _actions(tmp_path)[0]["profile"]["runs"]
    assert runs[0]["validate"]["usable_by"] == []
    assert runs[0]["validate"]["failed_check_ids"] == [CHECK_TRACE_HAS_OPS]
    assert runs[1]["validate"]["usable_by"] == ["bypass", "tracelens"]
    assert runs[1]["validate"]["decode_conclusions_valid"] is True
    assert runs[1]["validate"]["failed_check_ids"] == []


def test_crash_closes_the_event(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_crashed(RuntimeError("record_trace_analyze blew up"))

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "failed"
    action = event["ext"]["actions"][0]
    assert action["failure"]["error_class"] == "RuntimeError"
    assert action["failure"]["phase"] == "profile"
    assert action["failed_substep"] == "profile"


def test_a_crash_after_the_profile_was_adopted_blames_the_analysis(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.record_profile_run(
        run_index=0,
        attempt_reason=PROFILE_ATTEMPT_INITIAL,
        status="succeeded",
        started_at="2026-01-01T00:00:00+00:00",
        duration_sec=30.0,
        disable_cuda_graph=False,
        profile_result=_profile_result(),
    )
    recorder.adopt_profile_run(run_index=0, profile_result=_profile_result(), params={"reason": "kernel_followup"})
    recorder.finish_crashed(RuntimeError("trace_analyze_handler blew up"))

    action = _actions(tmp_path)[0]
    assert action["failed_substep"] == "analysis"
    assert action["failure"]["phase"] == "analysis"
    assert action["profile"]["runs"][0]["status"] == "succeeded"


def test_crash_does_not_overwrite_a_closed_action(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    _succeed(recorder)
    recorder.finish_crashed(RuntimeError("raised after the result was built"))

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["actions"][0]["failure"] is None


def test_open_ended_route_ext_is_size_capped(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    bloated = _ta_result()
    bloated["analysis_meta"]["route_ext"] = {
        "attribution": {f"kernel_{index}": {"share": 0.001, "note": "x" * 200} for index in range(500)},
        "rank_count": 8,
    }
    recorder.adopt_analysis_run(run_index=1, ta_result=bloated, trace_input="/w/traces/a.gz")
    _succeed(recorder)

    route_ext = _actions(tmp_path)[0]["analysis"]["effective_run"]["route_ext"]
    assert route_ext["omitted"] is True
    assert route_ext["keys"] == ["attribution", "rank_count"]


def test_no_sink_records_nothing(tmp_path: Path) -> None:
    assert make_roofline_recorder(None, reason="kernel_followup", framework="sglang") is None
    assert not (tmp_path / "reports").exists()


def test_recorder_write_failure_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    recorder = _recorder(reason="kernel_followup")

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.session.sbd_v6.write_timeline_event",
        _boom,
    )
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="unrelated")

    warnings = (tmp_path / "reports" / "sbd_v6" / "write_warnings.jsonl").read_text(encoding="utf-8")
    assert "timeline.roofline.open" in warnings
