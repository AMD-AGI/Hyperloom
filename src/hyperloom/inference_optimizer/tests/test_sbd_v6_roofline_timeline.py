# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the real-time SBD V6 ``roofline`` timeline event."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.orchestrator.actions.executors._roofline_timeline import (
    ANALYSIS_ATTEMPT_INITIAL,
    ANALYSIS_ATTEMPT_N26_RETRY,
    PROFILE_ATTEMPT_AFTER_ZERO_OPS,
    PROFILE_ATTEMPT_INITIAL,
    _summarize_trace_files,
    make_roofline_recorder,
)
from hyperloom.orchestrator.actions.executors.profile import (
    CHECK_GRAPH_LAUNCH_COVERAGE,
    CHECK_RANK_SHAPE,
    CHECK_TRACE_HAS_OPS,
    _build_trace_validate,
)
from hyperloom.orchestrator.kernel.request_handlers import _analysis_steady_state


def _roofline_events(session_dir: Path) -> list[dict[str, Any]]:
    return [event for event in read_timeline_events(session_dir) if event.get("type") == "roofline"]


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
        "orchestrator_mode": "deterministic",
        "hot_kernels": [{"name": "gemm", "gpu_time_us": 10.0, "gpu_pct": 50.0, "count": 2}],
        "trace_health_warnings": [],
        "analysis_meta": {
            "route": "deterministic",
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


def test_begin_writes_an_in_flight_event(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, task_id="t-1", reason="prelude_initial", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)

    events = _roofline_events(tmp_path)
    assert len(events) == 1
    event = events[0]
    # A session killed mid-roofline has to be readable as "the stage was running",
    # which is the whole point of writing before the work rather than after it.
    assert event["status"] == "running"
    assert event["end_time"] == ""
    assert event["ext"]["in_flight_substep"] == "profile"
    assert event["ext"]["request"]["arm"] == "baseline"
    assert event["ext"]["profile"]["max_attempts"] == 3


def test_shape_capture_dispatch_does_not_claim_a_measured_arm(tmp_path: Path) -> None:
    """The GEMM shape-capture path reuses this executor but measures no arm.

    It dispatches as ``gemm_shape_capture`` and carries no reason of its own, so
    defaulting the arm to current_best would state a measurement the run never
    made -- and a consumer counting roofline dispatches would have nothing in the
    event to exclude it by.
    """
    recorder = make_roofline_recorder(
        tmp_path,
        task_id="t-capture-1",
        task_kind="gemm_shape_capture",
        framework="vllm",
    )
    assert recorder is not None
    recorder.begin(max_profile_attempts=1)

    request = _roofline_events(tmp_path)[0]["ext"]["request"]
    assert request["task_kind"] == "gemm_shape_capture"
    assert request["arm"] == ""


def test_roofline_dispatch_without_a_reason_still_names_its_arm(tmp_path: Path) -> None:
    """A roofline task may carry no reason, and current_best remains its default.

    The arm is withheld by dispatch kind rather than by an absent reason, so this
    case must not be caught by the shape-capture rule above.
    """
    recorder = make_roofline_recorder(tmp_path, task_id="t-2", task_kind="roofline", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)

    request = _roofline_events(tmp_path)[0]["ext"]["request"]
    assert request["task_kind"] == "roofline"
    assert request["arm"] == "current_best"


def test_profile_retries_collapse_into_one_event(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
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

    events = _roofline_events(tmp_path)
    # Two attempts, one event: the retry is internal to the action, so it must not
    # look like a second dispatched roofline.
    assert len(events) == 1
    profile = events[0]["ext"]["profile"]
    assert profile["attempt_count"] == 2
    assert [row["attempt_reason"] for row in profile["runs"]] == [
        PROFILE_ATTEMPT_INITIAL,
        PROFILE_ATTEMPT_AFTER_ZERO_OPS,
    ]
    assert [row["effective"] for row in profile["runs"]] == [False, True]
    assert profile["effective_run_index"] == 2
    assert profile["eager_fallback_applied"] is True
    assert profile["effective_run"]["trace"]["main_path"].endswith("merged-a.pt.trace.json.gz")
    assert events[0]["ext"]["in_flight_substep"] == "analysis"


def test_analysis_retry_keeps_both_runs_and_one_conclusion(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
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

    analysis = _roofline_events(tmp_path)[0]["ext"]["analysis"]
    assert analysis["attempt_count"] == 2
    assert analysis["effective_run_index"] == 2
    assert [row["effective"] for row in analysis["runs"]] == [False, True]
    assert analysis["n26_auto_retry"]["to_mode"] == "decode_only"
    effective = analysis["effective_run"]
    assert effective["route"] == "deterministic"
    assert effective["tool"] == "tracelens"
    assert effective["hot_kernels"]["count"] == 1
    assert effective["preflight"]["trace_file_count"] == 1


def test_failed_event_names_the_failing_substep(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="prelude_initial", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="trace_analyze", message="splitter produced no steady-state chunks")

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "failed"
    assert event["end_time"]
    assert event["ext"]["failed_substep"] == "analysis"
    assert event["ext"]["failure"]["phase"] == "trace_analyze"
    assert event["ext"]["in_flight_substep"] is None


def test_each_dispatched_roofline_is_its_own_event(tmp_path: Path) -> None:
    for reason in ("prelude_initial", "kernel_followup", "close_post_opt"):
        recorder = make_roofline_recorder(tmp_path, reason=reason, framework="sglang")
        assert recorder is not None
        recorder.begin(max_profile_attempts=3)
        recorder.finish_succeeded(
            snapshot_id=1,
            hot_kernel_count=3,
            kernel_attribution_degraded=False,
            cached={"roofline_snapshot_id": 1},
            trace_path="/w/traces/a.gz",
        )

    events = _roofline_events(tmp_path)
    # Separate dispatches are separate events; only retries fold inward.
    assert [event["ext"]["request"]["reason"] for event in events] == [
        "prelude_initial",
        "kernel_followup",
        "close_post_opt",
    ]
    assert all(event["status"] == "succeeded" for event in events)


def test_degraded_when_attribution_folded(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=2,
        hot_kernel_count=0,
        kernel_attribution_degraded=True,
        cached={"roofline_snapshot_id": 2, "analysis_md_path": "/w/reports/analysis.md"},
        trace_path="/w/traces/a.gz",
    )

    event = _roofline_events(tmp_path)[0]
    # Zero routable candidates from folded attribution is a completed roofline
    # that cannot advance kernel work -- distinct from a clean run and from a failure.
    assert event["status"] == "degraded"
    assert event["ext"]["outcome"]["kernel_attribution_degraded"] is True


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
    # Same question, different mechanism: one selects a chunk file, the other a
    # window in memory, and a consumer must not have to branch on which ran.
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
    """A trace both consumers can route can still carry a false decode answer.

    This is the case a single healthy/degraded/unusable grade cannot express:
    nothing fails, the analysis completes, and the conclusion is wrong anyway.
    """
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
    # Chunks do not exist until the splitter runs, so the profile stage records
    # the forecast the analysis stage will later be measured against instead.
    assert out["chunk_level"] == []
    assert out["steady_state_forecast"]["viable_modes"] == ["mixed"]

    coverage = next(row for row in out["checks"] if row["check_id"] == CHECK_GRAPH_LAUNCH_COVERAGE)
    assert coverage["status"] == "failed"
    # Numerator and denominator are kept apart: a coverage ratio alone cannot
    # say whether the capture recorded two launches or two hundred.
    assert coverage["detail"]["graph_launch_count"] == 128
    assert coverage["detail"]["graph_launches_with_kernels"] == 1
    assert coverage["detail"]["coverage_max"] == 0.5


def test_eager_capture_skips_coverage_instead_of_failing_it() -> None:
    """No graph launches means no denominator, which is not a failed check."""
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
    """One certified rank is not a claim about the other seven."""
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
    """A probe that could not run must not leave an empty verdict looking clean."""
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
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
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

    runs = _roofline_events(tmp_path)[0]["ext"]["profile"]["runs"]
    # Each attempt keeps the verdict computed against the trace it produced, so
    # "attempt 1 was unusable by anything, attempt 2 was not" stays answerable.
    assert runs[0]["validate"]["usable_by"] == []
    assert runs[0]["validate"]["failed_check_ids"] == [CHECK_TRACE_HAS_OPS]
    assert runs[1]["validate"]["usable_by"] == ["bypass", "tracelens"]
    assert runs[1]["validate"]["decode_conclusions_valid"] is True
    assert runs[1]["validate"]["failed_check_ids"] == []


def test_crash_closes_the_event(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    recorder.finish_crashed(RuntimeError("record_trace_analyze blew up"))

    event = _roofline_events(tmp_path)[0]
    # An executor that raised must not be indistinguishable from a session that
    # was killed mid-roofline, which is what a dangling "running" event means.
    assert event["status"] == "failed"
    assert event["ext"]["failure"]["error_class"] == "RuntimeError"
    assert event["ext"]["failure"]["phase"] == "profile"


def test_crash_does_not_overwrite_a_closed_event(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    recorder.finish_succeeded(
        snapshot_id=1,
        hot_kernel_count=4,
        kernel_attribution_degraded=False,
        cached={"roofline_snapshot_id": 1},
        trace_path="/w/traces/a.gz",
    )
    recorder.finish_crashed(RuntimeError("raised after the result was built"))

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["failure"] is None


def test_open_ended_route_ext_is_size_capped(tmp_path: Path) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None
    recorder.begin(max_profile_attempts=3)
    bloated = _ta_result()
    bloated["analysis_meta"]["route_ext"] = {
        "attribution": {f"kernel_{index}": {"share": 0.001, "note": "x" * 200} for index in range(500)},
        "rank_count": 8,
    }
    recorder.adopt_analysis_run(run_index=1, ta_result=bloated, trace_input="/w/traces/a.gz")

    route_ext = _roofline_events(tmp_path)[0]["ext"]["analysis"]["effective_run"]["route_ext"]
    # ``route_ext`` is deliberately open, so a verbose tool must not be able to
    # multiply the SBD payload; the block is replaced by its shape instead.
    assert route_ext["omitted"] is True
    assert route_ext["keys"] == ["attribution", "rank_count"]


def test_unresolved_session_dir_records_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # ``_resolve_session_dir`` falls back to Path(".") when the runner context
    # carries no session dir; recording there would scatter events into whatever
    # the working directory happens to be.
    assert make_roofline_recorder(Path("."), reason="kernel_followup", framework="sglang") is None
    assert not (tmp_path / "reports").exists()


def test_recorder_write_failure_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    recorder = make_roofline_recorder(tmp_path, reason="kernel_followup", framework="sglang")
    assert recorder is not None

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "hyperloom.inference_optimizer.session.sbd_v6.write_timeline_event_at",
        _boom,
    )
    # Observability must never change roofline behavior, so a writer failure is
    # parked rather than propagated.
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="unrelated")

    warnings = (tmp_path / "reports" / "sbd_v6" / "write_warnings.jsonl").read_text(encoding="utf-8")
    assert "roofline.begin" in warnings
