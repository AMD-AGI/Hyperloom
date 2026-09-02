# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the SBD V6 ``roofline`` action, standalone and inline."""

from __future__ import annotations

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


def test_begin_puts_the_event_on_the_timeline(tmp_path: Path) -> None:
    """A session killed mid-roofline has to be readable as "this was running"."""
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
    """The retry budget distinguishes exhausting it from stopping early."""
    recorder = _recorder(reason="prelude_initial")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="never booted")

    action = _actions(tmp_path)[0]
    assert action["request"]["arm"] == "baseline"
    assert action["profile"]["max_attempts"] == 3


def test_shape_capture_dispatch_does_not_claim_a_measured_arm(tmp_path: Path) -> None:
    """The GEMM shape-capture path reuses this executor but measures no arm.

    It dispatches as ``gemm_shape_capture`` and carries no reason of its own, so
    defaulting the arm to current_best would state a measurement the run never
    made -- and a consumer counting roofline dispatches would have nothing in the
    event to exclude it by.
    """
    recorder = _recorder(task_id="t-capture-1", task_kind="gemm_shape_capture", framework="vllm")
    recorder.begin(max_profile_attempts=1)
    recorder.finish_failed(phase="profile", message="stopped")

    request = _actions(tmp_path)[0]["request"]
    assert request["task_kind"] == "gemm_shape_capture"
    assert request["arm"] == ""


def test_roofline_dispatch_without_a_reason_still_names_its_arm(tmp_path: Path) -> None:
    """A roofline task may carry no reason, and current_best remains its default.

    The arm is withheld by dispatch kind rather than by an absent reason, so this
    case must not be caught by the shape-capture rule above.
    """
    recorder = _recorder(task_id="t-2", task_kind="roofline")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_failed(phase="profile", message="stopped")

    request = _actions(tmp_path)[0]["request"]
    assert request["task_kind"] == "roofline"
    assert request["arm"] == "current_best"


def test_profile_retries_collapse_into_one_action(tmp_path: Path) -> None:
    """The retry is internal to the action, so it must not read as a second one."""
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
    assert profile["eager_fallback_applied"] is True
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
    assert effective["route"] == "deterministic"
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
    """A phase can dispatch roofline more than once, and the task id separates them.

    The event id names a phase in a macro cycle, which is what a reader looks
    for; the actions inside it are what a reader counts. Making each dispatch
    its own event would need the task id in the event id, and then nothing on
    the timeline would say those two rooflines belonged to the same phase.
    """
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
    """What separates two events is the phase and the cycle, nothing else."""
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
    """The event takes the worst status of its actions, not the last one."""
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
    """Zero routable candidates is a completed roofline that cannot advance work."""
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


def test_an_inline_action_leaves_no_roofline_event(tmp_path: Path) -> None:
    """The KERNEL entry's re-profile belongs to the kernel event, not beside it.

    A sibling event would claim the inline run is dispatchable on its own, and
    a reader counting roofline events would count a sub-step of another phase.
    """
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
    """Each attempt keeps the verdict computed against the trace it produced."""
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
    """An executor that raised must not read as a session killed mid-roofline."""
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    recorder.finish_crashed(RuntimeError("record_trace_analyze blew up"))

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "failed"
    action = event["ext"]["actions"][0]
    assert action["failure"]["error_class"] == "RuntimeError"
    assert action["failure"]["phase"] == "profile"


def test_crash_does_not_overwrite_a_closed_action(tmp_path: Path) -> None:
    recorder = _recorder(reason="kernel_followup")
    recorder.begin(max_profile_attempts=3)
    _succeed(recorder)
    recorder.finish_crashed(RuntimeError("raised after the result was built"))

    event = _roofline_events(tmp_path)[0]
    assert event["status"] == "succeeded"
    assert event["ext"]["actions"][0]["failure"] is None


def test_open_ended_route_ext_is_size_capped(tmp_path: Path) -> None:
    """A verbose tool must not be able to multiply the SBD payload."""
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
    """A caller with no event to write into declines rather than guessing one."""
    assert make_roofline_recorder(None, reason="kernel_followup", framework="sglang") is None
    assert not (tmp_path / "reports").exists()


def test_recorder_write_failure_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    """Observability must never change roofline behavior."""
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
