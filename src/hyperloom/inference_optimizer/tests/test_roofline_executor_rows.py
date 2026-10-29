# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The attempt rows and run indices a roofline action leaves on its timeline event, driven through the executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import (
    ANALYSIS_ATTEMPT_COMPUTE_BOUND,
    ANALYSIS_ATTEMPT_INITIAL,
    ANALYSIS_ATTEMPT_N26_RETRY,
    PROFILE_ATTEMPT_COMPUTE_BOUND,
    PROFILE_ATTEMPT_INITIAL,
)
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.orchestrator.actions.executors import roofline as roofline_mod
from hyperloom.orchestrator.actions.executors.roofline import RooflineExecutor
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task

TRACE = "/tmp/rows/a.pt.trace.json.gz"
CB_TRACE = "/tmp/rows/cb.pt.trace.json.gz"
_TIMING_KEYS = ("start_time", "end_time", "duration_sec")


def _ctx(tmp_path: Path) -> RunnerContext:
    task = Task(
        task_id="t-rows",
        kind="roofline",
        state="running",
        params={"base_extra_args": "", "reason": "kernel_followup"},
        idempotency_key="roofline:rows",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


def _profile_ok(trace: str = TRACE, **extra: Any) -> dict[str, Any]:
    return {"status": "succeeded", "main_trace_path": trace, "workspace": "/tmp/rows", **extra}


def _ta_ok(md: Path, *, hot: int = 0, host_bound: bool = False) -> dict[str, Any]:
    return {
        "status": "ok",
        "trace_report_path": str(md),
        "hot_kernels": [{"kernel_id": f"k{i}", "name": f"gemm_{i}", "gpu_pct": 10.0} for i in range(hot)],
        "trace_health_warnings": [{"code": "high_gpu_idle_pct"}] if host_bound else [],
    }


def _ta_empty_chunk() -> dict[str, Any]:
    return {
        "status": "failed",
        "error": "steady_state_chunk_low_quality",
        "trace_health_warnings": [
            {"code": "steady_state_chunk_low_quality", "requested_mode": "mixed", "non_empty_modes": ["decode_only"]}
        ],
    }


async def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profiles: list[Any],
    analyses: list[Any],
    *,
    multi_node: bool = False,
    disable_cuda_graph: str | None = None,
    profile_params: list[dict[str, Any]] | None = None,
    analysis_payloads: list[dict[str, Any]] | None = None,
    state: SharedState | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one roofline action against scripted sub-steps; return its result and its recorded action.

    ``profile_params`` and ``analysis_payloads``, when given, collect what each sub-step was called with; ``state``,
    when given, is the shared state the action runs against.
    """
    profiles, analyses = list(profiles), list(analyses)

    async def fake_profile(ctx: RunnerContext) -> Any:
        if profile_params is not None:
            profile_params.append(dict(ctx.task.params or {}))
        out = profiles.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    async def fake_ta(payload: dict[str, Any], *, session_dir: Path) -> Any:
        if analysis_payloads is not None:
            analysis_payloads.append(dict(payload))
        out = analyses.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out

    if disable_cuda_graph is None:
        monkeypatch.delenv("HYPERLOOM_PROFILE_DISABLE_CUDA_GRAPH", raising=False)
    else:
        monkeypatch.setenv("HYPERLOOM_PROFILE_DISABLE_CUDA_GRAPH", disable_cuda_graph)
    monkeypatch.delenv("HYPERLOOM_PROFILE_AUTO_COMPUTE_BOUND", raising=False)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.profile.profile_executor", fake_profile)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.trace_analyze.trace_analyze_handler", fake_ta)
    monkeypatch.setattr(roofline_mod, "is_multi_node", lambda: multi_node)
    if state is None:
        state = SharedState()
        state.baseline_tput = 100.0
    result = await RooflineExecutor(shared_state=state)(_ctx(tmp_path))
    assert profiles == [] and analyses == [], "every scripted sub-step result must be consumed"
    event = next(e for e in read_timeline_events(tmp_path) if e.get("type") == "roofline")
    (action,) = event["ext"]["actions"]
    return result, action


def _rows(action: dict[str, Any], half: str) -> list[tuple[Any, ...]]:
    """Each attempt as (run_index, attempt_reason, status, effective, failure stage, failure error_class)."""
    out = []
    for row in action[half]["runs"]:
        for key in _TIMING_KEYS:
            assert key in row, f"{half} run {row['run_index']} lost {key}"
        assert row["duration_sec"] >= 0
        assert row["start_time"] <= row["end_time"]
        failure = row["failure"] or {}
        out.append(
            (
                row["run_index"],
                row["attempt_reason"],
                row["status"],
                row["effective"],
                failure.get("stage"),
                failure.get("error_class"),
            )
        )
    return out


def _messages(action: dict[str, Any], half: str) -> list[str | None]:
    return [(row["failure"] or {}).get("message") for row in action[half]["runs"]]


_ZERO_OPS_MESSAGE = (
    "profile produced a metadata-only trace (PyTorch Profiler Op count == 0); the active capture window never "
    "overlapped execution"
)
_CAPTURE_ONLY_MESSAGE = (
    "profile produced only CUDA-graph capture sidecars under capture_traces/ (no annotated steady-state trace); "
    "the steady-state splitter cannot use these"
)
_NO_TRACE_MESSAGE = "profile succeeded but no trace_path in result (missing both main_trace_path and trace_files[0])"


@pytest.mark.parametrize(
    ("returned", "stage", "error_class", "message"),
    [
        (RuntimeError("boot"), "profile", "RuntimeError", "profile_executor raised: RuntimeError('boot')"),
        (42, "profile", "bad_return", "profile_executor returned non-dict: int"),
        (
            {"status": "failed", "error_class": "server_crashed", "error": "engine exited"},
            "profile",
            "server_crashed",
            "engine exited",
        ),
        ({"status": "succeeded", "workspace": "/tmp/rows"}, "profile_no_trace", "no_trace", _NO_TRACE_MESSAGE),
        (
            _profile_ok(profile_trace_selection_reason="capture_only_fallback"),
            "profile_capture_only",
            "capture_only",
            _CAPTURE_ONLY_MESSAGE,
        ),
        (_profile_ok(trace_health={"zero_ops": True}), "profile_zero_ops", "zero_ops", _ZERO_OPS_MESSAGE),
    ],
    ids=["raised", "bad_return", "failed", "no_trace", "capture_only", "zero_ops"],
)
@pytest.mark.asyncio
async def test_an_unusable_profile_rows_one_run_and_fails_without_reprofiling(
    tmp_path, monkeypatch, returned, stage, error_class, message
):
    result, action = await _run(tmp_path, monkeypatch, [returned], [])

    assert _rows(action, "profile") == [(1, PROFILE_ATTEMPT_INITIAL, "failed", False, stage, error_class)]
    assert _messages(action, "profile") == [message]
    assert action["profile"]["effective_run_index"] is None
    assert action["analysis"]["runs"] == []
    assert result["status"] == "failed"
    assert result["phase"] == stage
    assert result["error"] == message
    assert action["failed_substep"] == "profile"


@pytest.mark.asyncio
async def test_a_failed_profile_fails_on_the_result_the_profiler_returned(tmp_path, monkeypatch):
    failed = {"status": "failed", "error_class": "server_crashed", "error": "engine exited"}
    result, _action = await _run(tmp_path, monkeypatch, [failed], [])

    assert result["sub_result"] == {"status": "failed", "error": "engine exited", "error_class": "server_crashed"}


@pytest.mark.asyncio
async def test_a_recovered_profile_and_an_n26_retry_are_the_effective_runs(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    recovered = {"status": "failed", "error_class": "stop_profile", "error": "dup stop", "main_trace_path": TRACE}
    result, action = await _run(tmp_path, monkeypatch, [recovered], [_ta_empty_chunk(), _ta_ok(md, hot=2)])

    assert _rows(action, "profile") == [(1, PROFILE_ATTEMPT_INITIAL, "recovered", True, None, None)]
    assert action["profile"]["effective_run_index"] == 1
    assert action["profile"]["recovered"] is True
    assert _rows(action, "analysis") == [
        (1, ANALYSIS_ATTEMPT_INITIAL, "failed", False, "trace_analyze", ""),
        (2, ANALYSIS_ATTEMPT_N26_RETRY, "succeeded", True, None, None),
    ]
    assert [row["requested_steady_state_mode"] for row in action["analysis"]["runs"]] == ["", "decode_only"]
    assert [row["trace_input"] for row in action["analysis"]["runs"]] == [TRACE, TRACE]
    assert [row["hot_kernel_count"] for row in action["analysis"]["runs"]] == [0, 2]
    assert action["analysis"]["effective_run_index"] == 2
    assert action["analysis"]["n26_auto_retry"] == {
        "applied": True,
        "from_mode": "mixed",
        "to_mode": "decode_only",
        "source_warning_code": "steady_state_chunk_low_quality",
    }
    assert result["status"] == "recovered"
    assert result["profile_warning"] == {"status": "failed", "error_class": "stop_profile", "error": "dup stop"}


@pytest.mark.parametrize(
    ("fatal", "error_class"),
    [
        (
            {"status": "failed", "error_class": "primary_rank_trace_missing", "error": "no primary rank trace"},
            "primary_rank_trace_missing",
        ),
        (
            {
                "status": "failed",
                "error_class": "capture_failed",
                "error": "no primary rank trace",
                "trace_capture": {"reason": "api_port_allocation_failed"},
            },
            "capture_failed",
        ),
        (
            {
                "status": "failed",
                "error_class": ORCHESTRATOR_CANCELLED_CLASS,
                "error": "no primary rank trace",
            },
            ORCHESTRATOR_CANCELLED_CLASS,
        ),
    ],
    ids=["error_class", "capture_reason", "cancelled"],
)
@pytest.mark.asyncio
async def test_a_terminal_failure_rows_one_attempt_and_stops(tmp_path, monkeypatch, fatal, error_class):
    result, action = await _run(tmp_path, monkeypatch, [fatal], [])

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "failed", False, "profile", error_class),
    ]
    assert result["phase"] == "profile"
    assert result["error"] == "no primary rank trace"


@pytest.mark.asyncio
async def test_a_cancelled_profile_is_not_recovered_from_the_trace_it_flushed(tmp_path, monkeypatch):
    """Recovering it would run the analysis, and the re-profile that analysis can ask for, in the cancelled scope."""
    cancelled = {
        "status": "failed",
        "error_class": ORCHESTRATOR_CANCELLED_CLASS,
        "error": "the orchestrator cancelled this action while this round was running",
        "main_trace_path": TRACE,
    }
    result, action = await _run(tmp_path, monkeypatch, [cancelled], [])

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "failed", False, "profile", ORCHESTRATOR_CANCELLED_CLASS),
    ]
    assert action["profile"]["recovered"] is False
    assert action["analysis"]["runs"] == []
    assert result["status"] == "failed"
    assert "profile_warning" not in result


@pytest.mark.asyncio
async def test_a_capture_failure_rows_one_attempt_and_stops(tmp_path, monkeypatch):
    capture = RuntimeError("operation not permitted when stream is capturing")
    result, action = await _run(tmp_path, monkeypatch, [capture], [])

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "failed", False, "profile", "RuntimeError"),
    ]
    assert result["error_class"] == "profile_cuda_graph_capture_config_failed"
    assert (tmp_path / "diagnostics" / "roofline_cuda_graph_capture_t-rows.json").is_file()


@pytest.mark.parametrize(
    ("analyses", "error_class", "message"),
    [
        ([RuntimeError("tl")], "RuntimeError", "trace_analyze_handler raised: RuntimeError('tl')"),
        (["nope"], "bad_return", "trace_analyze_handler returned non-dict: str"),
        (
            [_ta_empty_chunk(), RuntimeError("tl")],
            "RuntimeError",
            "trace_analyze_handler raised on N26 auto-retry (mode=decode_only): RuntimeError('tl')",
        ),
        (
            [_ta_empty_chunk(), None],
            "bad_return",
            "trace_analyze_handler returned non-dict on N26 auto-retry (mode=decode_only): NoneType",
        ),
    ],
)
@pytest.mark.asyncio
async def test_an_analysis_that_cannot_conclude_rows_the_attempt_it_failed_on(
    tmp_path, monkeypatch, analyses, error_class, message
):
    result, action = await _run(tmp_path, monkeypatch, [_profile_ok()], analyses)

    runs = _rows(action, "analysis")
    assert runs[-1] == (len(analyses), runs[-1][1], "failed", False, "trace_analyze", error_class)
    assert [row[1] for row in runs] == [ANALYSIS_ATTEMPT_INITIAL, ANALYSIS_ATTEMPT_N26_RETRY][: len(analyses)]
    assert _messages(action, "analysis")[-1] == message
    assert action["analysis"]["effective_run_index"] is None
    assert action["profile"]["effective_run_index"] == 1
    assert result["phase"] == "trace_analyze"
    assert result["error"] == message
    assert "sub_result" not in result
    assert action["failed_substep"] == "analysis"


@pytest.mark.asyncio
async def test_an_analysis_that_reports_failure_ends_the_action_on_its_result(tmp_path, monkeypatch):
    failed = {"status": "failed", "error_class": "tl_error", "error": "no steady state"}
    result, action = await _run(tmp_path, monkeypatch, [_profile_ok()], [failed])

    assert _rows(action, "analysis") == [(1, ANALYSIS_ATTEMPT_INITIAL, "failed", False, "trace_analyze", "tl_error")]
    assert result["error"] == "no steady state"
    assert result["sub_result"] == {"status": "failed", "error": "no steady state", "error_class": "tl_error"}


@pytest.mark.asyncio
async def test_an_adopted_compute_bound_reprofile_is_the_effective_run_of_both_halves(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    state = SharedState()
    result, action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok(), _profile_ok(CB_TRACE, launch_evidence_path="/tmp/rows/cb_launch.json")],
        [_ta_ok(md, host_bound=True), _ta_ok(md, hot=3)],
        multi_node=True,
        state=state,
    )

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "succeeded", False, None, None),
        (2, PROFILE_ATTEMPT_COMPUTE_BOUND, "succeeded", True, None, None),
    ]
    assert _rows(action, "analysis") == [
        (1, ANALYSIS_ATTEMPT_INITIAL, "succeeded", False, None, None),
        (2, ANALYSIS_ATTEMPT_COMPUTE_BOUND, "succeeded", True, None, None),
    ]
    assert [row["trace_input"] for row in action["analysis"]["runs"]] == [TRACE, CB_TRACE]
    assert action["analysis"]["effective_run"]["trace_input"] == CB_TRACE
    assert action["analysis"]["compute_bound_reprofile"] == {
        "attempted": True,
        "adopted": True,
        "reason": "adopted: surfaced 3 hot kernel(s)",
    }
    assert result["last_profile_trace"] == CB_TRACE
    assert state.last_profile_trace == state.last_trace_analyze["trace_input"] == CB_TRACE
    assert state.last_profile_launch_evidence_path == "/tmp/rows/cb_launch.json"
    assert state.last_profile_status == "succeeded"


@pytest.mark.asyncio
async def test_a_compute_bound_reprofile_that_stays_host_bound_leaves_the_first_runs_effective(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    result, action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok(), _profile_ok(CB_TRACE)],
        [_ta_ok(md, host_bound=True), _ta_ok(md, host_bound=True)],
        multi_node=True,
    )

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "succeeded", True, None, None),
        (2, PROFILE_ATTEMPT_COMPUTE_BOUND, "succeeded", False, None, None),
    ]
    assert _rows(action, "analysis") == [
        (1, ANALYSIS_ATTEMPT_INITIAL, "succeeded", True, None, None),
        (2, ANALYSIS_ATTEMPT_COMPUTE_BOUND, "succeeded", False, None, None),
    ]
    assert action["analysis"]["compute_bound_reprofile"]["reason"] == (
        "still host-bound: re-analysis surfaced no hot kernels"
    )
    assert result["last_profile_trace"] == TRACE


@pytest.mark.parametrize(
    ("profiles", "analyses", "profile_row", "analysis_row", "reason"),
    [
        (
            [{"status": "succeeded"}],
            [],
            ("failed", "profile_no_trace", "no_trace"),
            None,
            "re-profile produced no usable trace",
        ),
        (
            [_profile_ok(CB_TRACE)],
            [{"status": "failed", "error": "tl said no"}],
            ("succeeded", None, None),
            ("failed", "trace_analyze", "compute_bound_reanalyze"),
            "re-profile produced no usable trace",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_compute_bound_reprofile_that_fails_still_rows_its_attempts(
    tmp_path, monkeypatch, profiles, analyses, profile_row, analysis_row, reason
):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    state = SharedState()
    result, action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok(), *profiles],
        [_ta_ok(md, host_bound=True), *analyses],
        multi_node=True,
        state=state,
    )

    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "succeeded", True, None, None),
        (2, PROFILE_ATTEMPT_COMPUTE_BOUND, *profile_row[:1], False, *profile_row[1:]),
    ]
    expected_analysis = [(1, ANALYSIS_ATTEMPT_INITIAL, "succeeded", True, None, None)]
    if analysis_row is not None:
        expected_analysis.append((2, ANALYSIS_ATTEMPT_COMPUTE_BOUND, analysis_row[0], False, *analysis_row[1:]))
    assert _rows(action, "analysis") == expected_analysis
    assert action["analysis"]["compute_bound_reprofile"] == {"attempted": True, "adopted": False, "reason": reason}
    assert result["status"] == "succeeded"
    assert result["last_profile_trace"] == TRACE
    assert state.last_profile_trace == state.last_trace_analyze["trace_input"] == TRACE


@pytest.mark.parametrize(
    ("profiles", "analyses", "profile_row", "analysis_row", "message"),
    [
        ([RuntimeError("cb boot")], [], ("failed", "profile", "RuntimeError"), None, "cb boot"),
        (
            [_profile_ok(CB_TRACE)],
            [RuntimeError("tl")],
            ("succeeded", None, None),
            ("failed", "trace_analyze", "RuntimeError"),
            "tl",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_compute_bound_reprofile_that_raises_crashes_with_its_attempts_rowed(
    tmp_path, monkeypatch, profiles, analyses, profile_row, analysis_row, message
):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"^{message}$"):
        await _run(
            tmp_path, monkeypatch, [_profile_ok(), *profiles], [_ta_ok(md, host_bound=True), *analyses], multi_node=True
        )

    event = next(e for e in read_timeline_events(tmp_path) if e.get("type") == "roofline")
    (action,) = event["ext"]["actions"]
    assert _rows(action, "profile") == [
        (1, PROFILE_ATTEMPT_INITIAL, "succeeded", True, None, None),
        (2, PROFILE_ATTEMPT_COMPUTE_BOUND, *profile_row[:1], False, *profile_row[1:]),
    ]
    expected_analysis = [(1, ANALYSIS_ATTEMPT_INITIAL, "succeeded", False, None, None)]
    if analysis_row is not None:
        expected_analysis.append((2, ANALYSIS_ATTEMPT_COMPUTE_BOUND, analysis_row[0], False, *analysis_row[1:]))
    assert _rows(action, "analysis") == expected_analysis


@pytest.mark.asyncio
async def test_a_capture_marker_only_in_the_server_log_fails_the_attempt(tmp_path, monkeypatch):
    """The engine's server.log is evidence too: a marker found only there still ends the action."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "server.log").write_text(
        "boot\nRuntimeError: operation not permitted when stream is capturing\n", encoding="utf-8"
    )
    failed = {
        "status": "failed",
        "error_class": "server_crashed",
        "error": "engine exited",
        "workspace": str(workspace),
    }
    result, action = await _run(tmp_path, monkeypatch, [failed], [])

    assert _rows(action, "profile") == [(1, PROFILE_ATTEMPT_INITIAL, "failed", False, "profile", "server_crashed")]
    assert result["error_class"] == "profile_cuda_graph_capture_config_failed"
    diagnosis = tmp_path / "diagnostics" / "roofline_cuda_graph_capture_t-rows.json"
    assert "server_log_tail" in json.loads(diagnosis.read_text(encoding="utf-8"))["sources"]


@pytest.mark.asyncio
async def test_the_graph_capture_override_reaches_the_profiled_args_and_its_row(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    seen: list[dict[str, Any]] = []
    result, action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok()],
        [_ta_ok(md, hot=1)],
        disable_cuda_graph="1",
        profile_params=seen,
    )

    assert result["status"] == "succeeded"
    assert [row["disable_cuda_graph"] for row in action["profile"]["runs"]] == [True]
    assert action["profile"]["graph_capture_disabled"] is True
    assert len(seen) == 1
    assert "--disable-cuda-graph" in seen[0]["base_extra_args"]


@pytest.mark.asyncio
async def test_under_the_graph_capture_override_a_capture_marker_is_a_plain_failure(tmp_path, monkeypatch):
    """With capture already off, a capture marker says nothing about this run."""
    marker = "operation not permitted when stream is capturing"
    result, action = await _run(tmp_path, monkeypatch, [RuntimeError(marker)], [], disable_cuda_graph="1")

    assert _rows(action, "profile") == [(1, PROFILE_ATTEMPT_INITIAL, "failed", False, "profile", "RuntimeError")]
    assert result["phase"] == "profile"
    assert result["error"] == f"profile_executor raised: RuntimeError({marker!r})"
    assert not (tmp_path / "diagnostics").exists()


@pytest.mark.asyncio
async def test_the_n26_retry_reissues_the_initial_request_with_only_the_mode_changed(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    payloads: list[dict[str, Any]] = []
    result, _action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok()],
        [_ta_empty_chunk(), _ta_ok(md, hot=1)],
        analysis_payloads=payloads,
    )

    assert result["status"] == "succeeded"
    initial, retry = payloads
    assert initial["roofline_arm"] == "current_best"
    assert initial["roofline_output_name"] == "kernel_roofline_current.json"
    assert retry == {**initial, "steady_state_mode": "decode_only"}


@pytest.mark.asyncio
async def test_a_compute_bound_reanalysis_that_returns_no_dict_rows_why(tmp_path, monkeypatch):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    result, action = await _run(
        tmp_path,
        monkeypatch,
        [_profile_ok(), _profile_ok(CB_TRACE)],
        [_ta_ok(md, host_bound=True), "nope"],
        multi_node=True,
    )

    assert _rows(action, "analysis")[-1] == (
        2,
        ANALYSIS_ATTEMPT_COMPUTE_BOUND,
        "failed",
        False,
        "trace_analyze",
        "compute_bound_reanalyze",
    )
    assert _messages(action, "analysis")[-1] == "non-dict result: str"
    assert result["last_profile_trace"] == TRACE
