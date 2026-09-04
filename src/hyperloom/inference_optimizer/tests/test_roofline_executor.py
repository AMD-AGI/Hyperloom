# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Roofline-v2 N2b: real RooflineExecutor orchestration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hyperloom.orchestrator.actions.executors import roofline as roofline_mod
from hyperloom.orchestrator.actions.executors.roofline import (
    RooflineExecutor,
    _extract_trace_path,
    _failed,
    make_roofline_executor,
)
from hyperloom.orchestrator.trace.task_progress import progress_scope
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


def _ctx(tmp_path: Path | None = None) -> RunnerContext:
    task = Task(
        task_id="t-roofline-1",
        kind="roofline",
        state="running",
        params={"base_extra_args": "--mem-fraction-static=0.92"},
        idempotency_key="roofline:t-1",
        requires_lanes=["profile_lane"],
    )
    extra = {}
    if tmp_path is not None:
        extra["session_dir"] = str(tmp_path)
    return RunnerContext(task=task, lease=None, extra=extra)


def _state() -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    return s


def _profile_success(trace_path: str = "/tmp/trace.json.gz") -> dict:
    return {
        "status": "succeeded",
        "main_trace_path": trace_path,
        "workspace": "/tmp/workspace",
        "output_throughput": 110.0,
    }


def _trace_analyze_success(*, snapshot_id_in_state: int = 1) -> dict:
    """`trace_analyze_handler` success result shape."""
    return {
        "status": "ok",
        "candidates_path": "/tmp/kc.json",
        "trace_report_path": "/tmp/analysis.md",
        "artifact_paths": {
            "tracelens_steady_state_trace": "/tmp/mixed_steady_state.trace.json.gz",
        },
        "hot_kernels": [],
        "trace_health_warnings": [],
    }


def _patch_subs(profile_result, ta_result):
    """Return patches for both sub-step callables."""

    async def fake_profile(ctx):
        if isinstance(profile_result, Exception):
            raise profile_result
        return profile_result

    async def fake_ta(payload, *, session_dir):
        if isinstance(ta_result, Exception):
            raise ta_result
        return ta_result

    return patch(
        "hyperloom.orchestrator.actions.executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
        new=fake_ta,
    )


@pytest.mark.asyncio
async def test_happy_path_promotes_profile_and_caches_trace_analyze(tmp_path):
    state = _state()
    state.cumulative_gain_validated = 2.5
    ctx = _ctx(tmp_path)

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 51%, Idle 48%\n", encoding="utf-8")
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)
    ta["kernel_roofline_path"] = "/tmp/reports/kernel_roofline.json"

    p1, p2 = _patch_subs(_profile_success("/tmp/trace.gz"), ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["snapshot_id"] == 1
    assert result["last_profile_trace"] == "/tmp/trace.gz"
    assert result["analysis_md_path"] == str(md)
    assert result["kernel_roofline_path"] == "/tmp/reports/kernel_roofline.json"
    assert result["profile_workspace"] == "/tmp/workspace"
    assert "executed_at_iso" in result

    assert state.last_profile_trace == "/tmp/trace.gz"
    assert state.last_profile_status == "succeeded"
    assert state.last_profile_args == "--mem-fraction-static=0.92"
    assert state.last_profile_workload == state.profile_workload_context(ctx.task.params)
    assert state.last_trace_analyze["steady_state_trace"] == ("/tmp/mixed_steady_state.trace.json.gz")
    assert result["steady_state_trace"] == "/tmp/mixed_steady_state.trace.json.gz"
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(md)
    assert cached["kernel_roofline_path"] == "/tmp/reports/kernel_roofline.json"
    assert "Executive Summary" in cached["analysis_md_text"]
    assert cached["roofline_snapshot_id"] == 1
    assert cached["roofline_baseline_gain_at_snapshot"] == 2.5


@pytest.mark.asyncio
async def test_roofline_passes_resolved_framework_to_trace_analysis(tmp_path, monkeypatch):
    """The trace-analysis payload carries the same framework as the profile."""
    state = _state()
    state.framework = "xdit"
    captured: list[dict] = []

    async def fake_profile(_ctx):
        return _profile_success("/tmp/xdit.trace.json.gz")

    async def fake_trace_analyze(payload, *, session_dir):
        captured.append(dict(payload))
        return _trace_analyze_success()

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.profile.profile_executor",
        fake_profile,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
        fake_trace_analyze,
    )

    result = await RooflineExecutor(shared_state=state)(_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert captured[0]["framework"] == "xdit"


@pytest.mark.asyncio
async def test_profile_retry_records_successful_child_runtime(tmp_path, monkeypatch):
    state = _state()
    state.framework = "vllm"
    ctx = _ctx(tmp_path)
    calls = 0

    async def fake_profile(profile_ctx):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "status": "failed",
                "error": "Capture cuda graph failed",
            }
        effective_args = str(profile_ctx.task.params.get("base_extra_args") or "")
        profile_ctx.task.params["extra_server_args"] = effective_args
        return _profile_success("/tmp/eager.trace.json.gz")

    async def fake_trace_analyze(payload, *, session_dir):
        return _trace_analyze_success()

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.profile.profile_executor",
        fake_profile,
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
        fake_trace_analyze,
    )

    result = await RooflineExecutor(shared_state=state)(ctx)

    assert result["status"] == "succeeded"
    assert calls == 2
    assert "--enforce-eager" in state.last_profile_workload["server_args"]
    assert state.last_profile_args == state.last_profile_workload["server_args"]


@pytest.mark.asyncio
async def test_happy_path_increments_snapshot_id_on_re_run(tmp_path):
    """Second roofline run on the same session bumps snapshot_id."""
    state = _state()
    ctx = _ctx(tmp_path)
    md = tmp_path / "a.md"
    md.write_text("first", encoding="utf-8")
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)

    p1, p2 = _patch_subs(_profile_success("/tmp/t1.gz"), ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result1 = await executor(ctx)
    assert result1["snapshot_id"] == 1

    md.write_text("second", encoding="utf-8")
    state.cumulative_gain_validated = 4.0
    p1b, p2b = _patch_subs(_profile_success("/tmp/t2.gz"), ta)
    with p1b, p2b:
        result2 = await executor(ctx)
    assert result2["snapshot_id"] == 2
    assert state.last_trace_analyze["roofline_baseline_gain_at_snapshot"] == 4.0


@pytest.mark.asyncio
async def test_profile_failed_does_not_mutate_shared_state(tmp_path):
    state = _state()
    state.last_profile_trace = "/old/trace.gz"
    state.last_trace_analyze = {"analysis_md_text": "old", "roofline_snapshot_id": 5}
    ctx = _ctx(tmp_path)

    profile_failed = {
        "status": "failed",
        "error": "magpie exited 1",
        "error_class": "subprocess_error",
    }
    p1, p2 = _patch_subs(profile_failed, _trace_analyze_success())
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_failed"
    assert result["phase"] == "profile"
    assert "magpie exited 1" in result["error"]
    assert result["sub_result"]["status"] == "failed"

    assert state.last_profile_trace == "/old/trace.gz"
    assert state.last_trace_analyze["roofline_snapshot_id"] == 5


@pytest.mark.asyncio
async def test_profile_failed_with_trace_continues_to_trace_analyze(tmp_path):
    """Duplicate stop_profile failures are non-fatal once a trace exists."""
    state = _state()
    ctx = _ctx(tmp_path)
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nRecovered trace\n", encoding="utf-8")
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)
    profile_failed_with_trace = {
        "status": "failed",
        "error_class": "subprocess_nonzero",
        "error": "RuntimeError: Profiling is not in progress. Call /start_profile first.",
        "main_trace_path": "/tmp/recovered.trace.json.gz",
        "workspace": "/tmp/workspace",
    }

    p1, p2 = _patch_subs(profile_failed_with_trace, ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["last_profile_trace"] == "/tmp/recovered.trace.json.gz"
    assert result["profile_recovered"] is True
    assert result["profile_warning"]["error_class"] == "subprocess_nonzero"
    assert "Profiling is not in progress" in result["profile_warning"]["error"]
    assert state.last_profile_trace == "/tmp/recovered.trace.json.gz"
    assert state.last_profile_status == "succeeded"
    assert state.last_trace_analyze["analysis_md_path"] == str(md)


@pytest.mark.asyncio
async def test_profile_failed_without_trace_never_calls_trace_analyze(tmp_path):
    """failed + no trace fields must stay profile_failed, never analyze a trace.

    Asserts trace_analyze is never reached and the canonical profile_failed
    shape is returned.
    """
    state = _state()
    state.last_profile_trace = "/old/trace.gz"
    ctx = _ctx(tmp_path)
    profile_failed_no_trace = {
        "status": "failed",
        "error_class": "no_trace_files",
        "error": "no .trace.json.gz under /tmp/workspace",
        "main_trace_path": None,
        "trace_files": [],
        "workspace": "/tmp/workspace",
    }
    ta_should_not_run = AssertionError("trace_analyze must not run without trace")

    p1, p2 = _patch_subs(profile_failed_no_trace, ta_should_not_run)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_failed"
    assert result["phase"] == "profile"
    assert "profile_recovered" not in result
    assert state.last_profile_trace == "/old/trace.gz"


@pytest.mark.asyncio
async def test_primary_rank_missing_is_not_retried(tmp_path):
    calls = 0

    async def fake_profile(_ctx):
        nonlocal calls
        calls += 1
        return {
            "status": "failed",
            "error_class": "primary_rank_trace_missing",
            "error": "no rank-0 trace",
            "trace_input_ready": False,
            "trace_files": ["/tmp/merged.trace.json.gz"],
        }

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            side_effect=AssertionError("trace_analyze must not run"),
        ),
    ):
        result = await executor(_ctx(tmp_path))

    assert calls == 1
    assert result["status"] == "failed"
    assert "no rank-0 trace" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "api_port_allocation_failed",
        "capture_status_missing",
        "profiler_output_unconfigured",
    ],
)
async def test_deterministic_capture_failures_are_not_retried(tmp_path, reason):
    calls = 0

    async def fake_profile(_ctx):
        nonlocal calls
        calls += 1
        return {
            "status": "failed",
            "error_class": "profile_capture_failed",
            "error": reason,
            "trace_input_ready": False,
            "trace_capture": {"status": "failed", "reason": reason},
        }

    executor = RooflineExecutor(shared_state=_state())
    with patch(
        "hyperloom.orchestrator.actions.executors.profile.profile_executor",
        new=fake_profile,
    ):
        result = await executor(_ctx(tmp_path))

    assert calls == 1
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_profile_no_trace_path(tmp_path):
    """Profile succeeded but result lacks main_trace_path / trace_files."""
    state = _state()
    ctx = _ctx(tmp_path)
    profile_bad = {"status": "succeeded", "output_throughput": 110.0}
    p1, p2 = _patch_subs(profile_bad, _trace_analyze_success())
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_no_trace_failed"
    assert "no trace_path" in result["error"]
    assert state.last_profile_trace == ""


@pytest.mark.asyncio
async def test_profile_raises_exception(tmp_path):
    state = _state()
    ctx = _ctx(tmp_path)
    p1, p2 = _patch_subs(RuntimeError("boom"), _trace_analyze_success())
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_failed"
    assert "boom" in result["error"]
    assert "raised" in result["error"]


@pytest.mark.asyncio
async def test_profile_returns_non_dict(tmp_path):
    state = _state()
    ctx = _ctx(tmp_path)
    p1, p2 = _patch_subs("garbage", _trace_analyze_success())
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "profile_failed"
    assert "non-dict" in result["error"]


@pytest.mark.asyncio
async def test_trace_analyze_failed_keeps_profile_promote(tmp_path):
    """trace_analyze failure keeps last_profile_trace but leaves last_trace_analyze empty."""
    state = _state()
    ctx = _ctx(tmp_path)
    ta_failed = {"status": "failed", "error": "tracelens crashed"}

    p1, p2 = _patch_subs(_profile_success("/tmp/new.gz"), ta_failed)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "trace_analyze_failed"
    assert result["phase"] == "trace_analyze"

    assert state.last_profile_trace == "/tmp/new.gz"
    assert state.last_profile_status == "succeeded"
    assert state.last_trace_analyze == {}


@pytest.mark.asyncio
async def test_trace_analyze_raises_exception(tmp_path):
    state = _state()
    ctx = _ctx(tmp_path)
    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), ValueError("bad payload"))
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "trace_analyze_failed"
    assert "bad payload" in result["error"]
    assert state.last_profile_trace == "/tmp/t.gz"


@pytest.mark.asyncio
async def test_trace_analyze_returns_non_dict(tmp_path):
    state = _state()
    ctx = _ctx(tmp_path)
    p1, p2 = _patch_subs(_profile_success("/tmp/t.gz"), 42)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)
    assert result["status"] == "failed"
    assert result["error_class"] == "trace_analyze_failed"
    assert "non-dict" in result["error"]


def test_extract_trace_path_prefers_main_trace_path():
    r = {
        "main_trace_path": "/main.gz",
        "trace_files": ["/alt-0.gz", "/alt-1.gz"],
    }
    assert _extract_trace_path(r) == "/main.gz"


def test_extract_trace_path_falls_back_to_first_trace_file():
    r = {"trace_files": ["/first.gz", "/second.gz"]}
    assert _extract_trace_path(r) == "/first.gz"


def test_extract_trace_path_refuses_explicitly_unready_trace():
    r = {
        "trace_input_ready": False,
        "main_trace_path": "/merged.gz",
        "trace_files": ["/merged.gz"],
    }
    assert _extract_trace_path(r) == ""


def test_extract_trace_path_empty_when_both_missing():
    assert _extract_trace_path({}) == ""
    assert _extract_trace_path({"trace_files": []}) == ""
    assert _extract_trace_path({"trace_files": [None]}) == ""


def test_extract_trace_path_handles_non_dict():
    assert _extract_trace_path(None) == ""  # type: ignore[arg-type]
    assert _extract_trace_path("garbage") == ""  # type: ignore[arg-type]


def test_failed_helper_constructs_canonical_shape():
    f = _failed("profile", "boom")
    assert f["status"] == "failed"
    assert f["error_class"] == "profile_failed"
    assert f["error"] == "boom"
    assert f["phase"] == "profile"
    assert "executed_at_iso" in f
    assert "sub_result" not in f

    f2 = _failed("trace_analyze", "x", sub_result={"status": "failed", "error": "y", "extra": "ignored"})
    assert f2["sub_result"] == {"status": "failed", "error": "y"}
    assert "extra" not in f2["sub_result"]


def test_make_roofline_executor_requires_shared_state():
    with pytest.raises(ValueError, match="requires a SharedState"):
        RooflineExecutor(shared_state=None)


def test_make_roofline_executor_factory_signature():
    state = SharedState()
    exe = make_roofline_executor(shared_state=state)
    assert isinstance(exe, RooflineExecutor)
    assert exe.shared_state is state


def test_wrap_profile_ctx_creates_child_task():
    state = SharedState()
    exe = RooflineExecutor(shared_state=state)
    parent = _ctx()
    parent.extra["session_dir"] = "/sess"
    child = exe._wrap_profile_ctx(parent)
    assert child.task.kind == "profile"
    assert child.task.task_id == "t-roofline-1-profile"
    assert child.task.idempotency_key == "roofline:t-1-profile"
    assert child.task.state == "running"
    assert child.task.params == {"base_extra_args": "--mem-fraction-static=0.92"}
    assert child.extra["session_dir"] == "/sess"
    assert child.lease is parent.lease


def test_resolve_session_dir_handles_missing_extra():
    state = SharedState()
    exe = RooflineExecutor(shared_state=state)
    ctx = _ctx()
    assert exe._resolve_session_dir(ctx) == Path(".")

    ctx.extra["session_dir"] = "/abc"
    assert exe._resolve_session_dir(ctx) == Path("/abc")


# N10: Coordinator persists roofline executor SharedState mutations to state.json.
import json

from hyperloom.orchestrator.roles import (
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import (
    _AUDIT_ACTIONS as SHARED_STATE_AUDIT_ACTIONS,
    _KEY_METRIC_MAP,
)
from hyperloom.inference_optimizer.session.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }


def _roofline_task(snapshot_id: int = 1) -> Task:
    return Task(
        task_id="t-rf-1",
        kind="roofline",
        state="running",
        params={"base_extra_args": "--mem-fraction-static=0.9"},
        idempotency_key=f"roofline-tick5-initial-{snapshot_id}",
    )


def _roofline_result(snapshot_id: int = 1) -> dict:
    """Mirrors RooflineExecutor's success result."""
    return {
        "status": "succeeded",
        "executed_at_iso": "2026-05-19T15:55:00+00:00",
        "snapshot_id": snapshot_id,
        "last_profile_trace": "/sessions/abc/runs/roofline/.../trace.json.gz",
        "analysis_md_path": "/sessions/abc/kernel-agent/runs/.../analysis.md",
        "profile_workspace": "/sessions/abc/runs/roofline/.../benchmark_sglang_...",
    }


def test_shared_state_has_roofline_audit_fields_by_default():
    """Both audit mirrors are declared fields, so both survive a save."""
    s = SharedState()
    assert s.roofline_attempts == []
    assert s.last_roofline == {}
    assert {"roofline_attempts", "last_roofline"} <= set(s.to_dict())


def test_audit_actions_includes_roofline():
    """One set gates both the recorder and writeback's failed-attempt row."""
    from hyperloom.orchestrator.loop import writeback as wb

    assert "roofline" in SHARED_STATE_AUDIT_ACTIONS
    assert wb._AUDIT_ACTIONS is SHARED_STATE_AUDIT_ACTIONS


def test_key_metric_map_has_roofline_snapshot_id():
    assert "roofline" in _KEY_METRIC_MAP
    key, label = _KEY_METRIC_MAP["roofline"]
    assert key == "snapshot_id"
    assert label == "snapshot_id"


def test_record_action_attempt_populates_last_roofline_and_history():
    s = SharedState()
    s.record_action_attempt(
        action="roofline",
        task_id="t-rf-1",
        status="succeeded",
        decision="promoted",
        result={"snapshot_id": 3},
        extras={"analysis_md_path": "/p/analysis.md"},
    )
    assert s.last_roofline
    assert s.last_roofline.get("status") == "succeeded"
    assert s.last_roofline.get("decision") == "promoted"
    assert s.last_roofline.get("key_metric") == 3
    assert s.last_roofline.get("key_metric_kind") == "snapshot_id"
    assert s.last_roofline.get("extras", {}).get("analysis_md_path") == "/p/analysis.md"
    assert len(s.roofline_attempts) == 1


def test_record_action_attempt_caps_roofline_history():
    s = SharedState()
    for i in range(25):
        s.record_action_attempt(
            action="roofline",
            task_id=f"t-{i}",
            status="succeeded",
            decision="promoted",
            result={"snapshot_id": i},
        )
    assert len(s.roofline_attempts) == 20
    assert s.roofline_attempts[-1].get("task_id") == "t-24"


@pytest.mark.asyncio
async def test_promote_roofline_flips_changed_and_saves(session_dir):
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state

    s.last_profile_trace = "/sessions/abc/runs/roofline/.../trace.json.gz"
    s.last_profile_status = "succeeded"
    s.last_profile_args = "--mem-fraction-static=0.9"
    s.last_trace_analyze = {
        "trace_input": "/sessions/abc/runs/roofline/.../trace.json.gz",
        "analysis_md_path": "/sessions/abc/.../analysis.md",
        "analysis_md_text": "# Executive Summary\nCompute 51%\n",
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
    }

    s.save(session_dir)
    pre = json.loads((session_dir / "state.json").read_text())
    assert pre.get("last_profile_trace", "") == s.last_profile_trace

    s.last_profile_trace = "/sessions/abc/.../NEW_trace.gz"
    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=1),
        task=_roofline_task(),
    )
    post = json.loads((session_dir / "state.json").read_text())
    assert post.get("last_profile_trace") == "/sessions/abc/.../NEW_trace.gz", (
        "N10 _promote_to_shared_state 'roofline' branch must trigger "
        "the tail-save; otherwise the post-promote state.json would "
        "still show the pre-mutate trace path"
    )


@pytest.mark.asyncio
async def test_promote_roofline_records_audit_attempt(session_dir):
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state
    s.last_trace_analyze = {
        "analysis_md_path": "/p/analysis.md",
        "roofline_snapshot_id": 1,
    }
    s.last_profile_trace = "/t/trace.gz"

    assert s.roofline_attempts == []
    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=1),
        task=_roofline_task(),
    )
    assert len(s.roofline_attempts) == 1, "roofline must enter _AUDIT_ACTIONS so record_action_attempt fires"
    attempt = s.roofline_attempts[-1]
    assert attempt.get("status") == "succeeded"
    assert attempt.get("decision") == "promoted"
    assert attempt.get("task_id") == "t-rf-1"
    extras = attempt.get("extras") or {}
    assert extras.get("snapshot_id") == 1
    assert extras.get("analysis_md_path") == "/p/analysis.md"


@pytest.mark.asyncio
async def test_promote_roofline_does_not_remutate_state(session_dir):
    coord = Coordinator(session_dir, backends=_silent_backends())
    s = coord.shared_state
    s.last_profile_trace = "/before/trace.gz"
    s.last_trace_analyze = {
        "analysis_md_text": "before report",
        "roofline_snapshot_id": 3,
    }
    s.last_profile_status = "succeeded"

    await coord._promote_to_shared_state(
        "roofline",
        _roofline_result(snapshot_id=3),
        task=_roofline_task(snapshot_id=3),
    )
    assert s.last_profile_trace == "/before/trace.gz"
    assert s.last_trace_analyze["analysis_md_text"] == "before report"
    assert s.last_trace_analyze["roofline_snapshot_id"] == 3
    assert s.last_profile_status == "succeeded"


@pytest.mark.asyncio
async def test_promote_roofline_non_dict_result_short_circuits(session_dir):
    coord = Coordinator(session_dir, backends=_silent_backends())
    await coord._promote_to_shared_state(
        "roofline",
        None,
        task=_roofline_task(),  # type: ignore[arg-type]
    )
    assert coord.shared_state.roofline_attempts == []


# N11: strip base64 image data URLs from analysis.md.
def test_strip_passes_text_through_when_no_base64_url():
    md = "# Analysis\n\nNo images here, just text.\nSome markdown link: [foo](https://example.com/bar)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert out == md


def test_strip_empty_input_returns_empty():
    assert SharedState._strip_base64_data_urls("") == ""


def test_strip_handles_none_gracefully():
    assert SharedState._strip_base64_data_urls(None) is None or SharedState._strip_base64_data_urls(None) == ""


def test_strip_replaces_data_image_png_payload():
    md = (
        "## Section\n"
        "![Performance Improvement](data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAACB8AAAN8CAYAAAAa5)\n"
        "Trailing text\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/png;base64" not in out
    assert "iVBORw0KGgo" not in out
    assert "stripped" in out.lower()
    assert "Performance Improvement" in out
    assert "Trailing text" in out
    assert "## Section" in out


def test_strip_replaces_data_image_jpeg_too():
    md = "![Foo](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "stripped" in out.lower()
    assert "Foo" in out


def test_strip_replaces_data_image_svg_xml_base64():
    md = "![chart](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "chart" in out


def test_strip_replaces_multiple_images_independently():
    md = "![a](data:image/png;base64,AAA)\ntext\n![b](data:image/png;base64,BBB)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "AAA" not in out and "BBB" not in out
    assert out.count("stripped") == 2
    assert "stripped: base64 image — a" in out
    assert "stripped: base64 image — b" in out


def test_strip_does_not_touch_regular_image_urls():
    md = "![chart](https://example.com/perf.png)\n![fig](./local/figure.svg)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert out == md


def test_strip_preserves_empty_alt_text():
    md = "![](data:image/png;base64,iVBOR)\n"
    out = SharedState._strip_base64_data_urls(md)
    assert "data:image/" not in out
    assert "image" in out


def test_strip_reduces_real_r1_report_by_90pct_plus():
    import os

    p = "/shared/user1/sessions/kernel-agent/runs/sessions/tracelens/analysis.md"
    if not os.path.exists(p):
        pytest.skip(f"sample analysis.md not present at {p}")
    with open(p, encoding="utf-8") as f:
        md = f.read()
    if "data:image/" not in md:
        pytest.skip(
            "sample analysis.md no longer contains base64 data URLs "
            "(file overwritten by a later session); can't validate "
            "the ≥80% reduction claim without the original artefact"
        )
    before = len(md)
    stripped = SharedState._strip_base64_data_urls(md)
    after = len(stripped)
    reduction_pct = (1 - after / before) * 100
    assert reduction_pct > 80, (
        f"N11 must reduce real R1 analysis.md by ≥80%; got before={before} after={after} reduction={reduction_pct:.1f}%"
    )
    assert "Executive Summary" in stripped
    assert "Idle %" in stripped
    assert "fmoe_fp8_blockscale_g1u1" in stripped or "MoE" in stripped


def test_format_analysis_md_full_strips_base64_before_injection(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE", "1")
    s = SharedState()
    s.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
        "analysis_md_text": (
            "# Test\n\n"
            "## Executive Summary\nGPU idle 64%\n"
            "![chart](data:image/png;base64,iVBORw0KGgoAAAANSUhEUg)\n"
            "## Recommendations\nP1: kernel_opt fmoe\n"
        ),
        "analysis_md_path": "/p/analysis.md",
    }
    rendered = s._format_analysis_md_full()
    assert "data:image/" not in rendered
    assert "iVBORw0KGgo" not in rendered
    assert "Executive Summary" in rendered
    assert "GPU idle 64%" in rendered
    assert "P1: kernel_opt fmoe" in rendered
    assert "=== TraceLens Analysis (snapshot #1" in rendered
    assert "=== End TraceLens Analysis ===" in rendered


def test_format_analysis_md_full_defaults_to_pointer_not_inline(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE", raising=False)
    s = SharedState()
    s.last_trace_analyze = {
        "roofline_snapshot_id": 7,
        "roofline_baseline_gain_at_snapshot": 3.0,
        "analysis_md_text": "# Report\nverbatim body text\n",
        "analysis_md_path": "/p/analysis.md",
    }
    rendered = s._format_analysis_md_full()
    assert "verbatim body text" not in rendered
    assert "show_analysis_md" in rendered
    assert "snapshot #7" in rendered


def test_format_analysis_md_full_no_strip_when_no_image(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE", "1")
    s = SharedState()
    s.last_trace_analyze = {
        "roofline_snapshot_id": 1,
        "roofline_baseline_gain_at_snapshot": 0.0,
        "analysis_md_text": "# Report\nText only\n",
        "analysis_md_path": "/p/analysis.md",
    }
    rendered = s._format_analysis_md_full()
    assert "# Report" in rendered
    assert "Text only" in rendered


def test_strip_keeps_surrounding_markdown_intact():
    md = (
        "# H1\n\n"
        "## H2\n"
        "Some text\n\n"
        "![chart](data:image/png;base64,xxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\n\n"
        "## H2-2\n"
        "| col | val |\n"
        "|-----|-----|\n"
        "| a   | 1   |\n\n"
        "```python\nprint('hi')\n```\n"
    )
    out = SharedState._strip_base64_data_urls(md)
    assert "# H1" in out
    assert "## H2" in out
    assert "## H2-2" in out
    assert "| col | val |" in out
    assert "```python" in out
    assert "print('hi')" in out
    assert "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in out


# N26: RooflineExecutor auto-retries trace_analyze on
# steady_state_chunk_{empty,missing} with an alternate mode.
from hyperloom.orchestrator.actions.executors.roofline import (
    _extract_steady_state_retry_mode,
)


def _n26_ctx(tmp_path: Path) -> RunnerContext:
    task = Task(
        task_id="t-n26-1",
        kind="roofline",
        state="running",
        params={"base_extra_args": ""},
        idempotency_key="roofline:n26-1",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


def _profile_ok(trace: str = "/tmp/t.gz") -> dict:
    return {
        "status": "succeeded",
        "main_trace_path": trace,
        "workspace": "/tmp/wsp",
        "output_throughput": 50.0,
    }


def _ta_empty_chunk_failure(
    *,
    requested: str = "mixed",
    non_empty: list[str] | None = None,
) -> dict:
    return {
        "status": "failed",
        "error": (
            f"RuntimeError: steady_state_chunk_empty: requested "
            f"--steady-state-mode={requested} but the selected chunk has "
            "zero GPU events"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_empty",
                "severity": "blocking",
                "requested_mode": requested,
                "selected_chunk": f"/tmp/{requested}.json.gz",
                "num_gpu_events": 0,
                "gpu_busy_duration": 0.0,
                "non_empty_modes": list(non_empty if non_empty is not None else []),
                "remediation": "Re-issue with one of non_empty_modes.",
                "message": "stub",
            },
        ],
    }


def _ta_missing_chunk_failure(
    *,
    requested: str = "decode_only",
    available: list[str] | None = None,
) -> dict:
    return {
        "status": "failed",
        "error": (
            f"RuntimeError: steady_state_chunk_missing: requested "
            f"--steady-state-mode={requested} but splitter produced no "
            "matching chunk"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_missing",
                "severity": "blocking",
                "requested_mode": requested,
                "requested_chunk_label": f"{requested}_steady_state",
                "available_modes": list(available if available is not None else []),
                "remediation": "Re-issue with one of available_modes.",
                "trace_input": "/tmp/raw.trace.json.gz",
                "split_dir": "/tmp/split",
            },
        ],
    }


def _ta_ok(*, report_md: Path) -> dict:
    return {
        "status": "ok",
        "candidates_path": "/tmp/kc.json",
        "trace_report_path": str(report_md),
        "hot_kernels": [],
        "trace_health_warnings": [],
    }


def _run_roofline_captured_payload(tmp_path, *, reason: str) -> dict:
    """Run RooflineExecutor with stubbed profile/trace_analyze; return the
    payload passed to record_trace_analyze."""
    import asyncio

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nbody\n", encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_profile(ctx):
        return _profile_ok()

    async def fake_ta(payload, *, session_dir):
        return _ta_ok(report_md=md)

    state = _state()
    orig_record = state.record_trace_analyze

    def record_spy(payload, result):
        captured["payload"] = dict(payload)
        return orig_record(payload, result)

    state.record_trace_analyze = record_spy  # type: ignore[assignment]

    task = Task(
        task_id=f"t-{reason}-1",
        kind="roofline",
        state="running",
        params={"base_extra_args": "", "reason": reason},
        idempotency_key=f"roofline:{reason}-1",
        requires_lanes=["profile_lane"],
    )
    ctx = RunnerContext(
        task=task,
        lease=None,
        extra={"session_dir": str(tmp_path)},
    )

    executor = RooflineExecutor(shared_state=state)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = asyncio.run(executor(ctx))

    assert result["status"] == "succeeded"
    return captured["payload"]  # type: ignore[return-value]


def test_prelude_roofline_records_baseline_arm(tmp_path):
    """A prelude_initial roofline tags trace_analyze payload roofline_arm=baseline."""
    payload = _run_roofline_captured_payload(tmp_path, reason="prelude_initial")
    assert payload.get("roofline_arm") == "baseline"


def test_watermark_roofline_tags_current_best_arm(tmp_path):
    """A non-prelude roofline explicitly tags arm=current_best (no reliance on
    transient recorder inference)."""
    payload = _run_roofline_captured_payload(
        tmp_path,
        reason="explore_keep_watermark",
    )
    assert payload.get("roofline_arm") == "current_best"


def test_extract_picks_first_non_empty_mode():
    res = _ta_empty_chunk_failure(non_empty=["prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    mode, w = out
    assert mode == "prefilldecode"
    assert w["code"] == "steady_state_chunk_empty"


def test_extract_handles_missing_chunk_warning():
    res = _ta_missing_chunk_failure(available=["mixed", "prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    mode, w = out
    assert mode == "mixed"
    assert w["code"] == "steady_state_chunk_missing"


def test_extract_returns_none_when_no_alternate():
    res = _ta_empty_chunk_failure(non_empty=[])
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_returns_none_for_unrelated_warning():
    res = {
        "status": "failed",
        "trace_health_warnings": [
            {"code": "tracelens_analysis_failed", "severity": "warning"},
        ],
    }
    assert _extract_steady_state_retry_mode(res) is None


def test_extract_returns_none_for_empty_result():
    assert _extract_steady_state_retry_mode({}) is None
    assert _extract_steady_state_retry_mode({"status": "ok"}) is None
    assert (
        _extract_steady_state_retry_mode(  # type: ignore[arg-type]
            None
        )
        is None
    )


def test_extract_skips_blank_mode_entries():
    res = _ta_empty_chunk_failure(non_empty=["", "   ", "prefilldecode"])
    out = _extract_steady_state_retry_mode(res)
    assert out is not None
    assert out[0] == "prefilldecode"


def _n26_patch_subs(profile_result, ta_results, *, on_ta_call=None):
    ta_calls = {"n": 0, "payloads": []}

    async def fake_profile(ctx):
        return profile_result

    async def fake_ta(payload, *, session_dir):
        idx = ta_calls["n"]
        ta_calls["payloads"].append(dict(payload))
        ta_calls["n"] += 1
        if on_ta_call is not None:
            on_ta_call()
        if idx >= len(ta_results):
            return ta_results[-1]
        return ta_results[idx]

    return (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
        ta_calls,
    )


@pytest.mark.asyncio
async def test_auto_retry_succeeds_on_alternate_mode(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="mixed",
        non_empty=["prefilldecode"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert result["snapshot_id"] == 1
    assert calls["n"] == 2
    assert calls["payloads"][0].get("steady_state_mode") is None
    assert calls["payloads"][1]["steady_state_mode"] == "prefilldecode"
    assert calls["payloads"][1].get("_n26_auto_retry") is True
    assert calls["payloads"][1].get("_n26_retry_from_mode") == "mixed"
    assert state.last_profile_trace == "/tmp/t.gz"
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(md)
    assert cached["roofline_snapshot_id"] == 1


@pytest.mark.asyncio
async def test_auto_retry_succeeds_on_missing_chunk(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 80%, Idle 20%\n", encoding="utf-8")
    fail = _ta_missing_chunk_failure(
        requested="decode_only",
        available=["mixed", "prefilldecode"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["payloads"][1]["steady_state_mode"] == "mixed"
    assert calls["payloads"][1]["_n26_retry_from_mode"] == "decode_only"


@pytest.mark.asyncio
async def test_no_retry_when_no_alternate_modes(tmp_path):
    fail = _ta_empty_chunk_failure(requested="mixed", non_empty=[])
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result.get("phase") == "trace_analyze"
    assert "steady_state_chunk_empty" in str(result.get("error") or "")
    assert calls["n"] == 1
    assert state.last_trace_analyze == {}


@pytest.mark.asyncio
async def test_no_retry_on_unrelated_failure(tmp_path):
    fail = {
        "status": "failed",
        "error": "RuntimeError: TraceLens skill crashed",
        "trace_health_warnings": [
            {"code": "tracelens_analysis_failed", "severity": "warning"},
        ],
    }
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "failed"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_failure_propagates_without_third_attempt(tmp_path):
    fail1 = _ta_empty_chunk_failure(
        requested="mixed",
        non_empty=["prefilldecode"],
    )
    fail2 = {
        "status": "failed",
        "error": (
            "RuntimeError: steady_state_chunk_empty: requested "
            "--steady-state-mode=prefilldecode but the selected chunk has "
            "zero GPU events"
        ),
        "trace_health_warnings": [
            {
                "code": "steady_state_chunk_empty",
                "requested_mode": "prefilldecode",
                "non_empty_modes": ["decode_only"],
            },
        ],
    }
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail1, fail2])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "failed"
    assert calls["n"] == 2
    err = str(result.get("error") or "")
    assert "prefilldecode" in err


@pytest.mark.asyncio
async def test_retry_exception_propagates(tmp_path):
    fail = _ta_empty_chunk_failure(
        requested="mixed",
        non_empty=["prefilldecode"],
    )

    async def fake_profile(ctx):
        return _profile_ok()

    call_count = {"n": 0}

    async def fake_ta(payload, *, session_dir):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return fail
        raise RuntimeError("network unreachable")

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "failed"
    err = str(result.get("error") or "")
    assert "N26 auto-retry" in err
    assert "prefilldecode" in err


@pytest.mark.asyncio
async def test_retry_success_stamps_n26_metadata(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="mixed",
        non_empty=["prefilldecode"],
    )
    succ = _ta_ok(report_md=md)

    captured = {"ta_result": None}

    async def fake_profile(ctx):
        return _profile_ok()

    async def fake_ta(payload, *, session_dir):
        return fail if not payload.get("_n26_auto_retry") else succ

    state = _state()

    orig_record = state.record_trace_analyze

    def record_spy(payload, result):
        captured["ta_result"] = dict(result)
        return orig_record(payload, result)

    state.record_trace_analyze = record_spy  # type: ignore[assignment]

    executor = RooflineExecutor(shared_state=state)
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    stamp = (captured["ta_result"] or {}).get("n26_auto_retry") or {}
    assert stamp.get("applied") is True
    assert stamp.get("from_mode") == "mixed"
    assert stamp.get("to_mode") == "prefilldecode"
    assert stamp.get("source_warning_code") == "steady_state_chunk_empty"


@pytest.mark.asyncio
async def test_retry_works_when_operator_started_with_non_mixed(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 60%, Idle 40%\n", encoding="utf-8")
    fail = _ta_empty_chunk_failure(
        requested="prefilldecode",
        non_empty=["decode_only"],
    )
    succ = _ta_ok(report_md=md)
    p1, p2, calls = _n26_patch_subs(_profile_ok(), [fail, succ])

    state = _state()
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert calls["payloads"][1]["steady_state_mode"] == "decode_only"
    assert calls["payloads"][1]["_n26_retry_from_mode"] == "prefilldecode"


# cuda-graph folding -> trace_analyze ok but 0 hot kernels
@pytest.mark.asyncio
async def test_431_zero_hot_with_degraded_trace_appends_warning(tmp_path):
    """trace_analyze succeeds but returns 0 hot kernels because cuda-graph
    capture folded per-kernel activity; the executor appends a
    ``cuda_graph_attribution_degraded`` trace_health_warnings entry so the LLM
    re-profiles eager instead of reading top=[] as 'no optimizable kernels'."""
    state = _state()
    ctx = _ctx(tmp_path)
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 80%, Idle 18%\n", encoding="utf-8")

    profile = _profile_success("/tmp/trace.gz")
    profile["trace_health"] = {
        "per_kernel_attribution_degraded": True,
        "capture_traces_present": True,
        "issues": ["[3] main trace ... no execute_*/user_annotation ..."],
    }
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)

    p1, p2 = _patch_subs(profile, ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["kernel_attribution_degraded"] is True
    warnings = state.last_trace_analyze.get("trace_health_warnings") or []
    codes = [w.get("code") for w in warnings if isinstance(w, dict)]
    assert "cuda_graph_attribution_degraded" in codes, warnings
    w = next(w for w in warnings if w.get("code") == "cuda_graph_attribution_degraded")
    assert w["capture_traces_present"] is True
    assert "--enforce-eager" in w["message"]


@pytest.mark.asyncio
async def test_431_zero_hot_without_degraded_health_no_warning(tmp_path):
    """Healthy trace (per_kernel_attribution_degraded=False) that genuinely
    finds 0 hot kernels must NOT be mislabeled as cuda-graph degradation."""
    state = _state()
    ctx = _ctx(tmp_path)
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    profile = _profile_success("/tmp/trace.gz")
    profile["trace_health"] = {
        "per_kernel_attribution_degraded": False,
        "capture_traces_present": True,
        "issues": [],
    }
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)

    p1, p2 = _patch_subs(profile, ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["kernel_attribution_degraded"] is False
    warnings = state.last_trace_analyze.get("trace_health_warnings") or []
    codes = [w.get("code") for w in warnings if isinstance(w, dict)]
    assert "cuda_graph_attribution_degraded" not in codes


def _progress_sink(notes: list[dict]):
    """Build an ambient progress sink appending every note to ``notes``."""

    async def _sink(**note):
        notes.append(note)

    return _sink


@pytest.mark.asyncio
async def test_the_n26_retry_reports_before_it_runs(tmp_path):
    """The retry re-runs TraceLens; unannounced, it is indistinguishable from a hang."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    notes: list[dict] = []
    at_call: list[list[str]] = []
    fail = _ta_empty_chunk_failure(requested="mixed", non_empty=["prefilldecode"])
    p1, p2, _calls = _n26_patch_subs(
        _profile_ok(),
        [fail, _ta_ok(report_md=md)],
        on_ta_call=lambda: at_call.append([n["label"] for n in notes]),
    )

    executor = RooflineExecutor(shared_state=_state())
    with p1, p2, progress_scope(_progress_sink(notes)):
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert at_call == [
        ["profile", "trace_analyze"],
        ["profile", "trace_analyze", "trace_analyze_n26_retry"],
    ]
    assert all(n["unit"] == "roofline_step" and n["status"] == "started" for n in notes)


@pytest.mark.asyncio
async def test_the_compute_bound_reprofile_reports_both_of_its_steps(tmp_path, monkeypatch):
    """A second profile and a second analysis, silent end to end until now."""
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    host_bound = _ta_ok(report_md=md)
    host_bound["trace_health_warnings"] = [
        {"code": "high_gpu_idle_pct", "severity": "warning"},
    ]
    compute_bound = _ta_ok(report_md=md)
    compute_bound["hot_kernels"] = [{"kernel_id": "k001", "name": "fused_moe", "gpu_pct": 30.0}]
    notes: list[dict] = []
    p1, p2, _calls = _n26_patch_subs(_profile_ok(), [host_bound, compute_bound])
    monkeypatch.setattr(roofline_mod, "is_multi_node", lambda: True)

    executor = RooflineExecutor(shared_state=_state())
    with p1, p2, progress_scope(_progress_sink(notes)):
        result = await executor(_n26_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert [n["label"] for n in notes] == [
        "profile",
        "trace_analyze",
        "profile_compute_bound",
        "trace_analyze_compute_bound",
    ]


@pytest.mark.asyncio
async def test_a_raising_compute_bound_reprofile_still_rows_the_attempt(tmp_path, monkeypatch):
    """An attempt that raises must still appear in the event's ``runs``.

    The fail-soft handler wrapping this branch only narrates the outcome into a
    reason string, so an unrowed attempt leaves ``attempt_count`` short of what
    actually ran -- and the main retry loop does row its raising attempts.
    """
    from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
    from hyperloom.inference_optimizer.breakdown.recorder.roofline_event import PROFILE_ATTEMPT_COMPUTE_BOUND

    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")
    host_bound = _ta_ok(report_md=md)
    host_bound["trace_health_warnings"] = [{"code": "high_gpu_idle_pct", "severity": "warning"}]
    profile_calls = {"n": 0}

    async def fake_profile(ctx):
        profile_calls["n"] += 1
        if profile_calls["n"] == 1:
            return _profile_ok()
        raise RuntimeError("server boot failed on the compute-bound retry")

    async def fake_ta(payload, *, session_dir):
        return host_bound

    monkeypatch.setattr(roofline_mod, "is_multi_node", lambda: True)

    executor = RooflineExecutor(shared_state=_state())
    with (
        patch(
            "hyperloom.orchestrator.actions.executors.profile.profile_executor",
            new=fake_profile,
        ),
        patch(
            "hyperloom.orchestrator.kernel.request_handlers.trace_analyze_handler",
            new=fake_ta,
        ),
    ):
        result = await executor(_n26_ctx(tmp_path))

    # Fail-soft is preserved: the host-bound analysis remains the outcome.
    assert result["status"] == "succeeded"

    event = next(e for e in read_timeline_events(tmp_path) if e.get("type") == "roofline")
    action = event["ext"]["actions"][0]
    profile = action["profile"]
    assert profile["attempt_count"] == 2
    raised = profile["runs"][-1]
    assert raised["attempt_reason"] == PROFILE_ATTEMPT_COMPUTE_BOUND
    assert raised["status"] == "failed"
    assert "server boot failed" in raised["failure"]["message"]
    assert action["analysis"]["compute_bound_reprofile"]["adopted"] is False


@pytest.mark.asyncio
async def test_431_nonzero_hot_never_flags_degraded(tmp_path):
    """Even if trace_health says degraded, a non-empty hot_kernels list
    proves attribution worked — do NOT append the warning."""
    state = _state()
    ctx = _ctx(tmp_path)
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n", encoding="utf-8")

    profile = _profile_success("/tmp/trace.gz")
    profile["trace_health"] = {
        "per_kernel_attribution_degraded": True,
        "capture_traces_present": False,
        "issues": [],
    }
    ta = _trace_analyze_success()
    ta["hot_kernels"] = [
        {"kernel_id": "k001", "name": "fused_moe", "gpu_pct": 30.0},
    ]
    ta["trace_report_path"] = str(md)

    p1, p2 = _patch_subs(profile, ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["kernel_attribution_degraded"] is False
    warnings = state.last_trace_analyze.get("trace_health_warnings") or []
    codes = [w.get("code") for w in warnings if isinstance(w, dict)]
    assert "cuda_graph_attribution_degraded" not in codes
