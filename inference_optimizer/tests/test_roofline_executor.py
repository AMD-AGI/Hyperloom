"""Roofline-v2 N2b: real RooflineExecutor orchestration tests.

These tests pin the contract N3 (Coordinator sequence_denial) and N5
(prompt rendering) build on top of:

* **Happy path** — profile succeeds + trace_analyze succeeds →
  SharedState carries `last_profile_trace` + `last_trace_analyze`
  (with `analysis_md_text` + `roofline_snapshot_id` from C1 path)
  and the executor returns `status=succeeded` with `snapshot_id`.
* **profile failure** — executor returns `_failed("profile", ...)` and
  SharedState is **not mutated** (preserves the pre-roofline state so
  subsequent ticks can retry).
* **profile success but no trace_path** — executor returns
  `_failed("profile_no_trace", ...)`; corner case where profile
  succeeds but doesn't surface a `main_trace_path` / `trace_files`.
* **trace_analyze failure (after profile succeeded)** — executor
  returns `_failed("trace_analyze", ...)`. SharedState **does** keep
  the newly-set `last_profile_trace` (profile artifact is
  independently useful) but NOT `last_trace_analyze` cache.
* **Sub-step exceptions** — `profile_executor` / `trace_analyze_handler`
  raising bubbles back as `_failed(..., error="<reason> raised: ...")`
  not as an executor crash.
* **trace_path extraction** — prefers `main_trace_path` over first
  `trace_files` entry; both missing → no_trace failure.
* **shared_state required** — constructor without shared_state raises
  ValueError (cli wiring contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    RooflineExecutor,
    _extract_trace_path,
    _failed,
    make_roofline_executor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _ctx(tmp_path: Path | None = None) -> RunnerContext:
    task = Task(
        task_id="t-roofline-1", kind="roofline", state="running",
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
    """`trace_analyze_handler` result shape per kernel_request_handlers."""
    return {
        "status": "ok",
        "candidates_path": "/tmp/kc.json",
        "trace_report_path": "/tmp/analysis.md",
        "hot_kernels": [],
        "trace_health_warnings": [],
    }


def _patch_subs(profile_result, ta_result):
    """Context manager patching both sub-step callables.

    Returns the two patches applied in order so a test can assert
    side-effects via the mock objects.
    """
    async def fake_profile(ctx):
        if isinstance(profile_result, Exception):
            raise profile_result
        return profile_result

    async def fake_ta(payload, *, session_dir):
        if isinstance(ta_result, Exception):
            raise ta_result
        return ta_result

    return patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_promotes_profile_and_caches_trace_analyze(tmp_path):
    state = _state()
    state.cumulative_gain_validated = 2.5
    ctx = _ctx(tmp_path)

    # Write a placeholder analysis.md so record_trace_analyze can read it
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 51%, Idle 48%\n", encoding="utf-8")
    ta = _trace_analyze_success()
    ta["trace_report_path"] = str(md)

    p1, p2 = _patch_subs(_profile_success("/tmp/trace.gz"), ta)
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "succeeded"
    assert result["snapshot_id"] == 1
    assert result["last_profile_trace"] == "/tmp/trace.gz"
    assert result["analysis_md_path"] == str(md)
    assert result["profile_workspace"] == "/tmp/workspace"
    assert "executed_at_iso" in result

    # SharedState mutations
    assert state.last_profile_trace == "/tmp/trace.gz"
    assert state.last_profile_status == "succeeded"
    assert state.last_profile_args == "--mem-fraction-static=0.92"
    cached = state.last_trace_analyze
    assert cached["analysis_md_path"] == str(md)
    assert "Executive Summary" in cached["analysis_md_text"]
    assert cached["roofline_snapshot_id"] == 1
    assert cached["roofline_baseline_gain_at_snapshot"] == 2.5


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


# ---------------------------------------------------------------------------
# Failure paths — profile
# ---------------------------------------------------------------------------
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

    # SharedState UNCHANGED
    assert state.last_profile_trace == "/old/trace.gz"
    assert state.last_trace_analyze["roofline_snapshot_id"] == 5


@pytest.mark.asyncio
async def test_profile_no_trace_path(tmp_path):
    """Profile succeeded but result lacks main_trace_path / trace_files."""
    state = _state()
    ctx = _ctx(tmp_path)
    profile_bad = {"status": "succeeded", "output_throughput": 110.0}
    # No main_trace_path, no trace_files

    p1, p2 = _patch_subs(profile_bad, _trace_analyze_success())
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(ctx)

    assert result["status"] == "failed"
    assert result["error_class"] == "profile_no_trace_failed"
    assert "no trace_path" in result["error"]
    # SharedState UNCHANGED (we caught this before promote)
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


# ---------------------------------------------------------------------------
# Failure paths — trace_analyze
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_trace_analyze_failed_keeps_profile_promote(tmp_path):
    """After profile succeeds but trace_analyze fails:
    - last_profile_trace IS promoted (profile artifact is useful)
    - last_trace_analyze stays empty (no fresh cache)"""
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

    # last_profile_trace IS updated (profile artifact retained)
    assert state.last_profile_trace == "/tmp/new.gz"
    assert state.last_profile_status == "succeeded"
    # last_trace_analyze is EMPTY (cleared during profile promote, never re-populated)
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
    # Profile promote still happened
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


# ---------------------------------------------------------------------------
# Helpers + factory
# ---------------------------------------------------------------------------
def test_extract_trace_path_prefers_main_trace_path():
    r = {
        "main_trace_path": "/main.gz",
        "trace_files": ["/alt-0.gz", "/alt-1.gz"],
    }
    assert _extract_trace_path(r) == "/main.gz"


def test_extract_trace_path_falls_back_to_first_trace_file():
    r = {"trace_files": ["/first.gz", "/second.gz"]}
    assert _extract_trace_path(r) == "/first.gz"


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

    f2 = _failed("trace_analyze", "x",
                 sub_result={"status": "failed", "error": "y", "extra": "ignored"})
    assert f2["sub_result"] == {"status": "failed", "error": "y"}
    # 'extra' key not in the pinned allowlist → dropped
    assert "extra" not in f2["sub_result"]


def test_make_roofline_executor_requires_shared_state():
    with pytest.raises(ValueError, match="requires a SharedState"):
        RooflineExecutor(shared_state=None)


def test_make_roofline_executor_factory_signature():
    state = SharedState()
    exe = make_roofline_executor(shared_state=state)
    assert isinstance(exe, RooflineExecutor)
    assert exe.shared_state is state


# ---------------------------------------------------------------------------
# Ctx wrap helper
# ---------------------------------------------------------------------------
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
    # Lease inherited (None in this test fixture)
    assert child.lease is parent.lease


def test_resolve_session_dir_handles_missing_extra():
    state = SharedState()
    exe = RooflineExecutor(shared_state=state)
    ctx = _ctx()
    assert exe._resolve_session_dir(ctx) == Path(".")

    ctx.extra["session_dir"] = "/abc"
    assert exe._resolve_session_dir(ctx) == Path("/abc")
