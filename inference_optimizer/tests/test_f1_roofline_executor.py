"""F1-2 — RooflineExecutor smoke tests.

Covers the wiring contract of the cherry-picked
:class:`RooflineExecutor`: it composes ``profile`` + ``trace_analyze``
into one atomic action, mutates SharedState only on the documented
paths, and surfaces the snapshot id from the new
``record_trace_analyze`` writer (F1-1).

The full executor's N26 auto-retry / failure-mode coverage is exercised
by ``test_roofline_executor.py`` cherry-picked from main alongside
F1-2; this file is the branch-local smoke that proves the alias-based
``trace_analyze_handler`` import (F1-2) resolves.

Reference: ``plan_roofline_framework/F1_roofline_composite.MD`` §F1-2.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    RooflineExecutor,
    RooflineStubExecutor,
    make_roofline_executor,
    make_roofline_stub_executor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


def _make_ctx(session_dir: Path) -> RunnerContext:
    task = Task(
        task_id="t1",
        kind="roofline",
        state="running",
        params={"base_extra_args": "--max-num-seqs 64"},
        idempotency_key="ik-roofline-1",
        requires_lanes=["profile_lane"],
        allowed_tools=["emit_intent"],
        side_effects=["reads_server", "writes_results"],
        lease_ttl_sec=2700,
    )
    return RunnerContext(
        task=task,
        lease=None,
        extra={"session_dir": str(session_dir)},
    )


@pytest.mark.asyncio
async def test_roofline_executor_happy_path(tmp_path: Path):
    md = tmp_path / "analysis.md"
    md.write_text("# TraceLens analysis\nbody.\n", encoding="utf-8")

    profile_result = {
        "status": "succeeded",
        "main_trace_path": str(tmp_path / "trace.json"),
        "workspace": str(tmp_path / "ws"),
    }
    ta_result = {
        "status": "ok",
        "trace_report_path": str(md),
        "candidates_path": str(tmp_path / "candidates.json"),
        "hot_kernels": [],
        "task_groups": [],
        "trace_health_warnings": [],
    }

    state = SharedState()
    executor = make_roofline_executor(shared_state=state)

    async def _fake_profile(ctx):
        return profile_result

    async def _fake_trace_analyze(payload, *, session_dir):
        assert payload["trace_input"] == str(tmp_path / "trace.json")
        return ta_result

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        side_effect=_fake_profile,
    ), patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        side_effect=_fake_trace_analyze,
    ):
        result = await executor(_make_ctx(tmp_path))

    assert result["status"] == "succeeded"
    assert result["snapshot_id"] == 1
    assert result["last_profile_trace"] == str(tmp_path / "trace.json")
    assert result["analysis_md_path"] == str(md)
    # SharedState was mutated by the executor's inline-promote +
    # record_trace_analyze writer.
    assert state.last_profile_trace == str(tmp_path / "trace.json")
    assert state.last_profile_status == "succeeded"
    assert state.last_profile_args == "--max-num-seqs 64"
    assert state.roofline_snapshot_id == 1
    assert state.last_trace_analyze["analysis_md_text"].startswith(
        "# TraceLens analysis",
    )


@pytest.mark.asyncio
async def test_roofline_executor_profile_failure_does_not_mutate_state(
    tmp_path: Path,
):
    state = SharedState()
    state.last_profile_trace = "/old/trace.json"
    executor = make_roofline_executor(shared_state=state)

    async def _failing_profile(ctx):
        return {"status": "failed", "error": "magpie crashed"}

    with patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        side_effect=_failing_profile,
    ):
        result = await executor(_make_ctx(tmp_path))

    assert result["status"] == "failed"
    assert result["phase"] == "profile"
    assert result["error_class"] == "profile_failed"
    assert state.last_profile_trace == "/old/trace.json"
    assert state.last_trace_analyze == {}


def test_roofline_stub_executor_factory():
    state = SharedState()
    stub = make_roofline_stub_executor(shared_state=state)
    assert isinstance(stub, RooflineStubExecutor)


def test_roofline_executor_requires_shared_state():
    with pytest.raises(ValueError, match="requires a SharedState"):
        RooflineExecutor(shared_state=None)
