# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``tracelens`` and ``roofline`` pointers carry the profiled commit's per-op shares; no other tool's does."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom_multibranch.contracts.notebook.records import MAX_OP_SHARES
from hyperloom_multibranch.toolbox.tools import (
    JobStatus,
    ToolJobs,
    ToolRequest,
    build_tool,
    roofline_tool,
    tracelens_tool,
)
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink

_TRACE_PKG = "hyperloom.orchestrator.trace_analysis"


@pytest.fixture(autouse=True)
def _loader_without_tracelens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the main flow's real ``analysis.json`` reader with TraceLens' source resolver replaced."""
    stub = types.ModuleType(f"{_TRACE_PKG}._kernel_source")
    stub.ResolveResult = SimpleNamespace  # type: ignore[attr-defined]
    stub.resolve_source_verdict = lambda *a, **k: SimpleNamespace(  # type: ignore[attr-defined]
        location=None, patchable=False, reason="", kind="", method="unresolved"
    )
    monkeypatch.setitem(sys.modules, f"{_TRACE_PKG}._kernel_source", stub)
    for name in ("_analysis_json", "analysis_json"):
        monkeypatch.delitem(sys.modules, f"{_TRACE_PKG}.{name}", raising=False)


def _task(operation: str, share: float, *, kernels: int = 1) -> dict[str, Any]:
    member = {
        "kernel_name": ["k"],
        "args_shapes": ["(1,2)"],
        "args_datatypes": ["bf16"],
        "time_ms": 1.0,
        "count": 1,
        "pct_e2e": share,
        "impact_score": share,
    }
    return {
        "priority": 1,
        "operation": operation,
        "impact": {"mid": 1.0, "low": 0.5, "high": 2.0},
        "members": [dict(member, pct_e2e=share / kernels) for _ in range(kernels)],
    }


def _report(tmp_path: Path, tasks: list[dict[str, Any]]) -> Path:
    (tmp_path / "analysis.json").write_text(json.dumps({"compute_optimizations": tasks}), encoding="utf-8")
    report = tmp_path / "analysis.md"
    report.write_text("# report\n", encoding="utf-8")
    return report


async def _tracelens_pointer(tmp_path: Path, result: dict, params: dict) -> Any:
    async def handler(payload: dict, *, session_dir: Path) -> dict:
        return result

    tool = tracelens_tool()
    tool._handler = handler
    sink = RecordingSink()
    jobs = ToolJobs([tool], sink, tmp_path / "jobs", push_delay_seconds=0)
    job = jobs.submit(ToolRequest("tracelens", "b1", {"trace_input": "/t", **params}), job_id="j1")
    assert await jobs.wait(job) is JobStatus.SUCCEEDED
    return sink.pointers[0]


_IDS = {"commit_id": "c9", "profile_request_id": "req-4"}


async def test_tracelens_pointer_carries_the_shares_the_commit_and_the_profile_request(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("aiter::gemm", 40.0), _task("rccl::all_reduce", 12.5), _task("silu", 3.25)])
    pointer = await _tracelens_pointer(tmp_path, {"status": "completed", "trace_report_path": str(report)}, _IDS)
    assert dict(pointer.op_shares) == {"aiter::gemm": 40.0, "rccl::all_reduce": 12.5, "silu": 3.25}
    assert (pointer.commit_id, pointer.profile_request_id) == ("c9", "req-4")


async def test_an_op_listed_in_several_members_has_its_shares_summed(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("aiter::gemm", 30.0, kernels=3), _task("silu", 5.0)])
    pointer = await _tracelens_pointer(tmp_path, {"status": "completed", "trace_report_path": str(report)}, _IDS)
    assert pointer.op_shares["aiter::gemm"] == pytest.approx(30.0)


async def test_more_than_fifty_ops_keep_the_fifty_largest(tmp_path: Path) -> None:
    ops = {f"op{n:02d}": float(n) for n in range(1, 61)}
    report = _report(tmp_path, [_task(name, share) for name, share in reversed(ops.items())])
    pointer = await _tracelens_pointer(tmp_path, {"status": "completed", "trace_report_path": str(report)}, _IDS)
    kept = {f"op{n:02d}" for n in range(11, 61)}
    assert len(pointer.op_shares) == MAX_OP_SHARES
    assert set(pointer.op_shares) == kept
    assert "op10" not in pointer.op_shares and pointer.op_shares["op60"] == 60.0


async def test_no_table_leaves_the_shares_empty_and_the_job_fine(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "analysis.md"
    for result in ({"status": "completed"}, {"status": "completed", "trace_report_path": str(missing)}):
        pointer = await _tracelens_pointer(tmp_path / str(len(result)), result, _IDS)
        assert pointer.op_shares == {} and pointer.commit_id == ""


async def test_an_unknown_commit_leaves_the_shares_empty_instead_of_failing(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("silu", 5.0)])
    result = {"status": "completed", "trace_report_path": str(report)}
    pointer = await _tracelens_pointer(tmp_path, result, {"profile_request_id": "req-4"})
    assert pointer.op_shares == {} and pointer.commit_id == "" and pointer.profile_request_id == ""


async def test_a_trace_that_did_not_come_from_a_profile_test_has_no_request_id(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("silu", 5.0)])
    result = {"status": "completed", "trace_report_path": str(report)}
    pointer = await _tracelens_pointer(tmp_path, result, {"commit_id": "c9"})
    assert pointer.op_shares == {"silu": 5.0} and pointer.profile_request_id == ""


async def test_roofline_pointer_reads_the_table_beside_its_analysis_report(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("aiter::gemm", 40.0)])

    class Executor:
        async def __call__(self, ctx: Any) -> dict:
            return {"status": "succeeded", "analysis_md_path": str(report)}

    tool = roofline_tool()
    tool._factory = lambda workdir: Executor()
    sink = RecordingSink()
    jobs = ToolJobs([tool], sink, tmp_path / "jobs", push_delay_seconds=0)
    job = jobs.submit(ToolRequest("roofline", "b1", {"commit_id": "c9"}), job_id="r1")
    assert await jobs.wait(job) is JobStatus.SUCCEEDED
    assert dict(sink.pointers[0].op_shares) == {"aiter::gemm": 40.0}
    assert (sink.pointers[0].commit_id, sink.pointers[0].profile_request_id) == ("c9", "")


async def test_another_tool_keeps_the_pointer_defaults_even_beside_a_table(tmp_path: Path) -> None:
    report = _report(tmp_path, [_task("silu", 5.0)])

    class Executor:
        async def __call__(self, ctx: Any) -> dict:
            return {"ok": True, "analysis_md_path": str(report)}

    tool = build_tool()
    tool._factory = lambda workdir: Executor()
    sink = RecordingSink()
    jobs = ToolJobs([tool], sink, tmp_path / "jobs", push_delay_seconds=0)
    await jobs.wait(jobs.submit(ToolRequest("build", "b1", dict(_IDS))))
    pointer = sink.pointers[0]
    assert (pointer.commit_id, pointer.profile_request_id, dict(pointer.op_shares)) == ("", "", {})
