# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``tracelens`` and ``roofline`` pointers carry the profiled commit's per-op shares; no other tool's does."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from hyperloom_multibranch.contracts.blackboard.records import MAX_OP_SHARES
from hyperloom_multibranch.toolbox.tools import (
    JobStatus,
    ToolJobs,
    ToolRequest,
    build_tool,
    tracelens_tool,
)
from hyperloom_multibranch.toolbox.tools.op_shares import OPS_SUMMARY
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink


def _report(tmp_path: Path, shares: dict[str, float]) -> Path:
    """An analysis report with the ops summary TraceLens' perf report writes beside it."""
    (tmp_path / OPS_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{name},{share}\n" for name, share in shares.items())
    (tmp_path / OPS_SUMMARY).write_text(f"name,Percentage (%)\n{rows}", encoding="utf-8")
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
    report = _report(tmp_path, {"aiter::gemm": 40.0, "rccl::all_reduce": 12.5, "silu": 3.25})
    pointer = await _tracelens_pointer(tmp_path, {"status": "completed", "trace_report_path": str(report)}, _IDS)
    assert dict(pointer.op_shares) == {"aiter::gemm": 40.0, "rccl::all_reduce": 12.5, "silu": 3.25}
    assert (pointer.commit_id, pointer.profile_request_id) == ("c9", "req-4")


async def test_more_than_fifty_ops_keep_the_fifty_largest(tmp_path: Path) -> None:
    ops = {f"op{n:02d}": float(n) for n in range(1, 61)}
    report = _report(tmp_path, dict(reversed(ops.items())))
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
    report = _report(tmp_path, {"silu": 5.0})
    result = {"status": "completed", "trace_report_path": str(report)}
    pointer = await _tracelens_pointer(tmp_path, result, {"profile_request_id": "req-4"})
    assert pointer.op_shares == {} and pointer.commit_id == "" and pointer.profile_request_id == ""


async def test_a_trace_that_did_not_come_from_a_profile_test_has_no_request_id(tmp_path: Path) -> None:
    report = _report(tmp_path, {"silu": 5.0})
    result = {"status": "completed", "trace_report_path": str(report)}
    pointer = await _tracelens_pointer(tmp_path, result, {"commit_id": "c9"})
    assert pointer.op_shares == {"silu": 5.0} and pointer.profile_request_id == ""


async def test_another_tool_keeps_the_pointer_defaults_even_beside_a_table(tmp_path: Path) -> None:
    report = _report(tmp_path, {"silu": 5.0})

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


async def test_tracelens_runs_on_local_disk_and_copies_back_only_what_is_read(tmp_path: Path) -> None:
    seen: dict[str, Path] = {}

    async def handler(payload: dict, *, session_dir: Path) -> dict:
        seen["dir"] = session_dir
        out = session_dir / "run"
        out.mkdir()
        (out / "merged.trace_iteration_0:4.json.gz").write_text("x", encoding="utf-8")
        report = _report(out, {"gemm": 60.0})
        return {
            "status": "ok",
            "trace_report_path": str(report),
            "split": str(out / "merged.trace_iteration_0:4.json.gz"),
        }

    tool = tracelens_tool()
    tool._handler = handler
    sink = RecordingSink()
    jobs = ToolJobs([tool], sink, tmp_path / "jobs", push_delay_seconds=0)
    job = jobs.submit(ToolRequest("tracelens", "b1", {"trace_input": "/t", "commit_id": "c9"}), job_id="j1")
    assert await jobs.wait(job) is JobStatus.SUCCEEDED
    workdir = tmp_path / "jobs" / "j1"
    assert not seen["dir"].is_relative_to(tmp_path)
    assert not seen["dir"].exists()
    assert (workdir / "run" / "analysis.md").is_file() and (workdir / "run" / OPS_SUMMARY).is_file()
    assert not [p for p in workdir.rglob("*") if ":" in p.name]
    pointer = sink.pointers[0]
    assert pointer.op_shares == {"gemm": 60.0}
