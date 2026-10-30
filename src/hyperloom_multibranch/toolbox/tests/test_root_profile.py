# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``analyze_root_profile``: the scheduler starts the TraceLens report of the PRELUDE child's trace as a tool job."""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.contracts.blackboard.records import ToolResultPointer
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.tests.helpers import Rig, build
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.op_shares import ANALYSIS_JSON, DEGRADED_NOTICE, OPS_SUMMARY, with_op_shares

OPS_SUMMARY_FIXTURE = Path(__file__).parent / "data" / "tracelens_ops_summary.csv"
COMMIT = "prelude-head"
SOURCE = "30ea279d"
POINT = """\
benchmark:
  envs:
    CONC: '4'
    ISL: '256'
    OSL: 64
    RANDOM_RANGE_RATIO: '1'
"""
HINTS = {"split_conc": "4", "split_osl": "64", "split_r": "1", "split_isl": "256"}
DEGRADED = {"mode": "deterministic-fallback", "trace_quality": "poor", "warnings": ["graph-under-recorded"]}


class _Report:
    """Stands in for the ``tracelens`` tool: writes the report files of its job and reads its shares as the tool does."""

    name = "tracelens"

    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []
        self.info: dict[str, Any] = {"mode": "agentic"}
        self.fails = False
        self.gate: asyncio.Event | None = None

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.fails:
            return ToolOutput(result={"status": "failed"}, summary="the splitter wrote no window", ok=False)
        report = workdir / "tracelens"
        (report / OPS_SUMMARY).parent.mkdir(parents=True)
        shutil.copy(OPS_SUMMARY_FIXTURE, report / OPS_SUMMARY)
        (report / ANALYSIS_JSON).write_text(json.dumps({"report_info": self.info}))
        (report / "analysis.md").write_text("report")
        result = {"status": "ok", "analysis_md_path": str(report / "analysis.md")}
        return with_op_shares(ToolOutput(result=result, summary="trace analysed"), request)


class _Inbox:
    """The inbox push: keeps every pointer, then writes it to the blackboard as the toolbox does."""

    def __init__(self, blackboard: LocalBlackboard) -> None:
        self.blackboard, self.pointers = blackboard, []

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self.pointers.append(pointer)
        await self.blackboard.notify_tool_result(pointer)


@dataclass
class Stack:
    rig: Rig
    jobs: ToolJobs
    report: _Report
    inbox: _Inbox
    session: Path

    @property
    def trace_dir(self) -> Path:
        return self.session / "runs" / "profile" / "torch_trace"

    @property
    def point(self) -> Path:
        return self.session / "runs" / "baseline" / "baseline_config.with_envs.yaml"

    async def analyze(self, **paths: Path) -> str:
        wanted = {"trace_dir": self.trace_dir, "point_config": self.point, **paths}
        return await self.rig.toolbox.analyze_root_profile(COMMIT, source_id=SOURCE, **wanted)

    async def finished(self, job_id: str) -> ToolResultPointer:
        await self.jobs.wait(job_id)
        (pointer,) = self.inbox.pointers
        return pointer


@pytest.fixture
async def stack(tmp_path: Path) -> Stack:
    report, made = _Report(), []

    def tools(blackboard: LocalBlackboard) -> ToolJobs:
        made.append(_Inbox(blackboard))
        return ToolJobs([report], made[0], tmp_path / "session" / "tool-results")

    rig = build(tmp_path, tools=tools)
    session = tmp_path / "session"
    rig.toolbox.trace_root = session
    built = Stack(rig, rig.toolbox._tools, report, made[0], session)  # type: ignore[arg-type]
    built.trace_dir.mkdir(parents=True)
    (built.trace_dir / "1792420256.799032-TP-0.trace.json.gz").write_bytes(b"")
    built.point.parent.mkdir(parents=True)
    built.point.write_text(POINT, encoding="utf-8")
    await rig.toolbox.start()
    yield built
    await rig.toolbox.close()


async def test_the_report_is_a_job_of_the_integration_branch_and_its_pointer_carries_the_op_shares(
    stack: Stack,
) -> None:
    job = await stack.analyze()
    assert job == f"prelude:{SOURCE}"
    pointer = await stack.finished(job)
    assert (pointer.branch, pointer.job_id, pointer.commit_id, pointer.profile_request_id) == (
        INTEGRATION_BRANCH,
        job,
        COMMIT,
        "",
    )
    assert pointer.op_shares["aten::mm"] == pytest.approx(25.32, abs=0.01)
    (request,) = stack.report.requests
    assert (request.tool, request.branch) == ("tracelens", INTEGRATION_BRANCH)
    assert request.params["analysis_route"] == "report"
    assert request.params["trace_input"] == str(stack.trace_dir.resolve())
    held = await stack.rig.blackboard.op_shares(COMMIT)
    assert held is not None and (held.job_id, held.profile_request_id) == (job, "")
    assert ":" not in Path(pointer.location).parent.name  # shared storage rejects ":" in a name


async def test_the_splitter_hints_are_the_point_configs_not_the_environment(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (("CONC", "99"), ("OSL", "7"), ("RANDOM_RANGE_RATIO", "0.5")):
        monkeypatch.setenv(name, value)
    await stack.finished(await stack.analyze())
    (request,) = stack.report.requests
    assert {k: v for k, v in request.params.items() if k.startswith("split_")} == HINTS


@pytest.mark.parametrize(
    ("point", "hints"),
    [
        (POINT.replace("    CONC: '4'\n", ""), {"split_osl": "64", "split_r": "1", "split_isl": "256"}),
        (POINT.replace("    ISL: '256'\n", ""), {"split_conc": "4", "split_osl": "64", "split_r": "1"}),
        ("benchmark: [not, a, mapping", {}),
        (POINT.replace("'4'", "many").replace("64", "0").replace("'1'", "2").replace("'256'", "-1"), {}),
    ],
)
async def test_a_point_that_cannot_give_a_hint_gives_none_and_the_analysis_still_runs(
    stack: Stack, point: str, hints: dict[str, str]
) -> None:
    stack.point.write_text(point, encoding="utf-8")
    await stack.finished(await stack.analyze())
    (request,) = stack.report.requests
    assert {k: v for k, v in request.params.items() if k.startswith("split_")} == hints


async def test_a_trace_directory_or_point_config_outside_the_session_is_refused(stack: Stack, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    (outside / "torch_trace").mkdir(parents=True)
    (outside / "torch_trace" / "0-TP-0.trace.json.gz").write_bytes(b"")
    (outside / "point.yaml").write_text(POINT, encoding="utf-8")
    (stack.session / "escape").symlink_to(outside)
    refused = [
        {"trace_dir": outside / "torch_trace"},
        {"trace_dir": stack.session / "escape" / "torch_trace"},
        {"trace_dir": stack.session / ".." / "outside" / "torch_trace"},
        {"point_config": outside / "point.yaml"},
        {"point_config": stack.session / "escape" / "point.yaml"},
    ]
    for paths in refused:
        with pytest.raises(RequestRejected) as raised:
            await stack.analyze(**paths)
        assert raised.value.code == "invalid_request", paths
    assert stack.report.requests == [] and not stack.jobs.has_job(f"prelude:{SOURCE}")


async def test_a_toolbox_without_a_session_directory_refuses(stack: Stack) -> None:
    stack.rig.toolbox.trace_root = None
    with pytest.raises(RequestRejected) as raised:
        await stack.analyze()
    assert raised.value.code == "invalid_request"


@pytest.mark.parametrize("files", [[], ["torch_trace.selfcert.json", "profile.json", "notes.trace"]])
async def test_a_trace_directory_without_a_trace_is_refused(stack: Stack, files: list[str]) -> None:
    empty = stack.session / "runs" / "empty"
    empty.mkdir(parents=True)
    for name in files:
        (empty / name).write_text("{}")
    with pytest.raises(RequestRejected) as raised:
        await stack.analyze(trace_dir=empty)
    assert raised.value.code == "profile_without_trace"
    assert stack.report.requests == [] and not stack.jobs.has_job(f"prelude:{SOURCE}")


async def test_a_second_call_with_the_same_source_id_returns_the_job_and_runs_nothing_again(stack: Stack) -> None:
    first = await stack.analyze()
    assert await stack.analyze() == first
    await stack.finished(first)
    assert await stack.analyze() == first
    assert len(stack.report.requests) == 1 and len(stack.inbox.pointers) == 1
    other = await stack.rig.toolbox.analyze_root_profile(
        COMMIT, stack.trace_dir, stack.point, source_id="another-child"
    )
    assert other == "prelude:another-child" != first


async def test_a_degraded_report_ends_the_job_with_a_summary_and_no_op_shares(stack: Stack) -> None:
    stack.report.info = DEGRADED
    pointer = await stack.finished(await stack.analyze())
    assert not pointer.op_shares and not pointer.commit_id
    assert DEGRADED_NOTICE in pointer.summary
    assert await stack.rig.blackboard.op_shares(COMMIT) is None


async def test_a_failed_report_ends_the_job_with_its_reason_and_raises_nothing(stack: Stack) -> None:
    stack.report.fails = True
    job = await stack.analyze()
    pointer = await stack.finished(job)
    assert stack.jobs.status(job) is JobStatus.FAILED
    assert not pointer.op_shares and "the splitter wrote no window" in pointer.summary


async def test_the_job_takes_no_gpu_lease_and_no_slot_of_any_branch(stack: Stack) -> None:
    stack.report.gate = asyncio.Event()
    job = await stack.analyze()
    state = await stack.rig.toolbox.queue_state()
    assert (state.running, state.queued) == (0, 0)
    assert stack.rig.executor.jobs == []
    stack.report.gate.set()
    await stack.finished(job)


async def test_a_toolbox_that_does_not_run_tracelens_refuses_by_name(tmp_path: Path) -> None:
    rig = build(tmp_path)
    with pytest.raises(RequestRejected) as raised:
        await rig.toolbox.analyze_root_profile(COMMIT, tmp_path, tmp_path, source_id=SOURCE)
    assert raised.value.code == "no_tool_executor"
