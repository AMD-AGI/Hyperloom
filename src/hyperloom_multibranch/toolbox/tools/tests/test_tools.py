# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each tool runs the orchestrator's existing implementation, and what that returns reaches the inbox."""

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor
from hyperloom.orchestrator.actions.executors.targeted_build_executor import TargetedBuildExecutor
from hyperloom_multibranch.toolbox.tools import (
    InvalidRequest,
    JobStatus,
    ToolJobs,
    ToolRequest,
    accuracy_tool,
    all_tools,
    build_tool,
    roofline_tool,
    tracelens_tool,
)
from hyperloom_multibranch.toolbox.tools.roofline import roofline_handler, tracelens_handler
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink


def test_the_tool_names_are_unique_and_complete() -> None:
    names = [tool.name for tool in all_tools()]
    assert sorted(names) == [
        "accuracy",
        "build",
        "profile",
        "roofline",
        "tracelens",
        "upstream_pr_search",
    ]


def test_each_tool_is_bound_to_the_existing_implementation(tmp_path: Path) -> None:
    assert roofline_tool()._handler is roofline_handler
    assert isinstance(build_tool()._factory(tmp_path), TargetedBuildExecutor)
    accuracy = accuracy_tool()._factory(tmp_path)
    assert isinstance(accuracy, BaselineExecutor)
    assert Path(accuracy.session_dir) == tmp_path
    assert tracelens_tool()._handler is tracelens_handler


def test_the_accuracy_tool_refuses_to_switch_the_evaluation_off() -> None:
    request = ToolRequest("accuracy", "s1", {"disable_run_eval": "1"})
    with pytest.raises(InvalidRequest, match="disable_run_eval is not allowed"):
        accuracy_tool().validate(request)
    accuracy_tool().validate(ToolRequest("accuracy", "s1", {"disable_run_eval": "0"}))


def test_tracelens_needs_a_trace() -> None:
    with pytest.raises(InvalidRequest, match="tracelens needs trace_input"):
        tracelens_tool().validate(ToolRequest("tracelens", "s1", {}))


async def test_a_handler_tool_gets_its_payload_and_the_job_directory(tmp_path: Path) -> None:
    seen: list[tuple[dict, Path]] = []

    async def handler(payload: dict, *, session_dir: Path) -> dict:
        seen.append((payload, session_dir))
        return {"status": "completed", "hot_kernels": ["gemm"]}

    tool = tracelens_tool()
    tool._handler = handler
    sink = RecordingSink()
    runner = ToolJobs([tool], sink, tmp_path, push_delay_seconds=0)
    job_id = runner.submit(ToolRequest("tracelens", "s3", {"trace_input": "/t"}), job_id="j9")
    assert await runner.wait(job_id) is JobStatus.SUCCEEDED
    assert [p for p, _ in seen] == [{"trace_input": "/t"}]
    assert seen[0][1] != tmp_path / "j9"  # run on local scratch, not the (possibly shared) job directory
    assert sink.pointers[0].summary == "succeeded: trace analysed"


async def test_a_failed_build_reaches_the_branch_with_the_builds_reason(tmp_path: Path) -> None:
    class FailedBuild:
        async def __call__(self, ctx) -> dict:
            return {"ok": False, "failure_class": "compile", "failure_summary": "ld: cannot find -lrocblas"}

    tool = build_tool()
    tool._factory = lambda workdir: FailedBuild()
    sink = RecordingSink()
    runner = ToolJobs([tool], sink, tmp_path, push_delay_seconds=0)
    assert await runner.wait(runner.submit(ToolRequest("build", "s4"))) is JobStatus.FAILED
    assert sink.pointers[0].summary == "failed: ld: cannot find -lrocblas"


async def test_an_executor_that_raises_still_tells_the_branch(tmp_path: Path) -> None:
    class Raising:
        async def __call__(self, ctx) -> dict:
            raise RuntimeError("targeted_build failed: failure_class='timeout'")

    tool = build_tool()
    tool._factory = lambda workdir: Raising()
    sink = RecordingSink()
    runner = ToolJobs([tool], sink, tmp_path, push_delay_seconds=0)
    assert await runner.wait(runner.submit(ToolRequest("build", "s4"))) is JobStatus.FAILED
    assert "failure_class='timeout'" in sink.pointers[0].summary
