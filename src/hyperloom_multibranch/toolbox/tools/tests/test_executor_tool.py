# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolRequest, profile_tool
from hyperloom_multibranch.toolbox.tools.executor_tool import ExecutorTool
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink


class FakeExecutor:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.contexts: list[RunnerContext] = []

    async def __call__(self, ctx: RunnerContext) -> dict:
        self.contexts.append(ctx)
        return self.result


def tool_with(executor: FakeExecutor) -> ExecutorTool:
    return ExecutorTool(
        "profile", "profile", lambda workdir: executor, ok_text="captured", summary_keys=("main_trace_path",)
    )


async def test_the_executor_gets_a_context_built_from_the_request(tmp_path: Path) -> None:
    executor = FakeExecutor({"status": "completed", "main_trace_path": "/t/a.trace.json.gz"})
    sink = RecordingSink()
    runner = ToolJobs([tool_with(executor)], sink, tmp_path, push_delay_seconds=0)
    job_id = runner.submit(ToolRequest("profile", "s2", {"extra_server_args": "--tp 2"}), job_id="job1")
    assert await runner.wait(job_id) is JobStatus.SUCCEEDED
    (ctx,) = executor.contexts
    assert ctx.task.kind == "profile"
    assert ctx.task.task_id == "job1"
    assert ctx.task.params == {"extra_server_args": "--tp 2"}
    assert ctx.extra["workspace"] == str(tmp_path / "job1") == ctx.extra["session_dir"]
    assert sink.pointers[0].branch == "s2"
    assert sink.pointers[0].summary == "succeeded: captured (main_trace_path=/t/a.trace.json.gz)"


async def test_the_executors_own_failure_reaches_the_branch_with_its_result(tmp_path: Path) -> None:
    executor = FakeExecutor({"status": "failed", "error": "server never became ready", "trace_dir": None})
    sink = RecordingSink()
    runner = ToolJobs([tool_with(executor)], sink, tmp_path, push_delay_seconds=0)
    job_id = runner.submit(ToolRequest("profile", "s2"))
    assert await runner.wait(job_id) is JobStatus.FAILED
    assert sink.pointers[0].summary == "failed: server never became ready"


async def test_the_params_the_request_gave_are_not_shared_with_the_caller(tmp_path: Path) -> None:
    executor = FakeExecutor({"status": "completed"})
    params = {"extra_server_args": "--tp 2"}
    runner = ToolJobs([tool_with(executor)], RecordingSink(), tmp_path, push_delay_seconds=0)
    await runner.wait(runner.submit(ToolRequest("profile", "s2", params)))
    executor.contexts[0].task.params["extra_server_args"] = "mutated"
    assert params == {"extra_server_args": "--tp 2"}


def test_the_profile_tool_runs_the_existing_profile_executor(tmp_path: Path) -> None:
    tool = profile_tool()
    assert tool.name == "profile"
    executor = tool._factory(tmp_path)
    assert isinstance(executor, ProfileExecutor)
    assert Path(executor.session_dir) == tmp_path
