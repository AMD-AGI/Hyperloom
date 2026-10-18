# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import json
from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools import (
    InvalidRequest,
    JobStatus,
    ToolFailure,
    ToolJobs,
    ToolOutput,
    ToolRequest,
    UnknownTool,
)
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink, ScriptedTool


def jobs(tmp_path: Path, tool: ScriptedTool, sink: RecordingSink, **kw) -> ToolJobs:
    return ToolJobs([tool], sink, tmp_path / "results", push_delay_seconds=0, **kw)


def request(branch: str = "s1", tool: str = "scripted") -> ToolRequest:
    return ToolRequest(tool=tool, branch=branch, params={"x": 1})


async def test_success_pushes_a_pointer_to_the_asking_branch(tmp_path: Path) -> None:
    sink, tool = RecordingSink(), ScriptedTool()
    runner = jobs(tmp_path, tool, sink)
    job_id = runner.submit(request("s7"))
    assert runner.status(job_id) is JobStatus.RUNNING
    assert sink.pointers == []
    assert await runner.wait(job_id) is JobStatus.SUCCEEDED
    (pointer,) = sink.pointers
    assert (pointer.branch, pointer.job_id) == ("s7", job_id)
    assert pointer.summary == "succeeded: done"
    record = json.loads(Path(pointer.location).read_text())
    assert record["result"] == {"n": 1}
    assert record["status"] == "succeeded"
    assert tool.seen[0][1] == Path(pointer.location).parent


async def test_a_tool_failure_is_pushed_with_its_reason(tmp_path: Path) -> None:
    sink = RecordingSink()
    runner = jobs(tmp_path, ScriptedTool(raises=ToolFailure("no GPU free")), sink)
    job_id = runner.submit(request())
    assert await runner.wait(job_id) is JobStatus.FAILED
    assert sink.pointers[0].summary == "failed: no GPU free"
    assert json.loads(Path(sink.pointers[0].location).read_text())["error"] == "no GPU free"


async def test_a_crash_is_pushed_and_the_runner_keeps_serving(tmp_path: Path) -> None:
    sink, tool = RecordingSink(), ScriptedTool(raises=KeyError("boom"))
    runner = jobs(tmp_path, tool, sink)
    first = runner.submit(request())
    assert await runner.wait(first) is JobStatus.FAILED
    assert "KeyError" in sink.pointers[0].summary
    tool.raises = None
    second = runner.submit(request())
    assert await runner.wait(second) is JobStatus.SUCCEEDED
    assert [p.job_id for p in sink.pointers] == [first, second]


async def test_an_unsuccessful_verdict_keeps_its_result(tmp_path: Path) -> None:
    sink = RecordingSink()
    tool = ScriptedTool(output=ToolOutput(result={"trace_dir": None}, summary="no trace", ok=False))
    runner = jobs(tmp_path, tool, sink)
    job_id = runner.submit(request())
    assert await runner.wait(job_id) is JobStatus.FAILED
    record = json.loads(Path(sink.pointers[0].location).read_text())
    assert record["result"] == {"trace_dir": None}
    assert sink.pointers[0].summary == "failed: no trace"


async def test_cancelling_a_running_job_still_tells_the_branch(tmp_path: Path) -> None:
    sink, tool = RecordingSink(), ScriptedTool()
    tool.block = asyncio.Event()
    runner = jobs(tmp_path, tool, sink)
    job_id = runner.submit(request())
    await tool.started.wait()
    assert runner.cancel(job_id) is True
    assert await runner.wait(job_id) is JobStatus.CANCELLED
    assert sink.pointers[0].summary.startswith("cancelled:")
    assert runner.cancel(job_id) is False


async def test_cancelling_before_the_tool_starts_never_runs_it(tmp_path: Path) -> None:
    sink, tool = RecordingSink(), ScriptedTool()
    runner = jobs(tmp_path, tool, sink)
    job_id = runner.submit(request())
    assert runner.cancel(job_id) is True
    assert await runner.wait(job_id) is JobStatus.CANCELLED
    assert tool.seen == []
    assert len(sink.pointers) == 1


async def test_a_refused_push_is_retried(tmp_path: Path) -> None:
    sink = RecordingSink(refuse=2)
    runner = jobs(tmp_path, ScriptedTool(), sink, push_attempts=3)
    job_id = runner.submit(request())
    await runner.wait(job_id)
    assert sink.calls == 3
    assert len(sink.pointers) == 1
    assert runner.undelivered() == ()


async def test_a_push_that_keeps_failing_is_kept_and_redelivered(tmp_path: Path) -> None:
    sink = RecordingSink(refuse=2)
    runner = jobs(tmp_path, ScriptedTool(), sink, push_attempts=2)
    job_id = runner.submit(request())
    await runner.wait(job_id)
    assert sink.pointers == []
    assert runner.undelivered() == (job_id,)
    assert Path(runner._jobs[job_id].pointer.location).exists()
    assert await runner.redeliver() == ()
    assert [p.job_id for p in sink.pointers] == [job_id]
    assert await runner.redeliver() == ()
    assert len(sink.pointers) == 1


@pytest.mark.parametrize(
    ("req", "job_id", "error", "match"),
    [
        (ToolRequest("nope", "s1"), None, UnknownTool, "no tool named 'nope'"),
        (ToolRequest("scripted", ""), None, InvalidRequest, "names the branch"),
        (ToolRequest("scripted", "s1"), "../escape", InvalidRequest, "not a plain file name"),
    ],
)
async def test_a_request_that_cannot_work_is_refused_and_nothing_starts(
    tmp_path: Path, req: ToolRequest, job_id: str | None, error: type[Exception], match: str
) -> None:
    tool = ScriptedTool()
    runner = jobs(tmp_path, tool, RecordingSink())
    with pytest.raises(error, match=match):
        runner.submit(req, job_id=job_id)
    await asyncio.sleep(0)
    assert tool.seen == []
    assert not (tmp_path / "results").exists()


async def test_a_job_id_is_used_once(tmp_path: Path) -> None:
    runner = jobs(tmp_path, ScriptedTool(), RecordingSink())
    runner.submit(request(), job_id="j1")
    with pytest.raises(InvalidRequest, match="already in use"):
        runner.submit(request(), job_id="j1")
    await runner.wait("j1")


async def test_validation_by_the_tool_refuses_at_submit(tmp_path: Path) -> None:
    class Picky(ScriptedTool):
        def validate(self, request: ToolRequest) -> None:
            raise InvalidRequest("params need a repo")

    runner = jobs(tmp_path, Picky(), RecordingSink())
    with pytest.raises(InvalidRequest, match="params need a repo"):
        runner.submit(request())
