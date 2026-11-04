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


async def test_a_job_id_may_carry_a_colon_and_its_directory_never_does(tmp_path: Path) -> None:
    sink = RecordingSink()
    runner = jobs(tmp_path, ScriptedTool(), sink)
    runner.submit(request(), job_id="prelude:abc")
    assert runner.has_job("prelude:abc") and not runner.has_job("prelude_abc")
    await runner.wait("prelude:abc")
    (pointer,) = sink.pointers
    assert pointer.job_id == "prelude:abc"
    assert Path(pointer.location).parent == tmp_path / "results" / "prelude%3Aabc"


class HeldSink(RecordingSink):
    """Holds the first push until ``release`` is set: the window in which a join or a redelivery runs."""

    def __init__(self) -> None:
        super().__init__()
        self.holding = asyncio.Event()
        self.release = asyncio.Event()

    async def notify_tool_result(self, pointer) -> None:
        if not self.holding.is_set():
            self.holding.set()
            await self.release.wait()
        await super().notify_tool_result(pointer)


class LateCancelTool(ScriptedTool):
    """A tool that returns its result even when cancelled: the cancel came too late to stop it."""

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        try:
            return await super().run(request, workdir)
        except asyncio.CancelledError:
            return self.output


class UnwindingTool(ScriptedTool):
    """A tool that, once cancelled, takes until ``unwound`` is set to end."""

    def __init__(self) -> None:
        super().__init__()
        self.block = asyncio.Event()
        self.unwound = asyncio.Event()

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        try:
            return await super().run(request, workdir)
        except asyncio.CancelledError:
            await self.unwound.wait()
            raise


async def test_a_branch_that_joins_during_a_push_and_the_branch_being_pushed_to_each_get_one_pointer(
    tmp_path: Path,
) -> None:
    sink = HeldSink()
    runner = jobs(tmp_path, ScriptedTool(), sink)
    job_id = runner.submit(request("a"))
    await sink.holding.wait()

    await runner.join(job_id, "b")
    sink.release.set()
    await runner.wait(job_id)

    assert sorted(pointer.branch for pointer in sink.pointers) == ["a", "b"]
    assert runner.undelivered() == ()


async def test_a_redelivery_during_a_push_does_not_push_the_pointer_again(tmp_path: Path) -> None:
    sink = HeldSink()
    runner = jobs(tmp_path, ScriptedTool(), sink)
    job_id = runner.submit(request("c"))
    await sink.holding.wait()

    still = await runner.redeliver()
    sink.release.set()
    await runner.wait(job_id)

    assert [pointer.branch for pointer in sink.pointers] == ["c"]
    assert still == (job_id,), "the push under way was not yet accepted"
    assert runner.undelivered() == ()


async def test_a_branch_a_redelivery_pushed_to_meanwhile_is_not_pushed_to_again_by_the_push_it_interrupted(
    tmp_path: Path,
) -> None:
    sink, tool = HeldSink(), ScriptedTool()
    tool.block = asyncio.Event()
    runner = jobs(tmp_path, tool, sink)
    job_id = runner.submit(request("a"))
    await tool.started.wait()
    await runner.join(job_id, "b")
    tool.block.set()
    await sink.holding.wait()

    await runner.redeliver()
    sink.release.set()
    await runner.wait(job_id)

    assert sorted(pointer.branch for pointer in sink.pointers) == ["a", "b"]


async def test_a_branch_that_asks_while_a_cancelled_job_unwinds_gets_a_new_job_and_the_key_keeps_naming_it(
    tmp_path: Path,
) -> None:
    sink, tool = RecordingSink(), UnwindingTool()
    runner = jobs(tmp_path, tool, sink)
    first = await runner.share(request("a"), "key")
    await tool.started.wait()
    assert runner.leave(first, "a"), "the last branch left, so the job is cancelled"

    second = await runner.share(request("b"), "key")
    tool.unwound.set()
    assert await runner.wait(first) is JobStatus.CANCELLED
    third = await runner.share(request("c"), "key")
    tool.block.set()

    assert second != first and third == second
    assert await runner.wait(second) is JobStatus.SUCCEEDED
    assert sorted(pointer.branch for pointer in sink.pointers) == ["b", "c"]


async def test_a_job_that_succeeded_although_a_cancel_was_asked_for_answers_its_key(tmp_path: Path) -> None:
    sink, tool = RecordingSink(), LateCancelTool()
    tool.block = asyncio.Event()
    runner = jobs(tmp_path, tool, sink)
    first = await runner.share(request("a"), "key")
    await tool.started.wait()
    assert runner.cancel(first)
    assert await runner.wait(first) is JobStatus.SUCCEEDED

    second = await runner.share(request("b"), "key")

    assert second == first, "the job has a result, so the key is not run again"
    assert len(tool.seen) == 1
    assert sorted(pointer.branch for pointer in sink.pointers) == ["a", "b"]
