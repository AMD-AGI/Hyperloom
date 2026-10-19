# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``Toolbox.close()`` ends the tool jobs it started and the child processes their threads own.

A tool that runs its child in a worker thread (the analysis tools do, through ``_run_subprocess``) is not stopped by
cancelling the task that awaits the thread: the child outlives the cancelled job, and the interpreter then waits for
the thread at exit.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors._kernel_agent_tool import _run_subprocess
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.tests.helpers import build
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest

CHILD = "child.pid"
GRANDCHILD = "grandchild.pid"
# The longest the stub's children live if nothing stops them, so a failing run cannot leave them behind.
LIFETIME_SECONDS = 40
CLOSE_BUDGET_SECONDS = 8.0


class _LongAnalysis:
    """A tool whose child (with a grandchild of its own) runs for a long time, in a worker thread."""

    name = "tracelens"

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        script = f"sleep {LIFETIME_SECONDS} & echo $! > {workdir / GRANDCHILD}; echo $$ > {workdir / CHILD}; wait"
        returncode, _, _ = await _run_subprocess(["sh", "-c", script], timeout_sec=LIFETIME_SECONDS)
        return ToolOutput(result={"returncode": returncode}, summary="analysed")


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


async def _pids(workdir: Path) -> tuple[int, int]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            return int((workdir / CHILD).read_text()), int((workdir / GRANDCHILD).read_text())
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(0.05)
    raise AssertionError("the analysis never started its child")


async def test_close_cancels_a_running_analysis_and_ends_its_child_processes(tmp_path: Path) -> None:
    rig = build(tmp_path, tools=lambda notebook: ToolJobs([_LongAnalysis()], notebook, tmp_path / "tool-results"))
    await rig.toolbox.start()
    job_id = rig.toolbox.submit_tool(ToolRequest("tracelens", "b1", {"trace_input": "t"}))
    child, grandchild = await _pids(tmp_path / "tool-results" / job_id)
    assert _running(child) and _running(grandchild)

    started = time.monotonic()
    await asyncio.wait_for(rig.toolbox.close(), CLOSE_BUDGET_SECONDS + 5)

    assert time.monotonic() - started < CLOSE_BUDGET_SECONDS
    assert not _running(child)
    assert not _running(grandchild)
    assert rig.toolbox.tool_status(job_id) is JobStatus.CANCELLED


async def test_close_waits_for_the_children_of_a_job_cancelled_earlier(tmp_path: Path) -> None:
    jobs: list[ToolJobs] = []

    def make(notebook: LocalNotebook) -> ToolJobs:
        jobs.append(ToolJobs([_LongAnalysis()], notebook, tmp_path / "tool-results"))
        return jobs[0]

    rig = build(tmp_path, tools=make)
    await rig.toolbox.start()
    job_id = rig.toolbox.submit_tool(ToolRequest("tracelens", "b1", {"trace_input": "t"}))
    child, grandchild = await _pids(tmp_path / "tool-results" / job_id)

    await rig.toolbox.cancel_branch("b1")
    assert await jobs[0].wait(job_id) is JobStatus.CANCELLED
    await asyncio.wait_for(rig.toolbox.close(), CLOSE_BUDGET_SECONDS + 5)

    assert not _running(child)
    assert not _running(grandchild)


async def test_a_closed_toolbox_starts_no_further_tool_job(tmp_path: Path) -> None:
    rig = build(tmp_path, tools=lambda notebook: ToolJobs([_LongAnalysis()], notebook, tmp_path / "tool-results"))
    await rig.toolbox.start()
    await rig.toolbox.close()
    with pytest.raises(InvalidRequest, match="closed"):
        rig.toolbox.submit_tool(ToolRequest("tracelens", "b1", {"trace_input": "t"}))
