# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.tests.helpers import build, settle
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolOutput, ToolRequest, all_tools


class Slow:
    name = "slow"

    def __init__(self) -> None:
        self.gate = asyncio.Event()

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        await self.gate.wait()
        return ToolOutput(result={"n": 1}, summary="done")


def _with_tools(tmp_path: Path, tool: Slow | None = None):
    made: list[ToolJobs] = []

    def make(notebook):
        made.append(ToolJobs([tool] if tool else list(all_tools()), notebook, tmp_path / "tool-results"))
        return made[0]

    return build(tmp_path, tools=make), made[0]


async def test_a_tool_result_reaches_the_inbox_and_takes_no_slot_of_k(tmp_path: Path) -> None:
    tool = Slow()
    rig, jobs = _with_tools(tmp_path, tool)
    job = rig.toolbox.submit_tool(ToolRequest("slow", "b1"))
    assert rig.toolbox.tool_status(job) is JobStatus.RUNNING
    assert (await rig.toolbox.queue_state()).running == 0
    tool.gate.set()
    assert await jobs.wait(job) is JobStatus.SUCCEEDED


async def test_a_stopped_branch_has_its_running_tool_jobs_cancelled(tmp_path: Path) -> None:
    tool = Slow()
    rig, jobs = _with_tools(tmp_path, tool)
    job = rig.toolbox.submit_tool(ToolRequest("slow", "b1"))
    await settle()
    assert await rig.toolbox.cancel_branch("b1") == 1
    assert await jobs.wait(job) is JobStatus.CANCELLED


async def test_every_registered_tool_is_served_and_a_toolbox_without_tools_refuses(tmp_path: Path) -> None:
    rig, jobs = _with_tools(tmp_path)
    assert {"profile", "upstream_pr_search"} <= set(jobs.tool_names)
    with pytest.raises(RequestRejected, match="no asynchronous tools"):
        build(tmp_path / "other").toolbox.submit_tool(ToolRequest("profile", "b1"))
