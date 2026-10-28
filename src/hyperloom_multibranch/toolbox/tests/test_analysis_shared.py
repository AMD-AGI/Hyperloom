# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One analysis per profile test and tool (board #153 c3447), on a toolbox built by the real local factory.

``submit_analysis`` of a profile test that is being analysed joins that job, of one already analysed shares its result as
it is, and of one whose analysis failed starts a new job. Every branch that asked gets the pointer in its own inbox. The
job runner is the real one; only TraceLens and the roofline are stand-ins, which can be held running or made to fail.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.tests.helpers import PARENT, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_analysis_route import SHARES
from hyperloom_multibranch.toolbox.tests.test_local_analysis import LOCATION, _toolbox
from hyperloom_multibranch.toolbox.tools import JobStatus, ToolJobs, ToolRequest
from hyperloom_multibranch.toolbox.tools import op_shares as op_shares_module
from hyperloom_multibranch.toolbox.tools.base import ToolFailure, ToolOutput
from hyperloom_multibranch.toolbox.tools.op_shares import with_op_shares
from hyperloom_multibranch.toolbox.tools.shared_jobs import SHARED_JOBS_FILE

BRANCHES = ("b1", "b2", "b3")
PROFILE = "p1"


class _Analysis:
    """Stands in for an analysis tool: it runs until ``release`` (at once by default) and fails ``failures`` times."""

    def __init__(self, name: str, *, held: bool = False, failures: int = 0) -> None:
        self.name, self.failures = name, failures
        self.requests: list[ToolRequest] = []
        self._gate = asyncio.Event()
        if not held:
            self._gate.set()

    def release(self) -> None:
        self._gate.set()

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        self.requests.append(request)
        await self._gate.wait()
        if self.failures:
            self.failures -= 1
            raise ToolFailure("the trace could not be split")
        return with_op_shares(ToolOutput(result={}, summary="analysed"), request)


@pytest.fixture(autouse=True)
def _shares(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op_shares_module, "read_op_shares", lambda result: dict(SHARES))


@dataclass
class Campaign:
    """One session directory and blackboard; :meth:`toolbox` builds (and rebuilds, as a restart does) its toolbox."""

    tmp_path: Path
    blackboard: LocalBlackboard
    repo: FakeRepo
    built: list[LocalHostedToolbox]

    async def toolbox(self, *tools: _Analysis) -> LocalHostedToolbox:
        toolbox = _toolbox(self.tmp_path, self.blackboard, self.repo, tools=list(tools))
        await toolbox.start()
        self.built.append(toolbox)
        return toolbox

    async def inbox(self, branch: str) -> list[tuple[str, str]]:
        """``(job id, summary)`` of every tool result in ``branch``'s inbox."""
        reader = LocalBlackboard(self.blackboard.root, Writer(Role.SPECIALIST_RUNTIME, branch, branch))
        page = await reader.poll(branch)
        return [(item.pointer, item.summary) for item in page.items if item.kind.value == "tool_result"]


@pytest.fixture
async def campaign(tmp_path: Path) -> AsyncIterator[Campaign]:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    repo = FakeRepo()
    for branch in BRANCHES:
        repo.heads[branch] = PARENT  # three branches on one commit, which the profile test measured
    request = replace(req("b1", purpose=Purpose.PROFILE, kind=TestKind.PROFILE), request_id=PROFILE)
    await blackboard.write_request(request)
    await blackboard.write_result(
        TestResult(request_id=PROFILE, validity=Validity.SUCCEEDED, profile_location=LOCATION)
    )
    built: list[LocalHostedToolbox] = []
    try:
        yield Campaign(tmp_path, blackboard, repo, built)
    finally:
        for toolbox in built:
            await toolbox.close()


def _jobs(toolbox: LocalHostedToolbox) -> ToolJobs:
    assert toolbox._tools is not None
    return toolbox._tools


async def _submit_all(toolbox: LocalHostedToolbox, tool: str = "tracelens") -> list[str]:
    calls = (toolbox.submit_analysis(branch, tool, {}, PROFILE) for branch in BRANCHES)
    return list(await asyncio.gather(*calls))


async def test_three_branches_asking_while_it_runs_start_one_job_and_each_gets_its_result(campaign: Campaign) -> None:
    tracelens = _Analysis("tracelens", held=True)
    toolbox = await campaign.toolbox(tracelens)

    jobs = await _submit_all(toolbox)

    assert len(set(jobs)) == 1 and len(tracelens.requests) == 1
    assert toolbox.tool_status(jobs[0]) is JobStatus.RUNNING
    tracelens.release()
    assert await _jobs(toolbox).wait(jobs[0]) is JobStatus.SUCCEEDED
    for branch in BRANCHES:
        (pointer,) = await campaign.inbox(branch)
        assert pointer[0] == jobs[0] and pointer[1].startswith(f"tool {jobs[0]}: succeeded: analysed")
    assert _jobs(toolbox).undelivered() == ()


async def test_a_finished_analysis_is_shared_as_it_is_without_running_again(campaign: Campaign) -> None:
    tracelens = _Analysis("tracelens")
    toolbox = await campaign.toolbox(tracelens)
    first = await toolbox.submit_analysis("b1", "tracelens", {}, PROFILE)
    await _jobs(toolbox).wait(first)

    later = await toolbox.submit_analysis("b2", "tracelens", {"ignored": True}, PROFILE)

    assert later == first and len(tracelens.requests) == 1
    assert await campaign.inbox("b2") == await campaign.inbox("b1")
    assert await campaign.inbox("b3") == [], "a branch that did not ask is sent nothing"


async def test_a_failed_analysis_is_not_reused_and_the_next_submission_starts_a_new_job(campaign: Campaign) -> None:
    tracelens = _Analysis("tracelens", failures=1)
    toolbox = await campaign.toolbox(tracelens)
    failed = await toolbox.submit_analysis("b1", "tracelens", {}, PROFILE)
    assert await _jobs(toolbox).wait(failed) is JobStatus.FAILED

    retried = await toolbox.submit_analysis("b2", "tracelens", {}, PROFILE)
    assert retried != failed and await _jobs(toolbox).wait(retried) is JobStatus.SUCCEEDED
    shared = await toolbox.submit_analysis("b3", "tracelens", {}, PROFILE)

    assert shared == retried and len(tracelens.requests) == 2
    assert [job for job, _ in await campaign.inbox("b3")] == [retried]


async def test_two_tools_on_one_profile_are_two_jobs(campaign: Campaign) -> None:
    tracelens, roofline = _Analysis("tracelens", held=True), _Analysis("roofline", held=True)
    toolbox = await campaign.toolbox(tracelens, roofline)

    analysed, bounded = await _submit_all(toolbox), await _submit_all(toolbox, "roofline")

    assert len(set(analysed)) == len(set(bounded)) == 1 and analysed[0] != bounded[0]
    assert (len(tracelens.requests), len(roofline.requests)) == (1, 1)
    tracelens.release()
    roofline.release()
    for job in (analysed[0], bounded[0]):
        assert await _jobs(toolbox).wait(job) is JobStatus.SUCCEEDED


async def test_a_restarted_toolbox_shares_what_the_one_before_it_finished(campaign: Campaign) -> None:
    before = await campaign.toolbox(_Analysis("tracelens"))
    first = await before.submit_analysis("b1", "tracelens", {}, PROFILE)
    await _jobs(before).wait(first)
    await before.close()
    tracelens = _Analysis("tracelens")

    after = await campaign.toolbox(tracelens)
    shared = await after.submit_analysis("b2", "tracelens", {}, PROFILE)

    assert shared == first and tracelens.requests == []
    assert after.tool_status(shared) is JobStatus.SUCCEEDED
    assert await campaign.inbox("b2") == await campaign.inbox("b1")


async def test_a_job_running_when_the_toolbox_stopped_is_not_shared_after_the_restart(campaign: Campaign) -> None:
    before = await campaign.toolbox(_Analysis("tracelens", held=True))
    lost = await before.submit_analysis("b1", "tracelens", {}, PROFILE)
    await before.close()
    tracelens = _Analysis("tracelens")

    after = await campaign.toolbox(tracelens)
    fresh = await after.submit_analysis("b2", "tracelens", {}, PROFILE)

    assert fresh != lost and await _jobs(after).wait(fresh) is JobStatus.SUCCEEDED and len(tracelens.requests) == 1


async def test_a_branch_that_stops_leaves_a_shared_job_running_for_the_others(campaign: Campaign) -> None:
    tracelens = _Analysis("tracelens", held=True)
    toolbox = await campaign.toolbox(tracelens)
    job = await toolbox.submit_analysis("b1", "tracelens", {}, PROFILE)
    assert await toolbox.submit_analysis("b2", "tracelens", {}, PROFILE) == job

    assert toolbox.cancel_tool_jobs("b1") == 0 and toolbox.tool_status(job) is JobStatus.RUNNING
    tracelens.release()
    assert await _jobs(toolbox).wait(job) is JobStatus.SUCCEEDED
    assert [found for found, _ in await campaign.inbox("b2")] == [job] and await campaign.inbox("b1") == []


async def test_the_last_branch_that_stops_cancels_the_job(campaign: Campaign) -> None:
    toolbox = await campaign.toolbox(_Analysis("tracelens", held=True))
    job = await toolbox.submit_analysis("b1", "tracelens", {}, PROFILE)
    await toolbox.submit_analysis("b2", "tracelens", {}, PROFILE)

    assert toolbox.cancel_tool_jobs("b1") + toolbox.cancel_tool_jobs("b2") == 1
    assert await _jobs(toolbox).wait(job) is JobStatus.CANCELLED


async def test_an_unreadable_ledger_shares_nothing_and_says_so(
    campaign: Campaign, caplog: pytest.LogCaptureFixture
) -> None:
    before = await campaign.toolbox(_Analysis("tracelens"))
    first = await before.submit_analysis("b1", "tracelens", {}, PROFILE)
    await _jobs(before).wait(first)
    await before.close()
    ledger = next((campaign.tmp_path / "session").rglob(SHARED_JOBS_FILE))
    assert first in json.dumps(json.loads(ledger.read_text()))
    ledger.write_text("{not json")
    tracelens = _Analysis("tracelens")

    after = await campaign.toolbox(tracelens)

    fresh = await after.submit_analysis("b2", "tracelens", {}, PROFILE)
    assert fresh != first and await _jobs(after).wait(fresh) is JobStatus.SUCCEEDED and len(tracelens.requests) == 1
    assert "the shared tool-job ledger" in caplog.text and "cannot be read" in caplog.text
