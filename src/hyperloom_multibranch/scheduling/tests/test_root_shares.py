# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.root_shares import WAIT_SECONDS, ShareSources, root_shares
from hyperloom_multibranch.specialist.agent_onboarding import ROOT_PROFILE_SOURCE
from hyperloom_multibranch.toolbox.tools.base import JobStatus

ROOT, JOB = "c0ffee", "prelude:p1"
OPS = OpShares(ROOT, JOB, "", {"gemm": 48.0, "attention": 31.0})


class Toolbox:
    """A toolbox whose job reports ``statuses`` in turn, then the last one for good; no job at all with none."""

    def __init__(self, *statuses: JobStatus) -> None:
        self.statuses, self.asked = list(statuses), 0

    def tool_status(self, job_id: str) -> JobStatus:
        self.asked += 1
        if not self.statuses:
            raise LookupError(f"no job {job_id}")
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


class Records:
    """The blackboard, holding the root's ``shares`` from its ``landing``-th read of them on."""

    def __init__(self, shares: OpShares | None, landing: int = 1) -> None:
        self.shares, self.landing, self.reads = shares, landing, 0

    async def op_shares(self, commit_id: str) -> OpShares | None:
        self.reads += 1
        return self.shares if commit_id == ROOT and self.reads >= self.landing else None


def _clock() -> Clock:
    now = [1000.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    return Clock(now=lambda: now[0], sleep=sleep)


def _sources(records: Records, toolbox: Toolbox, clock: Clock) -> ShareSources:
    return ShareSources(records, toolbox, clock)  # type: ignore[arg-type]


async def test_the_root_s_op_shares_are_read_once_its_profile_job_ended() -> None:
    toolbox, clock = Toolbox(JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.SUCCEEDED), _clock()

    read = await root_shares(_sources(Records(OPS), toolbox, clock), ROOT, JOB)

    assert read.shares is not None and (read.shares.shares, read.shares.source) == (OPS, ROOT_PROFILE_SOURCE)
    assert (read.detail["job"], read.detail["taken"], read.detail["ops"]) == ("succeeded", "op shares", 2)
    assert toolbox.asked == 3 and clock.now() == 1000.0 + 2 * 5.0  # two polls while it ran; its shares were there


async def test_shares_that_reach_the_blackboard_after_their_job_ended_are_waited_for() -> None:
    records, clock = Records(OPS, landing=3), _clock()

    read = await root_shares(_sources(records, Toolbox(JobStatus.SUCCEEDED), clock), ROOT, JOB)

    assert read.shares is not None and read.shares.shares == OPS and read.detail["taken"] == "op shares"
    assert (records.reads, clock.now()) == (3, 1000.0 + 2 * 5.0)


async def test_a_job_that_succeeded_without_shares_is_waited_on_until_the_deadline_and_a_failed_one_is_not() -> None:
    clock = _clock()
    empty = await root_shares(_sources(Records(None), Toolbox(JobStatus.SUCCEEDED), clock), ROOT, JOB)
    assert (empty.shares, empty.detail["job"], empty.detail["taken"]) == (None, "succeeded", "none: no op shares")
    assert clock.now() == 1000.0 + WAIT_SECONDS

    clock, records = _clock(), Records(None)
    failed = await root_shares(_sources(records, Toolbox(JobStatus.FAILED), clock), ROOT, JOB)
    assert (failed.shares, failed.detail["job"], records.reads, clock.now()) == (None, "failed", 1, 1000.0)


async def test_without_a_job_or_past_the_wait_there_is_nothing_to_map_from() -> None:
    none = await root_shares(_sources(Records(None), Toolbox(), _clock()), ROOT, "")
    assert none.shares is None and none.detail["job"] == "no root profile was started"

    unknown = await root_shares(_sources(Records(None), Toolbox(), _clock()), ROOT, JOB)
    assert unknown.detail["job"] == "the toolbox knows no such job"

    clock, records = _clock(), Records(None)
    late = await root_shares(_sources(records, Toolbox(JobStatus.RUNNING), clock), ROOT, JOB)
    assert (late.shares, late.detail["job"]) == (None, "still running after 600 s")
    assert (late.detail["taken"], records.reads, clock.now()) == ("none: no op shares", 1, 1000.0 + WAIT_SECONDS)
