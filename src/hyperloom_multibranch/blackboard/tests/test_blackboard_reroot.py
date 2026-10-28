# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch its runtime moved onto an integration commit: where its head stands, what waits on it, who may move it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom.contracts.measurement import RoundValue
from hyperloom_multibranch.contracts.blackboard import (
    INTEGRATION_BRANCH,
    ConflictingWrite,
    NotRecorded,
    Role,
    WriterNotAllowed,
)
from hyperloom_multibranch.contracts.blackboard.peers import Purpose, TestRequest, TestResult, Verdict
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    BranchProgress,
    LineageEvent,
    MergeAttempt,
    PendingCandidate,
)
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import intent, judgement, profile, request, result

Open = Callable[..., LocalBlackboard]
METRIC = "output_tps"
REROOT = LineageEvent(LineageKind.REROOT, "s1", commit_id="i1", from_commit="c1", reason="behind integration")
ON_I1 = intent("s1", "i1", "attention", "fused rope")


def _measured(request_id: str, value: float) -> TestResult:
    rounds = (RoundValue("warmup", 64, {METRIC: 1.0}), RoundValue("measure", 64, {METRIC: value}))
    return replace(result(request_id), rounds=rounds)


async def _commit(open_as: Open, tried: TestRequest, commit: str, value: float) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(tried)
    await toolbox.write_result(result(tried.request_id))
    kept = judgement(2.0, (tried.request_id,), recorded={METRIC: value})
    await open_as(Role.GATE).write_judgement(tried.request_id, kept, Conclusion.COMMITTED, commit)


async def _row(open_as: Open, branch: str) -> BranchProgress:
    return next(row for row in await open_as(Role.OBSERVER).progress() if row.branch == branch)


@pytest.fixture
async def session(open_as: Open) -> Open:
    """The root ``c0`` baselined at 1000 with a bar of 1.6%; ``s1`` committed ``c1`` (1020) on it and ``s2`` ``c2``,
    which integration merged as ``i1`` (1050)."""

    scheduler, toolbox, integrator = open_as(Role.SCHEDULER), open_as(Role.TOOLBOX), open_as(Role.INTEGRATOR)
    await scheduler.write_session_profile(profile())
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    baseline = BaselineRecord("c0", {METRIC: 1000.0}, {METRIC: 0.8}, {METRIC: 1.6}, ("b1", "b2"))
    await open_as(Role.GATE).write_baseline(baseline)
    for branch, commit, value in (("s1", "c1", 1020.0), ("s2", "c2", 1050.0)):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0"))
        await _commit(open_as, request(f"r-{commit}", intent(branch, "c0")), commit, value)
    await toolbox.write_request(request("m1", intent(INTEGRATION_BRANCH, "c0", "merge", "c2")))
    await integrator.record_merge_attempt(MergeAttempt(("c2",), "c0", request_id="m1"))
    await toolbox.write_result(result("m1"))
    merged = judgement(5.0, ("m1",), recorded={METRIC: 1050.0})
    await integrator.write_judgement("m1", merged, Conclusion.MERGED, "i1")
    return open_as


async def test_a_rerooted_branch_stands_on_the_integration_commit_and_a_late_commit_from_its_old_head_stays_off_it(
    session: Open,
) -> None:
    await session(Role.TOOLBOX).write_request(request("flat", intent("s1", "c1", "sampler", "fused top-k")))
    await session(Role.TOOLBOX).write_result(result("flat"))
    flat = judgement(0.2, ("flat",), verdict=Verdict.FLAT)
    await session(Role.GATE).write_judgement("flat", flat, Conclusion.RECORDED_ONLY)
    await session(Role.SPECIALIST_RUNTIME, "s1").record_lineage(REROOT)
    rerooted = await _row(session, "s1")
    await _commit(session, request("late", intent("s1", "c1", "norm", "fused rms norm")), "c3", 1030.0)
    after_late = await _row(session, "s1")
    await _commit(session, request("on-i1", ON_I1), "c4", 1071.0)

    assert (rerooted.start_commit, rerooted.head_commit, rerooted.head_value) == ("c0", "i1", {METRIC: 1050.0})
    assert rerooted.judged_since_commit == 1
    assert after_late.head_commit == "i1"
    assert (await session(Role.OBSERVER).snapshot()).lineage.commit("c3") is not None
    assert (await _row(session, "s1")).head_commit == "c4"


async def test_an_explore_on_the_new_head_waits_for_its_confirm_once_the_branch_is_rerooted(session: Open) -> None:
    toolbox = session(Role.TOOLBOX)
    await toolbox.write_request(request("r1", ON_I1))
    await toolbox.write_result(_measured("r1", 1071.0))
    await toolbox.write_request(replace(request("r1c", ON_I1), purpose=Purpose.CONFIRM, retest_of="r1"))
    before = (await _row(session, "s1")).pending
    await session(Role.SPECIALIST_RUNTIME, "s1").record_lineage(REROOT)

    assert before == ()
    assert (await _row(session, "s1")).pending == (
        PendingCandidate("r1", "r1c", "i1", METRIC, 1071.0, pytest.approx(2.0)),
    )


async def test_only_a_branchs_own_runtime_reroots_it_while_it_runs_and_only_onto_an_integration_commit(
    session: Open,
) -> None:
    runtime = session(Role.SPECIALIST_RUNTIME, "s1")
    with pytest.raises(WriterNotAllowed, match="may not write this"):
        await session(Role.SCHEDULER).record_lineage(REROOT)
    with pytest.raises(WriterNotAllowed, match="not s1"):
        await session(Role.SPECIALIST_RUNTIME, "s2").record_lineage(REROOT)
    for onto in ("c2", "zz"):
        with pytest.raises(NotRecorded, match=f"no commit {onto}"):
            await runtime.record_lineage(replace(REROOT, commit_id=onto))
    with pytest.raises(ValueError, match="needs from_commit"):
        await runtime.record_lineage(replace(REROOT, from_commit=""))
    with pytest.raises(ValueError, match="never by a re-root"):
        await session(Role.SPECIALIST_RUNTIME, INTEGRATION_BRANCH).record_lineage(
            replace(REROOT, branch=INTEGRATION_BRANCH)
        )
    with pytest.raises(NotRecorded, match="never started"):
        await session(Role.SPECIALIST_RUNTIME, "s9").record_lineage(replace(REROOT, branch="s9"))
    await runtime.record_lineage(REROOT)
    await runtime.record_lineage(REROOT)
    await session(Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="no progress"))

    with pytest.raises(ConflictingWrite, match="already stopped"):
        await runtime.record_lineage(replace(REROOT, commit_id="c0", from_commit="i1"))
    assert len(await session(Role.OBSERVER).search("reroot")) == 1
