# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The flow the end-to-end skeleton runs on the local blackboard: submit a test, judge it, commit, merge, split."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, ClaimKey, Role
from hyperloom.contracts.measurement import AttachmentRef
from hyperloom_multibranch.contracts.blackboard.peers import Purpose, TestKind, Verdict
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    LineageEvent,
    MergeAttempt,
    RegressionCase,
    ToolResultPointer,
)
from hyperloom_multibranch.contracts.blackboard.vocab import (
    ClaimStatus,
    CommitOrigin,
    Conclusion,
    DedupStatus,
    InboxKind,
    LineageKind,
    MergeOutcome,
)
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import TTL, delta, intent, judgement, request, result

BASELINE = {"output_tps": 1000.0}


@dataclass
class Session:
    scheduler: LocalBlackboard
    toolbox: LocalBlackboard
    gate: LocalBlackboard
    integrator: LocalBlackboard
    s1: LocalBlackboard
    s2: LocalBlackboard


async def _measure(session: Session, specialist: LocalBlackboard, request_id: str, of_intent, change) -> None:
    claim = await specialist.acquire(of_intent, change.fingerprint(), ttl_seconds=TTL)
    await session.toolbox.write_request(request(request_id, of_intent, change))
    await session.toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)
    await session.toolbox.write_result(result(request_id))


@pytest.fixture
async def session(open_as: Callable[..., LocalBlackboard]) -> Session:
    """Two specialists from the root c0; s1 commits c1, s2 records a flat result, the integrator merges c1 as i1."""

    s = Session(
        scheduler=open_as(Role.SCHEDULER),
        toolbox=open_as(Role.TOOLBOX),
        gate=open_as(Role.GATE),
        integrator=open_as(Role.INTEGRATOR),
        s1=open_as(Role.SPECIALIST_RUNTIME, "s1"),
        s2=open_as(Role.SPECIALIST_RUNTIME, "s2"),
    )
    await s.scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await s.gate.write_baseline(BaselineRecord("c0", BASELINE, {"output_tps": 0.8}, {"output_tps": 1.6}, ("b1", "b2")))
    for branch in ("s1", "s2"):
        await s.scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0", reason="first batch"))

    await _measure(s, s.s1, "r1", intent("s1", "c0"), delta("--max-num-seqs 256"))
    await s.gate.write_judgement(
        "r1", judgement(4.0, ("r1",), recorded={"output_tps": 1040.0}), Conclusion.COMMITTED, "c1"
    )

    await _measure(s, s.s2, "r2", intent("s2", "c0", "attention", "split kv"), delta("--block-size 32"))
    flat = judgement(0.2, ("r2",), verdict=Verdict.FLAT)
    await s.gate.write_judgement("r2", flat, Conclusion.RECORDED_ONLY)

    merge = intent(INTEGRATION_BRANCH, "c0", "merge", "c1")
    await s.toolbox.write_request(request("m1", merge, delta("--max-num-seqs 256")))
    await s.integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))
    await s.toolbox.write_result(result("m1"))
    merged = judgement(4.1, ("m1",), recorded={"output_tps": 1041.0})
    await s.integrator.write_judgement("m1", merged, Conclusion.MERGED, "i1")
    return s


async def test_a_committed_test_becomes_a_commit_and_the_merge_brings_it_into_integration(session: Session) -> None:
    view = await session.scheduler.lineage()

    commits = {node.commit_id: node for node in view.commits}
    assert {commit_id: node.origin for commit_id, node in commits.items()} == {
        "c0": CommitOrigin.ROOT,
        "c1": CommitOrigin.COMMIT,
        "i1": CommitOrigin.MERGE,
    }
    assert (commits["c1"].branch, commits["c1"].parent_commit, commits["c1"].request_id) == ("s1", "c0", "r1")
    assert (commits["i1"].parent_commit, commits["i1"].merged_from) == ("c0", ("c1",))
    assert {branch.branch: branch.head_commit for branch in view.branches} == {
        INTEGRATION_BRANCH: "i1",
        "s1": "c1",
        "s2": "c0",
    }
    assert view.ancestry("i1") == ("i1", "c0", "c1")


async def test_the_integrator_sees_each_commit_with_its_gain_and_where_it_was_merged(session: Session) -> None:
    (c1,) = await session.integrator.committed()

    assert (c1.commit_id, c1.branch, c1.gain_pct, c1.merged_into) == ("c1", "s1", 4.0, "i1")
    assert c1.last_merge is not None
    assert (c1.last_merge.outcome, c1.last_merge.merged_commit) == (MergeOutcome.MERGED, "i1")


async def test_recorded_values_come_from_the_baseline_and_from_the_judgements_that_made_each_commit(
    session: Session,
) -> None:
    root, c1, i1 = [await session.gate.recorded_value(commit) for commit in ("c0", "c1", "i1")]

    assert (dict(root.values), root.request_ids) == (BASELINE, ("b1", "b2"))
    assert (dict(c1.values), c1.request_ids) == ({"output_tps": 1040.0}, ("r1",))
    assert dict(i1.values) == {"output_tps": 1041.0}


async def test_progress_counts_each_branchs_tests_commits_and_gpu_time(session: Session) -> None:
    progress = {row.branch: row for row in await session.scheduler.progress()}

    assert (progress["s1"].tests, progress["s1"].commits, progress["s1"].gpu_seconds) == (1, 1, 600.0)
    assert dict(progress["s1"].head_value) == {"output_tps": 1040.0}
    assert (progress["s2"].tests, progress["s2"].commits, progress["s2"].head_commit) == (1, 0, "c0")
    assert (progress[INTEGRATION_BRANCH].commits, progress[INTEGRATION_BRANCH].head_commit) == (1, "i1")
    assert all(row.running and row.frontier is None for row in progress.values())


async def test_a_result_completes_the_claim_and_the_same_action_on_that_state_is_refused(session: Session) -> None:
    s1_intent = intent("s1", "c0")
    key = ClaimKey("c0", s1_intent.fingerprint)

    answer = await session.s2.lookup(key, delta("--max-num-seqs 256").fingerprint())
    claim = session.s1._leases.current(key)

    assert claim is not None
    assert (claim.status, claim.request_id) == (ClaimStatus.COMPLETED, "r1")
    assert answer.status is DedupStatus.TESTED
    # The merge test of c1 onto the root applied the same change to the same state, so it counts too.
    assert [record.request_id for record in answer.records] == ["r1", "m1"]


async def test_a_fill_specialist_is_briefed_on_everything_tried_on_the_states_it_builds_on(session: Session) -> None:
    brief = await session.scheduler.startup_brief("i1")

    tried = {(item.request_id, item.target, item.mechanism, item.conclusion) for item in brief.tried}
    assert tried == {
        ("r1", "moe gemm", "tuned kernel", Conclusion.COMMITTED),
        ("r2", "attention", "split kv", Conclusion.RECORDED_ONLY),
    }
    assert {item.request_id: item.change for item in brief.tried} == {
        "r1": "add args --max-num-seqs 256",
        "r2": "add args --block-size 32",
    }
    assert (brief.leads, brief.lessons) == ((), ())
    assert brief.op_shares is None


async def test_the_brief_carries_the_latest_op_shares_reported_on_its_start_commit(session: Session) -> None:
    for job, share in (("job-1", 30.0), ("job-2", 41.5)):
        await session.toolbox.notify_tool_result(
            ToolResultPointer("s1", job, "/a", "trace analysed", "c1", "", {"gemm": share})
        )

    brief = await session.scheduler.startup_brief("c1")

    assert brief.op_shares is not None and (brief.op_shares.job_id, dict(brief.op_shares.shares)) == (
        "job-2",
        {"gemm": 41.5},
    )
    assert (await session.scheduler.startup_brief("i1")).op_shares is None


async def test_a_branch_inbox_gets_its_results_judgements_and_lineage_once(session: Session) -> None:
    first = await session.s1.poll("s1")
    again = await session.s1.poll("s1", first.cursor)

    assert [(item.kind, item.pointer) for item in first.items if item.kind is not InboxKind.LINEAGE] == [
        (InboxKind.RESULT, "r1"),
        (InboxKind.JUDGEMENT, "r1"),
    ]
    assert [item.kind for item in first.items].count(InboxKind.LINEAGE) == 1
    assert again.items == ()


async def test_a_stopped_branch_reads_as_stopped_and_gives_up_its_claims(session: Session) -> None:
    await session.s2.acquire(
        intent("s2", "c0", "sampler", "fused topk"), delta("--top-k 8").fingerprint(), ttl_seconds=TTL
    )
    await session.scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s2", reason="no progress"))
    released = await session.scheduler.release_branch("s2")

    progress = {row.branch: row for row in await session.scheduler.progress()}
    view = await session.scheduler.lineage()

    assert not progress["s2"].running and progress["s1"].running
    assert {branch.branch: branch.stop_reason for branch in view.branches}["s2"] == "no progress"
    assert [claim.status for claim in released] == [ClaimStatus.RELEASED]


async def test_the_split_branch_is_the_same_whichever_the_gate_writes_first(session: Session) -> None:
    fix = intent("s1", "c1", "fp8 overflow", "clamp scales")
    await _measure(session, session.s1, "r3", fix, delta("--kv-cache-dtype fp8"))
    slower = judgement(-2.0, ("r3",), verdict=Verdict.REGRESSED, fixes_correctness=True)
    await session.gate.write_judgement("r3", slower, Conclusion.SPLIT_COMMITTED, "c2")
    split = LineageEvent(LineageKind.SPLIT, "s1-fix", commit_id="c2", from_commit="c1", from_branch="s1", reason="fix")
    await session.gate.record_lineage(split)

    view = await session.scheduler.lineage()

    branch = {node.branch: node for node in view.branches}["s1-fix"]
    assert (branch.origin, branch.start_commit, branch.from_branch, branch.head_commit) == (
        LineageKind.SPLIT,
        "c1",
        "s1",
        "c2",
    )
    assert view.commit("c2").branch == "s1-fix"


@pytest.mark.parametrize("waiting", [Conclusion.SPLIT_REQUESTED, Conclusion.SPLIT_PENDING])
async def test_a_correctness_fix_that_costs_performance_splits_and_its_case_guards_the_descendants(
    session: Session, waiting: Conclusion
) -> None:
    fix = intent("s1", "c1", "fp8 overflow", "clamp scales")
    await _measure(session, session.s1, "r3", fix, delta("--kv-cache-dtype fp8"))
    case = AttachmentRef("fp8_overflow.py", "file", "b" * 64)
    repro = request(
        "r4", fix, delta("--kv-cache-dtype fp8"), kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(case,)
    )
    await session.toolbox.write_request(repro)
    slower = judgement(-2.0, ("r3",), verdict=Verdict.REGRESSED, fixes_correctness=True, foundation_fix=True)
    await session.gate.write_judgement("r3", slower, waiting)
    split = LineageEvent(LineageKind.SPLIT, "s1-fix", commit_id="c2", from_commit="c1", from_branch="s1", reason="fix")
    await session.gate.record_lineage(split)
    await session.gate.write_judgement("r3", slower, Conclusion.SPLIT_COMMITTED, "c2")
    await session.gate.add_regression_case(RegressionCase("repro/fp8_overflow.py", "c2", "r4"))

    view = await session.scheduler.lineage()
    committed = await session.integrator.committed()
    inbox = await session.s1.poll("s1")

    c2 = view.commit("c2")
    assert c2 is not None
    assert (c2.branch, c2.parent_commit, c2.origin) == ("s1-fix", "c1", CommitOrigin.SPLIT)
    assert [commit.commit_id for commit in committed] == ["c2", "c1"]
    assert [case.case_ref for case in await session.gate.regression_cases_for("c2")] == ["repro/fp8_overflow.py"]
    assert await session.gate.regression_cases_for("c1") == ()
    assert any(item.kind is InboxKind.LINEAGE and "split s1-fix" in item.summary for item in inbox.items)
