# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may write what, what a rewrite does, and what each query returns on its own."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import DIGEST_PREFIX, PatchRef, RecipeDelta
from hyperloom_multibranch.contracts.blackboard import (
    INTEGRATION_BRANCH,
    ClaimKey,
    ConflictingWrite,
    NotRecorded,
    Role,
    UnknownRequest,
    Writer,
    WriterNotAllowed,
    interfaces,
)
from hyperloom_multibranch.contracts.blackboard.peers import (
    BranchLogEntry,
    BranchLogType,
    HypothesisStatus,
    IntentLayer,
    Purpose,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.repo import ResolvedPatch
from hyperloom_multibranch.contracts.blackboard.records import (
    BaselineRecord,
    BuildNotice,
    Learning,
    LineageEvent,
    LlmUsage,
    MergeAttempt,
    QuarantineEvent,
    TestRecord,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.blackboard.vocab import (
    BranchDriver,
    BuildOutcome,
    ClaimStatus,
    Conclusion,
    DedupStatus,
    InboxKind,
    LearningKind,
    LineageKind,
    MergeOutcome,
    QuarantineAction,
    UsageSource,
)
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import FakeClock, TTL, delta, intent, judgement, request, result
from hyperloom_multibranch.blackboard.tests.sessions import scripted_session

Open = Callable[..., LocalBlackboard]
ROOT_RECIPE = DIGEST_PREFIX + "a" * 64
CHILD_RECIPE = DIGEST_PREFIX + "b" * 64
#: A past session's lesson, as the RSI lesson pool names one.
LESSON = "lsn-0123456789ab"


async def _tested(open_as: Open, request_id: str, branch: str = "s1", parent: str = "c0") -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request(request_id, intent(branch, parent)))
    await toolbox.write_result(result(request_id))


@pytest.mark.parametrize(
    ("kind", "validity", "conclusion", "judged_at", "judged"),
    [
        (TestKind.TEST, Validity.SUCCEEDED, Conclusion.RECORDED_ONLY, 30.0, True),
        (TestKind.TEST, Validity.CANDIDATE_FAILED, Conclusion.RECORDED_ONLY, 30.0, True),
        (TestKind.TEST, Validity.SUCCEEDED, None, None, False),
        (TestKind.PROFILE, Validity.SUCCEEDED, Conclusion.RECORDED_ONLY, 30.0, False),
        (TestKind.TEST, Validity.MEASUREMENT_FAILED, Conclusion.RECORDED_ONLY, 30.0, False),
        (TestKind.TEST, Validity.SUCCEEDED, Conclusion.UNVERIFIED, 30.0, False),
    ],
)
def test_a_record_is_judged_once_the_gate_judged_a_test_it_measured(
    kind: TestKind, validity: Validity, conclusion: Conclusion | None, judged_at: float | None, judged: bool
) -> None:
    record = TestRecord(
        request("r1", intent("s1", "c0"), kind=kind),
        result=result("r1", validity=validity),
        conclusion=conclusion,
        judged_at=judged_at,
    )

    assert record.judged is judged


def test_the_local_blackboard_implements_every_interface_method() -> None:
    names = [name for name, member in inspect.getmembers(interfaces.Blackboard) if inspect.iscoroutinefunction(member)]

    assert len(names) == 46
    assert all(getattr(LocalBlackboard, name) is not getattr(interfaces.Blackboard, name) for name in names)


@pytest.mark.parametrize(
    ("writer", "message"), [(Writer(Role.SPECIALIST_RUNTIME, "s1"), "branch"), (Writer(Role.GATE, "a b"), "name")]
)
def test_a_writer_needs_a_safe_name_and_a_runtime_needs_its_branch(root: Path, writer: Writer, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LocalBlackboard(root, writer)


async def test_rewriting_a_request_or_result_is_idempotent_and_a_different_one_conflicts(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    asked = request("r1", intent("s1", "c0"))
    await toolbox.write_request(asked)
    await toolbox.write_request(asked)
    await toolbox.write_result(result("r1"))
    await toolbox.write_result(result("r1"))

    with pytest.raises(ConflictingWrite):
        await toolbox.write_request(replace(asked, parent_commit="c9"))
    with pytest.raises(ConflictingWrite):
        await toolbox.write_result(result("r1", gpu_seconds=1.0))
    assert len(toolbox._state().entries) == 2


async def test_results_and_judgements_need_the_request_and_judgements_need_the_result(open_as: Open) -> None:
    toolbox, gate = open_as(Role.TOOLBOX), open_as(Role.GATE)
    await toolbox.write_request(request("r1", intent("s1", "c0")))

    with pytest.raises(UnknownRequest):
        await toolbox.write_result(result("r-missing"))
    with pytest.raises(UnknownRequest):
        await gate.write_judgement("r-missing", judgement(1.0, ("r-missing",)), Conclusion.RECORDED_ONLY)
    with pytest.raises(NotRecorded):
        await gate.write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.RECORDED_ONLY)


async def test_the_gate_judges_specialist_branches_and_the_integrator_the_integration_branch(open_as: Open) -> None:
    await _tested(open_as, "r1")
    await _tested(open_as, "m1", branch=INTEGRATION_BRANCH)
    gate, integrator = open_as(Role.GATE), open_as(Role.INTEGRATOR)

    with pytest.raises(WriterNotAllowed):
        await integrator.write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.COMMITTED, "c1")
    with pytest.raises(WriterNotAllowed):
        await gate.write_judgement("m1", judgement(1.0, ("m1",)), Conclusion.MERGED, "i1")
    with pytest.raises(WriterNotAllowed):
        await gate.write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.MERGED, "i1")
    with pytest.raises(ValueError, match="names a commit"):
        await gate.write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.COMMITTED)
    with pytest.raises(ValueError, match="names a commit"):
        await gate.write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.RECORDED_ONLY, "c1")


async def test_a_retest_appends_a_judgement_and_the_latest_one_is_in_force(open_as: Open) -> None:
    await _tested(open_as, "r1")
    await _tested(open_as, "r1b")
    gate = open_as(Role.GATE)

    provisional = replace(judgement(5.0, ("r1",)), verdict=Verdict.NEEDS_CONFIRM)
    await gate.write_judgement("r1", provisional, Conclusion.RECORDED_ONLY)
    record = await gate.write_judgement("r1", judgement(3.5, ("r1", "r1b")), Conclusion.COMMITTED, "c1")

    assert (record.conclusion, record.commit_id, record.judgement.request_ids) == (
        Conclusion.COMMITTED,
        "c1",
        ("r1", "r1b"),
    )
    assert [write.judgement.gain_pct for write in gate._state().judgements["r1"]] == [5.0, 3.5]


@pytest.mark.parametrize(
    ("role", "event"),
    [
        (Role.TOOLBOX, LineageEvent(LineageKind.OPEN, "s1", from_commit="c0")),
        (Role.SCHEDULER, LineageEvent(LineageKind.SPLIT, "s1-fix", commit_id="c2", from_commit="c1", from_branch="s1")),
        (Role.GATE, LineageEvent(LineageKind.STOP, "s1")),
        (Role.GATE, LineageEvent(LineageKind.PROBATION, "s1", tests=4)),
    ],
)
async def test_each_lineage_event_has_one_writer(open_as: Open, role: Role, event: LineageEvent) -> None:
    with pytest.raises(WriterNotAllowed):
        await open_as(role).record_lineage(event)


@pytest.mark.parametrize(
    "event",
    [
        LineageEvent(LineageKind.OPEN, "s1"),
        LineageEvent(LineageKind.ROOT, "s1", commit_id="c0"),
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH),
        LineageEvent(LineageKind.OPEN, "s1", from_commit="c0", recipe_digest=ROOT_RECIPE),
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0", recipe_digest="sha256:c0"),
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0", driver=BranchDriver.AUTONOMOUS),
        LineageEvent(LineageKind.STOP, "s1", driver=BranchDriver.AUTONOMOUS),
        LineageEvent(LineageKind.PROBATION, "s1"),
        LineageEvent(LineageKind.STOP, "s1", tests=4),
        LineageEvent(LineageKind.PROBATION, INTEGRATION_BRANCH, tests=4),
    ],
)
async def test_a_lineage_event_names_what_its_kind_needs(open_as: Open, event: LineageEvent) -> None:
    with pytest.raises(ValueError):
        await open_as(Role.SCHEDULER).record_lineage(event)


async def test_a_branch_an_autonomous_agent_drives_is_known_from_its_open_even_when_it_stops_at_once(
    open_as: Open,
) -> None:
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b01", from_commit="c0"))
    await scheduler.record_lineage(
        LineageEvent(LineageKind.OPEN, "b02", from_commit="c0", driver=BranchDriver.AUTONOMOUS)
    )
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "b02", reason="its agent did not start"))

    drivers = {node.branch: node.driver for node in (await scheduler.lineage()).branches}
    assert drivers == {
        INTEGRATION_BRANCH: BranchDriver.SPECIALIST,
        "b01": BranchDriver.SPECIALIST,
        "b02": BranchDriver.AUTONOMOUS,
    }


async def test_a_running_branch_is_held_on_probation_once_a_stall(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    first = LineageEvent(LineageKind.PROBATION, "s1", reason="drawn", tests=4)

    for event in (first, replace(first, reason="drawn again", tests=6), replace(first, commit_id="c1")):
        await scheduler.record_lineage(event)
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s1", reason="judge: stalled"))

    branches = {node.branch: node for node in (await scheduler.lineage()).branches}
    assert [(item.tests, item.since_commit) for item in branches["s1"].probations] == [(4, ""), (4, "c1")]
    assert branches["s1"].stop_reason == "judge: stalled"
    with pytest.raises(ConflictingWrite):
        await scheduler.record_lineage(replace(first, commit_id="c2"))
    with pytest.raises(NotRecorded):
        await scheduler.record_lineage(replace(first, branch="s9"))


async def test_a_commit_keeps_the_recipe_digest_it_was_recorded_with(open_as: Open) -> None:
    scheduler, gate = open_as(Role.SCHEDULER), open_as(Role.GATE)
    root = LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0", recipe_digest=ROOT_RECIPE)
    await scheduler.record_lineage(root)
    await _tested(open_as, "r1")
    await _tested(open_as, "r2")
    await gate.write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1", recipe_digest=CHILD_RECIPE)

    with pytest.raises(ConflictingWrite):
        await gate.write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1", recipe_digest=ROOT_RECIPE)
    with pytest.raises(ConflictingWrite):
        await scheduler.record_lineage(replace(root, recipe_digest=CHILD_RECIPE))
    with pytest.raises(ValueError, match="names the recipe"):
        await gate.write_judgement("r2", judgement(0.1, ("r2",)), Conclusion.RECORDED_ONLY, recipe_digest=ROOT_RECIPE)
    with pytest.raises(ValueError, match="64 hex digits"):
        await gate.write_judgement("r2", judgement(3.0, ("r2",)), Conclusion.COMMITTED, "c2", recipe_digest="sha256:c2")
    lineage = await gate.lineage()
    assert {node.commit_id: node.recipe_digest for node in lineage.commits} == {"c0": ROOT_RECIPE, "c1": CHILD_RECIPE}


async def test_a_commit_no_tool_profiled_has_the_op_shares_of_the_last_profiled_commit_of_its_recipe(
    open_as: Open,
) -> None:
    scheduler, gate, toolbox = open_as(Role.SCHEDULER), open_as(Role.GATE), open_as(Role.TOOLBOX)
    await scheduler.record_lineage(
        LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0", recipe_digest=ROOT_RECIPE)
    )
    for n in (1, 2, 3):
        await _tested(open_as, f"r{n}", branch=f"s{n}")
        await gate.write_judgement(
            f"r{n}", judgement(3.0, (f"r{n}",)), Conclusion.COMMITTED, f"c{n}", recipe_digest=CHILD_RECIPE
        )

    async def profiled(job: str, commit: str) -> None:
        await toolbox.notify_tool_result(
            ToolResultPointer("s1", job, "/a", "trace analysed", commit, "", {"gemm": 30.0})
        )

    await profiled("job-1", "c1")
    await profiled("job-0", "c0")
    await profiled("job-3", "c3")

    shared = await toolbox.op_shares("c2")
    assert shared is not None and (shared.commit_id, shared.job_id) == ("c3", "job-3")
    assert (await toolbox.snapshot()).op_shares["c2"] == shared
    root = await toolbox.op_shares("c0")
    assert root is not None and root.job_id == "job-0"
    await profiled("job-2", "c2")
    own = await toolbox.op_shares("c2")
    assert own is not None and own.job_id == "job-2"


async def test_a_branch_starts_once_and_stops_once(open_as: Open) -> None:
    scheduler, gate = open_as(Role.SCHEDULER), open_as(Role.GATE)
    stopped = LineageEvent(LineageKind.STOP, "s1", reason="no progress")
    for writer, event in (
        (scheduler, LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0")),
        (scheduler, LineageEvent(LineageKind.OPEN, "s1", from_commit="c0")),
        (gate, LineageEvent(LineageKind.SPLIT, "s1-fix", commit_id="c2", from_commit="c1", from_branch="s1")),
        (scheduler, stopped),
    ):
        await writer.record_lineage(event)
        await writer.record_lineage(event)

    with pytest.raises(ConflictingWrite):
        await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c9"))
    with pytest.raises(ConflictingWrite):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1-fix", from_commit="c2"))
    with pytest.raises(ConflictingWrite):
        await scheduler.record_lineage(replace(stopped, reason="session ended"))
    with pytest.raises(NotRecorded):
        await scheduler.record_lineage(LineageEvent(LineageKind.STOP, "s9", reason="session ended"))
    await scheduler.record_lineage(LineageEvent(LineageKind.STOP, INTEGRATION_BRANCH, reason="session ended"))
    branches = {node.branch: node for node in (await scheduler.lineage()).branches}
    assert len(scheduler._state().lineage) == 5
    assert (branches["s1-fix"].origin, branches["s1-fix"].from_branch) == (LineageKind.SPLIT, "s1")
    assert (branches["s1"].stop_reason, branches[INTEGRATION_BRANCH].stop_reason) == ("no progress", "session ended")


async def test_a_withdrawn_test_is_settled_unmeasured_and_frees_its_claim(open_as: Open) -> None:
    runtime, toolbox = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    wanted = intent("s1", "c0")
    claim = await runtime.acquire(wanted, delta().fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("r1", wanted))
    await toolbox.attach_request(claim.key, "r1", ttl_seconds=TTL)

    record = await toolbox.write_withdrawal(Withdrawal("r1", "branch s1 stopped"))

    assert (record.in_flight, record.result, record.withdrawal_reason) == (False, None, "branch s1 stopped")
    assert await toolbox.write_withdrawal(Withdrawal("r1", "branch s1 stopped")) == record
    assert (await runtime.lookup(claim.key, delta().fingerprint())).status is DedupStatus.FREE


async def test_only_an_unmeasured_test_is_withdrawn_and_then_it_takes_no_result(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await _tested(open_as, "r1")
    await toolbox.write_request(request("r2", intent("s1", "c0")))
    await toolbox.write_withdrawal(Withdrawal("r2", "session stopped"))

    with pytest.raises(ConflictingWrite):
        await toolbox.write_withdrawal(Withdrawal("r1", "session stopped"))
    with pytest.raises(ConflictingWrite):
        await toolbox.write_result(result("r2"))
    with pytest.raises(ConflictingWrite):
        await toolbox.write_withdrawal(Withdrawal("r2", "branch s1 stopped"))
    with pytest.raises(UnknownRequest):
        await toolbox.write_withdrawal(Withdrawal("r-missing", "session stopped"))
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SCHEDULER).write_withdrawal(Withdrawal("r2", "session stopped"))


async def test_a_withdrawn_test_is_no_test_of_its_branch_and_no_direction_tried(open_as: Open) -> None:
    scheduler, toolbox = open_as(Role.SCHEDULER), open_as(Role.TOOLBOX)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    await _tested(open_as, "r1")
    await toolbox.write_request(request("r2", intent("s1", "c0", target="kv cache")))
    await toolbox.write_withdrawal(Withdrawal("r2", "branch s1 stopped"))

    (row,) = [row for row in await scheduler.progress() if row.branch == "s1"]
    brief = await scheduler.startup_brief("c0")

    assert row.tests == 1
    assert [item.request_id for item in brief.tried] == ["r1"]


async def test_the_brief_lists_the_start_commits_tests_first_then_the_nearest_commits_latest_first(
    tmp_path: Path,
) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())

    on_c1, on_merge = await blackboard.startup_brief("c1"), await blackboard.startup_brief("i1")

    assert [item.request_id for item in on_c1.tried] == ["r4", "r5", "r3", "r2", "r1"]
    assert [item.commit_id for item in on_merge.tried] == ["c1", "c0", "c0", "c0", "c0"]


async def test_a_brief_from_a_request_id_says_so_and_names_the_commit_its_test_was_made_on(tmp_path: Path) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())

    with pytest.raises(NotRecorded, match="that is a test's request_id, and start_commit names a commit, such as c0"):
        await blackboard.startup_brief("r1")
    with pytest.raises(NotRecorded, match="^no commit c404$"):
        await blackboard.startup_brief("c404")


async def test_a_failed_measurement_keeps_its_cost_frees_its_claim_and_takes_no_judgement(open_as: Open) -> None:
    runtime, toolbox = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    wanted = intent("s1", "c0")
    claim = await runtime.acquire(wanted, delta().fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("r1", wanted))
    await toolbox.attach_request(claim.key, "r1", ttl_seconds=TTL)
    failed = result("r1", validity=Validity.MEASUREMENT_FAILED, failure_reason="no healthy node", gpu_seconds=90.0)

    record = await toolbox.write_result(failed)

    assert (record.in_flight, record.measured, record.result) == (False, False, failed)
    assert (await runtime.lookup(claim.key, delta().fingerprint())).status is DedupStatus.FREE
    with pytest.raises(NotRecorded, match="failed"):
        await open_as(Role.GATE).write_judgement("r1", judgement(1.0, ("r1",)), Conclusion.RECORDED_ONLY)


async def test_a_failed_measurement_costs_its_branch_but_is_no_direction_tried(open_as: Open) -> None:
    scheduler, toolbox = open_as(Role.SCHEDULER), open_as(Role.TOOLBOX)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    await _tested(open_as, "r1")
    await toolbox.write_request(request("r2", intent("s1", "c0", target="kv cache")))
    await toolbox.write_result(result("r2", validity=Validity.MEASUREMENT_FAILED, gpu_seconds=90.0))

    (row,) = [row for row in await scheduler.progress() if row.branch == "s1"]
    brief = await scheduler.startup_brief("c0")

    assert (row.tests, row.gpu_seconds) == (2, 690.0)
    assert [item.request_id for item in brief.tried] == ["r1"]


async def test_a_result_the_gate_could_not_re_verify_is_no_test_of_its_change_and_no_direction_tried(
    open_as: Open,
) -> None:
    scheduler, runtime, toolbox = open_as(Role.SCHEDULER), open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    wanted = intent("s1", "c0")
    claim = await runtime.acquire(wanted, delta().fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("r1", wanted))
    await toolbox.attach_request(claim.key, "r1", ttl_seconds=TTL)
    await toolbox.write_result(result("r1", gpu_seconds=90.0))

    record = await open_as(Role.GATE).write_judgement("r1", judgement(4.0, ("r1",)), Conclusion.UNVERIFIED)
    (row,) = [row for row in await scheduler.progress() if row.branch == "s1"]

    assert (record.measured, record.conclusion) == (False, Conclusion.UNVERIFIED)
    assert (await runtime.lookup(claim.key, delta().fingerprint())).status is DedupStatus.FREE
    assert (await runtime.acquire(wanted, delta().fingerprint(), ttl_seconds=TTL)).status is ClaimStatus.HELD
    assert (await scheduler.startup_brief("c0")).tried == ()
    assert (row.tests, row.gpu_seconds) == (1, 90.0)


async def test_a_request_without_an_intent_carries_no_claim_and_matches_no_lookup(open_as: Open) -> None:
    runtime, toolbox = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    baseline = TestRequest(TestKind.TEST, Purpose.BASELINE, INTEGRATION_BRANCH, "c0", request_id="r0")
    await toolbox.write_request(baseline)
    await toolbox.write_result(result("r0"))
    key = ClaimKey("c0", intent("s1", "c0").fingerprint)

    with pytest.raises(ConflictingWrite, match="no intent"):
        await toolbox.attach_request(key, "r0", ttl_seconds=TTL)
    with pytest.raises(ValueError, match="accepted"):
        await toolbox.write_request(replace(baseline, request_id=None))
    answer = await runtime.lookup(key, RecipeDelta().fingerprint())
    assert (answer.status, (await toolbox.snapshot()).claims) == (DedupStatus.FREE, ())


async def test_a_branch_log_is_its_runtimes_alone_and_its_current_plan_is_the_latest_unsuperseded(
    open_as: Open,
) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    hypothesis = await s1.append_branch_log(
        BranchLogEntry("s1", BranchLogType.HYPOTHESIS, "moe gemm is the bottleneck")
    )
    plan = await s1.append_branch_log(BranchLogEntry("s1", BranchLogType.PLAN, "tune the moe gemm"))
    await s1.append_branch_log(
        BranchLogEntry("s1", BranchLogType.RESULT, "no gain", ("r1",), hypothesis_status=HypothesisStatus.FALSIFIED)
    )
    replan = await s1.append_branch_log(
        BranchLogEntry("s1", BranchLogType.PLAN, "try fp8 kv", supersedes=plan.entry_id)
    )

    assert await s1.current_plan("s1") == replan
    assert [r.entry.kind for r in await s1.branch_log("s1", after=hypothesis.entry_id)] == [
        BranchLogType.PLAN,
        BranchLogType.RESULT,
        BranchLogType.PLAN,
    ]
    assert len(await s1.branch_log("s1", kinds=frozenset({BranchLogType.PLAN}))) == 2
    assert await s1.current_plan("s2") is None
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SPECIALIST_RUNTIME, "s2").append_branch_log(BranchLogEntry("s1", BranchLogType.PLAN, "x"))
    with pytest.raises(NotRecorded):
        await s1.append_branch_log(BranchLogEntry("s1", BranchLogType.PLAN, "x", supersedes="nope:1"))
    with pytest.raises(NotRecorded):
        await s1.branch_log("s1", after="nope:1")


async def test_a_branch_writes_its_own_learnings_and_every_branch_reads_them_all(open_as: Open) -> None:
    await _tested(open_as, "r1")
    s1, s2 = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.SPECIALIST_RUNTIME, "s2")
    pitfall = await s1.append_learning(
        Learning(
            "s1",
            LearningKind.PITFALL,
            "  fp8 KV cache   fails the accuracy gate on this model  ",
            "kv cache",
            "fp8 kv cache",
            IntentLayer.CONFIG,
            ("r1",),
        )
    )
    tooling = await s2.append_learning(Learning("s2", LearningKind.TOOLING, "vLLM 0.11 ignores VLLM_USE_V1=0 here"))
    corrected = await s1.append_learning(
        Learning(
            "s1",
            LearningKind.PITFALL,
            "fp8 KV cache passes accuracy with --calculate-kv-scales",
            supersedes=pitfall.entry_id,
        )
    )

    assert pitfall.learning.content == "fp8 KV cache fails the accuracy gate on this model"
    assert [record.entry_id for record in await s2.learnings()] == [
        pitfall.entry_id,
        tooling.entry_id,
        corrected.entry_id,
    ]
    assert await s2.learnings(branch="s1") == (pitfall, corrected)
    assert (await s1.snapshot()).learnings == (pitfall, tooling, corrected)
    assert [hit.entry_id for hit in await s2.search("ignores")] == [tooling.entry_id]


@pytest.mark.parametrize(
    ("learning", "refusal"),
    [
        (Learning("s2", LearningKind.INSIGHT, "attention dominates decode at conc 64"), WriterNotAllowed),
        (Learning("s1", LearningKind.INSIGHT, "no gain"), ValueError),
        (Learning("s1", LearningKind.INSIGHT, "too short"), ValueError),
        (Learning("s1", LearningKind.INSIGHT, "x" * 601), ValueError),
        (
            Learning("s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", request_ids=("r9",)),
            NotRecorded,
        ),
        (
            Learning("s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", supersedes="nope:1"),
            NotRecorded,
        ),
        (Learning("s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", supersedes=LESSON), ValueError),
        (Learning("s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", retires=LESSON), ValueError),
        (Learning("s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", retires="s1:1"), ValueError),
        (
            Learning(
                "s1", LearningKind.INSIGHT, "attention dominates decode at conc 64", supersedes="s1:1", retires=LESSON
            ),
            ValueError,
        ),
    ],
)
async def test_a_learning_the_blackboard_cannot_use_is_refused(
    open_as: Open, learning: Learning, refusal: type[Exception]
) -> None:
    with pytest.raises(refusal):
        await open_as(Role.SPECIALIST_RUNTIME, "s1").append_learning(learning)


async def test_a_past_sessions_lesson_is_revised_or_retired_on_this_sessions_tests(open_as: Open) -> None:
    runtime, toolbox = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    wanted = intent("s1", "c0", target="cuda graph", mechanism="max batch size", layer=IntentLayer.CONFIG)
    await runtime.acquire(wanted, delta().fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request("r1", wanted))
    await toolbox.write_result(result("r1"))
    fixed = "--cuda-graph-max-bs is ambiguous here; the -decode form takes the decode graph size"

    revised = await runtime.append_learning(
        Learning("s1", LearningKind.PITFALL, fixed, request_ids=("r1",), supersedes=LESSON)
    )
    retired = await runtime.append_learning(
        Learning(
            "s1",
            LearningKind.OTHER,
            "page size 64 starts fine once the flag is spelt out",
            request_ids=("r1",),
            retires=LESSON,
        )
    )
    plain = await runtime.append_learning(Learning("s1", LearningKind.TOOLING, "vLLM 0.11 ignores VLLM_USE_V1=0 here"))

    assert (revised.learning.supersedes, retired.learning.retires) == (LESSON, LESSON)
    assert (revised.learning.target, revised.learning.mechanism, revised.learning.layer) == (
        "cuda graph",
        "max batch size",
        IntentLayer.CONFIG,
    )
    assert (plain.learning.target, plain.learning.mechanism, plain.learning.layer) == ("", "", None)


async def test_a_branch_supersedes_only_its_own_learnings(open_as: Open) -> None:
    other = await open_as(Role.SPECIALIST_RUNTIME, "s2").append_learning(
        Learning("s2", LearningKind.INSIGHT, "attention dominates decode at conc 64")
    )

    with pytest.raises(NotRecorded):
        await open_as(Role.SPECIALIST_RUNTIME, "s1").append_learning(
            Learning("s1", LearningKind.INSIGHT, "gemm dominates decode at conc 256", supersedes=other.entry_id)
        )


async def test_a_tool_result_pointer_reaches_only_its_branch_and_a_bad_cursor_is_refused(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.notify_tool_result(ToolResultPointer("s1", "job-7", "/artifacts/build-7.log", "build done"))

    mine, theirs = await toolbox.poll("s1"), await toolbox.poll("s2")

    assert [(item.kind, item.pointer) for item in mine.items] == [(InboxKind.TOOL_RESULT, "job-7")]
    assert theirs.items == ()
    with pytest.raises(ValueError):
        await toolbox.poll("s1", '["not", "a", "cursor"]')


async def test_a_build_notice_is_the_build_item_of_the_branch_whose_test_waits_on_it(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r1", intent("s1", "c0")))
    await toolbox.notify_build(BuildNotice("r1", "sgl-kernel-3f2a", "sgl_kernel", BuildOutcome.READY))
    await toolbox.notify_build(BuildNotice("r1", "vllm-9c1d", "vllm", "failed", "registry timeout"))

    mine, theirs = await toolbox.poll("s1"), await toolbox.poll("s2")

    assert [(item.kind, item.pointer, item.summary) for item in mine.items] == [
        (InboxKind.BUILD, "r1", "build sgl_kernel sgl-kernel-3f2a ready for r1"),
        (InboxKind.BUILD, "r1", "build vllm vllm-9c1d failed for r1: registry timeout"),
    ]
    assert theirs.items == ()
    assert [value.state for _, value in toolbox._state().decoded[1:]] == [BuildOutcome.READY, BuildOutcome.FAILED]


async def test_only_the_toolbox_sends_a_build_notice_for_a_test_it_holds(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r1", intent("s1", "c0")))
    ready = BuildNotice("r1", "sgl-kernel-3f2a", "sgl_kernel", BuildOutcome.READY)

    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SPECIALIST_RUNTIME, "s1").notify_build(ready)
    with pytest.raises(UnknownRequest):
        await toolbox.notify_build(replace(ready, request_id="r-missing"))
    with pytest.raises(ValueError, match="key and component"):
        await toolbox.notify_build(replace(ready, key=""))
    with pytest.raises(ValueError, match="BuildOutcome"):
        await toolbox.notify_build(replace(ready, state="building"))
    assert len(toolbox._state().entries) == 1


async def test_lifting_a_node_returns_all_its_gpus_to_service(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.record_quarantine(QuarantineEvent("node-2", QuarantineAction.QUARANTINE, "gpu fault", (0, 1)))
    await toolbox.record_quarantine(QuarantineEvent("node-2", QuarantineAction.QUARANTINE, "slow", (4,)))
    await toolbox.record_quarantine(
        QuarantineEvent("node-3", QuarantineAction.QUARANTINE, "baseline drift", (), ("r9",))
    )
    before = await toolbox.quarantined()
    await toolbox.record_quarantine(QuarantineEvent("node-2", QuarantineAction.LIFT, "replaced"))

    assert len(before) == 3
    assert [(event.node, event.request_ids) for event in await toolbox.quarantined()] == [("node-3", ("r9",))]
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.GATE).record_quarantine(QuarantineEvent("node-1", QuarantineAction.QUARANTINE, "x"))


async def test_a_specialist_records_its_own_llm_spend_and_the_scheduler_its_judges_and_resolvers(open_as: Open) -> None:
    scheduler, runtime = open_as(Role.SCHEDULER), open_as(Role.SPECIALIST_RUNTIME, "s1")
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "s1", from_commit="c0"))
    await _tested(open_as, "r1")

    tokens = TokenCounts(input=1_000, output=4_000, cache_read=80_000, cache_write=9_000)
    spent = await runtime.record_llm_usage(LlmUsage("s1", UsageSource.SPECIALIST, 1.25, tokens, "launch-1"))
    await runtime.record_llm_usage(LlmUsage("s1", UsageSource.SPECIALIST, 0.5))
    unpriced = await runtime.record_llm_usage(LlmUsage("s1", UsageSource.SPECIALIST, None, tokens))
    await scheduler.record_llm_usage(LlmUsage("", UsageSource.JUDGE, 0.125))
    await scheduler.record_llm_usage(LlmUsage("", UsageSource.RESOLVER, 0.125))
    await scheduler.record_llm_usage(LlmUsage("", UsageSource.JUDGE, None, tokens))
    snapshot = await scheduler.snapshot()
    rows = {row.branch: row for row in snapshot.progress}

    assert (spent.usage.launch, snapshot.llm_usage[0]) == ("launch-1", spent)
    assert snapshot.llm_usage[2] == unpriced and unpriced.usage.tokens == tokens
    assert [record.usage.source for record in snapshot.llm_usage] == [
        UsageSource.SPECIALIST,
        UsageSource.SPECIALIST,
        UsageSource.SPECIALIST,
        UsageSource.JUDGE,
        UsageSource.RESOLVER,
        UsageSource.JUDGE,
    ]
    assert (rows["s1"].llm_usd, rows[INTEGRATION_BRANCH].llm_usd, snapshot.llm_usd) == (1.75, 0.0, 2.0)
    assert snapshot.llm_unpriced_tokens == tokens + tokens
    assert snapshot.gpu_hours == pytest.approx(600.0 / 3600.0)


@pytest.mark.parametrize(
    ("role", "branch", "usage"),
    [
        (Role.SPECIALIST_RUNTIME, "s1", LlmUsage("s2", UsageSource.SPECIALIST, 1.0)),
        (Role.SPECIALIST_RUNTIME, "s1", LlmUsage("s1", UsageSource.JUDGE, 1.0)),
        (Role.SCHEDULER, "", LlmUsage("s1", UsageSource.JUDGE, 1.0)),
        (Role.SCHEDULER, "", LlmUsage("", UsageSource.SPECIALIST, 1.0)),
        (Role.TOOLBOX, "", LlmUsage("", UsageSource.JUDGE, 1.0)),
        (Role.SPECIALIST_RUNTIME, "s1", LlmUsage("s1", UsageSource.CRITIC, 1.0)),
        (Role.SCHEDULER, "", LlmUsage("s1", UsageSource.CRITIC, 1.0)),
    ],
)
async def test_llm_spend_is_recorded_only_by_whoever_spent_it(
    open_as: Open, role: Role, branch: str, usage: LlmUsage
) -> None:
    with pytest.raises(WriterNotAllowed):
        await open_as(role, branch).record_llm_usage(usage)


async def test_the_scheduler_records_the_rule_gates_fix_critic_with_its_own_spend(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)

    await scheduler.record_llm_usage(LlmUsage("", UsageSource.CRITIC, 0.25))
    snapshot = await scheduler.snapshot()

    assert ([record.usage.source for record in snapshot.llm_usage], snapshot.llm_usd) == ([UsageSource.CRITIC], 0.25)


@pytest.mark.parametrize("cost", [-0.01, float("nan"), float("inf")])
def test_an_llm_cost_is_a_finite_amount(cost: float) -> None:
    with pytest.raises(ValueError):
        LlmUsage("s1", UsageSource.SPECIALIST, cost)


def _ran_on(request_id: str, node: str, **facts: object) -> TestResult:
    measured = result(request_id, node=node)
    return replace(measured, sandbox=replace(measured.sandbox, **facts))


async def test_a_node_whose_image_or_power_cap_moves_between_results_shows_an_environment_change(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    readings = [
        ("r1", "node-1", {"image_digest": "sha256:aa", "gpu_model": "MI355X", "power_cap_w": 1000.0}),
        ("r2", "node-1", {"image_digest": "sha256:bb", "gpu_model": "MI355X"}),
        ("r3", "node-2", {"image_digest": "sha256:aa", "power_cap_w": 750.0}),
        ("r4", "node-1", {"image_digest": "sha256:bb", "power_cap_w": 900.0}),
        ("r5", "node-1", {"image_digest": "sha256:bb", "gpu_model": "MI355X", "power_cap_w": 900.0}),
    ]
    for request_id, node, facts in readings:
        await toolbox.write_request(request(request_id, intent("s1", "c0", target=request_id)))
        await toolbox.write_result(_ran_on(request_id, node, **facts))

    snapshot = await toolbox.snapshot()
    changes = snapshot.environment_changes

    assert [(change.node, change.request_id, dict(change.changed)) for change in changes] == [
        ("node-1", "r2", {"image_digest": ("sha256:aa", "sha256:bb")}),
        ("node-1", "r4", {"power_cap_w": ("1000.0", "900.0")}),
    ]
    measured_at = {record.request_id: record.measured_at for record in snapshot.records}
    assert [change.at for change in changes] == [measured_at["r2"], measured_at["r4"]]


async def test_the_latest_baseline_is_in_force(open_as: Open) -> None:
    gate = open_as(Role.GATE)

    with pytest.raises(NotRecorded):
        await gate.baseline()
    await gate.write_baseline(
        BaselineRecord("c0", {"output_tps": 1000.0}, {"output_tps": 1.2}, {"output_tps": 2.4}, ())
    )
    updated = BaselineRecord("c0", {"output_tps": 1000.0}, {"output_tps": 0.9}, {"output_tps": 1.8}, ("b4",))
    await gate.write_baseline(updated)

    assert await gate.baseline() == updated
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SCHEDULER).write_baseline(updated)


async def test_merges_that_did_not_land_say_why_in_the_commit_query(open_as: Open, clock: FakeClock) -> None:
    scheduler, gate, integrator = open_as(Role.SCHEDULER), open_as(Role.GATE), open_as(Role.INTEGRATOR)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    for request_id in ("r1", "r2"):
        await _tested(open_as, request_id)
    await gate.write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1")
    await gate.write_judgement("r2", judgement(2.0, ("r2",)), Conclusion.COMMITTED, "c2")
    await _tested(open_as, "m1", branch=INTEGRATION_BRANCH)
    await integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))
    await integrator.write_judgement("m1", judgement(0.1, ("m1",), verdict=Verdict.FLAT), Conclusion.NOT_STACKABLE)
    await integrator.record_merge_attempt(MergeAttempt(("c2",), "c0", skipped_reason="patch conflict in fused_moe.py"))
    clock.advance(600.0)
    await integrator.record_merge_attempt(MergeAttempt(("c2",), "c0", skipped_reason="patch conflict in qwen3.py"))

    by_commit = {commit.commit_id: commit for commit in await integrator.committed()}

    assert [commit.commit_id for commit in await integrator.committed()] == ["c1", "c2"]
    c1, c2 = by_commit["c1"].last_merge, by_commit["c2"].last_merge
    assert (c1.outcome, c1.reason, by_commit["c1"].merge_attempts) == (
        MergeOutcome.NOT_STACKABLE,
        "+0.10% on the integration head",
        1,
    )
    assert (c2.outcome, c2.reason, by_commit["c2"].merge_attempts) == (
        MergeOutcome.CONFLICT_SKIPPED,
        "patch conflict in qwen3.py",
        2,
    )
    assert c2.at - c1.at > 600.0
    assert by_commit["c1"].merged_into is None
    with pytest.raises(ValueError, match="says why"):
        await integrator.record_merge_attempt(MergeAttempt(("c2",), "c0"))
    with pytest.raises(ValueError, match="integration"):
        await integrator.record_merge_attempt(MergeAttempt(("c2",), "c0", request_id="r2"))


async def test_the_patches_a_model_wrote_for_a_tested_merge_reach_the_snapshot(open_as: Open) -> None:
    """CLOSE names them in the Recipe KB's provenance, so the snapshot carries every one the merges recorded."""

    scheduler, gate, integrator = open_as(Role.SCHEDULER), open_as(Role.GATE), open_as(Role.INTEGRATOR)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await _tested(open_as, "r1")
    await gate.write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1")
    await _tested(open_as, "m1", branch=INTEGRATION_BRANCH)
    written = ResolvedPatch(
        "sglang", "patch/sglang/qwen3.patch", PatchRef("patch/sglang/m1.patch", "e" * 64), "m", "v1", 2
    )

    await integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1", resolved_patches=(written,)))

    assert (await integrator.snapshot()).resolved_patches == (written,)
    with pytest.raises(ValueError, match="no model-written patch"):
        await integrator.record_merge_attempt(
            MergeAttempt(("c1",), "c0", skipped_reason="unresolved patch conflict", resolved_patches=(written,))
        )


async def test_a_merge_skipped_because_its_change_is_already_on_the_head_is_no_conflict(open_as: Open) -> None:
    await open_as(Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await _tested(open_as, "r1")
    await open_as(Role.GATE).write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1")
    integrator = open_as(Role.INTEGRATOR)
    skipped = MergeAttempt(
        ("c1",), "c0", skipped_reason="its change is already on the head", skipped_outcome=MergeOutcome.ALREADY_ON_HEAD
    )
    await integrator.record_merge_attempt(skipped)

    (c1,) = await integrator.committed()

    assert (c1.last_merge.outcome, c1.merged_into) == (MergeOutcome.ALREADY_ON_HEAD, None)
    with pytest.raises(ValueError, match="skipped as"):
        await integrator.record_merge_attempt(replace(skipped, skipped_outcome=MergeOutcome.MERGED))


async def test_a_pending_merge_shows_as_pending(open_as: Open) -> None:
    await open_as(Role.SCHEDULER).record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await _tested(open_as, "r1")
    await open_as(Role.GATE).write_judgement("r1", judgement(3.0, ("r1",)), Conclusion.COMMITTED, "c1")
    toolbox, integrator = open_as(Role.TOOLBOX), open_as(Role.INTEGRATOR)
    await toolbox.write_request(request("m1", intent(INTEGRATION_BRANCH, "c0")))
    await integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))

    (c1,) = await integrator.committed()

    assert c1.last_merge.outcome is MergeOutcome.PENDING


async def test_queries_about_commits_the_blackboard_never_saw_raise_not_recorded(open_as: Open) -> None:
    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))

    for query in (scheduler.recorded_value, scheduler.regression_cases_for, scheduler.startup_brief):
        with pytest.raises(NotRecorded):
            await query("c-unknown")
    with pytest.raises(NotRecorded, match="baseline"):
        await scheduler.recorded_value("c0")


async def test_search_finds_the_newest_matching_entries_first(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    for request_id, parent in (("r1", "c0"), ("r2", "c1"), ("r3", "c1")):
        await toolbox.write_request(request(request_id, intent("s1", parent)))

    hits = await toolbox.search("request&c1,nothing-like-this", limit=1)
    every = await toolbox.search("REQUEST")

    assert [hit.summary for hit in hits] == ["test r3 on s1 at c1"]
    assert len(every) == 3
    with pytest.raises(ValueError):
        await toolbox.search(" , & ")


async def test_profiles_of_a_commit_are_its_profile_tests_that_left_an_artifact(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    for request_id, location in (("p1", "/profiles/p1.json"), ("p2", "")):
        await toolbox.write_request(request(request_id, intent("s1", "c1"), kind=TestKind.PROFILE))
        await toolbox.write_result(result(request_id, profile_location=location))
    await _tested(open_as, "r1", parent="c1")

    found = await toolbox.profiles("c1")

    assert [record.request_id for record in found] == ["p1"]
    assert (await toolbox.record("p1")).result.profile_location == "/profiles/p1.json"
