# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import hashlib
import re

import pytest

from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    TestRequest,
    Validity,
)
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeDelta, apply
from hyperloom_multibranch.contracts.gate import RegressionOutcome
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, LineageEvent, RegressionCase
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import delta, judgement, result
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID
from hyperloom_multibranch.rulegate.regression import (
    RESUBMIT_SECONDS,
    FixNotProven,
    PooledCase,
    RegressionSet,
    ReproRun,
    ReproRunner,
    prove_fix,
    register_case,
)
from hyperloom_multibranch.rulegate.service import Polling
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingVerifier, ReproAnswer, ReproToolbox

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
ROOT = "c0"
OVERFLOW = (AttachmentRef("fp8_overflow", "dir", "b" * 64),)
NAN = (AttachmentRef("nan_logits", "dir", "d" * 64),)
FIX = delta("--kv-cache-dtype fp8 --clamp-scales")
GUARD = delta("--guard-nan-logits")
MERGE = delta("--max-num-seqs 256")
NO_DELTA = RecipeDelta()
OOB = (AttachmentRef("oob_8k", "dir", "e" * 64),)
BOUNDS = delta("--max-model-len 8192 --check-bounds")
ON_C1 = TestRequest(TestKind.TEST, Purpose.EXPLORE, "s1", "c1", BOUNDS, request_id="r9")
ON_ROOT = TestRequest(TestKind.TEST, Purpose.EXPLORE, "s1", ROOT, BOUNDS, request_id="r9")


def _fixed_by_bounds(*, root: ReproAnswer = REPRODUCED):
    """The case reproduces on every recipe without ``BOUNDS``, the root answering ``root``."""

    def answer(request):
        if request.delta == BOUNDS:
            return NOT_REPRODUCED
        return root if request.parent_commit == ROOT else REPRODUCED

    return answer


class _Toolbox(ReproToolbox):
    """A repro toolbox that also stores the cases it is given, unless it ``refuses_cases``."""

    refuses_cases = False

    async def put_attachment(self, attachment_id, kind, data):
        if self.refuses_cases:
            raise RequestRejected("the archive expands past 64 MiB", "attachment_too_large")
        return AttachmentRef(attachment_id, kind, hashlib.sha256(data).hexdigest())


def _pooled(data: bytes, origin: str) -> PooledCase:
    return PooledCase(hashlib.sha256(data).hexdigest(), data, origin)


class _Pool:
    """The cases other sessions registered, and how often they were asked for."""

    def __init__(self, *cases: PooledCase) -> None:
        self.given = cases
        self.reads = 0

    async def cases(self) -> tuple[PooledCase, ...]:
        self.reads += 1
        return self.given


class _Rig:
    """A toolbox that runs repro tests, writing to a local blackboard, and a runner reading it as the gate."""

    def __init__(self, root, answer=lambda request: NOT_REPRODUCED, verifier=None):
        self.toolbox = _Toolbox(LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox")), answer)
        self.blackboard = LocalBlackboard(root, Writer(Role.GATE, "gate"))
        self.verifier = verifier or RecordingVerifier()
        self.sleeps: list[float] = []
        self.runner = ReproRunner(self.toolbox, self.blackboard, self.verifier, polling=Polling(0.5, self._sleep))

    async def _sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


async def test_a_case_runs_as_a_repro_test_and_its_verified_verdict_comes_back(tmp_path):
    rig = _Rig(tmp_path, lambda request: REPRODUCED)

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("p1", verdict=REPRODUCED)
    (submitted,) = rig.toolbox.submitted
    assert (submitted.kind, submitted.purpose) == (TestKind.REPRO, Purpose.REPRO)
    assert (submitted.branch_id, submitted.parent_commit, submitted.delta) == ("s1", "c1", FIX)
    assert submitted.attachments == OVERFLOW
    assert rig.verifier.asked == ["p1"]


async def test_a_full_queue_is_waited_out(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.busy = 2

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run.verdict == NOT_REPRODUCED
    assert len(rig.toolbox.submitted) == 1
    assert rig.sleeps == [RESUBMIT_SECONDS, RESUBMIT_SECONDS]


async def test_the_runner_waits_for_the_test_to_end(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.hold = True
    running = asyncio.create_task(rig.runner.run("c1", FIX, OVERFLOW, branch="s1"))
    while not rig.sleeps:
        await asyncio.sleep(0)

    await rig.toolbox.finish()

    assert (await running).verdict == NOT_REPRODUCED
    assert rig.sleeps[0] == 0.5


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (None, "withdrawn (the branch was stopped)"),
        (Validity.CANDIDATE_FAILED, "ended candidate_failed (the server did not start)"),
        ("", "succeeded without a verdict"),
    ],
)
async def test_a_case_that_ran_to_no_verdict_says_why(tmp_path, answer, failure):
    rig = _Rig(tmp_path, lambda request: answer)

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("p1", failure=failure)
    assert run.outcome == failure


async def test_a_verdict_that_does_not_re_verify_is_no_verdict(tmp_path):
    rig = _Rig(tmp_path, verifier=RecordingVerifier({"p1"}))

    run = await rig.runner.run("c1", FIX, OVERFLOW, branch="s1")

    assert (run.verdict, run.failure) == ("", "p1 did not re-verify: the manifest hash is not the recorded one")


async def test_a_refused_test_has_no_request(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.refuse = lambda request: True

    run = await rig.runner.run("c9", FIX, OVERFLOW, branch="s1")

    assert run == ReproRun("", failure="refused by the toolbox (unknown_parent: c9 is not in the repository)")


async def test_a_cancelled_run_takes_its_test_back(tmp_path):
    rig = _Rig(tmp_path)
    rig.toolbox.hold = True
    running = asyncio.create_task(rig.runner.run("c1", FIX, OVERFLOW, branch="s1"))
    while not rig.sleeps:
        await asyncio.sleep(0)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert rig.toolbox.cancelled == ["p1"]
    assert (await rig.blackboard.record("p1")).withdrawal_reason == "cancelled by its caller"


class _Session(_Rig):
    """``c0`` is the root; ``s1``'s fix ``c1`` carries the case ``fp8_overflow`` (request ``p1``) and ``s2``'s fix
    ``c2`` the case ``nan_logits`` (``p2``). Each case's answer is in ``answers``, ``not_reproduced`` until set."""

    def __init__(self, root):
        self.answers = {}
        super().__init__(root, self._answer)
        self.scheduler = LocalBlackboard(root, Writer(Role.SCHEDULER, "scheduler"))
        self.repo = InMemoryRepo(BASELINE)
        self.regression = RegressionSet(self.blackboard, self.runner, self.repo)

    async def start(self) -> None:
        await self.scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT))
        await self.blackboard.write_baseline(BaselineRecord(ROOT, {"output_tps": 1000.0}, {}, {}, ("b1", "b2")))
        await self._fix("s1", FIX, OVERFLOW)
        await self._fix("s2", GUARD, NAN)

    async def _fix(self, branch, change, case) -> None:
        await self.scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit=ROOT, reason="opened"))
        await self.repo.create_branch(branch, ROOT)
        measured = TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, ROOT, change, request_id=f"r-{branch}")
        await self.toolbox.blackboard.write_request(measured)
        await self.toolbox.blackboard.write_result(result(measured.request_id))
        commit = await self.repo.commit(branch, ROOT, apply(BASELINE, change), "a correctness fix")
        flat = judgement(0.1, (measured.request_id,), verdict=Verdict.FLAT, fixes_correctness=True)
        await self.blackboard.write_judgement(measured.request_id, flat, Conclusion.COMMITTED_WITH_REPRO, commit)
        repro = TestRequest(TestKind.REPRO, Purpose.REPRO, branch, ROOT, change, attachments=case)
        case_request = await self.toolbox.submit_test(repro)
        await self.blackboard.add_regression_case(RegressionCase(case[0].id, commit, case_request))

    def _answer(self, request):
        """A case's answer, by the case's content or else its name, or what a callable answer makes of the request."""

        (case,) = request.attachments
        answer = self.answers.get(case.sha256, self.answers.get(case.id, NOT_REPRODUCED))
        return answer(request) if callable(answer) else answer

    def runs(self) -> list[TestRequest]:
        """The repro tests run after the session's own two."""

        return self.toolbox.submitted[2:]

    def pooled_set(self, *cases: PooledCase, pool: _Pool | None = None) -> RegressionSet:
        """A regression set of this session over a pool of ``cases``."""

        return RegressionSet(self.blackboard, self.runner, self.repo, pool=pool or _Pool(*cases))

    def seeded(self) -> list[TestRequest]:
        """The repro tests run on the root unchanged, the pool's cases."""

        return [run for run in self.runs() if (run.parent_commit, run.delta) == (ROOT, NO_DELTA)]

    async def prove(self, candidate: TestRequest):
        return await prove_fix(self.runner, self.blackboard, candidate, OOB)


@pytest.fixture
async def session(tmp_path):
    session = _Session(tmp_path / "blackboard")
    await session.start()
    return session


async def test_a_commit_no_fix_built_on_passes_without_a_test(session):
    outcome = await session.regression.check(ROOT, MERGE, sources=())

    assert outcome == RegressionOutcome(True)
    assert session.runs() == []


async def test_the_outcome_counts_the_cases_it_checked_so_an_empty_set_is_told_from_a_clean_one(session):
    none = await session.regression.check(ROOT, MERGE, sources=())
    one = await session.regression.check("c1", MERGE, sources=())
    both = await session.regression.check("c1", MERGE, sources=("c2",))

    assert (none.cases, one.cases, both.cases) == (0, 1, 2)
    assert none.passed and one.passed and both.passed


async def test_each_case_of_the_base_and_the_sources_runs_once_on_the_candidate_concurrently(session):
    session.toolbox.hold = True
    checking = asyncio.create_task(session.regression.check("c1", MERGE, sources=("c2", "c1")))
    while len(session.runs()) < 2:
        await asyncio.sleep(0)

    await session.toolbox.finish()

    assert await checking == RegressionOutcome(True, cases=2)
    runs = session.runs()
    assert [(run.branch_id, run.parent_commit, run.delta) for run in runs] == [(INTEGRATION_BRANCH, "c1", MERGE)] * 2
    assert {run.attachments for run in runs} == {OVERFLOW, NAN}
    assert all((run.kind, run.purpose) == (TestKind.REPRO, Purpose.REPRO) for run in runs)


@pytest.mark.parametrize(
    ("answer", "cause"),
    [
        (REPRODUCED, "fp8_overflow: reproduced"),
        (None, "fp8_overflow: withdrawn (the branch was stopped)"),
        (Validity.CANDIDATE_FAILED, "fp8_overflow: ended candidate_failed (the server did not start)"),
        ("", "fp8_overflow: succeeded without a verdict"),
    ],
)
async def test_a_case_that_does_not_run_clean_fails_the_set_with_its_cause(session, answer, cause):
    session.answers["fp8_overflow"] = answer

    outcome = await session.regression.check("c2", MERGE, sources=("c1",))

    assert outcome == RegressionOutcome(False, cause, cases=2)


async def test_a_case_whose_result_does_not_re_verify_fails_the_set(session):
    session.verifier.rejects = frozenset({"p3"})

    outcome = await session.regression.check("c1", MERGE, sources=())

    assert outcome == RegressionOutcome(
        False, "fp8_overflow: p3 did not re-verify: the manifest hash is not the recorded one", cases=1
    )


async def test_a_case_the_toolbox_refuses_fails_the_set(session):
    session.toolbox.refuse = lambda request: True

    outcome = await session.regression.check("c1", MERGE, sources=())

    assert not outcome.passed
    assert outcome.reason.startswith("fp8_overflow: refused by the toolbox (unknown_parent:")


async def test_every_failed_case_is_named(session):
    session.answers.update({"fp8_overflow": REPRODUCED, "nan_logits": None})

    outcome = await session.regression.check("c1", MERGE, sources=("c2",))

    assert outcome.reason == "fp8_overflow: reproduced; nan_logits: withdrawn (the branch was stopped)"


def _fixed_on(fix: str, *, root: ReproAnswer = REPRODUCED):
    """A case that reproduces on every commit but ``fix``, the root answering ``root``."""

    def answer(request):
        if request.parent_commit == fix:
            return NOT_REPRODUCED
        return root if request.parent_commit == ROOT else REPRODUCED

    return answer


async def test_every_case_runs_on_a_commit_that_lacks_its_fix_and_one_the_root_reproduces_too_is_carried(session):
    session.answers.update({"fp8_overflow": _fixed_on("c1"), "nan_logits": _fixed_on("c2")})

    outcome = await session.regression.check_every_case("c2")

    assert outcome == RegressionOutcome(True, cases=2, carried=("fp8_overflow",))
    ran = sorted((run.parent_commit, run.attachments[0].id) for run in session.runs())
    assert ran == [("c0", "fp8_overflow"), ("c2", "fp8_overflow"), ("c2", "nan_logits")]


async def test_a_case_that_reproduces_on_a_commit_but_runs_clean_on_the_root_fails_every_case_check(session):
    session.answers.update({"fp8_overflow": _fixed_on("c1", root=NOT_REPRODUCED), "nan_logits": _fixed_on("c2")})

    outcome = await session.regression.check_every_case("c2")

    assert outcome == RegressionOutcome(False, "fp8_overflow: reproduced", cases=2)


@pytest.mark.parametrize("verdict", [NOT_REPRODUCED, REPRODUCED])
async def test_a_case_s_verdict_on_a_recipe_is_not_asked_for_again(session, verdict):
    session.answers["fp8_overflow"] = verdict
    first = await session.regression.check("c1", NO_DELTA, sources=())

    again = await session.regression.check(ROOT, FIX, sources=("c1",))

    assert again == first
    assert len(session.runs()) == 1


async def test_a_case_runs_again_on_another_recipe(session):
    await session.regression.check("c1", NO_DELTA, sources=())

    await session.regression.check("c1", MERGE, sources=())

    assert [run.delta for run in session.runs()] == [NO_DELTA, MERGE]


async def test_a_case_that_ran_to_no_verdict_runs_again(session):
    session.answers["fp8_overflow"] = None
    assert not (await session.regression.check("c1", MERGE, sources=())).passed
    session.answers["fp8_overflow"] = NOT_REPRODUCED

    assert (await session.regression.check("c1", MERGE, sources=())).passed

    assert len(session.runs()) == 2


async def test_a_cancelled_check_takes_every_case_back(session):
    session.toolbox.hold = True
    checking = asyncio.create_task(session.regression.check("c1", MERGE, sources=("c2",)))
    while len(session.runs()) < 2:
        await asyncio.sleep(0)

    checking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checking

    assert sorted(session.toolbox.cancelled) == ["p3", "p4"]


async def test_a_fix_is_proven_on_its_parent_its_candidate_and_the_root_at_once(session):
    session.answers["oob_8k"] = _fixed_by_bounds()
    session.toolbox.hold = True
    proving = asyncio.create_task(session.prove(ON_C1))
    while len(session.runs()) < 3:
        await asyncio.sleep(0)

    await session.toolbox.finish()

    proof = await proving
    assert (proof.parent.verdict, proof.candidate.verdict, proof.root.verdict) == (
        REPRODUCED,
        NOT_REPRODUCED,
        REPRODUCED,
    )
    assert proof.foundation
    assert sorted((run.parent_commit, run.delta.fingerprint()) for run in session.runs()) == sorted(
        [("c1", NO_DELTA.fingerprint()), ("c1", BOUNDS.fingerprint()), (ROOT, NO_DELTA.fingerprint())]
    )
    assert {(run.branch_id, run.attachments) for run in session.runs()} == {("s1", OOB)}
    assert proof.candidate.request_id == next(run.request_id for run in session.runs() if run.delta == BOUNDS)


@pytest.mark.parametrize(
    ("root", "cause"), [(NOT_REPRODUCED, NOT_REPRODUCED), (None, "withdrawn (the branch was stopped)")]
)
async def test_a_fix_of_a_bug_the_root_does_not_show_is_no_foundation_fix(session, root, cause):
    session.answers["oob_8k"] = _fixed_by_bounds(root=root)

    proof = await session.prove(ON_C1)

    assert not proof.foundation
    assert proof.root.outcome == cause


async def test_a_fix_whose_parent_is_the_root_runs_the_parent_once(session):
    session.answers["oob_8k"] = _fixed_by_bounds()

    proof = await session.prove(ON_ROOT)

    assert proof.root == proof.parent
    assert proof.foundation
    assert len(session.runs()) == 2


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (lambda request: NOT_REPRODUCED, "the case did not reproduce on the parent c1: not_reproduced"),
        (lambda request: None, "the case did not reproduce on the parent c1: withdrawn (the branch was stopped)"),
        (lambda request: REPRODUCED, "the case did not run clean on the candidate: reproduced"),
        (
            lambda request: Validity.MEASUREMENT_FAILED if request.delta == BOUNDS else REPRODUCED,
            "the case did not run clean on the candidate: ended measurement_failed (the server did not start)",
        ),
    ],
)
async def test_a_fix_its_case_does_not_tell_from_its_parent_is_not_proven(session, answer, reason):
    session.answers["oob_8k"] = answer

    with pytest.raises(FixNotProven, match=re.escape(reason)) as raised:
        await session.prove(ON_C1)

    assert raised.value.reason == reason


async def test_a_fix_s_clean_run_is_registered_as_the_case_of_its_commit(session):
    run = await session.runner.run("c1", BOUNDS, OOB, branch="s1")

    case = await register_case(session.blackboard, "c2", run.request_id)

    assert case == RegressionCase("oob_8k", "c2", run.request_id)
    assert case in await session.blackboard.regression_cases_for("c2")


@pytest.mark.parametrize("answer", [REPRODUCED, None, ""])
async def test_only_a_repro_test_whose_case_ran_clean_is_registered(session, answer):
    session.answers["oob_8k"] = answer
    run = await session.runner.run("c1", BOUNDS, OOB, branch="s1")

    with pytest.raises(ValueError, match="not a repro test whose case ran clean"):
        await register_case(session.blackboard, "c2", run.request_id)

    assert [case.case_ref for case in await session.blackboard.regression_cases_for("c2")] == ["nan_logits"]


async def test_a_performance_test_is_no_case(session):
    with pytest.raises(ValueError, match="r-s1 is not a repro test"):
        await register_case(session.blackboard, "c1", "r-s1")


async def test_before_its_first_check_the_set_runs_the_pools_cases_on_the_root_and_a_clean_one_guards_every_check(
    session,
):
    clean, broken = _pooled(b"clean case", "past-1 c7"), _pooled(b"broken case", "past-2 c9")
    session.answers[broken.case_id] = REPRODUCED

    outcome = await session.pooled_set(clean, broken).check("c1", MERGE, sources=())

    seeded = session.seeded()
    assert len(seeded) == 2 and {run.branch_id for run in seeded} == {INTEGRATION_BRANCH}
    assert {run.attachments for run in seeded} == {
        (AttachmentRef(FIX_CASE_ID, "dir", case.case_id),) for case in (clean, broken)
    }
    # The broken case reproduces on the root: a defect the root carries, which no merge can be held to.
    on_root = await session.blackboard.regression_cases_for(ROOT)
    assert [(case.case_ref, case.fix_commit) for case in on_root] == [("repro-case of past-1 c7", ROOT)]
    # c1 builds on the root, so its own case and the clean pooled one both run on it.
    assert outcome == RegressionOutcome(True, cases=2)


async def test_the_pools_cases_run_on_the_root_once_and_a_case_on_the_root_already_never_runs_there_again(session):
    pool = _Pool(_pooled(b"clean case", "past-1 c7"))

    first = session.pooled_set(pool=pool)
    await asyncio.gather(first.check("c1", MERGE, sources=()), first.check("c2", MERGE, sources=()))
    # A session resumed after a crash builds its regression set again.
    await session.pooled_set(pool=pool).check("c1", NO_DELTA, sources=())

    assert pool.reads == 2
    assert len(session.seeded()) == 1


@pytest.mark.parametrize("answer", [REPRODUCED, None, ""])
async def test_a_pooled_case_that_does_not_run_clean_on_the_root_is_no_case_of_it(session, answer):
    case = _pooled(b"clean case", "past-1 c7")
    session.answers[case.case_id] = answer

    outcome = await session.pooled_set(case).check("c1", MERGE, sources=())

    assert outcome == RegressionOutcome(True, cases=1)
    assert await session.blackboard.regression_cases_for(ROOT) == ()


async def test_a_pooled_case_the_toolbox_will_not_store_is_left_out_and_the_check_goes_on(session):
    session.toolbox.refuses_cases = True

    outcome = await session.pooled_set(_pooled(b"clean case", "past-1 c7")).check("c1", MERGE, sources=())

    assert outcome == RegressionOutcome(True, cases=1)
    assert session.seeded() == [] and await session.blackboard.regression_cases_for(ROOT) == ()
