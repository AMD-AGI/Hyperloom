# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio

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
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, LineageEvent, RegressionCase
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, judgement, result
from hyperloom_multibranch.rulegate.regression import RESUBMIT_SECONDS, RegressionSet, ReproRun, ReproRunner
from hyperloom_multibranch.rulegate.service import Polling
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingVerifier, ReproToolbox

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
ROOT = "c0"
OVERFLOW = (AttachmentRef("fp8_overflow", "dir", "b" * 64),)
NAN = (AttachmentRef("nan_logits", "dir", "d" * 64),)
FIX = delta("--kv-cache-dtype fp8 --clamp-scales")
GUARD = delta("--guard-nan-logits")
MERGE = delta("--max-num-seqs 256")
NO_DELTA = RecipeDelta()


class _Rig:
    """A toolbox that runs repro tests, writing to a local notebook, and a runner reading it as the gate."""

    def __init__(self, root, answer=lambda request: NOT_REPRODUCED, verifier=None):
        self.toolbox = ReproToolbox(LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox")), answer)
        self.notebook = LocalNotebook(root, Writer(Role.GATE, "gate"))
        self.verifier = verifier or RecordingVerifier()
        self.sleeps: list[float] = []
        self.runner = ReproRunner(self.toolbox, self.notebook, self.verifier, polling=Polling(0.5, self._sleep))

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
    assert (await rig.notebook.record("p1")).withdrawal_reason == "cancelled by its caller"


class _Session(_Rig):
    """``c0`` is the root; ``s1``'s fix ``c1`` carries the case ``fp8_overflow`` (request ``p1``) and ``s2``'s fix
    ``c2`` the case ``nan_logits`` (``p2``). Each case's answer is in ``answers``, ``not_reproduced`` until set."""

    def __init__(self, root):
        self.answers = {}
        super().__init__(root, lambda request: self.answers.get(request.attachments[0].id, NOT_REPRODUCED))
        self.scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"))
        self.repo = InMemoryRepo(BASELINE)
        self.regression = RegressionSet(self.notebook, self.runner, self.repo)

    async def start(self) -> None:
        await self.scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT))
        await self.notebook.write_baseline(BaselineRecord(ROOT, {"output_tps": 1000.0}, {}, {}, ("b1", "b2")))
        await self._fix("s1", FIX, OVERFLOW)
        await self._fix("s2", GUARD, NAN)

    async def _fix(self, branch, change, case) -> None:
        await self.scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit=ROOT, reason="opened"))
        await self.repo.create_branch(branch, ROOT)
        measured = TestRequest(TestKind.TEST, Purpose.EXPLORE, branch, ROOT, change, request_id=f"r-{branch}")
        await self.toolbox.notebook.write_request(measured)
        await self.toolbox.notebook.write_result(result(measured.request_id))
        commit = await self.repo.commit(branch, ROOT, apply(BASELINE, change), "a correctness fix")
        flat = judgement(0.1, (measured.request_id,), verdict=Verdict.FLAT, fixes_correctness=True)
        await self.notebook.write_judgement(measured.request_id, flat, Conclusion.COMMITTED_WITH_REPRO, commit)
        repro = TestRequest(TestKind.REPRO, Purpose.REPRO, branch, ROOT, change, attachments=case)
        case_request = await self.toolbox.submit_test(repro)
        await self.notebook.add_regression_case(RegressionCase(case[0].id, commit, case_request))

    def runs(self) -> list[TestRequest]:
        """The repro tests run after the session's own two."""

        return self.toolbox.submitted[2:]


@pytest.fixture
async def session(tmp_path):
    session = _Session(tmp_path / "notebook")
    await session.start()
    return session


async def test_a_commit_no_fix_built_on_passes_without_a_test(session):
    outcome = await session.regression.check(ROOT, MERGE, sources=())

    assert outcome == RegressionOutcome(True)
    assert session.runs() == []


async def test_each_case_of_the_base_and_the_sources_runs_once_on_the_candidate_concurrently(session):
    session.toolbox.hold = True
    checking = asyncio.create_task(session.regression.check("c1", MERGE, sources=("c2", "c1")))
    while len(session.runs()) < 2:
        await asyncio.sleep(0)

    await session.toolbox.finish()

    assert await checking == RegressionOutcome(True)
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

    assert outcome == RegressionOutcome(False, cause)


async def test_a_case_whose_result_does_not_re_verify_fails_the_set(session):
    session.verifier.rejects = frozenset({"p3"})

    outcome = await session.regression.check("c1", MERGE, sources=())

    assert outcome == RegressionOutcome(
        False, "fp8_overflow: p3 did not re-verify: the manifest hash is not the recorded one"
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
