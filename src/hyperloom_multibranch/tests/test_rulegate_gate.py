# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import dataclasses

import pytest

from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    TestResult,
    Validity,
)
from hyperloom.contracts.recipe import LaunchConfig, Recipe, apply
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import RegressionCase
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.decision import Action
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import FixProof, ReproRun
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
GEMM = intent("s1", "c0")
OVERFLOW = (AttachmentRef("fp8_overflow", "dir", "b" * 64),)


def _judged(verdict: Verdict, *, accuracy_ok: bool = True, fixes_correctness: bool = False) -> Judgement:
    return Judgement(
        verdict,
        3.0,
        2.0,
        ("r1", "r1b"),
        {"output_throughput": 1030.0},
        accuracy_ok,
        fixes_correctness=fixes_correctness,
    )


@pytest.fixture
async def setup(tmp_path):
    root = tmp_path / "notebook"
    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"))
    measured = request("r1", GEMM)
    await toolbox.write_request(measured)
    await toolbox.write_result(result("r1"))
    repo = InMemoryRepo(BASELINE)
    await repo.create_branch("s1", await repo.head(INTEGRATION_BRANCH))
    return LocalNotebook(root, Writer(Role.GATE, "gate")), repo, measured


async def test_an_improvement_lands_on_its_branch_and_in_the_notebook(setup):
    notebook, repo, measured = setup
    verifier = RecordingVerifier()
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), verifier)

    assert await gate.settle(measured, _judged(Verdict.IMPROVED)) is Action.COMMIT

    assert verifier.asked == ["r1", "r1b"]

    tip = await repo.head("s1")
    assert tip != "c0"
    assert await repo.read_recipe(tip) == apply(BASELINE, measured.delta)
    record = await notebook.record("r1")
    assert (record.conclusion, record.commit_id) == (Conclusion.COMMITTED, tip)


@pytest.mark.parametrize(
    ("judgement", "conclusion"),
    [
        (_judged(Verdict.REGRESSED), Conclusion.RECORDED_ONLY),
        (_judged(Verdict.FLAT, accuracy_ok=False, fixes_correctness=True), Conclusion.ACCURACY_FAILED),
        (_judged(Verdict.INCONCLUSIVE, accuracy_ok=False), Conclusion.RECORDED_ONLY),
    ],
)
async def test_a_candidate_only_recorded_leaves_its_branch_alone(setup, judgement, conclusion):
    notebook, repo, measured = setup
    verifier = RecordingVerifier()
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), verifier)

    assert await gate.settle(measured, judgement) is Action.RECORD_ONLY

    assert await repo.head("s1") == "c0"
    assert (await notebook.record("r1")).conclusion is conclusion
    assert verifier.asked == []


async def test_a_retest_is_the_callers_next_step(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())

    assert await gate.settle(measured, _judged(Verdict.NEEDS_CONFIRM, accuracy_ok=False)) is Action.RETEST

    assert (await notebook.record("r1")).judgement is None


async def test_a_blocked_gain_is_held_for_its_specialist_s_diagnosis(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    blocked = _judged(Verdict.BLOCKED, accuracy_ok=False)

    assert await gate.settle(measured, blocked) is Action.DIAGNOSE

    record = await notebook.record("r1")
    assert (record.judgement, record.conclusion) == (blocked, Conclusion.DIAGNOSIS_PENDING)
    assert await repo.head("s1") == "c0"


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        (AdmissionStatus.ACCEPTED, Conclusion.REPAIR_REQUESTED),
        (AdmissionStatus.PENDING, Conclusion.REPAIR_REQUESTED),
        (AdmissionStatus.REJECTED, Conclusion.ACCURACY_FAILED),
    ],
)
async def test_a_blocked_gain_goes_to_the_scheduler_for_a_repair(setup, status, conclusion):
    notebook, repo, measured = setup
    scheduler = RecordingScheduler(status)
    gate = RuleGate(notebook, repo, scheduler, RecordingVerifier())
    blocked = _judged(Verdict.BLOCKED, accuracy_ok=False)

    assert (await gate.repair(measured, blocked, GEMM, "per-tensor KV scales; scale per head")).status is status

    assert [repair.intent for repair in scheduler.repairs] == [GEMM]
    assert scheduler.repairs[0].candidate_delta == measured.delta
    assert (await notebook.record("r1")).conclusion is conclusion


async def test_a_blocked_gain_with_no_repair_is_recorded(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    blocked = dataclasses.replace(_judged(Verdict.BLOCKED), refused_by=("p90_regressed",))

    record = await gate.decline(measured, blocked)

    assert record.conclusion is Conclusion.RECORDED_ONLY


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        (AdmissionStatus.ACCEPTED, Conclusion.SPLIT_REQUESTED),
        (AdmissionStatus.PENDING, Conclusion.SPLIT_PENDING),
        (AdmissionStatus.REJECTED, Conclusion.RECORDED_ONLY),
    ],
)
async def test_a_fix_that_costs_performance_asks_for_a_fork(setup, tmp_path, status, conclusion):
    notebook, repo, measured = setup
    scheduler = RecordingScheduler(status)
    gate = RuleGate(notebook, repo, scheduler, RecordingVerifier())
    fix = await _proven(tmp_path / "notebook")

    admission = await gate.fork(measured, _judged(Verdict.REGRESSED, fixes_correctness=True), "oob at 8k", fix=fix)

    assert [(fork.from_branch, fork.parent_commit, fork.reason) for fork in scheduler.forks] == [
        ("s1", "c0", "oob at 8k")
    ]
    assert bool(admission.branch) is (status is not AdmissionStatus.REJECTED)
    assert (await notebook.record("r1")).conclusion is conclusion
    assert await repo.head("s1") == "c0"


async def test_a_gain_whose_retest_does_not_re_verify_is_not_committed(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier({"r1b"}))

    with pytest.raises(ResultUnverified, match="r1b did not re-verify"):
        await gate.settle(measured, _judged(Verdict.IMPROVED))

    assert await repo.head("s1") == "c0"
    record = await notebook.record("r1")
    assert (record.conclusion, record.commit_id) == (Conclusion.UNVERIFIED, None)


async def test_a_fork_or_a_repair_resting_on_an_unverified_result_is_not_asked_for(setup, tmp_path):
    notebook, repo, measured = setup
    scheduler = RecordingScheduler(AdmissionStatus.ACCEPTED)
    gate = RuleGate(notebook, repo, scheduler, RecordingVerifier({"r1"}))
    fix = await _proven(tmp_path / "notebook")

    with pytest.raises(ResultUnverified):
        await gate.fork(measured, _judged(Verdict.REGRESSED, fixes_correctness=True), "oob at 8k", fix=fix)
    with pytest.raises(ResultUnverified):
        await gate.repair(measured, _judged(Verdict.BLOCKED, accuracy_ok=False), GEMM, "scale per head")

    assert (scheduler.forks, scheduler.repairs) == ([], [])
    assert (await notebook.record("r1")).conclusion is Conclusion.UNVERIFIED


class _FullDisk:
    async def write_judgement(self, *args, **kwargs):
        raise OSError("no space left on device")


class _Stalled:
    """Holds every judgement write until ``release`` is set."""

    def __init__(self, inner):
        self.inner = inner
        self.writing = asyncio.Event()
        self.release = asyncio.Event()

    async def write_judgement(self, *args, **kwargs):
        self.writing.set()
        await self.release.wait()
        return await self.inner.write_judgement(*args, **kwargs)


async def test_a_gain_whose_branch_moved_on_is_recorded_only(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    landed_first = await repo.commit("s1", "c0", BASELINE, "another candidate of c0")

    assert await gate.settle(measured, _judged(Verdict.IMPROVED)) is Action.RECORD_ONLY

    assert await repo.head("s1") == landed_first
    record = await notebook.record("r1")
    assert (record.conclusion, record.commit_id) == (Conclusion.RECORDED_ONLY, None)


async def test_a_commit_whose_judgement_cannot_be_written_is_taken_back(setup):
    _notebook, repo, measured = setup
    gate = RuleGate(_FullDisk(), repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())

    with pytest.raises(OSError, match="no space"):
        await gate.settle(measured, _judged(Verdict.IMPROVED))

    assert await repo.head("s1") == "c0"


async def test_a_specialist_stopped_mid_commit_still_records_the_commit(setup):
    notebook, repo, measured = setup
    stalled = _Stalled(notebook)
    gate = RuleGate(stalled, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    settling = asyncio.create_task(gate.settle(measured, _judged(Verdict.IMPROVED)))
    await stalled.writing.wait()

    settling.cancel()
    stalled.release.set()
    with pytest.raises(asyncio.CancelledError):
        await settling
    for _ in range(50):
        if (await notebook.record("r1")).judgement is not None:
            break
        await asyncio.sleep(0)

    record = await notebook.record("r1")
    assert record.commit_id == await repo.head("s1") != "c0"


async def _proven(root) -> FixProof:
    """The proof of a fix whose clean run ``p1`` the toolbox wrote."""

    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"))
    clean = request("p1", GEMM, kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=OVERFLOW)
    await toolbox.write_request(clean)
    await toolbox.write_result(TestResult("p1", Validity.SUCCEEDED, repro_verdict=NOT_REPRODUCED))
    reproduced = ReproRun("p0", verdict=REPRODUCED)
    return FixProof(reproduced, ReproRun("p1", verdict=NOT_REPRODUCED), reproduced)


@pytest.mark.parametrize(
    ("judgement", "conclusion"),
    [
        (_judged(Verdict.FLAT, fixes_correctness=True), Conclusion.COMMITTED_WITH_REPRO),
        (_judged(Verdict.IMPROVED), Conclusion.COMMITTED),
    ],
)
async def test_a_proven_fix_lands_with_its_case_in_the_regression_set(setup, tmp_path, judgement, conclusion):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())

    assert await gate.settle(measured, judgement, fix=await _proven(tmp_path / "notebook")) is Action.COMMIT

    tip = await repo.head("s1")
    record = await notebook.record("r1")
    assert (record.conclusion, record.commit_id) == (conclusion, tip)
    assert await notebook.regression_cases_for(tip) == (RegressionCase("fp8_overflow", tip, "p1"),)


async def test_a_correctness_fix_without_its_proof_is_refused(setup):
    notebook, repo, measured = setup
    verifier = RecordingVerifier()
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), verifier)

    with pytest.raises(ValueError, match="without the proof of its case"):
        await gate.settle(measured, _judged(Verdict.FLAT, fixes_correctness=True))
    with pytest.raises(ValueError, match="without the proof of its case"):
        await gate.fork(measured, _judged(Verdict.REGRESSED, fixes_correctness=True), "oob at 8k")

    assert await repo.head("s1") == "c0"
    assert ((await notebook.record("r1")).judgement, verifier.asked) == (None, [])


async def test_a_commit_whose_case_cannot_be_registered_is_taken_back(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())
    not_a_case = FixProof(ReproRun("p0", REPRODUCED), ReproRun("r1", NOT_REPRODUCED), ReproRun("p0", REPRODUCED))

    with pytest.raises(ValueError, match="r1 is not a repro test"):
        await gate.settle(measured, _judged(Verdict.FLAT, fixes_correctness=True), fix=not_a_case)

    assert await repo.head("s1") == "c0"
    assert (await notebook.record("r1")).judgement is None


async def test_the_gate_refuses_a_step_the_table_does_not_call_for(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(AdmissionStatus.ACCEPTED), RecordingVerifier())

    with pytest.raises(ValueError, match="calls for commit"):
        await gate.fork(measured, _judged(Verdict.IMPROVED), "not a fix")
    with pytest.raises(ValueError, match="calls for commit"):
        await gate.decline(measured, _judged(Verdict.IMPROVED))
