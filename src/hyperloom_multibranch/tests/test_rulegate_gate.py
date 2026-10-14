# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import dataclasses

import pytest

from hyperloom.contracts.recipe import LaunchConfig, Recipe, apply
from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.decision import Action
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
GEMM = intent("s1", "c0")


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
    gate = RuleGate(notebook, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))

    assert await gate.settle(measured, _judged(Verdict.IMPROVED)) is Action.COMMIT

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
    gate = RuleGate(notebook, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))

    assert await gate.settle(measured, judgement) is Action.RECORD_ONLY

    assert await repo.head("s1") == "c0"
    assert (await notebook.record("r1")).conclusion is conclusion


@pytest.mark.parametrize(
    ("verdict", "action"), [(Verdict.NEEDS_CONFIRM, Action.RETEST), (Verdict.BLOCKED, Action.DIAGNOSE)]
)
async def test_a_retest_or_a_diagnosis_is_the_callers_next_step(setup, verdict, action):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))

    assert await gate.settle(measured, _judged(verdict, accuracy_ok=False)) is action

    assert (await notebook.record("r1")).judgement is None


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
    scheduler = RecordingScheduler(Admission(status))
    gate = RuleGate(notebook, repo, scheduler)
    blocked = _judged(Verdict.BLOCKED, accuracy_ok=False)

    assert (await gate.repair(measured, blocked, GEMM, "per-tensor KV scales; scale per head")).status is status

    assert [repair.intent for repair in scheduler.repairs] == [GEMM]
    assert scheduler.repairs[0].candidate_delta == measured.delta
    assert (await notebook.record("r1")).conclusion is conclusion


async def test_a_blocked_gain_with_no_repair_is_recorded(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))
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
async def test_a_fix_that_costs_performance_asks_for_a_fork(setup, status, conclusion):
    notebook, repo, measured = setup
    scheduler = RecordingScheduler(Admission(status))
    gate = RuleGate(notebook, repo, scheduler)

    await gate.fork(measured, _judged(Verdict.REGRESSED, fixes_correctness=True), "repro/moe-oob.py", "oob at 8k")

    assert [(fork.parent_commit, fork.repro_case) for fork in scheduler.forks] == [("c0", "repro/moe-oob.py")]
    assert (await notebook.record("r1")).conclusion is conclusion
    assert await repo.head("s1") == "c0"


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


async def test_a_commit_whose_judgement_cannot_be_written_is_taken_back(setup):
    _notebook, repo, measured = setup
    gate = RuleGate(_FullDisk(), repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))

    with pytest.raises(OSError, match="no space"):
        await gate.settle(measured, _judged(Verdict.IMPROVED))

    assert await repo.head("s1") == "c0"


async def test_a_specialist_stopped_mid_commit_still_records_the_commit(setup):
    notebook, repo, measured = setup
    stalled = _Stalled(notebook)
    gate = RuleGate(stalled, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))
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


async def test_the_gate_refuses_a_step_the_table_does_not_call_for(setup):
    notebook, repo, measured = setup
    gate = RuleGate(notebook, repo, RecordingScheduler(Admission(AdmissionStatus.ACCEPTED)))

    with pytest.raises(ValueError, match="calls for commit"):
        await gate.fork(measured, _judged(Verdict.IMPROVED), "repro/x.py", "not a fix")
    with pytest.raises(ValueError, match="calls for commit"):
        await gate.decline(measured, _judged(Verdict.IMPROVED))
