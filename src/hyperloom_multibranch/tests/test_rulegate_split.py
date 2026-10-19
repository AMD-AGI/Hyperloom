# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import logging
from dataclasses import dataclass

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
from hyperloom_multibranch.contracts.fork import AdmissionStatus, ReservedBranches
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, RegressionCase
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta, intent, request, result
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import FixProof, ReproRun
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier

BASELINE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
BRANCH, ROOT, FORK = "s1", "c0", "fork-1"
FIX = Judgement(Verdict.REGRESSED, -2.0, 2.0, ("r1", "r1b"), {"output_throughput": 980.0}, True, fixes_correctness=True)
BLOCKED = Judgement(Verdict.BLOCKED, 3.0, 2.0, ("r1", "r1b"), {"output_throughput": 1030.0}, False)
GATE_LOG = "hyperloom_multibranch.rulegate.gate"
CASE = AttachmentRef(FIX_CASE_ID, "dir", "a" * 64)
PROOF = FixProof(ReproRun("p0", REPRODUCED), ReproRun("p1", NOT_REPRODUCED), ReproRun("p0", REPRODUCED))


class _Stalls:
    """Holds the judgement writes of ``conclusion`` until ``release`` is set; refuses them instead with ``refuse``."""

    def __init__(self, inner: LocalNotebook, conclusion: Conclusion, *, refuse: bool = False) -> None:
        self.inner, self.conclusion, self.refuse = inner, conclusion, refuse
        self.writing, self.release = asyncio.Event(), asyncio.Event()

    async def write_judgement(self, request_id, judgement, conclusion, *args):
        if conclusion is self.conclusion:
            if self.refuse:
                raise OSError("no space left on device")
            self.writing.set()
            await self.release.wait()
        return await self.inner.write_judgement(request_id, judgement, conclusion, *args)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclass
class _Rig:
    gate: RuleGate
    notebook: LocalNotebook
    scheduler: LocalNotebook
    repo: InMemoryRepo
    writes: object

    async def conclusion(self) -> Conclusion | None:
        return (await self.notebook.record("r1")).conclusion

    async def branch(self, branch: str = FORK):
        return {node.branch: node for node in (await self.scheduler.lineage()).branches}.get(branch)


async def _rig(root, status: AdmissionStatus = AdmissionStatus.ACCEPTED, wrap=None) -> _Rig:
    toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"))
    await toolbox.write_request(request("r1", intent(BRANCH, ROOT), delta("--kv-cache-dtype fp8")))
    await toolbox.write_result(result("r1"))
    clean = request("p1", intent(BRANCH, ROOT), kind=TestKind.REPRO, purpose=Purpose.REPRO, attachments=(CASE,))
    await toolbox.write_request(clean)
    await toolbox.write_result(TestResult("p1", Validity.SUCCEEDED, repro_verdict=NOT_REPRODUCED))
    scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=ROOT))
    repo = InMemoryRepo(BASELINE)
    await repo.create_branch(BRANCH, await repo.head(INTEGRATION_BRANCH))
    notebook = LocalNotebook(root, Writer(Role.GATE, "gate"))
    writes = wrap(notebook) if wrap else notebook
    gate = RuleGate(writes, repo, RecordingScheduler(status), RecordingVerifier())
    return _Rig(gate, notebook, scheduler, repo, writes)


async def _fork(rig: _Rig) -> str:
    candidate = (await rig.notebook.record("r1")).request
    return (await rig.gate.fork(candidate, FIX, "fp8 scales overflow at 8k", fix=PROOF)).branch


async def _repair(rig: _Rig) -> str:
    candidate = (await rig.notebook.record("r1")).request
    return (await rig.gate.repair(candidate, BLOCKED, intent(BRANCH, ROOT), "scale per head")).branch


@pytest.mark.parametrize("status", [AdmissionStatus.ACCEPTED, AdmissionStatus.PENDING])
async def test_a_landed_fork_carries_the_fix_on_its_own_branch_cut_at_the_parent(tmp_path, status):
    rig = await _rig(tmp_path, status)
    gate: ReservedBranches = rig.gate
    branch = await _fork(rig)

    assert await gate.land_split(branch)

    head = await rig.repo.head(branch)
    assert await rig.repo.read_recipe(head) == apply(BASELINE, delta("--kv-cache-dtype fp8"))
    assert await rig.repo.head(BRANCH) == ROOT
    record = await rig.notebook.record("r1")
    assert (record.conclusion, record.commit_id) == (Conclusion.SPLIT_COMMITTED, head)
    node = await rig.branch(branch)
    assert (node.origin, node.start_commit, node.from_branch, node.head_commit) == (
        LineageKind.SPLIT,
        ROOT,
        BRANCH,
        head,
    )
    assert (await rig.scheduler.lineage()).commit(head).branch == branch
    assert await rig.notebook.regression_cases_for(head) == (RegressionCase(FIX_CASE_ID, head, "p1"),)
    assert not await gate.land_split(branch)


async def test_only_a_reserved_fork_is_landed(tmp_path):
    rig = await _rig(tmp_path, AdmissionStatus.REJECTED)
    await _fork(rig)
    repair = await _rig(tmp_path / "repair")
    repair_branch = await _repair(repair)

    assert not await rig.gate.land_split(FORK)
    assert not await repair.gate.land_split(repair_branch)

    assert await rig.branch() is None and await repair.branch(repair_branch) is None
    assert await repair.conclusion() is Conclusion.REPAIR_REQUESTED


async def test_a_landing_waits_for_the_fork_to_record_where_the_candidate_stands(tmp_path):
    rig = await _rig(tmp_path, wrap=lambda notebook: _Stalls(notebook, Conclusion.SPLIT_REQUESTED))
    stalled = rig.writes
    forking = asyncio.create_task(_fork(rig))
    await stalled.writing.wait()

    landing = asyncio.create_task(rig.gate.land_split(FORK))
    await asyncio.sleep(0)
    stalled.release.set()

    assert await forking == FORK and await landing
    assert await rig.conclusion() is Conclusion.SPLIT_COMMITTED


async def test_a_fix_whose_judgement_cannot_be_written_is_taken_back(tmp_path):
    rig = await _rig(tmp_path, wrap=lambda notebook: _Stalls(notebook, Conclusion.SPLIT_COMMITTED, refuse=True))
    branch = await _fork(rig)

    with pytest.raises(OSError, match="no space"):
        await rig.gate.land_split(branch)

    assert await rig.repo.head(branch) == ROOT
    assert await rig.branch(branch) is None
    assert await rig.conclusion() is Conclusion.SPLIT_REQUESTED


async def test_a_landing_whose_waiter_gives_up_still_lands(tmp_path):
    rig = await _rig(tmp_path, wrap=lambda notebook: _Stalls(notebook, Conclusion.SPLIT_COMMITTED))
    branch = await _fork(rig)
    landing = asyncio.create_task(rig.gate.land_split(branch))
    await rig.writes.writing.wait()

    landing.cancel()
    rig.writes.release.set()
    with pytest.raises(asyncio.CancelledError):
        await landing
    for _ in range(50):
        if await rig.branch(branch) is not None:
            break
        await asyncio.sleep(0)

    assert await rig.conclusion() is Conclusion.SPLIT_COMMITTED
    assert (await rig.branch(branch)).head_commit == await rig.repo.head(branch)


@pytest.mark.parametrize(("ask", "voided"), [(_fork, Conclusion.RECORDED_ONLY), (_repair, Conclusion.ACCURACY_FAILED)])
async def test_a_voided_fork_or_repair_is_recorded_only(tmp_path, ask, voided):
    rig = await _rig(tmp_path, AdmissionStatus.PENDING)
    branch = await ask(rig)

    await rig.gate.settle_voided(branch, "parent s1 stopped: no progress")

    assert await rig.conclusion() is voided
    assert not await rig.gate.land_split(branch)


async def test_a_void_of_a_landed_or_unknown_branch_records_nothing(tmp_path, caplog):
    rig = await _rig(tmp_path)
    branch = await _fork(rig)
    await rig.gate.land_split(branch)

    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        await rig.gate.settle_voided(branch, "start failed")
        await rig.gate.settle_voided("fork-9", "start failed")

    assert await rig.conclusion() is Conclusion.SPLIT_COMMITTED
    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        f"no fork or repair waits on {branch}",
        "no fork or repair waits on fork-9",
    ]
