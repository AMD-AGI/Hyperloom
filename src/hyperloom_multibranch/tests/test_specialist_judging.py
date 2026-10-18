# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from dataclasses import replace

import pytest

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta
from hyperloom_multibranch.rulegate.baseline import write_inferencex_baseline
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.inferencex import GAIN_NOT_CONFIRMED
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.specialist.judging import NOTICE, BranchJudge, Judging
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig, NoiseModel

ROOT, BRANCH, CONC = "c0", "s1", 64
RECIPE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
QUIET = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
FAST, FAST_TOO, FLAT, WRONG, BROKEN = (delta(f"--max-num-seqs {n}") for n in (256, 384, 136, 512, 1024))
EFFECTS = {change.fingerprint(): 0.05 for change in (FAST, FAST_TOO, WRONG)} | {FLAT.fingerprint(): 0.005}


def _request(
    branch: str,
    purpose: Purpose,
    change: RecipeDelta | None = None,
    accuracy: bool = True,
    parent: str = ROOT,
    retest_of: str | None = None,
    concurrency: tuple[int, ...] = (),
) -> TestRequest:
    return TestRequest(
        kind=TestKind.TEST,
        purpose=purpose,
        branch_id=branch,
        parent_commit=parent,
        delta=change or RecipeDelta(),
        accuracy=accuracy,
        retest_of=retest_of,
        concurrency=concurrency,
    )


class _Session:
    """PRELUDE done and s1 open at the root: the fake toolbox measures, the real gate settles, s1's judge acts."""

    def __init__(self, root, **config):
        self.toolbox = FakeToolbox(
            FakeToolboxConfig(
                noise=QUIET,
                accuracy_metric=ACCURACY_METRICS[0],
                effects=EFFECTS,
                accuracy_scores={WRONG.fingerprint(): 0.80},
                **config,
            ),
            notebook=LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox")),
            autorun=False,
        )
        self.scheduler = LocalNotebook(root, Writer(Role.SCHEDULER, "scheduler"))
        self.gate_notebook = LocalNotebook(root, Writer(Role.GATE, "gate"))
        self.runtime = LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, BRANCH, BRANCH))
        self.repo = InMemoryRepo(RECIPE)
        self.verifier = RecordingVerifier()
        self.gate = RuleGate(self.gate_notebook, self.repo, RecordingScheduler(AdmissionStatus.ACCEPTED), self.verifier)
        self.judge = self.new_judge()

    def new_judge(self) -> BranchJudge:
        return BranchJudge(BRANCH, self.runtime, Judging(self.gate, self.toolbox, InferenceXRule(self.runtime, CONC)))

    async def prelude(self) -> None:
        root_event = LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT, recipe_digest=RECIPE.digest())
        await self.scheduler.record_lineage(root_event)
        await self.repo.create_branch(BRANCH, ROOT)
        await self.scheduler.record_lineage(LineageEvent(LineageKind.OPEN, BRANCH, from_commit=ROOT))
        ids = [await self.toolbox.submit_test(_request(INTEGRATION_BRANCH, Purpose.BASELINE)) for _ in range(3)]
        await self.toolbox.drain()
        baselines = [await self.gate_notebook.record(request_id) for request_id in ids]
        await write_inferencex_baseline(self.gate_notebook, ROOT, baselines, CONC)

    async def explore(self, change: RecipeDelta) -> str:
        return await self.toolbox.submit_test(_request(BRANCH, Purpose.EXPLORE, change))

    async def step(self) -> None:
        """Measure what is queued, then let the judge act on it."""
        await self.toolbox.drain()
        await self.judge.reconcile()

    def retests(self) -> list[TestRequest]:
        return [request for request in self.toolbox.requests.values() if request.purpose is Purpose.CONFIRM]

    async def notices(self) -> list[str]:
        return [r.entry.content for r in await self.runtime.branch_log(BRANCH) if r.entry.content.startswith(NOTICE)]


@pytest.fixture
async def session(tmp_path):
    session = _Session(tmp_path / "notebook")
    await session.prelude()
    return session


async def test_a_gain_is_retested_on_another_node_and_lands_on_its_branch(session):
    first = await session.explore(FAST)
    await session.step()
    (retest,) = session.retests()
    first_node = (await session.runtime.record(first)).result.sandbox.node_name
    assert (retest.retest_of, retest.parent_commit, retest.delta, retest.accuracy) == (first, ROOT, FAST, True)
    assert retest.avoid_nodes == (first_node,)

    await session.step()

    record = await session.runtime.record(first)
    assert record.judgement.verdict is Verdict.IMPROVED
    assert set(record.judgement.recorded_value) == {GRADED_OUTPUT, ACCURACY_SCORE}
    assert (record.conclusion, record.commit_id) == (Conclusion.COMMITTED, await session.repo.head(BRANCH))
    assert (await session.runtime.record(retest.request_id)).result.sandbox.node_name != first_node
    (notice,) = await session.notices()
    assert notice.startswith(f"{NOTICE} {first} landed on your branch at +") and f"% over {ROOT};" in notice


async def test_the_retest_carries_the_files_the_candidate_was_measured_with(session):
    attached = (AttachmentRef("tuned-moe-config", "file", "a" * 64),)
    await session.toolbox.submit_test(replace(_request(BRANCH, Purpose.EXPLORE, FAST), attachments=attached))
    await session.step()

    (retest,) = session.retests()
    assert retest.attachments == attached


async def test_a_gain_whose_result_does_not_re_verify_is_not_committed_and_the_specialist_is_told(session):
    first = await session.explore(FAST)
    session.verifier.rejects = frozenset({first})
    await session.step()
    await session.step()
    await session.judge.reconcile()

    record = await session.runtime.record(first)
    assert (record.conclusion, record.commit_id) == (Conclusion.UNVERIFIED, None)
    assert await session.repo.head(BRANCH) == ROOT
    (notice,) = await session.notices()
    assert f"{first} was not committed: a result behind it did not re-verify against its signed bundle" in notice


async def test_a_flat_candidate_is_recorded_without_a_retest(session):
    first = await session.explore(FLAT)
    await session.step()

    record = await session.runtime.record(first)
    assert (record.judgement.verdict, record.conclusion) == (Verdict.FLAT, Conclusion.RECORDED_ONLY)
    assert session.retests() == []
    assert await session.repo.head(BRANCH) == ROOT


async def test_a_gain_the_retest_does_not_repeat_is_recorded_only(tmp_path):
    session = _Session(tmp_path / "notebook", effect_fn=lambda r: 0.05 if r.purpose is Purpose.EXPLORE else 0.0)
    await session.prelude()
    first = await session.explore(FAST)
    await session.step()
    await session.step()

    record = await session.runtime.record(first)
    assert (record.judgement.verdict, record.judgement.refused_by) == (Verdict.FLAT, (GAIN_NOT_CONFIRMED,))
    assert record.conclusion is Conclusion.RECORDED_ONLY


async def test_a_gain_the_accuracy_gate_holds_back_twice_is_held_for_the_specialist_once(session):
    first = await session.explore(WRONG)
    await session.step()
    await session.step()
    await session.new_judge().reconcile()

    record = await session.runtime.record(first)
    assert (record.judgement.verdict, record.conclusion) == (Verdict.BLOCKED, Conclusion.DIAGNOSIS_PENDING)
    assert len(session.retests()) == 1
    (notice,) = await session.notices()
    assert notice == (
        f"{NOTICE} {first} cleared the bar twice and is held back by the accuracy gate; "
        "diagnose it, then request_repair or decline_repair"
    )


async def test_a_failed_measurement_takes_no_judgement(session):
    session.toolbox.script(lambda request: request.delta == BROKEN, [Fate.MEASUREMENT])
    first = await session.explore(BROKEN)
    await session.step()

    record = await session.runtime.record(first)
    assert record.result.validity is Validity.MEASUREMENT_FAILED
    assert record.judgement is None
    assert session.retests() == []
    assert await session.notices() == []


async def test_an_inferencex_test_without_the_graded_concurrency_is_not_judged(session):
    first = await session.toolbox.submit_test(_request(BRANCH, Purpose.EXPLORE, FAST, concurrency=(CONC * 2,)))
    await session.step()

    assert (await session.runtime.record(first)).judgement is None
    assert session.retests() == []
    (notice,) = await session.notices()
    assert f"{first} cannot be judged" in notice


async def test_of_two_gains_on_one_parent_the_second_does_not_land(session):
    first, second = await session.explore(FAST), await session.explore(FAST_TOO)
    await session.step()
    await session.step()

    landed, moved = await session.runtime.record(first), await session.runtime.record(second)
    assert landed.conclusion is Conclusion.COMMITTED
    assert (moved.judgement.verdict, moved.conclusion, moved.commit_id) == (
        Verdict.IMPROVED,
        Conclusion.RECORDED_ONLY,
        None,
    )
    assert await session.repo.head(BRANCH) == landed.commit_id
    first_landed, second_moved = await session.notices()
    assert f"{first} landed on your branch" in first_landed
    assert (
        f"{second} improved on {ROOT}, but the branch has moved on; submit it again on the head as a new test"
        in second_moved
    )


async def test_a_candidate_on_a_commit_the_notebook_does_not_know_does_not_stop_the_branch(session):
    stray = await session.toolbox.submit_test(_request(BRANCH, Purpose.EXPLORE, FLAT, parent=BRANCH))
    await session.explore(FAST)
    await session.step()

    assert len(session.retests()) == 1
    (notice,) = await session.notices()
    assert f"{stray} cannot be judged" in notice


@pytest.mark.parametrize(
    ("change", "attached"),
    [(FAST_TOO, ()), (FAST, (AttachmentRef("other", "file", "b" * 64),))],
    ids=["another change", "other files"],
)
async def test_a_retest_of_another_candidate_does_not_settle_it(session, change, attached):
    first = await session.explore(FAST)
    await session.toolbox.drain()
    other = replace(_request(BRANCH, Purpose.CONFIRM, change, retest_of=first), attachments=attached)
    stray = await session.toolbox.submit_test(other)
    await session.step()
    await session.step()

    (own,) = [retest.request_id for retest in session.retests() if retest.request_id != stray]
    record = await session.runtime.record(first)
    assert (record.conclusion, record.judgement.request_ids) == (Conclusion.COMMITTED, (first, own))


async def test_a_restarted_judge_takes_each_step_once(session):
    await session.explore(FAST)
    await session.step()

    await session.new_judge().reconcile()

    assert len(session.retests()) == 1


async def test_the_judge_wakes_when_a_result_arrives(session):
    running = asyncio.create_task(session.judge.run(poll_seconds=0.01))
    first = await session.explore(FLAT)
    await session.toolbox.drain()
    for _ in range(500):
        if (await session.runtime.record(first)).judgement is not None:
            break
        await asyncio.sleep(0.01)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert (await session.runtime.record(first)).conclusion is Conclusion.RECORDED_ONLY
