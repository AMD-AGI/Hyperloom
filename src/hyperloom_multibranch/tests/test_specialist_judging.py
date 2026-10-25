# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
from dataclasses import replace

import pytest

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.contracts.measurement import (
    NOT_REPRODUCED,
    REPRODUCED,
    AttachmentRef,
    Purpose,
    TestKind,
    TestRequest,
    Validity,
)
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.fork import AdmissionStatus
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, Verdict
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import delta
from hyperloom_multibranch.rulegate.baseline import write_inferencex_baseline
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.inferencex import DROPPED_REQUESTS, GAIN_NOT_CONFIRMED
from hyperloom_multibranch.rulegate.critic import CriticVerdict, FixReview
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, pack_case
from hyperloom_multibranch.rulegate.regression import ReproRunner
from hyperloom_multibranch.rulegate.rules import InferenceXRule
from hyperloom_multibranch.rulegate.service import Polling
from hyperloom_multibranch.specialist.fix_check import FixChecking
from hyperloom_multibranch.specialist.judging import NOTICE, BranchJudge, Judging
from hyperloom_multibranch.tests.fakes import InMemoryRepo, RecordingScheduler, RecordingVerifier, ReproToolbox
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig, NoiseModel

ROOT, BRANCH, CONC = "c0", "s1", 64
RECIPE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
QUIET = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
FAST, FAST_TOO, FLAT, WRONG, BROKEN = (delta(f"--max-num-seqs {n}") for n in (256, 384, 136, 512, 1024))
SLOW = delta("--max-num-seqs 64")
EFFECTS = {change.fingerprint(): 0.05 for change in (FAST, FAST_TOO, WRONG)} | {
    FLAT.fingerprint(): 0.005,
    SLOW.fingerprint(): -0.05,
}


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
        self.fixes: FixChecking | None = None
        self.judge = self.new_judge()

    def new_judge(self) -> BranchJudge:
        rule = InferenceXRule(self.runtime, CONC)
        return BranchJudge(BRANCH, self.runtime, Judging(self.gate, self.toolbox, rule, self.fixes))

    def check_fixes(self, root, critic: "_Critic", *, parent_reproduces: bool = True) -> AttachmentRef:
        """Check declared fixes with a repro toolbox whose case reproduces on an unchanged commit (when
        ``parent_reproduces``) and runs clean on a changed one; returns the case to attach."""

        def answer(request: TestRequest) -> str:
            unchanged = request.delta == RecipeDelta()
            return REPRODUCED if unchanged and parent_reproduces else NOT_REPRODUCED

        blobs = LocalBlobStore(root / "blobs")
        repro = ReproToolbox(LocalNotebook(root / "notebook", Writer(Role.TOOLBOX, "repro")), answer)
        runner = ReproRunner(repro, self.runtime, RecordingVerifier(), polling=Polling(0.001))
        self.fixes = FixChecking(runner, critic, blobs, "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64")
        self.judge = self.new_judge()
        sha = blobs.put(pack_case("curl -s $HL_BASE_URL/v1/completions -d @long.json | grep -q nan && exit 1", "nan"))
        return AttachmentRef(FIX_CASE_ID, "dir", sha)

    async def declare(self, change: RecipeDelta, case: AttachmentRef) -> str:
        return await self.toolbox.submit_test(replace(_request(BRANCH, Purpose.EXPLORE, change), attachments=(case,)))

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


async def test_a_gain_whose_retest_fails_as_a_candidate_is_recorded_regressed(session):
    session.toolbox.script(lambda request: request.purpose is Purpose.CONFIRM, [Fate.CANDIDATE] * 4)
    first = await session.explore(FAST)
    await session.step()
    (retest,) = session.retests()
    await session.step()

    record = await session.runtime.record(first)
    assert record.conclusion is Conclusion.RECORDED_ONLY
    assert (record.judgement.verdict, record.judgement.refused_by[-1]) == (Verdict.REGRESSED, "candidate_failed")
    assert record.judgement.request_ids == (first, retest.request_id)
    assert await session.repo.head(BRANCH) == ROOT
    (notice,) = await session.notices()
    assert notice.startswith(f"{NOTICE} the retest of {first} ended candidate_failed")
    assert notice.endswith("recorded as regressed")


async def test_a_gain_whose_retest_failed_to_measure_is_left_unjudged(session):
    session.toolbox.script(lambda request: request.purpose is Purpose.CONFIRM, [Fate.MEASUREMENT])
    first = await session.explore(FAST)
    await session.step()
    await session.step()

    assert (await session.runtime.record(first)).judgement is None
    (notice,) = await session.notices()
    assert notice.endswith("nothing was judged")


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
        f"{NOTICE} {first} held its gain on the retest and is held back by the accuracy gate; "
        "diagnose it, then request_repair or decline_repair"
    )


async def test_a_gain_that_drops_requests_on_both_measurements_is_held_for_the_specialist(tmp_path):
    session = _Session(tmp_path / "notebook", error_rates={FAST.fingerprint(): 1.0})
    await session.prelude()
    first = await session.explore(FAST)
    await session.step()
    await session.step()

    record = await session.runtime.record(first)
    assert (record.judgement.verdict, record.judgement.refused_by) == (Verdict.BLOCKED, (DROPPED_REQUESTS,))
    assert record.conclusion is Conclusion.DIAGNOSIS_PENDING
    assert await session.repo.head(BRANCH) == ROOT
    (notice,) = await session.notices()
    assert f"{first} held its gain on the retest and is held back by {DROPPED_REQUESTS};" in notice


class _Critic:
    def __init__(self, reachable: bool = True) -> None:
        self.reachable, self.reviews = reachable, []

    async def review(self, review: FixReview) -> CriticVerdict:
        self.reviews.append(review)
        return CriticVerdict(self.reachable, "a 4k prompt triggers it" if self.reachable else "only 64k prompts do")


async def test_a_proven_reachable_fix_that_costs_nothing_lands_with_its_case(session, tmp_path):
    critic = _Critic()
    first = await session.declare(FLAT, session.check_fixes(tmp_path, critic))
    await session.step()

    record = await session.runtime.record(first)
    head = await session.repo.head(BRANCH)
    assert (record.conclusion, record.commit_id) == (Conclusion.COMMITTED_WITH_REPRO, head)
    assert (record.judgement.fixes_correctness, record.judgement.foundation_fix) == (True, True)
    assert [case.fix_commit for case in await session.runtime.regression_cases_for(head)] == [head]
    assert critic.reviews[0].case.diagnosis == "nan"
    (notice,) = await session.notices()
    assert "with its repro case, which now guards every later commit" in notice


async def test_a_proven_reachable_fix_that_costs_performance_asks_for_its_fork(session, tmp_path):
    first = await session.declare(SLOW, session.check_fixes(tmp_path, _Critic()))
    await session.step()

    record = await session.runtime.record(first)
    assert (record.conclusion, await session.repo.head(BRANCH)) == (Conclusion.SPLIT_REQUESTED, ROOT)
    (notice,) = await session.notices()
    assert f"{first} fixes a defect of {ROOT}: it goes to a new branch, fork-1, cut at that commit" in notice


@pytest.mark.parametrize(
    ("checked", "reachable", "parent_reproduces", "why"),
    [
        (True, True, False, "its fix is not proven: the case did not reproduce on the parent"),
        (True, False, True, "the reviewer refused its fix: only 64k prompts do"),
        (False, True, True, "this session does not check correctness fixes"),
    ],
    ids=["not-proven", "refused", "unchecked"],
)
async def test_a_fix_that_does_not_count_is_judged_on_its_performance_alone(
    session, tmp_path, checked, reachable, parent_reproduces, why
):
    case = session.check_fixes(tmp_path, _Critic(reachable), parent_reproduces=parent_reproduces)
    if not checked:
        session.fixes = None
        session.judge = session.new_judge()
    first = await session.declare(FLAT, case)
    await session.step()

    record = await session.runtime.record(first)
    assert (record.conclusion, record.judgement.fixes_correctness) == (Conclusion.RECORDED_ONLY, False)
    (notice,) = await session.notices()
    assert notice.startswith(f"{NOTICE} {first} is judged on its performance alone: {why}")


async def test_a_gain_that_carries_a_proven_case_lands_on_its_gain_and_keeps_the_case_unreviewed(session, tmp_path):
    critic = _Critic()
    first = await session.declare(FAST, session.check_fixes(tmp_path, critic))
    await session.step()
    await session.step()

    record = await session.runtime.record(first)
    head = await session.repo.head(BRANCH)
    assert (record.conclusion, record.judgement.fixes_correctness) == (Conclusion.COMMITTED, False)
    assert len(await session.runtime.regression_cases_for(head)) == 1
    assert critic.reviews == []


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


def test_only_a_notice_that_no_judgement_follows_closes_a_candidate():
    from hyperloom_multibranch.specialist.judging import left_unjudged

    assert left_unjudged(f"{NOTICE} r1 cannot be judged: no score")
    assert left_unjudged(f"{NOTICE} the retest of r1 failed; nothing was judged")
    assert left_unjudged(f"{NOTICE} r1 was not committed: a result behind it did not re-verify")
    assert not left_unjudged(f"{NOTICE} r1 is judged on its performance alone: the case is not available")
    assert not left_unjudged(f"{NOTICE} r1 held its gain on the retest and is held back by the accuracy gate")
