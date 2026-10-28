# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY_P50,
)
from hyperloom.contracts.measurement import Purpose, RoundValue, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import LaunchConfig, Recipe, RecipeDelta
from hyperloom_multibranch.contracts.gate import NotMeasured, PreludeMeasurement, ResultUnverified
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.interfaces import NotRecorded
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, LineageKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.tests.factories import delta, result
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule
from hyperloom_multibranch.rulegate.service import PRELUDE_SOURCE_CAVEAT, GateService, Polling
from hyperloom_multibranch.tests.fakes import RecordingVerifier
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig, NoiseModel

ROOT, CONC = "c0", 64
RECIPE = Recipe("rocm/vllm@sha256:" + "c" * 64, LaunchConfig(server_args="--max-num-seqs 128"))
QUIET = NoiseModel(node_bias_median=0.0, node_bias_p90=0.0, test_noise_sigma=0.0)
FAST, FLAT, WRONG = (delta(f"--max-num-seqs {n}") for n in (256, 136, 512))
AGENTX_RUN = {GRADED_INTVTY_P50: 132.1028, GRADED_DURATION: 3601.52, GRADED_ERROR_RATE: 0.06925}
EFFECTS = {FAST.fingerprint(): 0.05, WRONG.fingerprint(): 0.05, FLAT.fingerprint(): 0.005}


def _request(
    purpose: Purpose,
    change: RecipeDelta | None = None,
    *,
    kind: TestKind = TestKind.TEST,
    concurrency: tuple[int, ...] = (),
) -> TestRequest:
    return TestRequest(
        kind=kind,
        purpose=purpose,
        branch_id=INTEGRATION_BRANCH,
        parent_commit=ROOT,
        delta=change or RecipeDelta(),
        accuracy=True,
        concurrency=concurrency,
    )


class _Session:
    """The scheduler's side of a session: it submits tests to the fake toolbox and asks the gate about them.

    Nothing is measured before the first drain; from then on every test is measured as soon as it is queued.
    """

    def __init__(self, root, accuracy_score: float = 0.93, *, agentx: bool = False):
        self.agentx = agentx
        self.toolbox = FakeToolbox(
            FakeToolboxConfig(
                noise=QUIET,
                accuracy_metric=ACCURACY_METRICS[0],
                effects=EFFECTS,
                accuracy_score=accuracy_score,
                accuracy_scores={WRONG.fingerprint(): 0.80},
            ),
            blackboard=LocalBlackboard(root, Writer(Role.TOOLBOX, "toolbox")),
            autorun=False,
        )
        self.scheduler = LocalBlackboard(root, Writer(Role.SCHEDULER, "scheduler"))
        self.blackboard = LocalBlackboard(root, Writer(Role.GATE, "gate"))
        self.verifier = RecordingVerifier()
        self.gate = self.new_gate()

    def new_gate(self) -> GateService:
        rule = AgentXRule() if self.agentx else InferenceXRule(self.blackboard, CONC)
        return GateService(self.blackboard, self.toolbox, rule, self.verifier, polling=Polling(0.001))

    async def record_root(self) -> None:
        event = LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=ROOT, recipe_digest=RECIPE.digest())
        await self.scheduler.record_lineage(event)

    async def root_tests(self, failing: int = 0) -> list[str]:
        """PRELUDE's root tests, measured, the first ``failing`` of them failing to measure."""
        await self.record_root()
        count = await self.gate.baseline_test_count()
        ids = [await self.toolbox.submit_test(_request(Purpose.BASELINE)) for _ in range(count)]
        self.toolbox.script(lambda request: request.request_id in ids[:failing], [Fate.MEASUREMENT])
        await self.toolbox.drain()
        return ids

    def retests(self) -> list[TestRequest]:
        return [request for request in self.toolbox.requests.values() if request.purpose is Purpose.CONFIRM]


@pytest.fixture
async def session(tmp_path):
    session = _Session(tmp_path / "blackboard")
    await session.gate.establish_baseline(ROOT, await session.root_tests())
    yield session
    await session.toolbox.close()


async def test_prelude_sets_the_bar_from_the_root_tests_that_measured(tmp_path):
    session = _Session(tmp_path / "blackboard")
    ids = await session.root_tests(failing=1)

    record = await session.gate.establish_baseline(ROOT, ids)

    assert record.request_ids == tuple(ids[1:])
    assert await session.blackboard.baseline() == record
    assert session.verifier.asked == ids[1:]
    await session.toolbox.close()


async def test_prelude_with_a_root_test_that_does_not_re_verify_sets_no_bar(tmp_path):
    session = _Session(tmp_path / "blackboard")
    ids = await session.root_tests()
    session.verifier.rejects = frozenset(ids[1:2])

    with pytest.raises(ResultUnverified, match=f"{ids[1]} did not re-verify"):
        await session.gate.establish_baseline(ROOT, ids)
    with pytest.raises(NotRecorded):
        await session.blackboard.baseline()
    await session.toolbox.close()


async def test_prelude_on_root_tests_scoring_below_the_floor_sets_no_bar(tmp_path):
    session = _Session(tmp_path / "blackboard", accuracy_score=0.3)
    ids = await session.root_tests()

    with pytest.raises(NotMeasured, match="below the floor"):
        await session.gate.establish_baseline(ROOT, ids)
    await session.toolbox.close()


async def test_prelude_with_one_root_test_measured_sets_no_bar(tmp_path):
    session = _Session(tmp_path / "blackboard")
    ids = await session.root_tests(failing=2)

    with pytest.raises(NotMeasured, match="1 of 3 root tests measured"):
        await session.gate.establish_baseline(ROOT, ids)
    await session.toolbox.close()


async def test_an_agentx_prelude_asks_for_no_root_test(tmp_path):
    session = _Session(tmp_path / "blackboard", agentx=True)

    assert await session.gate.baseline_test_count() == 0
    assert await session.root_tests() == []
    await session.toolbox.close()


def _child_run(tmp_path, metrics=AGENTX_RUN, validity=Validity.SUCCEEDED) -> PreludeMeasurement:
    report, export = tmp_path / "benchmark_report.json", tmp_path / "profile_export.jsonl"
    report.write_text('{"scenario": "agentx"}')
    export.write_text("{}\n")
    run = replace(result("prelude-child", validity=validity), rounds=(RoundValue("measure", 10, metrics),))
    return PreludeMeasurement(
        run,
        "/child/runs/baseline/r1",
        str(report),
        hashlib.sha256(report.read_bytes()).hexdigest(),
        str(export),
        hashlib.sha256(export.read_bytes()).hexdigest(),
        5287.67,
    )


async def test_an_agentx_baseline_is_the_prelude_childs_run_of_the_root_and_names_it(tmp_path):
    session = _Session(tmp_path / "blackboard", agentx=True)
    measured = _child_run(tmp_path)

    record = await session.gate.establish_prelude_baseline(ROOT, measured)

    assert (record.values, record.request_ids, record.wall_seconds) == (AGENTX_RUN, (), 5287.67)
    assert record.threshold_pct == {GRADED_INTVTY_P50: AGENTX_KEEP_P50_THRESHOLD_PCT}
    assert record.source == (
        f"prelude-child:/child/runs/baseline/r1 sha256:{measured.report_sha256}/{measured.export_sha256}; "
        f"{PRELUDE_SOURCE_CAVEAT}"
    )
    assert session.verifier.asked == []
    assert await session.blackboard.baseline() == record
    await session.toolbox.close()


async def test_a_child_run_whose_copy_changed_since_it_was_read_anchors_nothing(tmp_path):
    session = _Session(tmp_path / "blackboard", agentx=True)
    measured = _child_run(tmp_path)
    Path(measured.export).write_text("{}\n{}\n")

    with pytest.raises(NotMeasured, match="profile_export.jsonl has sha256"):
        await session.gate.establish_prelude_baseline(ROOT, measured)
    await session.toolbox.close()


@pytest.mark.parametrize(
    ("metrics", "validity", "error"),
    [
        ({GRADED_DURATION: 3601.5}, Validity.SUCCEEDED, GRADED_INTVTY_P50),
        (AGENTX_RUN, Validity.MEASUREMENT_FAILED, "did not measure"),
    ],
)
async def test_a_child_run_without_a_reading_anchors_nothing(tmp_path, metrics, validity, error):
    session = _Session(tmp_path / "blackboard", agentx=True)

    with pytest.raises(NotMeasured, match=error):
        await session.gate.establish_prelude_baseline(ROOT, _child_run(tmp_path, metrics, validity))
    await session.toolbox.close()


async def test_an_inferencex_baseline_is_never_read_from_the_child_run(session, tmp_path):
    with pytest.raises(ValueError, match="measured by 3 root tests"):
        await session.gate.establish_prelude_baseline(ROOT, _child_run(tmp_path))


async def test_a_merge_gain_counts_once_another_node_repeats_it(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST))

    judgement = await session.gate.compare(merge)

    (retest,) = session.retests()
    assert (judgement.verdict, judgement.request_ids) == (Verdict.IMPROVED, (merge, retest.request_id))
    assert retest.avoid_nodes == ((await session.blackboard.record(merge)).result.sandbox.node_name,)
    assert session.verifier.asked[-2:] == [merge, retest.request_id]


async def test_a_merge_whose_retest_does_not_re_verify_cannot_be_judged(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST))
    await session.gate.compare(merge)
    (retest,) = session.retests()
    session.verifier.rejects = frozenset({retest.request_id})

    with pytest.raises(NotMeasured, match=f"{retest.request_id} did not re-verify"):
        await session.new_gate().compare(merge)


async def test_a_flat_merge_is_judged_without_a_retest(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FLAT))

    assert (await session.gate.compare(merge)).verdict is Verdict.FLAT
    assert session.retests() == []


async def test_asking_again_after_a_restart_reuses_the_retest_asked_for(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST))
    judgement = await session.gate.compare(merge)

    assert await session.new_gate().compare(merge) == judgement
    assert len(session.retests()) == 1


async def test_a_confirm_asked_before_the_first_is_measured_is_the_retest_its_gain_is_judged_with(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST))

    confirm = await session.gate.request_confirm(merge)
    judgement = await session.gate.compare(merge)

    (retest,) = session.retests()
    assert (retest.request_id, retest.retest_of) == (confirm, merge)
    assert (judgement.verdict, judgement.request_ids) == (Verdict.IMPROVED, (merge, confirm))


async def test_a_flat_first_is_judged_without_the_confirm_asked_ahead_of_it(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FLAT))

    confirm = await session.gate.request_confirm(merge)
    judgement = await session.gate.compare(merge)

    assert (judgement.verdict, judgement.request_ids) == (Verdict.FLAT, (merge,))
    assert [retest.request_id for retest in session.retests()] == [confirm]


async def test_a_withdrawn_test_is_not_measured(tmp_path):
    session = _Session(tmp_path / "blackboard")
    await session.record_root()
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST))
    await session.toolbox.cancel(merge)

    with pytest.raises(NotMeasured, match="withdrawn"):
        await session.gate.compare(merge)
    await session.toolbox.close()


async def test_a_merge_measured_off_the_graded_concurrency_is_not_measured(session):
    merge = await session.toolbox.submit_test(_request(Purpose.INTEGRATION, FAST, concurrency=(CONC // 8,)))

    with pytest.raises(NotMeasured, match=f"no measure-phase round at concurrency {CONC}"):
        await session.gate.compare(merge)


@pytest.mark.parametrize(("change", "passes"), [(FAST, True), (WRONG, False)], ids=["accurate", "inaccurate"])
async def test_an_accuracy_test_is_judged_on_its_eval_score(session, change, passes):
    check = await session.toolbox.submit_test(_request(Purpose.ACCURACY, change, kind=TestKind.ACCURACY))

    assert await session.gate.accuracy_passes(check) is passes


async def test_an_accuracy_test_that_does_not_re_verify_is_not_judged(session):
    check = await session.toolbox.submit_test(_request(Purpose.ACCURACY, FAST, kind=TestKind.ACCURACY))
    session.verifier.rejects = frozenset({check})

    with pytest.raises(ResultUnverified):
        await session.gate.accuracy_passes(check)
