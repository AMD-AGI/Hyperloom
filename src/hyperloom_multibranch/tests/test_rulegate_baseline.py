# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import dataclasses

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.common.perf_metric import (
    AGENTX_KEEP_P50_THRESHOLD_PCT,
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom.contracts.measurement import AccuracyValue, Purpose, RoundValue, TestKind, TestRequest
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord
from hyperloom_multibranch.notebook import LocalNotebook, codec
from hyperloom_multibranch.notebook.tests.factories import delta, result
from hyperloom_multibranch.rulegate.accuracy import BaselineAccuracyError
from hyperloom_multibranch.rulegate.baseline import (
    read_inferencex_baseline,
    write_agentx_prelude_baseline,
    write_inferencex_baseline,
)
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.tests.short_prompt_checks import short_prompt_check

ROOT = "c0"
CONC = 64
STRICT = ACCURACY_METRICS[0]
AGENTX = {GRADED_INTVTY_P50: 40.0, GRADED_DURATION: 600.0, GRADED_ERROR_RATE: 1.0}


class _Session:
    """The toolbox writes the measured baselines; the gate reads them and writes the baseline record."""

    def __init__(self, root):
        self.toolbox = LocalNotebook(root, Writer(Role.TOOLBOX, "toolbox"))
        self.gate = LocalNotebook(root, Writer(Role.GATE, "gate"))

    async def measured(
        self, request_id, *, metrics, score=None, change=None, parent=ROOT, purpose=Purpose.BASELINE, check=None
    ):
        await self.toolbox.write_request(
            TestRequest(
                kind=TestKind.TEST,
                purpose=purpose,
                branch_id=INTEGRATION_BRANCH,
                parent_commit=parent,
                delta=change or RecipeDelta(),
                accuracy=score is not None,
                request_id=request_id,
            )
        )
        rounds = (RoundValue("measure", CONC if score is not None else None, metrics),)
        accuracy = () if score is None else (AccuracyValue("gsm8k", STRICT, score),)
        measured = dataclasses.replace(result(request_id), rounds=rounds, accuracy=accuracy)
        await self.toolbox.write_result(dataclasses.replace(measured, short_prompt_consistency=check))
        return await self.gate.record(request_id)

    async def inferencex(self, request_id, throughput, score=0.90, **fields):
        return await self.measured(request_id, metrics={GRADED_OUTPUT: throughput}, score=score, **fields)


@pytest.fixture
def session(tmp_path):
    return _Session(tmp_path / "notebook")


async def _prelude(session, throughputs=(1000.0, 1010.0, 990.0), scores=(0.90, 0.91, 0.89), checks=None):
    checks = checks or (None,) * len(throughputs)
    baselines = [
        await session.inferencex(f"b{i}", t, s, check=c)
        for i, (t, s, c) in enumerate(zip(throughputs, scores, checks, strict=True))
    ]
    return await write_inferencex_baseline(session.gate, ROOT, baselines, CONC)


async def test_the_prelude_baselines_become_the_record_the_gate_judges_by(session):
    record = await _prelude(session)

    assert record.values == {GRADED_OUTPUT: 1000.0, ACCURACY_SCORE: pytest.approx(0.90)}
    assert record.dispersion_pct == {GRADED_OUTPUT: pytest.approx(1.0)}
    assert record.threshold_pct == {GRADED_OUTPUT: pytest.approx(2.0)}
    assert record.request_ids == ("b0", "b1", "b2")
    assert await session.gate.baseline() == record
    baseline = await read_inferencex_baseline(session.gate)
    assert baseline.bar_pct == pytest.approx(2.0)
    assert baseline.accuracy == pytest.approx(0.90)
    assert baseline.short_prompt_floor is None, "no root test ran the short-prompt check"


@pytest.mark.parametrize(
    ("checks", "floor"),
    [
        ((short_prompt_check(0, 1, 0, 0, 0, 0), short_prompt_check(0, 0, 2, 0, 0, 0), None), 2),
        ((short_prompt_check(0, 0, 0, 0, 0, 0), None, short_prompt_check(0, 0, 0, 0, 0, 0)), 0),
        ((short_prompt_check(0, 3, planned=6), short_prompt_check(1, 0, 0, 0, 0, 0), None), 3),
        ((short_prompt_check(0, planned=6), None, None), None),
    ],
    ids=["largest-of-those-checked", "clean-is-zero", "incomplete-still-counted", "nothing-compared"],
)
async def test_the_short_prompt_floor_is_the_largest_inconsistency_a_root_test_measured(session, checks, floor):
    record = await _prelude(session, checks=checks)

    assert record.short_prompt_floor == floor
    assert (await read_inferencex_baseline(session.gate)).short_prompt_floor == floor


@pytest.mark.parametrize(
    ("throughputs", "scores", "error"),
    [((1000.0,), (0.90,), ValueError), ((1000.0, 1010.0), (0.45, 0.46), BaselineAccuracyError)],
    ids=["one-baseline", "score-below-floor"],
)
async def test_prelude_fails_on_too_few_baselines_or_an_eval_score_below_the_floor(session, throughputs, scores, error):
    with pytest.raises(error):
        await _prelude(session, throughputs, scores)


@pytest.mark.parametrize(
    "fields",
    [{"purpose": Purpose.EXPLORE, "change": delta()}, {"purpose": Purpose.HEALTH}, {"parent": "c1"}],
    ids=["a-candidate", "a-spot-check", "another-commit"],
)
async def test_only_a_prelude_baseline_of_the_root_is_a_baseline(session, fields):
    baselines = [await session.inferencex("b0", 1000.0), await session.inferencex("b1", 1010.0, **fields)]
    with pytest.raises(ValueError, match="is not a baseline test of the root"):
        await write_inferencex_baseline(session.gate, ROOT, baselines, CONC)


def _child_run(metrics):
    return dataclasses.replace(result("prelude-child"), rounds=(RoundValue("measure", 10, metrics),))


async def test_an_agentx_baseline_read_from_the_child_run_names_it_and_anchors_on_no_root_test(session):
    record = await write_agentx_prelude_baseline(
        session.gate, ROOT, _child_run(AGENTX), source="prelude-child:/runs/baseline/r1", wall_seconds=5287.67
    )

    assert (record.values, record.request_ids, record.dispersion_pct) == (AGENTX, (), {})
    assert record.threshold_pct == {GRADED_INTVTY_P50: AGENTX_KEEP_P50_THRESHOLD_PCT}
    assert (record.source, record.wall_seconds) == ("prelude-child:/runs/baseline/r1", 5287.67)
    assert record.short_prompt_floor is None, "no BASELINE test ran the short-prompt check"
    assert await session.gate.baseline() == record


@pytest.mark.parametrize("dropped", [GRADED_INTVTY_P50, GRADED_DURATION, GRADED_ERROR_RATE])
async def test_a_child_run_without_what_every_comparison_reads_anchors_nothing(session, dropped):
    partial = {key: value for key, value in AGENTX.items() if key != dropped}
    with pytest.raises(UnreadableResult, match=dropped):
        await write_agentx_prelude_baseline(session.gate, ROOT, _child_run(partial), source="s", wall_seconds=1.0)


def test_a_baseline_written_before_it_named_a_source_reads_back_with_none():
    stored = codec.encode(BaselineRecord(ROOT, AGENTX, {}, {GRADED_INTVTY_P50: 1.0}, ("b0",)))
    del stored["source"], stored["wall_seconds"], stored["short_prompt_floor"]

    decoded = codec.decode(BaselineRecord, stored)

    assert (decoded.source, decoded.wall_seconds, decoded.request_ids) == ("", None, ("b0",))
    assert decoded.short_prompt_floor is None
