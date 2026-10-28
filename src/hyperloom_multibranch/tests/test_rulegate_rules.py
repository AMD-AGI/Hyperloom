# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS, AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.perf_metric import GRADED_ERROR_RATE, GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.long_context import (
    LongContextResult,
    LongContextRoot,
)
from hyperloom.contracts.measurement import AccuracyValue, Purpose, RoundValue
from hyperloom.contracts.tests.long_context_checks import long_context_check
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, Verdict
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.blackboard.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.inferencex import LONG_CONTEXT_DROPPED
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule, retest_request
from hyperloom_multibranch.rulegate.short_prompts import SHORT_PROMPTS_INCONSISTENT
from hyperloom_multibranch.tests.short_prompt_checks import short_prompt_check


def _measured(metrics: dict[str, float]) -> TestRecord:
    rounds = (RoundValue("measure", 8, {GRADED_INTVTY_P50: 40.0, **metrics}),)
    return TestRecord(request("r1", intent("s1", "c0")), replace(result("r1"), rounds=rounds))


@pytest.mark.parametrize(
    ("metrics", "passes"),
    [({GRADED_ERROR_RATE: 0.0}, True), ({GRADED_ERROR_RATE: AGENTX_ERROR_RATE_THRESHOLD_PCT + 1}, False), ({}, False)],
    ids=["clean", "erroring", "unreported"],
)
async def test_an_agentx_run_is_accurate_while_its_error_rate_stays_within_the_bar(metrics, passes):
    assert await AgentXRule().accuracy_passes(_measured(metrics)) is passes


def _unscored(purpose: Purpose, accuracy: bool) -> TestRecord:
    retest_of = "r0" if purpose is Purpose.CONFIRM else None
    asked = replace(request("r1", intent("s1", "c0")), purpose=purpose, accuracy=accuracy, retest_of=retest_of)
    rounds = (RoundValue("measure", 64, {GRADED_OUTPUT: 1000.0}, failed_requests=0),)
    return TestRecord(asked, replace(result("r1"), rounds=rounds, accuracy=()))


@pytest.mark.parametrize(
    ("purpose", "accuracy"),
    [
        (Purpose.EXPLORE, True),
        (Purpose.CONFIRM, True),
        (Purpose.INTEGRATION, True),
        (Purpose.FINAL, True),
        (Purpose.CONFIRM, False),
        (Purpose.INTEGRATION, False),
    ],
)
async def test_an_inferencex_test_without_a_score_is_unreadable_unless_it_is_an_explore_that_skipped_its_eval(
    purpose, accuracy
):
    with pytest.raises(UnreadableResult, match="no eval score"):
        await InferenceXRule(_Baseline(None), 64).first(_unscored(purpose, accuracy), {GRADED_OUTPUT: 1000.0})


PARENT = {GRADED_OUTPUT: 1000.0}


class _Baseline:
    """The session's baseline record, holding candidates to ``floor``."""

    def __init__(self, floor):
        self.record = BaselineRecord(
            "c0",
            {GRADED_OUTPUT: 1000.0, ACCURACY_SCORE: 0.90},
            {},
            {GRADED_OUTPUT: 2.0},
            ("b0", "b1", "b2"),
            short_prompt_floor=floor,
        )

    async def baseline(self):
        return self.record


def _checked(throughput, check, *, request_id="r1", retest_of=None, long_context=None) -> TestRecord:
    purpose = Purpose.EXPLORE if retest_of is None else Purpose.CONFIRM
    asked = replace(request(request_id, intent("s1", "c0")), purpose=purpose, accuracy=True, retest_of=retest_of)
    rounds = (RoundValue("measure", 64, {GRADED_OUTPUT: throughput}, failed_requests=0),)
    scores = (AccuracyValue("gsm8k", ACCURACY_METRICS[0], 0.90),)
    measured = replace(
        result(request_id), rounds=rounds, accuracy=scores, short_prompt_consistency=check, long_context=long_context
    )
    return TestRecord(asked, measured)


SHA = "a" * 64
TASK = "niah_single"
#: 64 of 64 on the root: the lower end of its 95% interval is 0.94, so a candidate is refused below 58 of 64.
ROOT = LongContextRoot(SHA, ("b0", "b1", "b2"), 64, 64, 1.0, 0.94, 1.0)


def _long_context(passed, *, root=ROOT, sha=SHA, errored=0, n=64) -> LongContextResult:
    """A check of ``n`` prompts, the first ``passed`` passed and the last ``errored`` not answered."""
    return long_context_check(n, passed, errored=errored, sha=sha, root=root)


@pytest.mark.parametrize(
    ("check", "refused"),
    [
        (_long_context(57), True),
        (_long_context(58), False),
        (_long_context(10, root=None), False),
        (_long_context(10, sha="b" * 64), False),
        (_long_context(10, errored=1), False),
        (None, False),
    ],
    ids=["below-the-bar", "at-the-bar", "no-root-reading", "another-prompt-set", "incomplete", "no-check"],
)
async def test_an_inferencex_candidate_is_refused_on_a_long_context_drop_beyond_the_checks_noise(check, refused):
    judgement = await InferenceXRule(_Baseline(None), 64).first(_checked(1000.0, None, long_context=check), PARENT)

    assert (LONG_CONTEXT_DROPPED in judgement.refused_by) is refused
    assert judgement.verdict is (Verdict.REGRESSED if refused else Verdict.FLAT)


async def test_a_gain_with_a_long_context_drop_on_both_measurements_is_blocked():
    rule = InferenceXRule(_Baseline(None), 64)
    first = await rule.first(_checked(1100.0, None, long_context=_long_context(50)), PARENT)
    assert first.verdict is Verdict.NEEDS_CONFIRM and first.refused_by == (LONG_CONTEXT_DROPPED,)

    retest = _checked(1100.0, None, request_id="r2", retest_of="r1", long_context=_long_context(50))
    settled = await rule.retest(first, retest, PARENT)

    assert (settled.verdict, decide(settled)) == (Verdict.BLOCKED, Action.DIAGNOSE)


@pytest.mark.parametrize(
    ("floor", "check", "refused"),
    [
        (1, short_prompt_check(0, 2, 0), True),
        (0, short_prompt_check(0, 1, 0), True),
        (2, short_prompt_check(0, 2, 0), False),
        (0, short_prompt_check(0, 0, 0), False),
        (None, short_prompt_check(0, 9, 0), False),
        (1, None, False),
        (1, short_prompt_check(0, 9, complete=False), False),
        (1, short_prompt_check(0, complete=False), False),
        (1, short_prompt_check(0, 9, 0, 0), False),
    ],
    ids=[
        "above",
        "above-zero",
        "at",
        "clean",
        "no-floor",
        "no-check",
        "incomplete",
        "nothing-compared",
        "more-processes-than-the-floor",
    ],
)
async def test_an_inferencex_candidate_is_refused_only_above_the_baseline_short_prompt_floor(floor, check, refused):
    judgement = await InferenceXRule(_Baseline(floor), 64).first(_checked(1000.0, check), PARENT)

    assert (SHORT_PROMPTS_INCONSISTENT in judgement.refused_by) is refused
    assert judgement.verdict is (Verdict.REGRESSED if refused else Verdict.FLAT)


@pytest.mark.parametrize(
    ("retest_check", "verdict", "action"),
    [
        (short_prompt_check(0, 3, 0), Verdict.BLOCKED, Action.DIAGNOSE),
        (short_prompt_check(0, 0, 0), Verdict.INCONCLUSIVE, Action.RECORD_ONLY),
        (None, Verdict.INCONCLUSIVE, Action.RECORD_ONLY),
    ],
    ids=["both-inconsistent", "retest-consistent", "retest-unchecked"],
)
async def test_a_gain_inconsistent_beyond_the_floor_on_both_measurements_is_blocked(retest_check, verdict, action):
    rule = InferenceXRule(_Baseline(1), 64)
    first = await rule.first(_checked(1100.0, short_prompt_check(0, 3, 0)), PARENT)
    assert first.verdict is Verdict.NEEDS_CONFIRM and first.refused_by == (SHORT_PROMPTS_INCONSISTENT,)

    settled = await rule.retest(first, _checked(1100.0, retest_check, request_id="r2", retest_of="r1"), PARENT)

    assert (settled.verdict, decide(settled)) == (verdict, action)
    assert settled.refused_by == (SHORT_PROMPTS_INCONSISTENT,)


async def test_an_agentx_candidate_is_refused_nothing_on_its_short_prompt_check_while_no_floor_exists():
    record = _measured({GRADED_ERROR_RATE: 0.0})
    record = replace(record, result=replace(record.result, short_prompt_consistency=short_prompt_check(0, 9, 0)))

    judgement = await AgentXRule().first(record, {GRADED_INTVTY_P50: 40.0, GRADED_ERROR_RATE: 0.0})

    assert SHORT_PROMPTS_INCONSISTENT not in judgement.refused_by
    assert record.result.short_prompt_consistency.inconsistency == 9, "the check stays on the test's result"


def test_a_retest_asked_before_its_first_has_a_result_runs_on_any_node():
    unmeasured = replace(_checked(1100.0, None), result=None)

    retest = retest_request(unmeasured, InferenceXRule(None, 64))

    assert (retest.retest_of, retest.avoid_nodes, retest.purpose) == ("r1", (), Purpose.CONFIRM)


async def test_the_retest_of_a_test_that_forced_the_short_prompt_check_forces_it_too():
    first = _checked(1100.0, short_prompt_check(0, 3, 0))
    forced = replace(first, request=replace(first.request, consistency_check=True))

    assert retest_request(forced, InferenceXRule(None, 64)).consistency_check is True
    assert retest_request(first, InferenceXRule(None, 64)).consistency_check is None
