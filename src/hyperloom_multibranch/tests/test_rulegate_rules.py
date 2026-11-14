# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from hyperloom.common.gain_math import ACCURACY_METRICS, AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    TOTAL_TOKEN_THROUGHPUT,
)
from hyperloom.contracts.long_context import (
    LongContextResult,
    LongContextRoot,
)
from hyperloom.contracts.measurement import AccuracyValue, AttachmentRef, Purpose, RoundValue
from hyperloom.contracts.tests.long_context_checks import long_context_check
from hyperloom_multibranch.contracts.judgement import ACCURACY_SCORE, AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.contracts.blackboard.records import BaselineRecord, TestRecord
from hyperloom_multibranch.blackboard.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.decision import Action, decide
from hyperloom_multibranch.rulegate.agentx import P50_BELOW_THRESHOLD, AgentXObjective
from hyperloom_multibranch.rulegate.inferencex import DROPPED_REQUESTS, GAIN_BELOW_THRESHOLD, LONG_CONTEXT_DROPPED
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import (
    RETRACTION_ID,
    AgentXRule,
    InferenceXRule,
    lands_as_retraction,
    retest_request,
    retracts,
)
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
    assert await AgentXRule(AgentXObjective.P50_INTERACTIVITY).accuracy_passes(_measured(metrics)) is passes


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
        await InferenceXRule(_Baseline(), 64).first(_unscored(purpose, accuracy), {GRADED_OUTPUT: 1000.0})


PARENT = {GRADED_OUTPUT: 1000.0}


class _Baseline:
    """The session's baseline record."""

    def __init__(self):
        self.record = BaselineRecord(
            "c0", {GRADED_OUTPUT: 1000.0, ACCURACY_SCORE: 0.90}, {}, {GRADED_OUTPUT: 2.0}, ("b0", "b1", "b2")
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
    judgement = await InferenceXRule(_Baseline(), 64).first(_checked(1000.0, None, long_context=check), PARENT)

    assert (LONG_CONTEXT_DROPPED in judgement.refused_by) is refused
    assert judgement.verdict is (Verdict.REGRESSED if refused else Verdict.FLAT)


async def test_a_gain_with_a_long_context_drop_on_both_measurements_is_blocked():
    rule = InferenceXRule(_Baseline(), 64)
    first = await rule.first(_checked(1100.0, None, long_context=_long_context(50)), PARENT)
    assert first.verdict is Verdict.NEEDS_CONFIRM and first.refused_by == (LONG_CONTEXT_DROPPED,)

    retest = _checked(1100.0, None, request_id="r2", retest_of="r1", long_context=_long_context(50))
    settled = await rule.retest(first, retest, PARENT)

    assert (settled.verdict, decide(settled)) == (Verdict.BLOCKED, Action.DIAGNOSE)


#: What the gate once refused a candidate for; the check is recorded on the test's result and refuses nothing.
SHORT_PROMPTS_INCONSISTENT = "short_prompts_inconsistent"


@pytest.mark.parametrize(
    "check",
    [short_prompt_check(0, 41, 0), short_prompt_check(0, 9, 0, 0), short_prompt_check(0, 0, 0), None],
    ids=["far-above-any-floor", "a-larger-pool", "clean", "no-check"],
)
async def test_an_inferencex_candidate_is_never_refused_for_its_short_prompt_check(check):
    judgement = await InferenceXRule(_Baseline(), 64).first(_checked(1000.0, check), PARENT)

    assert SHORT_PROMPTS_INCONSISTENT not in judgement.refused_by
    assert judgement.verdict is Verdict.FLAT


async def test_a_gain_whose_short_prompts_differ_between_processes_on_both_measurements_still_lands():
    """Run-to-run nondeterministic kernels (atomic split-K GEMMs) make two boots of one recipe answer 30 to 41 of 64
    prompts differently while every eval passes; a gain must not be held back for it."""
    rule = InferenceXRule(_Baseline(), 64)
    first = await rule.first(_checked(1100.0, short_prompt_check(0, 41, 0)), PARENT)
    assert first.verdict is Verdict.NEEDS_CONFIRM and first.refused_by == ()

    retest = _checked(1100.0, short_prompt_check(0, 37, 0), request_id="r2", retest_of="r1")
    settled = await rule.retest(first, retest, PARENT)

    assert (settled.verdict, settled.refused_by) == (Verdict.IMPROVED, ())
    assert retest.result.short_prompt_consistency.inconsistency == 37, "the check stays on the test's result"


async def test_an_agentx_candidate_is_refused_nothing_on_its_short_prompt_check():
    record = _measured({GRADED_ERROR_RATE: 0.0})
    record = replace(record, result=replace(record.result, short_prompt_consistency=short_prompt_check(0, 9, 0)))

    judgement = await AgentXRule(AgentXObjective.P50_INTERACTIVITY).first(
        record, {GRADED_INTVTY_P50: 40.0, GRADED_ERROR_RATE: 0.0}
    )

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


def _refused(verdict: Verdict, accuracy: AccuracyVerdict, *refused_by: str) -> Judgement:
    return Judgement(verdict, -4.0, 1.0, ("r1",), {GRADED_OUTPUT: 96.0}, accuracy, refused_by)


@pytest.mark.parametrize(
    ("rule", "judgement", "lands"),
    [
        (InferenceXRule(None, 64), _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, GAIN_BELOW_THRESHOLD), True),
        (InferenceXRule(None, 64), _refused(Verdict.FLAT, AccuracyVerdict.PASSED, GAIN_BELOW_THRESHOLD), True),
        (
            AgentXRule(AgentXObjective.P50_INTERACTIVITY),
            _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, "p90_regressed", P50_BELOW_THRESHOLD),
            True,
        ),
        (InferenceXRule(None, 64), _refused(Verdict.REGRESSED, AccuracyVerdict.FAILED, GAIN_BELOW_THRESHOLD), False),
        (InferenceXRule(None, 64), _refused(Verdict.REGRESSED, AccuracyVerdict.NOT_MEASURED), False),
        (InferenceXRule(None, 64), _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, LONG_CONTEXT_DROPPED), False),
        (InferenceXRule(None, 64), _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, DROPPED_REQUESTS), False),
        (
            AgentXRule(AgentXObjective.P50_INTERACTIVITY),
            _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, "extra_failed_requests"),
            False,
        ),
        (InferenceXRule(None, 64), _refused(Verdict.NEEDS_CONFIRM, AccuracyVerdict.PASSED), False),
    ],
    ids=[
        "slower",
        "flat",
        "agentx-slower",
        "accuracy-failed",
        "accuracy-unmeasured",
        "long-context-dropped",
        "dropped-requests",
        "agentx-failed-requests",
        "provisional",
    ],
)
def test_a_retraction_lands_when_only_the_speed_gates_refused_it_on_passed_accuracy(rule, judgement, lands):
    assert lands_as_retraction(_checked(96.0, None, long_context=_long_context(64)), judgement, rule) is lands


def test_an_agentx_retraction_whose_long_context_check_drops_does_not_land():
    slower = _refused(Verdict.REGRESSED, AccuracyVerdict.PASSED, P50_BELOW_THRESHOLD)

    assert not lands_as_retraction(
        _checked(96.0, None, long_context=_long_context(57)), slower, AgentXRule(AgentXObjective.P50_INTERACTIVITY)
    )


def _retraction_record(check, long_context, forced: bool = True) -> TestRecord:
    record = _checked(96.0, check, long_context=long_context)
    marked = (AttachmentRef(RETRACTION_ID, "file", "b" * 64),)
    asked = replace(record.request, attachments=marked, consistency_check=forced, long_context_check=forced)
    return replace(record, request=asked)


@pytest.mark.parametrize(
    ("record", "counts"),
    [
        (_retraction_record(short_prompt_check(), _long_context(64)), True),
        (_retraction_record(short_prompt_check(), _long_context(64), forced=False), False),
        (_retraction_record(short_prompt_check(complete=False), _long_context(64)), False),
        (_retraction_record(short_prompt_check(), _long_context(63, errored=1)), False),
        (_retraction_record(None, _long_context(64)), False),
        (_retraction_record(short_prompt_check(), None), False),
    ],
    ids=["both-checks-ran", "checks-not-forced", "short-incomplete", "long-incomplete", "no-short", "no-long"],
)
def test_a_test_counts_as_a_retraction_only_when_both_forced_checks_ran_to_the_end(record, counts):
    assert retracts(record, InferenceXRule(None, 64)) is counts


@pytest.mark.parametrize(
    ("record", "counts"),
    [
        (_retraction_record(short_prompt_check(complete=False), _long_context(63, errored=1)), True),
        (_retraction_record(short_prompt_check(complete=False), _long_context(63, errored=1), forced=False), False),
    ],
    ids=["checks-incomplete", "checks-not-forced"],
)
def test_an_agentx_retraction_counts_with_its_checks_incomplete_since_an_agentx_test_cannot_run_them(record, counts):
    assert retracts(record, AgentXRule(AgentXObjective.P50_INTERACTIVITY)) is counts


AGENTX_PARENT = {
    GRADED_INTVTY_P50: 100.0,
    GRADED_INTVTY: 80.0,
    GRADED_DURATION: 600.0,
    GRADED_ERROR_RATE: 0.2,
    TOTAL_TOKEN_THROUGHPUT: 5000.0,
}
#: A P50 gain past the bar with its P90 guard broken beyond any noise band: a gain held back for a retest.
GUARD_BROKEN = {GRADED_INTVTY_P50: 106.0, GRADED_INTVTY: 60.0}


def _agentx_test(purpose, long_context, *, changes=None, request_id="r1", retest_of=None) -> TestRecord:
    """An AgentX test whose P50 clears the bar over ``AGENTX_PARENT`` unless ``changes`` say otherwise."""
    asked = replace(request(request_id, intent("s1", "c0")), purpose=purpose, retest_of=retest_of)
    rounds = (RoundValue("measure", 8, {**AGENTX_PARENT, GRADED_INTVTY_P50: 104.0, **(changes or {})}),)
    return TestRecord(asked, replace(result(request_id), rounds=rounds, long_context=long_context))


@pytest.mark.parametrize("purpose", [Purpose.INTEGRATION, Purpose.FINAL])
@pytest.mark.parametrize(
    ("check", "accuracy"),
    [
        (_long_context(64), AccuracyVerdict.PASSED),
        (_long_context(58), AccuracyVerdict.PASSED),
        (_long_context(57), AccuracyVerdict.FAILED),
        (_long_context(63, errored=1), AccuracyVerdict.NOT_MEASURED),
        (_long_context(64, root=None), AccuracyVerdict.NOT_MEASURED),
        (_long_context(64, sha="b" * 64), AccuracyVerdict.NOT_MEASURED),
        (None, AccuracyVerdict.NOT_MEASURED),
    ],
    ids=["clean", "at-the-bar", "below-the-bar", "incomplete", "no-root-reading", "another-prompt-set", "no-check"],
)
async def test_an_agentx_merge_or_final_test_passes_accuracy_only_on_a_complete_long_context_check_without_a_drop(
    purpose, check, accuracy
):
    rule = AgentXRule(AgentXObjective.P50_INTERACTIVITY)
    record = _agentx_test(purpose, check)

    assert (await rule.first(record, AGENTX_PARENT)).accuracy is accuracy
    assert await rule.accuracy_passes(record) is (accuracy is AccuracyVerdict.PASSED)


@pytest.mark.parametrize(
    ("changes", "refused_by"),
    [(None, (LONG_CONTEXT_DROPPED,)), (GUARD_BROKEN, ("p90_regressed", LONG_CONTEXT_DROPPED))],
    ids=["gain", "gain-held-by-a-guard"],
)
async def test_an_agentx_merge_test_whose_long_context_check_drops_regresses_with_no_retest(changes, refused_by):
    record = _agentx_test(Purpose.INTEGRATION, _long_context(57), changes=changes)

    judgement = await AgentXRule(AgentXObjective.P50_INTERACTIVITY).first(record, AGENTX_PARENT)

    assert (judgement.verdict, judgement.refused_by) == (Verdict.REGRESSED, refused_by)


async def test_an_agentx_explore_is_held_to_its_error_rate_whatever_its_long_context_check_shows():
    record = _agentx_test(Purpose.EXPLORE, _long_context(57))

    judgement = await AgentXRule(AgentXObjective.P50_INTERACTIVITY).first(record, AGENTX_PARENT)

    assert (judgement.verdict, judgement.accuracy, judgement.refused_by) == (
        Verdict.IMPROVED,
        AccuracyVerdict.PASSED,
        (),
    )


@pytest.mark.parametrize(
    ("check", "accuracy"),
    [(_long_context(64), AccuracyVerdict.PASSED), (_long_context(63, errored=1), AccuracyVerdict.NOT_MEASURED)],
    ids=["checked", "check-incomplete"],
)
async def test_the_retest_of_an_agentx_merge_test_keeps_the_accuracy_its_long_context_check_gave(check, accuracy):
    rule = AgentXRule(AgentXObjective.P50_INTERACTIVITY)
    first = await rule.first(_agentx_test(Purpose.INTEGRATION, check, changes=GUARD_BROKEN), AGENTX_PARENT)
    assert (first.verdict, first.accuracy) == (Verdict.NEEDS_CONFIRM, accuracy)

    retest = _agentx_test(Purpose.CONFIRM, None, changes=GUARD_BROKEN, request_id="r2", retest_of="r1")
    settled = await rule.retest(first, retest, AGENTX_PARENT)

    assert (settled.verdict, settled.accuracy) == (Verdict.BLOCKED, accuracy)
