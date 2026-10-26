# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from hyperloom.common.accuracy import AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.perf_metric import GRADED_ERROR_RATE, GRADED_INTVTY_P50, GRADED_OUTPUT
from hyperloom.contracts.measurement import Purpose, RoundValue
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.readings import UnreadableResult
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule


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
        await InferenceXRule(None, 64).first(_unscored(purpose, accuracy), {GRADED_OUTPUT: 1000.0})
