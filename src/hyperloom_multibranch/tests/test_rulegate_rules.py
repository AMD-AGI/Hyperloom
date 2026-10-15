# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from hyperloom.common.accuracy import AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.perf_metric import GRADED_ERROR_RATE, GRADED_INTVTY_P50
from hyperloom.contracts.measurement import RoundValue
from hyperloom_multibranch.contracts.notebook.records import TestRecord
from hyperloom_multibranch.notebook.tests.factories import intent, request, result
from hyperloom_multibranch.rulegate.rules import AgentXRule


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
