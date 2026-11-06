# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math

import pytest

from hyperloom_multibranch.rulegate.accuracy import (
    BaselineAccuracyError,
    agentx_accuracy_ok,
    baseline_accuracy,
    inferencex_accuracy_ok,
)


def test_the_baseline_reference_is_the_mean_of_the_prelude_scores():
    assert baseline_accuracy([0.80, 0.82, 0.78]) == pytest.approx(0.80)


@pytest.mark.parametrize("scores", [[], [0.80, None, 0.78], [0.80, math.nan], [0.40, 0.45, 0.42]])
def test_prelude_fails_on_a_missing_or_low_baseline_score(scores):
    with pytest.raises(BaselineAccuracyError):
        baseline_accuracy(scores)


@pytest.mark.parametrize(
    ("baseline", "candidate", "ok"),
    [
        (0.80, 0.76, True),
        (0.80, 0.83, True),
        (0.80, 0.74, False),
        (0.52, 0.49, False),
        (0.80, None, False),
    ],
)
def test_an_inferencex_score_holds_within_the_drop_and_above_the_floor(baseline, candidate, ok):
    assert inferencex_accuracy_ok(baseline, candidate) is ok


@pytest.mark.parametrize(("rate", "ok"), [(0.7, True), (10.0, True), (25.0, False), (None, False), (math.nan, False)])
def test_an_agentx_run_holds_upstreams_error_rate_bar(rate, ok):
    assert agentx_accuracy_ok(rate) is ok
