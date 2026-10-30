# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom_multibranch.contracts.judgement import AccuracyVerdict, Judgement, Verdict
from hyperloom_multibranch.rulegate.decision import Action, decide

PASSED, FAILED, NOT_MEASURED = AccuracyVerdict.PASSED, AccuracyVerdict.FAILED, AccuracyVerdict.NOT_MEASURED


@pytest.mark.parametrize(
    ("verdict", "accuracy", "fixes_correctness", "action"),
    [
        (Verdict.NEEDS_CONFIRM, FAILED, False, Action.RETEST),
        (Verdict.IMPROVED, PASSED, False, Action.COMMIT),
        (Verdict.BLOCKED, FAILED, False, Action.DIAGNOSE),
        (Verdict.BLOCKED, PASSED, True, Action.DIAGNOSE),
        (Verdict.INCONCLUSIVE, FAILED, True, Action.RECORD_ONLY),
        (Verdict.FLAT, PASSED, True, Action.COMMIT),
        (Verdict.FLAT, FAILED, True, Action.RECORD_ONLY),
        (Verdict.FLAT, PASSED, False, Action.RECORD_ONLY),
        (Verdict.REGRESSED, PASSED, True, Action.REQUEST_FORK),
        (Verdict.REGRESSED, FAILED, True, Action.RECORD_ONLY),
        (Verdict.REGRESSED, PASSED, False, Action.RECORD_ONLY),
        (Verdict.FLAT, NOT_MEASURED, True, Action.RECORD_ONLY),
        (Verdict.REGRESSED, NOT_MEASURED, True, Action.RECORD_ONLY),
    ],
)
def test_the_commit_table(verdict, accuracy, fixes_correctness, action):
    judgement = Judgement(
        verdict,
        gain_pct=0.0,
        threshold_pct=2.0,
        request_ids=("r1",),
        recorded_value={"output_throughput": 1000.0},
        accuracy=accuracy,
        fixes_correctness=fixes_correctness,
    )
    assert decide(judgement) is action
