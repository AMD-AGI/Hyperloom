# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.rulegate.decision import Action, decide


@pytest.mark.parametrize(
    ("verdict", "accuracy_ok", "fixes_correctness", "action"),
    [
        (Verdict.NEEDS_CONFIRM, False, False, Action.RETEST),
        (Verdict.IMPROVED, True, False, Action.COMMIT),
        (Verdict.BLOCKED, False, False, Action.DIAGNOSE),
        (Verdict.BLOCKED, True, True, Action.DIAGNOSE),
        (Verdict.INCONCLUSIVE, False, True, Action.RECORD_ONLY),
        (Verdict.FLAT, True, True, Action.COMMIT),
        (Verdict.FLAT, False, True, Action.RECORD_ONLY),
        (Verdict.FLAT, True, False, Action.RECORD_ONLY),
        (Verdict.REGRESSED, True, True, Action.REQUEST_FORK),
        (Verdict.REGRESSED, False, True, Action.RECORD_ONLY),
        (Verdict.REGRESSED, True, False, Action.RECORD_ONLY),
    ],
)
def test_the_commit_table(verdict, accuracy_ok, fixes_correctness, action):
    judgement = Judgement(
        verdict,
        gain_pct=0.0,
        threshold_pct=2.0,
        request_ids=("r1",),
        recorded_value={"output_throughput": 1000.0},
        accuracy_ok=accuracy_ok,
        fixes_correctness=fixes_correctness,
    )
    assert decide(judgement) is action
