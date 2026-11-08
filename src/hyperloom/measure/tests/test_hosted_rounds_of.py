# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rounds a landed result lists agree with its signed outcome: a round that failed under an outcome that says
success is a result the runner did not write, so it is rejected, never read as a success."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.measure.bundle import RejectReason, ResultRejected
from hyperloom.measure.hosted import _rounds_of
from hyperloom.measure.runner.rounds import ROUNDS_FILE

_SCORE = {"task": "gsm8k", "metric": "exact_match,flexible-extract", "score": 0.95}


def _verified(outcome_rc: int, accuracy_rc: int) -> SimpleNamespace:
    document = [
        {"point": 0, "phase": "measure", "dir": "m", "returncode": 0},
        {"point": 0, "phase": "accuracy", "dir": "e", "returncode": accuracy_rc, "accuracy": [_SCORE]},
    ]
    result = SimpleNamespace(files=(SimpleNamespace(path=ROUNDS_FILE),), outcome=SimpleNamespace(returncode=outcome_rc))
    return SimpleNamespace(result=result, read_file=lambda _path: json.dumps(document).encode("utf-8"))


def test_a_failed_round_under_an_outcome_that_says_success_is_rejected() -> None:
    with pytest.raises(ResultRejected) as raised:
        _rounds_of(_verified(outcome_rc=0, accuracy_rc=1), Path("/out"))
    assert raised.value.reason is RejectReason.MANIFEST_INVALID


@pytest.mark.parametrize(("outcome_rc", "accuracy_rc"), [(0, 0), (1, 1)])
def test_rounds_that_agree_with_the_outcome_are_read(outcome_rc: int, accuracy_rc: int) -> None:
    rounds = _rounds_of(_verified(outcome_rc, accuracy_rc), Path("/out"))
    assert [(r.phase, r.returncode) for r in rounds] == [("measure", 0), ("accuracy", accuracy_rc)]
