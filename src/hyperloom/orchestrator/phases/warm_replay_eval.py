# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy score a PRELUDE warm replay is judged on: the one its result carries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.common.gain_math import finite_score

from ..actions.executors._accuracy_gate import (
    ACCURACY_DEFINITION_DIGEST_KEY,
    EVAL_DEFINITION_DIGEST_KEY,
    AccuracyReference,
)


@dataclass(frozen=True)
class ReplayEval:
    """A replay's accuracy as its attempt handed it over.

    ``ran`` says the result carries a score, so it separates the two ways ``accuracy`` can be absent: a score the
    baseline's eval definition cannot judge, and no score at all.
    """

    accuracy: float | None
    ran: bool
    error: str


def read_replay_eval(result: Mapping[str, Any], baseline: AccuracyReference) -> ReplayEval:
    """The score ``result`` carries, which is that of the round of the replay's attempt that ran the eval.

    Against a baseline score, only a score taken under the baseline's eval definition counts.
    """
    if not isinstance(result.get("accuracy"), (int, float)):
        return ReplayEval(None, False, "the replay's result carries no accuracy score")
    scored = {
        "accuracy": result["accuracy"],
        EVAL_DEFINITION_DIGEST_KEY: result.get(ACCURACY_DEFINITION_DIGEST_KEY),
    }
    return _graded(scored, baseline)


def _graded(eval_results: Mapping[str, Any], baseline: AccuracyReference) -> ReplayEval:
    """An eval that produced a number: its score, unless the baseline's score was taken under another definition."""
    score = baseline.comparable(eval_results) if baseline.score > 0 else finite_score(eval_results.get("accuracy"))
    if score is None:
        return ReplayEval(None, True, "no finite score under the baseline's eval definition")
    return ReplayEval(score, True, "")
