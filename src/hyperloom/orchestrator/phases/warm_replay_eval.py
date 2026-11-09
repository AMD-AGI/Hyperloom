# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy score a PRELUDE warm replay took, read back from its result or the eval output its rounds left."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.gain_math import finite_score

from ..actions.executors import _accuracy_gate
from ..actions.executors._accuracy_gate import (
    ACCURACY_DEFINITION_DIGEST_KEY,
    EVAL_DEFINITION_DIGEST_KEY,
    AccuracyReference,
)
from ..actions.executors.benchmark_run.records import _MEASURE_ROUND_DIR, _WARMUP_ROUND_DIR

# The ``parse_eval_results`` reasons that prove an eval produced output: a results file it could not decode, and one
# carrying no metric it recognises.
_EVAL_RAN_BUT_UNSCORABLE = ("parse error:", "no recognized metric in")


@dataclass(frozen=True)
class ReplayEval:
    """A replay's accuracy as read back from its eval output.

    ``ran`` separates the two ways ``accuracy`` can be absent: an eval that ran but yielded no usable score, and no
    evidence an eval ran at all.
    """

    accuracy: float | None
    ran: bool
    error: str


def read_replay_eval(result: Mapping[str, Any], baseline: AccuracyReference) -> ReplayEval:
    """The replay's score, from ``result`` or else the eval output its round left on disk.

    Against a baseline score, only a score taken under the baseline's eval definition counts.
    """
    if isinstance(result.get("accuracy"), (int, float)):
        scored = {
            "accuracy": result["accuracy"],
            EVAL_DEFINITION_DIGEST_KEY: result.get(ACCURACY_DEFINITION_DIGEST_KEY),
        }
        return _graded(scored, baseline)
    root = _search_root(result)
    if root is None:
        return ReplayEval(None, False, "replay result names no round directory")
    try:
        eval_out = _accuracy_gate.parse_eval_results(root)
    except (OSError, AttributeError, TypeError, ValueError) as exc:  # an unreadable eval is "no verdict"
        eval_out = {"error": f"eval parse raised: {type(exc).__name__}"}
    if isinstance(eval_out.get("accuracy"), (int, float)):
        return _graded(eval_out, baseline)
    eval_error = str(eval_out.get("error") or "no accuracy in eval output")
    # Only a results file the parser reached can say the eval ran; a parser that never got one says nothing either way.
    return ReplayEval(None, eval_error.startswith(_EVAL_RAN_BUT_UNSCORABLE), eval_error)


def _graded(eval_results: Mapping[str, Any], baseline: AccuracyReference) -> ReplayEval:
    """An eval that produced a number: its score, unless the baseline's score was taken under another definition."""
    score = baseline.comparable(eval_results) if baseline.score > 0 else finite_score(eval_results.get("accuracy"))
    if score is None:
        return ReplayEval(None, True, "no finite score under the baseline's eval definition")
    return ReplayEval(score, True, "")


def _search_root(result: Mapping[str, Any]) -> Path | None:
    """Directory holding every round of this replay task."""
    round_dirs = {_WARMUP_ROUND_DIR, _MEASURE_ROUND_DIR}
    for key in ("output_dir", "workspace"):
        raw = str(result.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        for candidate in (path, *path.parents):
            if candidate.name in round_dirs:
                return candidate.parent
        return path
    return None
