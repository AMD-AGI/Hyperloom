# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy values of the round that evaluated, from its inference_suite run_dir's ``report.json``.

Only the run_dir of the round that evaluated is read: the evaluation round a graded test runs after its measured bench
on a held server, which is its one evaluation, or else the measured round.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.gain_math import ACCURACY_METRICS
from hyperloom.contracts.measurement import AccuracyValue
from hyperloom.orchestrator.actions.executors import RoundFolder, is_round_folder
from inference_suite.contracts import EvalResult

__all__ = ["RESULT_METRICS", "read_accuracy"]

#: The lm-eval metric keys that are accuracy scores, in the order they are reported; the same tuple the rule gate
#: picks a score from, so the toolbox never reports a metric it would not read.
RESULT_METRICS = ACCURACY_METRICS
_HARNESS = "lm_eval"


def _report_values(result: EvalResult) -> tuple[AccuracyValue, ...]:
    version = f"{_HARNESS}={result.definition.lm_eval_version}"
    if result.task_version:
        version += f";{result.task}={result.task_version}"
    return tuple(
        AccuracyValue(result.task, key, result.scores[key], version) for key in RESULT_METRICS if key in result.scores
    )


def read_accuracy(directory: Path) -> tuple[AccuracyValue, ...]:
    """The scores the run_dir's report holds; none for a directory that is no run_dir, a round whose report is
    unreadable, or one that did not evaluate."""
    if not is_round_folder(Path(directory)):
        return ()
    try:
        report = RoundFolder.read(Path(directory)).report
    except (OSError, ValueError):
        return ()
    return () if report.eval is None else _report_values(report.eval)
