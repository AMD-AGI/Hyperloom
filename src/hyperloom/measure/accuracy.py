# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy values of one measuring round, read from the ``results*.json`` that lm-eval wrote for it (design 4.4.3).

Only the directory of the round that was measured is read. ``run_grid`` also evaluates in the warm-up round it
discards; that round's files sit below a ``warmup_round`` directory and never count, even when they are the only
result there is: a score with nothing behind it from the measured round is no score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from hyperloom.common.accuracy import ACCURACY_METRICS
from hyperloom.contracts.measurement import AccuracyValue

__all__ = ["DISCARDED_DIRS", "RESULT_METRICS", "read_accuracy"]

#: Directories of rounds the old flow discards; nothing below them is a result.
DISCARDED_DIRS = frozenset({"warmup_round", "mn_warmup"})
#: The lm-eval metric keys that are accuracy scores, in the order they are reported; the same tuple the rule gate and
#: the old flow pick a score from, so the toolbox never reports a metric they would not read.
RESULT_METRICS = ACCURACY_METRICS
_RESULTS_GLOB = "results*.json"
_HARNESS = "lm_eval"


def _result_files(directory: Path) -> list[Path]:
    root = Path(directory)
    try:
        found = [p for p in root.rglob(_RESULTS_GLOB) if p.is_file()]
    except OSError:
        return []
    return sorted(p for p in found if DISCARDED_DIRS.isdisjoint(p.relative_to(root).parts))


def _version_text(value: object) -> str:
    """A version as lm-eval prints it: ``3.0`` is task version ``3``; anything unusable is empty."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _version(document: dict, task: str) -> str:
    """``lm_eval=<harness version>;<task>=<task version>`` from what the results file says; empty when it says none."""
    parts = []
    harness = _version_text(document.get("lm_eval_version"))
    if harness:
        parts.append(f"{_HARNESS}={harness}")
    versions = document.get("versions")
    task_version = _version_text(versions.get(task)) if isinstance(versions, dict) else ""
    if task_version:
        parts.append(f"{task}={task_version}")
    return ";".join(parts)


def _values(document: object) -> tuple[AccuracyValue, ...]:
    results = document.get("results") if isinstance(document, dict) else None
    if not isinstance(results, dict):
        return ()
    values = []
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key in RESULT_METRICS:
            score = metrics.get(key)
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                values.append(AccuracyValue(str(task), key, float(score), _version(document, str(task))))
    return tuple(values)


def read_accuracy(directory: Path) -> tuple[AccuracyValue, ...]:
    """Every accuracy score in the newest readable ``results*.json`` below ``directory``; empty when there is none.

    Files below a discarded warm-up directory are skipped. lm-eval names its files by time, so the last name is the
    newest; a file that cannot be read or holds no recognised metric falls back to the one before it.
    """
    for path in reversed(_result_files(directory)):
        try:
            values = _values(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if values:
            return values
    return ()
