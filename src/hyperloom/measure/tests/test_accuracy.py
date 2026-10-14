# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy of a round is read from the measured round's own lm-eval output; the warm-up round's never counts."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.contracts.measurement import AccuracyValue
from hyperloom.measure.accuracy import read_accuracy

STRICT, FLEX = "exact_match,strict-match", "exact_match,flexible-extract"


def _results(directory: Path, name: str = "results_2026-10-14T10-00-00.json", **metrics: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps({"results": {"gsm8k": {"alias": "gsm8k", **metrics}}}), encoding="utf-8")
    return path


def test_every_recognised_metric_of_every_task_becomes_a_value(tmp_path: Path) -> None:
    _results(tmp_path / "eval_x" / "model", **{STRICT: 0.61, FLEX: 0.64, "exact_match_stderr,strict-match": 0.01})
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", STRICT, 0.61), AccuracyValue("gsm8k", FLEX, 0.64))


def test_the_warmup_rounds_evaluation_is_ignored_even_beside_a_measured_one(tmp_path: Path) -> None:
    _results(tmp_path / "warmup_round" / "bench" / "eval", **{FLEX: 0.10})
    _results(tmp_path / "bench" / "eval", **{FLEX: 0.64})
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", FLEX, 0.64),)


def test_a_round_whose_only_evaluation_is_the_warmups_has_no_accuracy(tmp_path: Path) -> None:
    _results(tmp_path / "warmup_round" / "bench", **{FLEX: 0.10})
    _results(tmp_path / "mn_warmup", **{FLEX: 0.10})
    assert read_accuracy(tmp_path) == ()


def test_the_newest_file_wins_and_an_unreadable_or_empty_one_falls_back_to_the_one_before(tmp_path: Path) -> None:
    _results(tmp_path, "results_2026-10-14T09-00-00.json", **{FLEX: 0.5})
    _results(tmp_path, "results_2026-10-14T10-00-00.json", **{FLEX: 0.7})
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", FLEX, 0.7),)
    (tmp_path / "results_2026-10-14T11-00-00.json").write_text("{not json")
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", FLEX, 0.7),)
    _results(tmp_path, "results_2026-10-14T12-00-00.json", **{"other,none": 1.0})
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", FLEX, 0.7),)


def test_a_score_that_is_not_a_finite_number_is_not_a_value(tmp_path: Path) -> None:
    _results(tmp_path, **{STRICT: "0.5", FLEX: float("nan")})
    assert read_accuracy(tmp_path) == ()


def test_a_missing_directory_has_no_accuracy(tmp_path: Path) -> None:
    assert read_accuracy(tmp_path / "nope") == ()
