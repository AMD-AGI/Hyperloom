# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The accuracy of a round is read from its run_dir's report; nothing else below the directory is a score."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.contracts.measurement import AccuracyValue
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.measure.tests.run_dirs import run_dir_report, write_run_dir
from inference_suite.contracts import RunLayout

STRICT, FLEX = "exact_match,strict-match", "exact_match,flexible-extract"
VERSION = "lm_eval=0.4.9;gsm8k=3"


def _lm_eval_file(directory: Path, score: float) -> None:
    results = directory / "eval" / "results_2000-01-01T10-00-00.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps({"results": {"gsm8k": {FLEX: score}}}), encoding="utf-8")


def test_every_recognised_metric_the_report_holds_becomes_a_value_with_its_version(tmp_path: Path) -> None:
    write_run_dir(tmp_path, scores={STRICT: 0.61, FLEX: 0.64, "exact_match_stderr,strict-match": 0.01})
    assert read_accuracy(tmp_path) == (
        AccuracyValue("gsm8k", STRICT, 0.61, VERSION),
        AccuracyValue("gsm8k", FLEX, 0.64, VERSION),
    )


def test_a_report_without_a_task_version_names_only_the_harness(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    RunLayout(directory).report.write_bytes(run_dir_report(scores={FLEX: 0.64}, task_version=None))
    assert {v.version for v in read_accuracy(directory)} == {"lm_eval=0.4.9"}


def test_a_directory_that_is_no_run_dir_has_no_accuracy_even_beside_an_lm_eval_file(tmp_path: Path) -> None:
    _lm_eval_file(tmp_path, 0.64)
    assert read_accuracy(tmp_path) == ()


def test_a_round_that_did_not_evaluate_has_no_accuracy_even_beside_an_lm_eval_file(tmp_path: Path) -> None:
    write_run_dir(tmp_path)
    _lm_eval_file(tmp_path, 0.64)
    assert read_accuracy(tmp_path) == ()


def test_an_unreadable_report_has_no_accuracy(tmp_path: Path) -> None:
    tmp_path.joinpath("report.json").write_text("{not json", encoding="utf-8")
    assert read_accuracy(tmp_path) == ()


def test_a_missing_directory_has_no_accuracy(tmp_path: Path) -> None:
    assert read_accuracy(tmp_path / "nope") == ()
