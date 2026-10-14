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


VERSION = "lm_eval=0.4.13;gsm8k=3"


def _lm_eval_015_document(**top: object) -> dict[str, object]:
    """Shaped like the top level of an lm-eval 0.4.13 ``results_*.json``: results, versions, lm_eval_version."""
    return {
        "results": {"gsm8k": {"alias": "gsm8k", FLEX: 0.64, STRICT: 0.61}},
        "group_subtasks": {"gsm8k": []},
        "configs": {"gsm8k": {"task": "gsm8k"}},
        "versions": {"gsm8k": 3.0},
        "lm_eval_version": "0.4.13",
        **top,
    }


def _write(directory: Path, document: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results_2026-10-14T10-00-00.json").write_text(json.dumps(document), encoding="utf-8")


def test_the_version_is_the_harness_and_the_task_version_as_lm_eval_wrote_them(tmp_path: Path) -> None:
    _write(tmp_path, _lm_eval_015_document())
    assert read_accuracy(tmp_path) == (
        AccuracyValue("gsm8k", STRICT, 0.61, VERSION),
        AccuracyValue("gsm8k", FLEX, 0.64, VERSION),
    )


def test_a_fractional_task_version_keeps_its_fraction(tmp_path: Path) -> None:
    _write(tmp_path, _lm_eval_015_document(versions={"gsm8k": 0.5}))
    assert {v.version for v in read_accuracy(tmp_path)} == {"lm_eval=0.4.13;gsm8k=0.5"}


def test_a_file_that_names_only_one_of_the_versions_reports_only_that_one(tmp_path: Path) -> None:
    _write(tmp_path, {k: v for k, v in _lm_eval_015_document().items() if k != "lm_eval_version"})
    assert {v.version for v in read_accuracy(tmp_path)} == {"gsm8k=3"}
    _write(tmp_path, {k: v for k, v in _lm_eval_015_document().items() if k != "versions"})
    assert {v.version for v in read_accuracy(tmp_path)} == {"lm_eval=0.4.13"}


def test_a_file_with_no_version_info_gives_an_empty_version(tmp_path: Path) -> None:
    _write(tmp_path, {"results": {"gsm8k": {FLEX: 0.64}}})
    assert read_accuracy(tmp_path) == (AccuracyValue("gsm8k", FLEX, 0.64, ""),)
    _write(tmp_path, _lm_eval_015_document(versions=["not", "a", "mapping"], lm_eval_version=True))
    assert {v.version for v in read_accuracy(tmp_path)} == {""}
