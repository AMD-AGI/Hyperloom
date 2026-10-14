# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner lists the rounds of a test next to its files, so a reader can tell a warm-up round from a measuring one."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.contracts.measurement import AccuracyValue, SandboxFacts
from hyperloom.measure.runner.rounds import ROUNDS_FILE, WARMUP_DIR, rounds_document, write_rounds
from hyperloom.measure.socket import RoundOutcome, TestOutcome


def _outcome(*rounds: RoundOutcome) -> TestOutcome:
    return TestOutcome(0, "[]", "", rounds, SandboxFacts(sandbox_id="x"), request_id="r")


def test_each_round_is_listed_with_its_phase_and_a_directory_relative_to_the_files(tmp_path: Path) -> None:
    files = tmp_path / "files"
    variant = files / "variant_00_test"
    outcome = _outcome(
        RoundOutcome(0, "warmup", variant / WARMUP_DIR, 0),
        RoundOutcome(0, "measure", variant / "benchmark_sglang_1", 0),
        RoundOutcome(1, "accuracy", variant / "benchmark_sglang_1", 3),
    )
    assert rounds_document(outcome, files) == [
        {"point": 0, "phase": "warmup", "dir": f"variant_00_test/{WARMUP_DIR}", "returncode": 0},
        {"point": 0, "phase": "measure", "dir": "variant_00_test/benchmark_sglang_1", "returncode": 0},
        {"point": 1, "phase": "accuracy", "dir": "variant_00_test/benchmark_sglang_1", "returncode": 3, "accuracy": []},
    ]


def test_a_round_outside_the_files_is_not_listed_and_a_missing_outcome_lists_nothing(tmp_path: Path) -> None:
    files = tmp_path / "files"
    elsewhere = _outcome(RoundOutcome(0, "measure", tmp_path / "elsewhere", 0))
    assert rounds_document(elsewhere, files) == []
    assert rounds_document(None, files) == []


def test_the_file_is_written_into_the_output_files(tmp_path: Path) -> None:
    files = tmp_path / "files"
    write_rounds(files, _outcome(RoundOutcome(0, "measure", files / "a", 0)))
    assert json.loads((files / ROUNDS_FILE).read_text()) == [
        {"point": 0, "phase": "measure", "dir": "a", "returncode": 0}
    ]


def test_an_accuracy_round_lists_the_version_of_a_score_that_has_one(tmp_path: Path) -> None:
    files = tmp_path / "files"
    value = AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64, "lm_eval=0.4.13;gsm8k=3")
    document = rounds_document(_outcome(RoundOutcome(0, "accuracy", files / "a", 0, (value,))), files)
    assert document[0]["accuracy"] == [{**value.to_document()}]
    assert document[0]["accuracy"][0]["version"] == "lm_eval=0.4.13;gsm8k=3"


def test_an_accuracy_round_lists_its_scores_and_no_other_round_does(tmp_path: Path) -> None:
    files = tmp_path / "files"
    value = AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64)
    outcome = _outcome(
        RoundOutcome(0, "measure", files / "a", 0, (value,)),  # a score on a measuring round is not listed
        RoundOutcome(0, "accuracy", files / "a", 0, (value,)),
        RoundOutcome(1, "accuracy", files / "b", 0),
    )
    measure, scored, empty = rounds_document(outcome, files)
    assert "accuracy" not in measure
    assert scored["accuracy"] == [{"task": "gsm8k", "metric": "exact_match,flexible-extract", "score": 0.64}]
    assert empty["accuracy"] == [], "a round that scored nothing says so: the reader must not guess"
