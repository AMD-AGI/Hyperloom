# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measure readers of a round read an inference_suite run_dir through its ``report.json``.

The contract: accuracy, the accuracy pass's exit code and a landed bundle's rounds come from the run_dir's report, never
from the lm-eval file beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.measure import hosted, local_outcome
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.orchestrator.actions.executors import VariantResult
from inference_suite.contracts import ExitCodes, Report, RunLayout
from inference_suite.tests.test_report_contract import _report


def _run_dir(path: Path, report: Report) -> Path:
    path.mkdir(parents=True)
    RunLayout(path).report.write_text(json.dumps(report.to_document()), encoding="utf-8")
    return path


def _lm_eval_file(directory: Path, score: float) -> None:
    results = directory / "eval" / "model" / "results_2026.json"
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}), encoding="utf-8")


def _variant(workspace: Path, returncode: int) -> VariantResult:
    return VariantResult("test", "", {}, "succeeded", workspace=str(workspace), returncode=returncode)


def test_a_run_dirs_accuracy_is_its_reports_eval(tmp_path):
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None))
    _lm_eval_file(run_dir, 0.11)

    values = read_accuracy(run_dir)

    assert [(value.task, value.metric, value.score, value.version) for value in values] == [
        ("gsm8k", "exact_match,strict-match", 0.94, "lm_eval=0.4.9;gsm8k=3")
    ]


def test_a_run_dir_that_did_not_evaluate_has_no_accuracy_even_beside_an_lm_eval_file(tmp_path):
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None, eval=None, codes=ExitCodes(client=0)))
    _lm_eval_file(run_dir, 0.94)

    assert read_accuracy(run_dir) == ()


@pytest.mark.parametrize(("eval_code", "expected"), [(3, 3), (None, 7)])
def test_the_accuracy_pass_exit_code_of_a_run_dir_is_its_reports(tmp_path, eval_code, expected):
    codes = ExitCodes(client=0, eval=eval_code)
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None, codes=codes))

    assert local_outcome._eval_returncode(_variant(run_dir, 7)) == expected


def test_a_landed_bundle_without_a_round_list_finds_its_rounds_by_their_reports(tmp_path):
    listed = [
        "files/c8/variant_00_c8/suite_run_0/report.json",
        "files/c8/variant_00_c8/warmup_round/suite_run_0/report.json",
        "files/c4/variant_00_c4/benchmark_x/benchmark_report.json",
        "files/c4/variant_00_c4/suite_run_0/report.json",
        "files/c4/variant_00_c4/suite_run_0/eval/model/results_2026.json",
    ]
    files = [SimpleNamespace(path=path) for path in listed]
    verified = SimpleNamespace(result=SimpleNamespace(files=files, outcome=SimpleNamespace(returncode=0)))

    rounds = hosted._rounds_of(verified, tmp_path)

    assert [r.output_dir for r in rounds] == [
        tmp_path / "files/c4/variant_00_c4/suite_run_0",
        tmp_path / "files/c8/variant_00_c8/suite_run_0",
    ]
    assert {r.phase for r in rounds} == {"measure"}
