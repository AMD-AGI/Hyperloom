# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measure readers of a round read an inference_suite run_dir through its ``report.json``.

The contract: accuracy, the accuracy pass's exit code, the consistency check's measurement and a landed bundle's
rounds come from the run_dir's report, never from the lm-eval file beside it. A landed bundle's rounds are found by
either report, so an archived v1 bundle keeps its rounds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.measure import consistency, hosted, local
from hyperloom.measure.accuracy import read_accuracy
from hyperloom.measure.socket import RoundOutcome, TestOutcome
from hyperloom.orchestrator.actions.executors import VariantResult
from inference_suite.contracts import ExitCodes, FixedMetrics, Report, RunLayout
from inference_suite.tests.test_report_contract import _report

_SANDBOX = SandboxFacts(sandbox_id="local-t", backend="local")
_FIXED = {"output_throughput": 1234.5, "completed_requests": 640, "requested_requests": 640, "request_error_rate": 0.0}


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

    (value,) = read_accuracy(run_dir)

    assert (value.task, value.metric, value.score) == ("gsm8k", "exact_match,strict-match", 0.94)
    assert value.version == "lm_eval=0.4.9;gsm8k=3"


def test_a_run_dir_that_did_not_evaluate_has_no_accuracy_even_beside_an_lm_eval_file(tmp_path):
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None, eval=None, codes=ExitCodes(client=0)))
    _lm_eval_file(run_dir, 0.94)

    assert read_accuracy(run_dir) == ()


@pytest.mark.parametrize(("eval_code", "expected"), [(3, 3), (None, 7)])
def test_the_accuracy_pass_exit_code_of_a_run_dir_is_its_reports(tmp_path, eval_code, expected):
    codes = ExitCodes(client=0, eval=eval_code)
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None, codes=codes))

    assert local._eval_returncode(_variant(run_dir, 7)) == expected


def test_the_consistency_check_measures_a_run_dir_from_its_report(tmp_path):
    report = _report(profile=None, eval=None, codes=ExitCodes(client=0), fixed=FixedMetrics.of(_FIXED))
    run_dir = _run_dir(tmp_path / "suite_run_0", report)
    outcome = TestOutcome(0, "", "", (RoundOutcome(0, "measure", run_dir, 0),), _SANDBOX, "t")

    side = consistency.read_side(outcome, tmp_path)

    assert side.problem is None, side
    assert side.throughput == pytest.approx(1234.5)
    assert side.completed == (640,) and side.error_rate == (0.0,)


def test_the_consistency_check_names_a_run_dir_whose_round_did_not_measure(tmp_path):
    served_nothing = FixedMetrics.of({"output_throughput": 0.0, "completed_requests": 0})
    run_dir = _run_dir(tmp_path / "suite_run_0", _report(profile=None, eval=None, fixed=served_nothing))
    outcome = TestOutcome(0, "", "", (RoundOutcome(0, "measure", run_dir, 0),), _SANDBOX, "t")

    assert consistency.read_side(outcome, tmp_path).problem == "invalid_measurement: suite_run_0"


def test_a_landed_bundle_without_a_round_list_finds_its_rounds_by_either_report(tmp_path):
    listed = [
        "files/c8/variant_00_c8/suite_run_0/report.json",
        "files/c8/variant_00_c8/warmup_round/suite_run_0/report.json",
        "files/c4/variant_00_c4/benchmark_x/benchmark_report.json",
        "files/c4/variant_00_c4/suite_run_0/eval/model/results_2026.json",
    ]
    files = [SimpleNamespace(path=path) for path in listed]
    verified = SimpleNamespace(result=SimpleNamespace(files=files, outcome=SimpleNamespace(returncode=0)))

    rounds = hosted._rounds_of(verified, tmp_path)

    assert [r.output_dir for r in rounds] == [
        tmp_path / "files/c4/variant_00_c4/benchmark_x",
        tmp_path / "files/c8/variant_00_c8/suite_run_0",
    ]
    assert {r.phase for r in rounds} == {"measure"}
