# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the MLPerf load: the harness's argv and env, what it needs before the server starts, and the result
it leaves mapped into the run_dir, or the step its output fails at."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from inference_suite.contracts import FailedStep, MlperfResult, RunLayout
from inference_suite.load.mlperf import HARNESS_SCRIPT, harness_argv, harness_env, mlperf_result, refusal
from inference_suite.tests.test_config_contract import _MLPERF, _mlperf_config
from inference_suite.tests.test_mlperf_report import _FIXTURES


def test_the_harness_runs_the_perf_flow_with_every_knob_in_its_env(tmp_path):
    config = _mlperf_config()
    layout = RunLayout(tmp_path / "run")

    assert harness_argv(_MLPERF) == ("bash", str(HARNESS_SCRIPT), "smoke_test", "glm", "mi355x", "perf")
    assert harness_env(config, layout) == {
        "RESULTS_DIR": str(layout.mlperf_artifacts),
        "AGENTIC_DATASET_PATH": "/data/agentic_combined_v6.jsonl",
        "MLPERF_TOKENIZER_DIR": "/data/tokenizer",
        "AGENTIC_NUM_TRAJECTORIES": "150",
        "AGENTIC_CONCURRENCY": "16",
        "PORT": "30000",
    }


def test_a_harness_without_its_script_corpus_or_tokenizer_is_refused(tmp_path):
    endpoints, dataset, tokenizer = tmp_path / "endpoints", tmp_path / "corpus.jsonl", tmp_path / "tokenizer"
    workload = replace(_MLPERF, endpoints_dir=str(endpoints), dataset=str(dataset), tokenizer=str(tokenizer))

    refused = refusal(workload)
    assert refused is not None and refused.step is FailedStep.CONFIG_INVALID
    assert all(str(path) in refused.detail for path in (endpoints / HARNESS_SCRIPT, dataset, tokenizer))
    (endpoints / HARNESS_SCRIPT).parent.mkdir(parents=True)
    (endpoints / HARNESS_SCRIPT).write_text("")
    dataset.write_text("")
    tokenizer.mkdir()
    assert refusal(workload) is None


def _harness_output(layout: RunLayout, *names: str) -> None:
    run = layout.mlperf_artifacts / "run"
    run.mkdir(parents=True)
    for name in names:
        shutil.copy(_FIXTURES / name, run / name)


def test_the_result_maps_the_summary_and_scores_into_the_client_result(tmp_path):
    layout = RunLayout(tmp_path)
    _harness_output(layout, "result_summary.json", "scores.json")

    result = mlperf_result(layout, _MLPERF)

    assert isinstance(result, MlperfResult)
    assert result.summary_file == "client/mlperf_artifacts/run/result_summary.json"
    assert result.scores_file == "client/mlperf_artifacts/run/scores.json"
    assert result.result_file == "client/inferencex_result.json"
    written = json.loads(layout.inferencex_result.read_text())
    assert written == result.result
    assert written["corpus_loader"] == "agentic_combined_v6"
    assert written["accuracy_score"] == pytest.approx(0.7178)


def test_a_run_without_scores_maps_no_accuracy(tmp_path):
    layout = RunLayout(tmp_path)
    _harness_output(layout, "result_summary.json")

    result = mlperf_result(layout, _MLPERF)

    assert isinstance(result, MlperfResult) and result.scores_file is None
    assert "accuracy_score" not in result.result


@pytest.mark.parametrize("output", ["none", "two summaries", "not json", "off schema"])
def test_output_the_round_cannot_map_is_client_output_missing(tmp_path, output):
    layout = RunLayout(tmp_path)
    if output != "none":
        _harness_output(layout, "result_summary.json")
    summary = layout.mlperf_artifacts / "run" / "result_summary.json"
    if output == "two summaries":
        (layout.mlperf_artifacts / "again").mkdir()
        shutil.copy(summary, layout.mlperf_artifacts / "again" / "result_summary.json")
    elif output == "not json":
        summary.write_text("{")
    elif output == "off schema":
        summary.write_text("{}")

    result = mlperf_result(layout, _MLPERF)

    assert not isinstance(result, MlperfResult) and result.step is FailedStep.CLIENT_OUTPUT_MISSING
    assert not layout.inferencex_result.exists()
