# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PRELUDE child's AgentX baseline run, read as a test's measured round is.

The data is the child's run of GLM-5.3 at TP4, CONC 10: its point config and report as they landed, and its per-request
export cut to two warmup and five profiling requests, one of them failed with no content.
"""

import shutil
from pathlib import Path

import pytest

from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
)
from hyperloom.contracts.measurement import Validity
from hyperloom.measure.socket import RoundOutcome
from hyperloom_multibranch.rulegate.readings import GRADED_REQUESTS_TOTAL, agentx_reading
from hyperloom_multibranch.toolbox.hosted_executor import _round_values_at, landed_agentx_baseline

CHILD_RUN = Path(__file__).parent / "data" / "agentx_child_baseline"
POINT = "baseline_config.with_envs.yaml"
#: The child's ``state.json`` ``baseline_perf`` for this run.
CHILD_PERF = {
    GRADED_INTVTY_P50: 132.1028,
    GRADED_INTVTY: 81.72424,
    GRADED_OUTPUT: 442.30405,
    GRADED_DURATION: 3601.52025,
    GRADED_ERROR_RATE: 0.06925207756232687,
}


def test_the_child_run_reads_as_the_baseline_it_measured():
    result = landed_agentx_baseline(CHILD_RUN, CHILD_RUN / POINT, "prelude-child")

    assert (result.validity, result.failure_reason) == (Validity.SUCCEEDED, None)
    reading = agentx_reading(result)
    assert {key: reading[key] for key in CHILD_PERF} == pytest.approx(CHILD_PERF)
    assert reading[GRADED_REQUESTS_TOTAL] == 5.0
    (classes,) = result.request_classes
    assert (classes.concurrency, classes.failed_total, classes.failed_no_content) == (10, 1, 1)
    assert classes.export == "benchmark_sglang_20261026_075016/aiperf_artifacts/profile_export.jsonl"


def test_the_child_run_reads_as_the_measured_round_of_a_test_does():
    result = landed_agentx_baseline(CHILD_RUN, CHILD_RUN / POINT, "prelude-child")

    rounds = _round_values_at((RoundOutcome(point=0, phase="measure", output_dir=CHILD_RUN, returncode=0),), (10,))

    assert result.rounds == rounds


def test_a_run_without_a_report_did_not_measure(tmp_path):
    shutil.copy(CHILD_RUN / POINT, tmp_path / POINT)

    result = landed_agentx_baseline(tmp_path, tmp_path / POINT, "prelude-child")

    assert result.validity is Validity.MEASUREMENT_FAILED
    assert "benchmark_report.json" in (result.failure_reason or "")


def test_a_run_served_with_the_prefix_cache_off_is_refused(tmp_path):
    run = Path(shutil.copytree(CHILD_RUN, tmp_path / "run"))
    server_log = "server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=4)\n"
    (run / "benchmark_sglang_20261026_075016" / "server.log").write_text(server_log)

    result = landed_agentx_baseline(run, run / POINT, "prelude-child")

    assert result.validity is Validity.CANDIDATE_FAILED
    assert "disable_radix_cache=True" in (result.failure_reason or "")
