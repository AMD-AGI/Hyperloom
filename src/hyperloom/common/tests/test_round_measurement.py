# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``round_measurement``: each kind of round report maps to one measurement under the Magpie measurement's keys."""

from __future__ import annotations

import pytest

from hyperloom.common.round_measurement import LATENCY_SOURCE, round_measurement
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement
from inference_suite.contracts import (
    FIXED_METRIC_KEYS,
    AgentxResult,
    FailedStep,
    FixedMetrics,
    RunStatus,
    ScriptableResult,
    WorkloadKind,
)
from inference_suite.contracts.report import AGENTX_COUNT_KEYS, AGENTX_LATENCY_KEYS, AGENTX_THROUGHPUT_KEYS
from inference_suite.tests.test_compose import _launch
from inference_suite.tests.test_report_contract import _report

_FIXED = FixedMetrics.of({"output_throughput": 1234.5, "completed_requests": 640, "ttft_mean_ms": 80.0})


def _fixed(**overrides: object):
    return _report(**{"profile": None, "eval": None, "fixed": _FIXED, **overrides})


def test_a_completed_fixed_round_is_a_valid_measurement_with_every_fixed_key() -> None:
    launch = _launch()
    measurement = round_measurement(_fixed(), launch)
    assert measurement["valid_measurement"] is True
    assert measurement["round_succeeded"] is True
    assert measurement["diagnostic_only"] is False
    assert measurement["reported_success"] is True
    assert {key: measurement[key] for key in FIXED_METRIC_KEYS} == dict(_FIXED.values)
    assert measurement["framework"] == launch.framework.name.value
    assert measurement["model"] == launch.base.model.id
    assert (measurement["tp"], measurement["conc"]) == (launch.base.topology.tp, 64)
    assert measurement["ttft_e2el_source"] == measurement["tpot_source"] == LATENCY_SOURCE
    assert measurement["failed_step"] is None


def test_the_measurement_carries_every_key_a_magpie_measurement_carries() -> None:
    magpie = extract_benchmark_measurement({"success": True, "throughput": {"output_throughput": 1.0}})
    assert set(magpie) - {"gpu_power_avg_w", "gpu_power_by_gpu_w"} <= set(round_measurement(_fixed(), None))


@pytest.mark.parametrize(
    "metrics",
    [{"output_throughput": 0.0, "completed_requests": 640}, {"output_throughput": 10.0, "completed_requests": 0}, {}],
)
def test_a_round_that_served_nothing_is_invalid(metrics: dict[str, float]) -> None:
    measurement = round_measurement(_fixed(fixed=FixedMetrics.of(metrics)), None)
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is False


def test_a_profiled_round_is_diagnostic_only() -> None:
    measurement = round_measurement(_report(eval=None), None)
    assert measurement["diagnostic_only"] is True
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is True


@pytest.mark.parametrize("step", list(FailedStep))
def test_every_failed_step_is_carried_and_never_valid(step: FailedStep) -> None:
    report = _fixed(run_status=RunStatus.FAILED, failed_step=step, failure_detail="why", fixed=None)
    measurement = round_measurement(report, None)
    assert (measurement["failed_step"], measurement["failure_detail"]) == (step.value, "why")
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is False
    assert measurement["reported_success"] is False


def test_a_partial_round_keeps_the_load_it_measured() -> None:
    report = _fixed(run_status=RunStatus.PARTIAL, failed_step=FailedStep.EVAL_FAILED)
    measurement = round_measurement(report, None)
    assert measurement["valid_measurement"] is True
    assert measurement["reported_success"] is False
    assert measurement["failed_step"] == "eval_failed"


def test_an_agentx_round_maps_its_throughput_and_latency_in_milliseconds() -> None:
    agentx = AgentxResult(
        counts={**dict.fromkeys(AGENTX_COUNT_KEYS, 0), "records_total": 12, "successful": 10},
        throughput={**dict.fromkeys(AGENTX_THROUGHPUT_KEYS), "output_tokens_per_second": 900.0},
        latency_seconds={**dict.fromkeys(AGENTX_LATENCY_KEYS), "ttft_p90": 0.25},
        e2e_norm_intvty={"p50": 30.0, "p90": None},
        stopped_by_threshold=False,
        fast_mode=False,
    )
    report = _fixed(workload_kind=WorkloadKind.AGENTX, fixed=None, agentx=agentx)
    measurement = round_measurement(report, None)
    assert measurement["output_throughput"] == 900.0
    assert measurement["completed_requests"] == 10
    assert measurement["ttft_p90_ms"] == 250.0
    assert measurement["ttft_mean_ms"] is None
    assert measurement["e2e_norm_intvty_p50"] == 30.0
    assert measurement["agentx_counts"]["records_total"] == 12
    assert measurement["workload_identity"] == report.workload_digest
    assert measurement["valid_measurement"] is True


def test_a_scriptable_round_maps_its_script_result_and_carries_the_quality_scores_raw() -> None:
    quality = {"reference": "ref.png", "image": "out.png", "mse": 0.01, "ssim": 0.98, "lpips": None}
    result = {
        "output_throughput": 0.5,
        "completed": 4,
        "latency_s": 2.0,
        "throughput_unit": "img/s",
        "quality": quality,
    }
    report = _fixed(
        workload_kind=WorkloadKind.SCRIPTABLE, fixed=None, scriptable=ScriptableResult("client/r.json", result)
    )
    measurement = round_measurement(report, None)
    assert (measurement["output_throughput"], measurement["completed_requests"]) == (0.5, 4)
    assert (measurement["latency_s"], measurement["throughput_unit"]) == (2.0, "img/s")
    assert measurement["quality_gate"] == quality
    assert measurement["workload_kind"] == "scriptable"
    assert measurement["valid_measurement"] is True


def test_a_round_without_a_load_measures_nothing_and_succeeds_when_it_completes() -> None:
    measurement = round_measurement(_fixed(workload_kind=WorkloadKind.NONE, fixed=None), None)
    assert measurement["output_throughput"] is None
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is True
    failed = _fixed(workload_kind=WorkloadKind.NONE, fixed=None, run_status=RunStatus.FAILED, failed_step="eval_failed")
    assert round_measurement(failed, None)["round_succeeded"] is False
