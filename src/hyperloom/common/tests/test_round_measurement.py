# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``round_measurement``: each kind of round report maps to one measurement under the Magpie measurement's keys."""

from __future__ import annotations

import pytest

from hyperloom.common.gain_math import AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.round_measurement import LATENCY_SOURCE, round_measurement
from hyperloom.orchestrator.actions.executors.benchmark_result import (
    extract_benchmark_measurement,
    is_valid_measurement,
)
from inference_suite.contracts import (
    FIXED_METRIC_KEYS,
    AgentxResult,
    FailedStep,
    FixedMetrics,
    LaunchDelta,
    MlperfResult,
    ProfileResult,
    RunStatus,
    ScriptableResult,
    WorkloadKind,
)
from inference_suite.contracts.report import AGENTX_LATENCY_KEYS, AGENTX_THROUGHPUT_KEYS
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


_AGENTX_COUNTS = {
    "records_total": 112,
    "successful": 90,
    "error_dropped": 10,
    "warmup_dropped": 14,
    "dropped_total": 22,
    "error_osl_exported": 10,
}


def _agentx_result(counts: dict[str, int] | None = None, **overrides: object) -> AgentxResult:
    fields: dict[str, object] = dict(
        counts=counts or _AGENTX_COUNTS,
        throughput={
            **dict.fromkeys(AGENTX_THROUGHPUT_KEYS),
            "output_tokens_per_second": 900.0,
            "total_tokens_per_second": 9000.0,
            "duration_seconds": 600.0,
        },
        latency_seconds={**dict.fromkeys(AGENTX_LATENCY_KEYS), "ttft_p90": 0.25},
        e2e_norm_intvty={"p50": 30.0, "p90": None},
        stopped_by_threshold=False,
        fast_mode=False,
    )
    fields.update(overrides)
    return AgentxResult(**fields)


def _agentx(result: AgentxResult | None = None, **overrides: object):
    fields = {"workload_kind": WorkloadKind.AGENTX, "fixed": None, "agentx": result or _agentx_result(), **overrides}
    return _fixed(**fields)


def test_an_agentx_round_maps_its_throughput_and_latency_in_milliseconds() -> None:
    report = _agentx()
    measurement = round_measurement(report, None)
    assert measurement["output_throughput"] == 900.0
    assert measurement["completed_requests"] == 90
    assert measurement["ttft_p90_ms"] == 250.0
    assert measurement["ttft_mean_ms"] is None
    assert measurement["e2e_norm_intvty_p50"] == 30.0
    assert measurement["agentx_counts"] == _AGENTX_COUNTS
    assert measurement["workload_identity"] == report.workload_digest
    assert measurement["valid_measurement"] is True
    assert measurement["round_succeeded"] is True


def test_the_agentx_error_rate_leaves_warmup_drops_out() -> None:
    # (dropped - warmup) / (successful + errors): (22 - 14) / (90 + 10).
    assert round_measurement(_agentx(), None)["request_error_rate"] == pytest.approx(8.0)


def _counts(**changes: int) -> dict[str, int]:
    return {**_AGENTX_COUNTS, **changes}


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_agentx_result(stopped_by_threshold=True), id="stopped-by-threshold"),
        pytest.param(_agentx_result(fast_mode=True), id="fast-mode"),
        pytest.param(
            _agentx_result(_counts(error_dropped=12, dropped_total=26, records_total=116)), id="error-rate-over-10pct"
        ),
        pytest.param(_agentx_result(_counts(records_total=113)), id="counts-do-not-add-up"),
        pytest.param(_agentx_result(_counts(warmup_dropped=23)), id="more-warmup-than-dropped"),
        pytest.param(
            _agentx_result(_counts(successful=0, error_dropped=0, dropped_total=14, records_total=14)),
            id="nothing-profiled",
        ),
        pytest.param(
            _agentx_result(throughput={**dict.fromkeys(AGENTX_THROUGHPUT_KEYS), "output_tokens_per_second": 900.0}),
            id="no-duration-or-total-throughput",
        ),
    ],
)
def test_an_invalid_agentx_round_is_not_a_measurement(result: AgentxResult) -> None:
    measurement = round_measurement(_agentx(result), None)
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is False
    assert measurement["diagnostic_only"] is False


def test_an_error_rate_at_the_threshold_is_still_valid() -> None:
    at_threshold = _agentx_result(_counts(error_dropped=10, successful=90, dropped_total=24, records_total=114))
    measurement = round_measurement(_agentx(at_threshold), None)
    assert measurement["request_error_rate"] == pytest.approx(AGENTX_ERROR_RATE_THRESHOLD_PCT)
    assert measurement["valid_measurement"] is True


def test_a_profiled_agentx_round_is_diagnostic_only_and_succeeds() -> None:
    profiled = _agentx(profile=ProfileResult(trace_dirs=("profile/torch_trace",), has_step_traces=False))
    measurement = round_measurement(profiled, None)
    assert measurement["diagnostic_only"] is True
    assert measurement["valid_measurement"] is False
    assert measurement["round_succeeded"] is True


def test_an_agentx_round_served_with_the_prefix_cache_off_stays_a_valid_measurement() -> None:
    # The baseline prefix-cache refusal only judges a valid measurement, so the counts rule must not hide this round.
    launch = _launch(candidate=LaunchDelta())
    assert "--disable-radix-cache" in launch.argv
    measurement = round_measurement(_agentx(), launch)
    assert measurement["valid_measurement"] is True
    assert measurement["framework"] == launch.framework.name.value


def test_a_scriptable_round_maps_its_script_result_and_carries_its_quality_gate() -> None:
    quality = {"reference": "ref.png", "image": "out.png", "mse": 0.01, "ssim": 0.98, "lpips": None}
    result = {
        "output_throughput": 0.5,
        "completed": 4,
        "latency_s": 2.0,
        "throughput_unit": "img/s",
        "quality_gate": quality,
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


_MLPERF_RESULT = {
    "completed": 172,
    "request_error_rate": 0.0,
    "submission_valid": True,
    "submission_invalid_reasons": [],
    "corpus_loader": "agentic_combined_v6",
    "duration": 666.45,
    "request_throughput": 0.258,
    "output_throughput": 78.98,
    "mean_ttft_ms": 9000.0,
    "median_ttft_ms": 3325.7,
    "p90_ttft_ms": 8220.4,
    "p99_ttft_ms": 42970.9,
    "median_tpot_ms": 76.7,
    "mean_e2el_ms": 43923.4,
    "p99_e2el_ms": 277600.9,
    "total_output_tokens": 52640,
    "accuracy_score": 0.7178,
    "accuracy_missing_turns": 0,
}


def _mlperf(**result: object):
    section = MlperfResult(
        "client/mlperf_artifacts/run/result_summary.json",
        None,
        "client/inferencex_result.json",
        {**_MLPERF_RESULT, **result},
    )
    return _report(workload_kind=WorkloadKind.MLPERF, fixed=None, profile=None, eval=None, mlperf=section)


def test_an_mlperf_round_maps_the_harness_result_under_the_measurement_keys() -> None:
    measurement = round_measurement(_mlperf(), None)

    assert measurement["valid_measurement"] is True and measurement["round_succeeded"] is True
    assert (measurement["output_throughput"], measurement["completed_requests"]) == (78.98, 172)
    assert (measurement["duration_seconds"], measurement["request_error_rate"]) == (666.45, 0.0)
    assert (measurement["ttft_p50_ms"], measurement["ttft_p99_ms"], measurement["tpot_p50_ms"]) == (
        3325.7,
        42970.9,
        76.7,
    )
    assert (measurement["e2el_mean_ms"], measurement["total_output_tokens"]) == (43923.4, 52640)
    assert measurement["tpot_mean_ms"] is None, "a figure the harness did not report stays null"


@pytest.mark.parametrize(
    ("result", "valid"),
    [
        ({}, True),
        ({"submission_valid": False, "submission_invalid_reasons": ["interrupted"]}, False),
        ({"completed": 0}, False),
        ({"output_throughput": 0.0}, False),
    ],
)
def test_the_harness_submission_verdict_reaches_the_measurement_gate(result: dict, valid: bool) -> None:
    measurement = round_measurement(_mlperf(**result), None)

    assert measurement["valid_measurement"] is valid
    assert is_valid_measurement(measurement) is valid
    assert measurement["submission_invalid_reasons"] == result.get("submission_invalid_reasons", [])
