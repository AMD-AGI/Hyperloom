# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``report.json``: fixed key tables, the closed ``FailedStep`` list, and the ``gpu_monitor`` blocks."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from inference_suite.contracts import (
    FIXED_METRIC_KEYS,
    GPU_METRIC_KEYS,
    REPORT_KEYS,
    REPORT_SCHEMA,
    AgentxResult,
    EvalDefinition,
    EvalProbe,
    EvalResult,
    ExitCodes,
    FailedStep,
    FixedMetrics,
    GpuMonitorBlock,
    LogRange,
    MetricStats,
    MlperfResult,
    ProbeOutcome,
    ProfileResult,
    Report,
    ReportError,
    RunStatus,
    ScriptableResult,
    ServerFacts,
    WorkloadKind,
)
from inference_suite.contracts.report import (
    AGENTX_COUNT_KEYS,
    AGENTX_LATENCY_KEYS,
    AGENTX_THROUGHPUT_KEYS,
    POWER_SUMMARY_KEYS,
    PowerSummary,
)

_DIGEST = "sha256:" + "a" * 64
_SHA = "b" * 64


def _block(gpu: int, samples: int = 100) -> GpuMonitorBlock:
    def stats(base: float) -> MetricStats:
        return MetricStats(min=base - 10, max=base + 10, avg=base)

    return GpuMonitorBlock(
        gpu=gpu,
        sample_count=samples,
        power_watts=stats(600.0 + gpu),
        temperature_c=stats(70.0),
        gpu_clock_mhz=stats(2100.0),
        mem_clock_mhz=stats(1300.0),
        gpu_util_pct=stats(80.0),
        vram_pct=stats(50.0),
    )


def _eval() -> EvalResult:
    definition = EvalDefinition(
        task_yaml_sha256=_SHA,
        lm_eval_version="0.4.9",
        model_class_sha256=_SHA,
        chat_template_sha256=None,
        concurrency=16,
        max_length=32768,
        max_tokens_cap=4096,
        limit=None,
        num_fewshot=5,
        golden_applied=True,
        probe=EvalProbe(enabled=True, min_samples=20, length_ratio=0.9),
    )
    return EvalResult(
        task="gsm8k",
        scores={"exact_match,strict-match": 0.94},
        task_version="3",
        definition=definition,
        definition_digest=_DIGEST,
        results_file="eval/results_2026.json",
        probe=ProbeOutcome(tripped=False, samples_seen=1319, length_ratio_observed=0.12),
    )


def _report(**overrides: object) -> Report:
    fields: dict[str, object] = dict(
        run_status=RunStatus.COMPLETED,
        workload_kind=WorkloadKind.FIXED,
        started_unix=1000.0,
        ended_unix=1600.0,
        codes=ExitCodes(server=None, client=0, eval=0),
        base_digest=_DIGEST,
        launch_digest=_DIGEST,
        workload_digest=_DIGEST,
        concurrency=64,
        server=ServerFacts("127.0.0.1", 30000, "glm", reused=False, log_range=LogRange(0, 4096)),
        gpu_monitor=(_block(0), _block(1)),
        fixed=FixedMetrics.of({"output_throughput": 1234.5, "completed_requests": 640}),
        eval=_eval(),
        profile=ProfileResult(trace_dirs=("profile/torch_trace",), has_step_traces=False),
    )
    fields.update(overrides)
    return Report(**fields)


def _document(report: Report) -> dict[str, object]:
    return json.loads(json.dumps(report.to_document(), allow_nan=False))


def test_round_trip_through_json() -> None:
    report = _report()
    document = _document(report)
    assert document["schema"] == REPORT_SCHEMA
    assert tuple(document) == REPORT_KEYS
    assert Report.from_document(document) == report


def test_fixed_metrics_hold_the_fixed_key_table_with_null_for_unmeasured() -> None:
    document = _document(_report())["fixed"]
    assert tuple(document) == FIXED_METRIC_KEYS
    assert document["output_throughput"] == 1234.5
    assert document["ttft_p99_ms"] is None
    assert document["failed_requests"] is None


def test_fixed_metrics_refuse_an_unknown_or_missing_key() -> None:
    with pytest.raises(ReportError, match="unknown fixed metric"):
        FixedMetrics.of({"output_tput": 1.0})
    document = _document(_report())
    del document["fixed"]["ttft_p50_ms"]
    with pytest.raises(ReportError, match="missing ttft_p50_ms"):
        Report.from_document(document)


def test_fixed_counts_are_integers() -> None:
    with pytest.raises(ReportError, match="completed_requests"):
        FixedMetrics.of({"completed_requests": 1.5})


@pytest.mark.parametrize("key", ["gpu_monitor", "eval", "server"])
def test_every_top_level_key_is_written_and_required(key: str) -> None:
    document = _document(_report())
    del document[key]
    with pytest.raises(ReportError, match=f"missing {key}"):
        Report.from_document(document)


def test_unknown_top_level_key_is_refused() -> None:
    document = _document(_report())
    document["dropped"] = []
    with pytest.raises(ReportError, match="unknown key 'dropped'"):
        Report.from_document(document)


def test_failed_step_is_a_closed_list() -> None:
    assert {step.value for step in FailedStep} == {
        "config_invalid",
        "base_digest_mismatch",
        "framework_version_too_old",
        "framework_version_unknown",
        "reuse_not_owned",
        "eval_context_too_short",
        "staging_missing",
        "source_files_changed",
        "flag_unknown",
        "golden_owned_touched",
        "port_in_use",
        "gpus_unavailable",
        "server_exited_before_ready",
        "server_silent_before_ready",
        "server_boot_timeout",
        "server_stalled",
        "server_died",
        "client_failed",
        "client_output_missing",
        "eval_failed",
        "round_timeout",
        "deadline_exhausted",
        "cancelled",
        "teardown_incomplete",
    }
    document = _document(_report(run_status=RunStatus.FAILED, failed_step=FailedStep.SERVER_DIED, fixed=None))
    document["failed_step"] = "profile_no_trace"
    with pytest.raises(ReportError, match="the failed step"):
        Report.from_document(document)


def test_status_and_failed_step_agree() -> None:
    with pytest.raises(ReportError, match="completed round names"):
        _report(failed_step=FailedStep.EVAL_FAILED)
    with pytest.raises(ReportError, match="failed round names None"):
        _report(run_status=RunStatus.FAILED, fixed=None)
    with pytest.raises(ReportError, match="failure detail needs"):
        _report(failure_detail="boom")
    failed = _report(
        run_status=RunStatus.FAILED,
        failed_step="config_invalid",
        failure_detail="unknown key",
        fixed=None,
        eval=None,
        server=None,
        launch_digest=None,
        gpu_monitor=(),
    )
    assert failed.failed_step is FailedStep.CONFIG_INVALID
    assert Report.from_document(_document(failed)) == failed


def test_workload_sections_follow_the_workload_kind() -> None:
    with pytest.raises(ReportError, match="carries the scriptable section"):
        _report(scriptable=ScriptableResult("client/result.json", {"images": 4}))
    with pytest.raises(ReportError, match="completed fixed round carries its fixed section"):
        _report(fixed=None)
    partial = _report(run_status=RunStatus.PARTIAL, failed_step=FailedStep.EVAL_FAILED, eval=None)
    assert Report.from_document(_document(partial)) == partial


def test_mlperf_section_round_trips_and_belongs_to_an_mlperf_round() -> None:
    mlperf = MlperfResult(
        summary_file="client/mlperf_artifacts/run/result_summary.json",
        scores_file=None,
        result_file="client/inferencex_result.json",
        result={"output_throughput": 812.5, "submission_valid": True},
    )
    report = _report(workload_kind=WorkloadKind.MLPERF, fixed=None, mlperf=mlperf)

    assert Report.from_document(_document(report)) == report
    with pytest.raises(ReportError, match="carries the mlperf section"):
        _report(mlperf=mlperf)
    with pytest.raises(ReportError, match="completed mlperf round carries its mlperf section"):
        _report(workload_kind=WorkloadKind.MLPERF, fixed=None)


def test_agentx_section_round_trips_with_its_error_rate_definition() -> None:
    agentx = AgentxResult(
        counts=dict.fromkeys(AGENTX_COUNT_KEYS, 1),
        throughput=dict.fromkeys(AGENTX_THROUGHPUT_KEYS),
        latency_seconds=dict.fromkeys(AGENTX_LATENCY_KEYS, 0.5),
        e2e_norm_intvty={"p50": 30.0, "p90": None},
        stopped_by_threshold=False,
        fast_mode=False,
        request_export="client/aiperf_artifacts/profile_export.jsonl",
    )
    report = _report(workload_kind=WorkloadKind.AGENTX, fixed=None, agentx=agentx)
    document = _document(report)
    assert document["agentx"]["error_rate_definition"] == "inferencex-agentx/1"
    assert Report.from_document(document) == report
    document["agentx"]["error_rate_definition"] = "elsewhere/1"
    with pytest.raises(ReportError, match="error rate definition"):
        Report.from_document(document)


_POWER_JSON = {
    "scope": {"serving_gpus": [0, 1], "energy_gpus": [0, 1], "unread_gpus": []},
    "energy": {
        "window": {"duration_s": 100.0},
        "per_gpu": {"0": {"integral_j": 70_000.0}, "1": {"integral_j": 60_000.0}},
        "total_gpu_energy_j": 130_000.0,
        "avg_power_w": 650.0,
        "joules_per_output_token": 1.3,
        "accumulator_check": {"relative_error": 0.012},
        "power_valid": 1,
        "reasons": ["invalid_input_token_count"],
        "warnings": [],
    },
    "phases": {"measured": {"per_gpu": {"0": {"throttle": {"ppt": 0.25}}, "1": {"throttle": {"ppt": 0.5}}}}},
}


def test_the_power_section_summarizes_power_json_and_round_trips() -> None:
    power = PowerSummary.of("power.json", _POWER_JSON)
    assert power.per_gpu_avg_power_w == {"0": 700.0, "1": 600.0}
    assert power.power_throttle_fraction_max == 0.5 and power.accumulator_relative_error == 0.012
    assert power.metrics["total_gpu_energy_j"] == 130_000.0 and power.metrics["joules_per_input_token"] is None
    report = _report(power=power)
    document = _document(report)
    assert tuple(document["power"]) == POWER_SUMMARY_KEYS
    assert Report.from_document(document) == report


def test_an_invalid_energy_has_no_per_card_power() -> None:
    invalid = {**_POWER_JSON, "energy": {**_POWER_JSON["energy"], "power_valid": 0}}
    assert PowerSummary.of("power.json", invalid).per_gpu_avg_power_w is None


def test_power_valid_is_zero_or_one() -> None:
    document = _document(_report(power=PowerSummary.of("power.json", _POWER_JSON)))
    document["power"]["power_valid"] = True
    with pytest.raises(ReportError, match="power_valid"):
        Report.from_document(document)


def test_a_report_written_before_the_power_section_and_token_counts_reads_them_as_null() -> None:
    agentx = AgentxResult(
        counts={**dict.fromkeys(AGENTX_COUNT_KEYS, 1), "input_tokens": None, "output_tokens": None},
        throughput=dict.fromkeys(AGENTX_THROUGHPUT_KEYS),
        latency_seconds=dict.fromkeys(AGENTX_LATENCY_KEYS, 0.5),
        e2e_norm_intvty={"p50": 30.0, "p90": None},
        stopped_by_threshold=False,
        fast_mode=False,
    )
    report = _report(workload_kind=WorkloadKind.AGENTX, fixed=None, agentx=agentx)
    document = _document(report)
    del document["power"]
    for key in ("input_tokens", "output_tokens"):
        del document["agentx"]["counts"][key]
    assert Report.from_document(document) == report


def test_gpu_monitor_block_holds_the_gpu_its_sample_count_and_the_metric_table() -> None:
    document = _document(_report())
    assert all(tuple(block) == ("gpu", "sample_count", *GPU_METRIC_KEYS) for block in document["gpu_monitor"])


def test_gpu_monitor_records_an_unread_metric_as_null() -> None:
    report = _report(gpu_monitor=(replace(_block(0), gpu_util_pct=None),))
    assert _document(report)["gpu_monitor"][0]["gpu_util_pct"] is None
    assert Report.from_document(_document(report)) == report


def test_metric_stats_accept_a_mean_rounded_outside_the_range() -> None:
    avg = sum([0.1] * 3) / 3
    assert avg > 0.1
    assert MetricStats.from_document(MetricStats(min=0.1, max=0.1, avg=avg).to_document()).avg == avg


def test_gpu_monitor_refuses_inconsistent_blocks() -> None:
    with pytest.raises(ReportError, match="from no samples"):
        _block(0, samples=0)
    with pytest.raises(ReportError, match="duplicate monitored GPU"):
        _report(gpu_monitor=(_block(0), _block(0)))


def test_eval_definition_carries_the_callers_probe_thresholds() -> None:
    document = _document(_report())
    assert document["eval"]["definition"]["probe"] == {"enabled": True, "min_samples": 20, "length_ratio": 0.9}
    broken = copy.deepcopy(document)
    del broken["eval"]["definition"]["probe"]["min_samples"]
    with pytest.raises(ReportError, match="probe"):
        Report.from_document(broken)


def test_round_times_and_digests_are_checked() -> None:
    with pytest.raises(ReportError, match="ends at"):
        _report(ended_unix=999.0)
    with pytest.raises(ReportError, match="launch digest"):
        _report(launch_digest="a" * 64)


@pytest.mark.parametrize(
    ("step", "load_measured", "status"),
    [
        (None, False, RunStatus.COMPLETED),
        (FailedStep.CANCELLED, True, RunStatus.STOPPED),
        (FailedStep.EVAL_FAILED, True, RunStatus.PARTIAL),
        (FailedStep.CLIENT_FAILED, False, RunStatus.FAILED),
    ],
)
def test_run_status_follows_the_failed_step_and_whether_the_load_measured(
    step: FailedStep | None, load_measured: bool, status: RunStatus
) -> None:
    assert RunStatus.of(step, load_measured) is status
