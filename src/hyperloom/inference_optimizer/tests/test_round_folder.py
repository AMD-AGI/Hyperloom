# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading an inference_suite run_dir back: the measurement, the GPU artifact beside it, and an archived v1 report unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.common.round_measurement import round_measurement
from hyperloom.orchestrator.actions.executors import _grid_runner
from hyperloom.orchestrator.actions.executors._gpu_metrics import GPU_ARTIFACT_NAME
from hyperloom.orchestrator.actions.executors._round_folder import RoundFolder, is_round_folder
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement
from inference_suite.contracts import FailedStep, FixedMetrics, ReportError, RunLayout, RunStatus
from inference_suite.tests.test_compose import _launch
from inference_suite.tests.test_report_contract import _report

_FIXED = FixedMetrics.of({"output_throughput": 1234.5, "completed_requests": 640})


def _run_dir(tmp_path: Path, *, launch: bool = True, **report_overrides: object) -> Path:
    run_dir = tmp_path / "round" / "run"
    layout = RunLayout(run_dir)
    layout.client_dir.mkdir(parents=True)
    report = _report(**{"profile": None, "eval": None, "fixed": _FIXED, **report_overrides})
    layout.report.write_text(json.dumps(report.to_document()), encoding="utf-8")
    if launch:
        layout.launch.write_text(json.dumps(_launch().to_document()), encoding="utf-8")
    return run_dir


def test_a_run_dir_is_measured_through_the_mapper(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    RunLayout(run_dir).inferencex_result.write_text("{}", encoding="utf-8")
    folder = RoundFolder.read(run_dir)
    measurement = extract_benchmark_measurement(None, workspace=run_dir)
    assert measurement == {
        **round_measurement(folder.report, folder.launch),
        "raw_result_path": str(RunLayout(run_dir).inferencex_result),
        "gpu_power_avg_w": 600.5,
        "gpu_power_by_gpu_w": None,
    }
    assert measurement["valid_measurement"] is True


def test_a_run_dir_with_a_power_section_is_measured_from_it(tmp_path: Path) -> None:
    from inference_suite.contracts import PowerSummary
    from inference_suite.tests.test_report_contract import _POWER_JSON

    run_dir = _run_dir(tmp_path, power=PowerSummary.of("power.json", _POWER_JSON))
    measurement = extract_benchmark_measurement(None, workspace=run_dir)
    assert (measurement["gpu_power_avg_w"], measurement["gpu_energy_j"], measurement["power_valid"]) == (
        650.0,
        130_000.0,
        1,
    )
    assert measurement["gpu_power_by_gpu_w"] == {"0": 700.0, "1": 600.0}
    assert measurement["joules_per_output_token"] == 1.3 and measurement["joules_per_input_token"] is None


def test_a_round_refused_before_its_launch_reads_without_one(tmp_path: Path) -> None:
    run_dir = _run_dir(
        tmp_path, launch=False, run_status=RunStatus.FAILED, failed_step=FailedStep.CONFIG_INVALID, fixed=None
    )
    assert RoundFolder.read(run_dir).launch is None
    measurement = extract_benchmark_measurement(None, workspace=run_dir)
    assert (measurement["failed_step"], measurement["framework"]) == ("config_invalid", None)


def test_a_report_off_its_contract_raises(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    document = json.loads(RunLayout(run_dir).report.read_text(encoding="utf-8"))
    del document["fixed"]
    RunLayout(run_dir).report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReportError):
        extract_benchmark_measurement(None, workspace=run_dir)


def test_gpu_metrics_are_written_from_the_report_into_the_round_dir_not_the_run_dir(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    path = RoundFolder.read(run_dir).write_gpu_metrics(run_dir.parent)
    assert path == str(run_dir.parent / GPU_ARTIFACT_NAME)
    assert not (run_dir / GPU_ARTIFACT_NAME).exists()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["source"] == "report.json"
    assert (payload["samples"], payload["avg_power_w"]) == (200, 600.5)
    assert (payload["avg_temp_c"], payload["avg_gpu_util_pct"], payload["avg_vram_pct"]) == (70.0, 80.0, 50.0)


@pytest.mark.asyncio
async def test_the_grid_reads_a_finished_run_dir_once_and_writes_its_gpu_artifact(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, fixed=FixedMetrics.of({}))
    report, measurement = await _grid_runner._settled_measurement(
        run_dir, subprocess_started_unix=None, settle_seconds=60.0, poll_seconds=30.0
    )
    assert report is None
    assert measurement["valid_measurement"] is False
    assert (run_dir.parent / GPU_ARTIFACT_NAME).is_file()


def test_a_v1_report_maps_field_by_field_as_before(tmp_path: Path) -> None:
    report = {
        "success": True,
        "framework": "sglang",
        "model": "m",
        "throughput": {
            "request_throughput": 2.5,
            "output_throughput": 1200.0,
            "total_token_throughput": 2400.0,
            "completed_requests": 640,
            "duration_seconds": 256.0,
            "num_prompts": 640,
        },
        "latency": {
            "ttft": {"mean_ms": 80.0, "p99_ms": 200.0},
            "tpot": {"mean_ms": 12.0},
            "e2el": {"mean_ms": 3000.0, "p99_ms": 5000.0},
        },
        "gpu_monitor": [{"gpu": 0, "sample_count": 10, "power_watts": {"min": 500, "max": 700, "avg": 600}}],
    }
    assert not is_round_folder(tmp_path)
    assert extract_benchmark_measurement(report, workspace=tmp_path) == {
        "reported_success": True,
        "framework": "sglang",
        "model": "m",
        "workload_kind": None,
        "throughput_unit": None,
        "quality_gate": None,
        "latency_s": None,
        "request_throughput": 2.5,
        "output_throughput": 1200.0,
        "total_token_throughput": 2400.0,
        "completed_requests": 640,
        "duration_seconds": 256.0,
        "requested_requests": 640,
        "ttft_mean_ms": 80.0,
        "ttft_p99_ms": 200.0,
        "tpot_mean_ms": 12.0,
        "e2el_mean_ms": 3000.0,
        "e2el_p99_ms": 5000.0,
        "gpu_power_avg_w": 600.0,
        "gpu_power_by_gpu_w": None,
        "raw_result_path": None,
        "nonfatal_warnings": [],
        "valid_measurement": True,
        "ttft_e2el_source": "benchmark_report",
        "tpot_source": "benchmark_report",
    }
