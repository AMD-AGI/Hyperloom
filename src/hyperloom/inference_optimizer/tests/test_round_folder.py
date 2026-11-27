# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading an inference_suite run_dir back: the measurement, its power, and the GPU artifact beside it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.common.round_measurement import round_measurement
from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors.grid import _round_report
from hyperloom.orchestrator.actions.executors.gpu._gpu_metrics import GPU_ARTIFACT_NAME
from hyperloom.orchestrator.actions.executors.gpu._gpu_power import POWER_FIELDS
from hyperloom.orchestrator.actions.executors._round_folder import RoundFolder
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
    measurement = extract_benchmark_measurement(run_dir)
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
    measurement = extract_benchmark_measurement(run_dir)
    assert (measurement["gpu_power_avg_w"], measurement["gpu_energy_j"], measurement["power_valid"]) == (
        650.0,
        130_000.0,
        1,
    )
    assert measurement["gpu_power_by_gpu_w"] == {"0": 700.0, "1": 600.0}
    assert measurement["joules_per_output_token"] == 1.3 and measurement["joules_per_input_token"] is None
    assert set(POWER_FIELDS) <= set(measurement) and type(measurement["power_valid"]) is int


def test_a_sampled_run_dir_whose_energy_is_invalid_stays_unmeasured(tmp_path: Path) -> None:
    """The ``gpu_monitor`` mean is not substituted for a recording that measured nothing trustworthy."""
    from inference_suite.contracts import PowerSummary
    from inference_suite.tests.test_report_contract import _POWER_JSON

    energy = dict(_POWER_JSON["energy"], power_valid=0, reasons=["sampling_gap_exceeded"])
    energy.update(dict.fromkeys(key for key in energy if key.endswith(("_w", "_j", "_token"))))
    run_dir = _run_dir(tmp_path, power=PowerSummary.of("power.json", {**_POWER_JSON, "energy": energy}))
    measurement = extract_benchmark_measurement(run_dir)
    assert measurement["power_valid"] == 0
    assert measurement["gpu_power_avg_w"] is None and measurement["gpu_power_by_gpu_w"] is None
    assert measurement["gpu_energy_j"] is None and measurement["joules_per_output_token"] is None


def test_a_run_dir_with_no_gpu_telemetry_has_no_power_figure(tmp_path: Path) -> None:
    measurement = extract_benchmark_measurement(_run_dir(tmp_path, gpu_monitor=()))
    assert measurement["gpu_power_avg_w"] is None and measurement["gpu_power_by_gpu_w"] is None
    assert "power_valid" not in measurement and "gpu_energy_j" not in measurement


def test_a_round_refused_before_its_launch_reads_without_one(tmp_path: Path) -> None:
    run_dir = _run_dir(
        tmp_path, launch=False, run_status=RunStatus.FAILED, failed_step=FailedStep.CONFIG_INVALID, fixed=None
    )
    assert RoundFolder.read(run_dir).launch is None
    measurement = extract_benchmark_measurement(run_dir)
    assert (measurement["failed_step"], measurement["framework"]) == ("config_invalid", None)


def test_a_report_off_its_contract_raises(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    document = json.loads(RunLayout(run_dir).report.read_text(encoding="utf-8"))
    del document["fixed"]
    RunLayout(run_dir).report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReportError):
        extract_benchmark_measurement(run_dir)


def test_gpu_metrics_are_written_from_the_report_into_the_round_dir_not_the_run_dir(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    path = RoundFolder.read(run_dir).write_gpu_metrics(run_dir.parent)
    assert path == str(run_dir.parent / GPU_ARTIFACT_NAME)
    assert not (run_dir / GPU_ARTIFACT_NAME).exists()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["source"] == "report.json"
    assert (payload["samples"], payload["avg_power_w"]) == (200, 600.5)
    assert (payload["avg_temp_c"], payload["avg_gpu_util_pct"], payload["avg_vram_pct"]) == (70.0, 80.0, 50.0)


def test_the_grid_reads_a_finished_run_dir_and_writes_its_gpu_artifact(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, fixed=FixedMetrics.of({}))
    measurement = _round_report.record_round(run_dir, started_unix=0.0)
    assert measurement["valid_measurement"] is False
    assert (run_dir.parent / GPU_ARTIFACT_NAME).is_file()


def test_the_grid_merges_a_multi_node_rounds_pod_gpu_samples_into_its_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    monkeypatch.setattr(_multi_node_env, "pd_topology_from_state", dict)
    pods = tmp_path / "server_logs"
    pods.mkdir()
    monkeypatch.setenv("HYPERLOOM_MN_SERVER_LOG_DIR", str(pods))
    (pods / "gpu_metrics_10.0.0.1.csv").write_text("timestamp,power\n1000,300\n", encoding="utf-8")
    run_dir = _run_dir(tmp_path)
    _round_report.record_round(run_dir, started_unix=900.0)
    assert (run_dir / "gpu_metrics.csv").read_text(encoding="utf-8").splitlines() == [
        "host,timestamp,power",
        "10.0.0.1,1000,300",
    ]
