# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured-window GPU power and energy: the reading ``--max-power-w`` is graded on, and how a round carries it."""

from __future__ import annotations

import json
import time

import pytest

from hyperloom_common import gpu_telemetry
from inference_suite.contracts.layout import POWER
from inference_suite.server_log import LogCursor
from hyperloom.orchestrator.actions.executors.gpu._gpu_power import (
    POWER_FIELDS,
    build_gpu_power_recorder,
    round_power_summary,
)
from hyperloom.orchestrator.actions.executors.benchmark_result import extract_benchmark_measurement


def _power_json(directory, *, started_unix: float, valid: bool = True, cards: int = 4) -> None:
    """A ``power.json`` whose measured window was 10 s at 700 W per card, valid or not."""
    energy = {
        "window": {"duration_s": 10.0},
        "per_gpu": {str(gpu): {"integral_j": 7000.0} for gpu in range(cards)},
        "total_gpu_energy_j": 7000.0 * cards,
        "avg_power_w": 700.0,
        "p75_power_w": 710.0,
        "p90_power_w": 720.0,
        "avg_total_gpu_power_w": 700.0 * cards,
        "p75_total_gpu_power_w": 710.0 * cards,
        "p90_total_gpu_power_w": 720.0 * cards,
        "joules_per_input_token": 0.5,
        "joules_per_output_token": 2.0,
        "joules_per_total_token": 0.4,
        "joules_per_successful_query": 280.0,
        "node_energy_j": None,
        "cpu_energy_j": None,
        "accumulator_check": {"relative_error": 0.01},
        "power_valid": 1,
        "reasons": [],
        "warnings": [],
    }
    if not valid:
        figures = {key for key in energy if key.endswith(("_w", "_j", "_token", "_query"))}
        energy.update(dict.fromkeys(figures), power_valid=0, reasons=["sampling_gap_exceeded"])
    scope = {"serving_gpus": list(range(cards)), "energy_gpus": list(range(cards)), "unread_gpus": []}
    document = {"started_unix": started_unix, "scope": scope, "energy": energy, "phases": {}}
    (directory / POWER).write_text(json.dumps(document))


def test_the_reader_finds_power_json_beside_the_server_log_or_one_level_up(tmp_path):
    workspace = tmp_path / "benchmark_vllm_20260929"
    workspace.mkdir()
    _power_json(tmp_path, started_unix=1000.0)
    summary = round_power_summary(None, workspace, subprocess_started_unix=999.0)
    assert summary is not None and summary.metrics["avg_power_w"] == 700.0
    assert summary.per_gpu_avg_power_w == {"0": 700.0, "1": 700.0, "2": 700.0, "3": 700.0}


def test_the_reader_ignores_power_json_from_an_earlier_round(tmp_path):
    _power_json(tmp_path, started_unix=1000.0)
    assert round_power_summary(None, tmp_path, subprocess_started_unix=2000.0) is None


_V1_REPORT = {"gpu_monitor": {"sample_count": 157, "power_watts": {"min": 225.0, "max": 790.0, "avg": 313.7}}}


def test_the_measurement_carries_the_windows_power_energy_and_joules_per_token(tmp_path):
    _power_json(tmp_path, started_unix=time.time())
    measurement = extract_benchmark_measurement(_V1_REPORT, workspace=tmp_path)
    assert {key: measurement[key] for key in (*POWER_FIELDS, "power_valid")} == {
        "gpu_power_avg_w": 700.0,
        "gpu_power_p75_w": 710.0,
        "gpu_power_p90_w": 720.0,
        "gpu_energy_j": 28000.0,
        "joules_per_input_token": 0.5,
        "joules_per_output_token": 2.0,
        "joules_per_total_token": 0.4,
        "joules_per_successful_query": 280.0,
        "power_valid": 1,
    }
    assert type(measurement["power_valid"]) is int
    assert measurement["gpu_power_by_gpu_w"] == {str(gpu): 700.0 for gpu in range(4)}


def test_a_sampled_round_whose_energy_is_invalid_stays_unmeasured(tmp_path):
    _power_json(tmp_path, started_unix=time.time(), valid=False)
    measurement = extract_benchmark_measurement(_V1_REPORT, workspace=tmp_path)
    assert measurement["power_valid"] == 0
    assert measurement["gpu_power_avg_w"] is None and measurement["gpu_power_by_gpu_w"] is None
    assert measurement["gpu_energy_j"] is None and measurement["joules_per_output_token"] is None


def test_the_measurement_falls_back_to_the_report_when_no_recorder_ran(tmp_path):
    measurement = extract_benchmark_measurement(_V1_REPORT, workspace=tmp_path)
    assert measurement["gpu_power_avg_w"] == 313.7
    assert measurement["gpu_power_by_gpu_w"] is None, "a v1 host-wide mean cannot be split per card"
    assert "power_valid" not in measurement and "gpu_energy_j" not in measurement


def test_the_builder_follows_amd_smi_and_the_switch(tmp_path, monkeypatch):
    """On wherever amd-smi exists, unless the operator turns it off; nothing to sample without amd-smi."""
    from hyperloom.orchestrator.actions.executors.gpu import _gpu_power

    log_path = str(tmp_path / "server.log")
    monkeypatch.delenv("HYPERLOOM_GPU_POWER_SAMPLING", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert build_gpu_power_recorder(log_path, {}) is None
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
    built: list[dict] = []
    monkeypatch.setattr(_gpu_power, "PowerRecorder", lambda **kwargs: built.append(kwargs) or kwargs)
    monkeypatch.setattr("hyperloom.common.gpu_power_settings.gpus_in_rocm_order", lambda: list(range(8)))
    assert build_gpu_power_recorder(log_path, {"HIP_VISIBLE_DEVICES": "4,5"}) is not None
    scope = built[0]["scope"]
    assert scope.gpus == (4, 5) and scope.window_source == "server_log_poll"
    assert built[0]["output_path"] == str(tmp_path / POWER) and built[0]["timing"].fast_sec == 1.0
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_SAMPLING", "0")
    assert build_gpu_power_recorder(log_path, {}) is None
    assert build_gpu_power_recorder(None, {}) is None


@pytest.mark.parametrize(("value", "sampled"), [("0.2", 0.5), ("1.5", 1.5), ("1.6", None), ("3", None)])
def test_an_interval_too_long_for_the_sample_gap_limit_is_refused_where_it_is_set(
    tmp_path, monkeypatch, value, sampled
):
    """A round sampled at an interval whose gaps could reach the 3 s limit would be invalid every time, so the
    setting is refused and the round samples nothing; below the floor it is raised to the floor."""
    from hyperloom.orchestrator.actions.executors.gpu._gpu_power import round_gpu_telemetry

    monkeypatch.delenv("HYPERLOOM_GPU_POWER_SAMPLING", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_INTERVAL_S", value)
    telemetry = round_gpu_telemetry()
    assert (telemetry.interval_sec if telemetry.enabled else None) == sampled
    recorder = build_gpu_power_recorder(str(tmp_path / "server.log"), {})
    assert (None if recorder is None else recorder._timing.fast_sec) == sampled


def test_a_recorder_refuses_an_interval_too_long_for_the_sample_gap_limit(tmp_path):
    with pytest.raises(ValueError, match="1.5"):
        gpu_telemetry.PowerRecorder(
            output_path=None,
            scope=gpu_telemetry.RecorderScope(gpus=(0,)),
            timing=gpu_telemetry.RecorderTiming(fast_sec=2.0),
        )


class _PhaseLog:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def note_phase(self, phase, mono):
        self.phases.append(phase)


def _ready_line() -> str:
    from inference_suite.server_log import READY_MARKERS

    return READY_MARKERS[0] + "\n"


def _unlistable_logs() -> list[str]:
    raise PermissionError("replay output is not readable")


def test_an_unreadable_replay_log_is_warned_once_and_polling_continues(caplog):
    from hyperloom.orchestrator.actions.executors.gpu._gpu_power import ServerLogPhaseDriver

    driver = ServerLogPhaseDriver(_PhaseLog(), _unlistable_logs, cursor=LogCursor())
    with caplog.at_level("DEBUG", logger="hyperloom.orchestrator.actions.executors.gpu._gpu_power"):
        driver._poll_logs()
        driver._poll_logs()
    levels = [record.levelname for record in caplog.records if "cannot read the replay" in record.getMessage()]
    assert levels == ["WARNING", "DEBUG"]


def test_a_phase_driver_bug_is_not_swallowed():
    from hyperloom.orchestrator.actions.executors.gpu._gpu_power import ServerLogPhaseDriver

    class _BrokenCursor(LogCursor):
        def advance(self, server_log_path):
            raise KeyError("scan bug")

    driver = ServerLogPhaseDriver(_PhaseLog(), lambda: ["server.log"], cursor=_BrokenCursor())
    with pytest.raises(KeyError):
        driver.stop()


def test_the_gpu_monitor_block_is_read_whole_by_the_gpu_metrics_reader(tmp_path):
    from hyperloom_common.gpu_telemetry import gpu_monitor
    from hyperloom_common.power_recording import PowerRecording, RecorderScope
    from hyperloom.orchestrator.actions.executors.gpu._gpu_metrics import gpu_metrics_from_report

    fast = [(float(t), 0.0, 4, 700.0, 62.0, 50.0, 1730.0, 2000.0, 52790.0, 98.0, 30.0, 2.5e11, 1e7) for t in range(3)]
    recording = PowerRecording(
        host="h",
        started_unix=0.0,
        unix_offset=0.0,
        ended_mono=3.0,
        scope=RecorderScope(gpus=(4,)),
        recorder={},
        marks=[("measured", 0.0, 0.0)],
        fast=fast,
        vram_total_b={4: 3.0e11},
    )
    metrics = gpu_metrics_from_report({"gpu_monitor": [block.as_dict() for block in gpu_monitor(recording)]})
    assert all(value is not None for value in metrics.values()), metrics
    assert (metrics["avg_power_w"], metrics["max_temp_c"], metrics["avg_gpu_util_pct"]) == (700.0, 62.0, 98.0)
