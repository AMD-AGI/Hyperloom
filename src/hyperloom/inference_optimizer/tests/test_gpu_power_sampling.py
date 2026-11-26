# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured-window GPU power and energy: the reading ``--max-power-w`` is graded on, and how a round carries it."""

from __future__ import annotations

import pytest

from hyperloom_common import gpu_telemetry
from inference_suite.contracts import PowerSummary
from inference_suite.contracts.layout import POWER
from hyperloom.orchestrator.actions.executors.gpu._gpu_power import POWER_FIELDS, round_power_fields


def _power_summary(*, valid: bool = True, cards: int = 4) -> PowerSummary:
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
    return PowerSummary.of(POWER, {"started_unix": 1000.0, "scope": scope, "energy": energy, "phases": {}})


def test_a_rounds_power_fields_carry_the_windows_power_energy_and_joules_per_token():
    measurement = round_power_fields(_power_summary())
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


def test_a_sampled_round_whose_energy_is_invalid_stays_unmeasured():
    measurement = round_power_fields(_power_summary(valid=False))
    assert measurement["power_valid"] == 0
    assert measurement["gpu_power_avg_w"] is None and measurement["gpu_power_by_gpu_w"] is None
    assert measurement["gpu_energy_j"] is None and measurement["joules_per_output_token"] is None


@pytest.mark.parametrize(("value", "sampled"), [("0.2", 0.5), ("1.5", 1.5), ("1.6", None), ("3", None)])
def test_an_interval_too_long_for_the_sample_gap_limit_is_refused_where_it_is_set(monkeypatch, value, sampled):
    """A round sampled at an interval whose gaps could reach the 3 s limit would be invalid every time, so the
    setting is refused and the round samples nothing; below the floor it is raised to the floor."""
    from hyperloom.orchestrator.actions.executors.gpu._gpu_power import round_gpu_telemetry

    monkeypatch.delenv("HYPERLOOM_GPU_POWER_SAMPLING", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/amd-smi")
    monkeypatch.setenv("HYPERLOOM_GPU_POWER_INTERVAL_S", value)
    telemetry = round_gpu_telemetry()
    assert (telemetry.interval_sec if telemetry.enabled else None) == sampled


def test_a_recorder_refuses_an_interval_too_long_for_the_sample_gap_limit(tmp_path):
    with pytest.raises(ValueError, match="1.5"):
        gpu_telemetry.PowerRecorder(
            output_path=None,
            scope=gpu_telemetry.RecorderScope(gpus=(0,)),
            timing=gpu_telemetry.RecorderTiming(fast_sec=2.0),
        )


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
