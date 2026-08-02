# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the engine/memory clock fields of the GPU-monitor aggregate.

These feed the effective-frequency roofline derate, so the aggregate has to
distinguish "no clocks sampled" from "clocks sampled and low".
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.collectors.telemetry import (
    _aggregate_gpu_monitor,
)


def _write_report(tmp_path: Path, samples: list[dict], name: str = "benchmark_report.json") -> Path:
    """Write a benchmark report carrying *samples* under ``gpu_monitor``."""
    path = tmp_path / name
    path.write_text(json.dumps({"gpu_monitor": samples}), encoding="utf-8")
    return path


def test_aggregate_reports_clock_stats(tmp_path):
    report = _write_report(
        tmp_path,
        [
            {"power_w": 900.0, "temperature_c": 70.0, "clock_mhz": 1800.0, "mclk_mhz": 2000.0},
            {"power_w": 1100.0, "temperature_c": 80.0, "clock_mhz": 2000.0, "mclk_mhz": 2000.0},
        ],
    )
    agg = _aggregate_gpu_monitor([report], [])
    assert agg["avg_clock_mhz"] == 1900.0
    assert agg["max_clock_mhz"] == 2000.0
    assert agg["avg_mclk_mhz"] == 2000.0
    assert agg["clock_samples"] == 2
    assert agg["samples"] == 2


def test_clock_samples_distinguishes_unsampled_clocks(tmp_path):
    # A sampler predating ``--showclocks`` still yields power/temp rows; the
    # derate must be able to tell that apart from a genuinely low clock.
    report = _write_report(
        tmp_path,
        [
            {"power_w": 900.0, "temperature_c": 70.0},
            {"power_w": 950.0, "temperature_c": 72.0, "clock_mhz": 1900.0},
        ],
    )
    agg = _aggregate_gpu_monitor([report], [])
    assert agg["samples"] == 2
    assert agg["clock_samples"] == 1
    assert agg["avg_clock_mhz"] == 1900.0


def test_aggregate_accepts_alternate_sclk_key(tmp_path):
    report = _write_report(tmp_path, [{"sclk_mhz": 1750.0}])
    agg = _aggregate_gpu_monitor([report], [])
    assert agg["avg_clock_mhz"] == 1750.0
    assert agg["clock_samples"] == 1


def test_aggregate_without_clocks_reports_zero(tmp_path):
    report = _write_report(tmp_path, [{"power_w": 800.0}])
    agg = _aggregate_gpu_monitor([report], [])
    assert agg["avg_clock_mhz"] == 0.0
    assert agg["max_clock_mhz"] == 0.0
    assert agg["avg_mclk_mhz"] == 0.0
    assert agg["clock_samples"] == 0


def test_aggregate_empty_when_no_samples(tmp_path):
    assert _aggregate_gpu_monitor([_write_report(tmp_path, [])], []) == {}
