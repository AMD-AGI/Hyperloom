# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``_aggregate_gpu_monitor``.

The function had no coverage while shipping every single-node session's GPU
section as all-zeros, so the first case here is the real Magpie block that used
to produce them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.collectors.telemetry import (
    _aggregate_gpu_monitor,
)

# Verbatim from a production ``benchmark_report.json``: Magpie's GPUMonitor
# returns one pre-aggregated block per metric, not a list of flat samples.
MAGPIE_BLOCK: dict[str, Any] = {
    "sample_count": 1391,
    "duration_sec": 2879.94,
    "temperature_c": {"min": 65.0, "max": 66.0, "avg": 65.7},
    "gpu_clock_mhz": {"min": 93, "max": 2384, "avg": 1984.3},
    "power_watts": {"min": 256.0, "max": 316.0, "avg": 300.6},
}


def _report(tmp_path: Path, name: str, gpu_monitor: Any) -> Path:
    """Write a minimal ``benchmark_report.json`` carrying ``gpu_monitor``."""
    path = tmp_path / name
    path.write_text(json.dumps({"gpu_monitor": gpu_monitor}), encoding="utf-8")
    return path


def test_magpie_block_yields_real_numbers(tmp_path):
    """The regression: this exact shape used to aggregate to all zeros."""
    warnings: list[str] = []
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", MAGPIE_BLOCK)], warnings)

    assert out["avg_power_w"] == 300.6
    assert out["max_power_w"] == 316.0
    assert out["avg_temp_c"] == 65.7
    assert out["max_temp_c"] == 66.0
    assert out["avg_clock_mhz"] == 1984.3
    assert warnings == []


def test_samples_counts_underlying_samples_not_blocks(tmp_path):
    """``samples`` is the measurement behind the numbers; ``blocks`` is the read count."""
    reports = [
        _report(tmp_path, "a.json", MAGPIE_BLOCK),
        _report(tmp_path, "b.json", MAGPIE_BLOCK),
    ]
    out = _aggregate_gpu_monitor(reports, [])

    assert out["blocks"] == 2
    assert out["samples"] == 2 * 1391


def test_flat_scalar_samples_still_supported(tmp_path):
    """The older per-sample shape must keep parsing; its scalar is both mean and peak."""
    flat = [
        {"power_w": 100.0, "temperature_c": 50.0, "clock_mhz": 1000.0},
        {"power_w": 200.0, "temperature_c": 60.0, "clock_mhz": 2000.0},
    ]
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", flat)], [])

    assert out["avg_power_w"] == 150.0
    assert out["max_power_w"] == 200.0
    assert out["avg_temp_c"] == 55.0
    assert out["max_temp_c"] == 60.0
    assert out["avg_clock_mhz"] == 1500.0
    assert out["blocks"] == 2
    assert out["samples"] == 2


def test_absent_metric_is_none_not_zero(tmp_path):
    """A metric nobody sampled must stay distinguishable from one that read zero."""
    out = _aggregate_gpu_monitor(
        [_report(tmp_path, "r.json", {"power_watts": {"avg": 300.0, "max": 310.0}})],
        [],
    )

    assert out["avg_power_w"] == 300.0
    assert out["avg_temp_c"] is None
    assert out["max_temp_c"] is None
    assert out["avg_clock_mhz"] is None


def test_measured_zero_does_not_fall_through_to_alias(tmp_path):
    """A real 0.0 is a reading, not a miss.

    The old ``_avg("power_w") or _avg("power")`` form could not say this: 0.0 is
    falsy, so a card genuinely drawing no power reported the alias's value
    instead.
    """
    out = _aggregate_gpu_monitor(
        [_report(tmp_path, "r.json", {"power_w": 0.0, "power": 500.0})],
        [],
    )

    assert out["avg_power_w"] == 0.0
    assert out["max_power_w"] == 0.0


def test_average_is_weighted_by_sample_count(tmp_path):
    """A 1000-sample block must not be averaged against a 10-sample block as equals."""
    heavy = {"sample_count": 1000, "power_watts": {"avg": 300.0, "max": 300.0}}
    light = {"sample_count": 10, "power_watts": {"avg": 100.0, "max": 100.0}}
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", [heavy, light])], [])

    # Unweighted this would be 200.0.
    assert out["avg_power_w"] == 298.02
    assert out["samples"] == 1010


def test_no_gpu_monitor_yields_empty_dict(tmp_path):
    """No block at all is a different answer from a block full of ``None``."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"throughput": 1.0}), encoding="utf-8")

    assert _aggregate_gpu_monitor([path], []) == {}


def test_one_alias_wins_per_block_so_max_cannot_fall_below_avg(tmp_path):
    """Resolving per statistic lets the mean come from one key and the peak from
    a stale sibling, which reports a maximum below the average."""
    block = {"power_watts": {"avg": 300.0, "max": 316.0}, "power_w": 12.0}
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", block)], [])

    assert out["avg_power_w"] == 300.0
    assert out["max_power_w"] == 316.0


def test_a_statistic_the_winning_alias_omits_stays_none(tmp_path):
    """Not borrowed from another key: the producer did not report it."""
    block = {"power_watts": {"avg": 300.0}, "power_w": 12.0}
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", block)], [])

    assert out["avg_power_w"] == 300.0
    assert out["max_power_w"] is None


def test_a_block_that_measured_nothing_is_not_counted_as_a_sample(tmp_path):
    """``sample_count`` beside no recognised metric would put a large count next
    to a row of ``None``."""
    out = _aggregate_gpu_monitor(
        [_report(tmp_path, "r.json", {"sample_count": 27000, "duration_sec": 10.0})],
        [],
    )

    assert out["blocks"] == 1
    assert out["samples"] == 0
    assert out["avg_power_w"] is None


def test_zero_sample_count_is_not_promoted_to_one(tmp_path):
    """A monitor that started and sampled nothing reported zero, not one -- the
    same absent-versus-zero conflation this change exists to remove."""
    block = {"sample_count": 0, "power_watts": {"avg": 300.0, "max": 300.0}}
    out = _aggregate_gpu_monitor([_report(tmp_path, "r.json", block)], [])

    assert out["samples"] == 0


def test_section_is_omitted_when_no_block_was_found(tmp_path):
    """The documented contract is absence, not an empty object: a consumer
    testing for the key must agree with one testing its contents."""
    from hyperloom.inference_optimizer.breakdown.collectors.telemetry import collect_telemetry

    (tmp_path / "runs").mkdir()
    section = collect_telemetry(tmp_path, {}, [])

    assert "gpu_monitor_aggregate" not in section


def test_malformed_report_warns_without_raising(tmp_path):
    """A bad report degrades to a warning; a good one alongside it still counts."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    warnings: list[str] = []

    out = _aggregate_gpu_monitor([bad, _report(tmp_path, "ok.json", MAGPIE_BLOCK)], warnings)

    assert out["avg_power_w"] == 300.6
    assert out["blocks"] == 1
    assert warnings
