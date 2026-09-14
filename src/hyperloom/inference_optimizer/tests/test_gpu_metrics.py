# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the per-round GPU telemetry artifact.

The blocks below are the two shapes that actually occur, taken from production
``benchmark_report.json`` files rather than invented: Magpie's pre-aggregated
``{min,max,avg}`` form, and the multi-node harvester's flat per-sample form.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hyperloom.orchestrator.actions.executors._gpu_metrics import (
    GPU_ARTIFACT_NAME,
    gpu_metrics_from_report,
    write_gpu_metrics,
)

# Verbatim from a production single-node round. Note what is *not* here: no
# utilization and no VRAM -- Magpie's monitor samples neither.
MAGPIE_BLOCK: dict[str, Any] = {
    "sample_count": 35,
    "duration_sec": 70.55,
    "temperature_c": {"min": 56.0, "max": 63.0, "avg": 58.6},
    "gpu_clock_mhz": {"min": 94, "max": 2402, "avg": 1030.1},
    "mem_clock_mhz": {"min": 2000, "max": 2000, "avg": 2000.0},
    "power_watts": {"min": 253.0, "max": 943.0, "avg": 476.5},
}

# The second block from that same session, so the weighting below is real.
MAGPIE_BLOCK_2: dict[str, Any] = {
    "sample_count": 126,
    "duration_sec": 259.37,
    "temperature_c": {"min": 56.0, "max": 63.0, "avg": 58.8},
    "gpu_clock_mhz": {"min": 94, "max": 2400, "avg": 1682.3},
    "mem_clock_mhz": {"min": 2000, "max": 2000, "avg": 2000.0},
    "power_watts": {"min": 253.0, "max": 848.0, "avg": 471.6},
}


def _round(tmp_path: Path, gpu_monitor: Any) -> Path:
    """A round workspace carrying a benchmark report."""
    (tmp_path / "benchmark_report.json").write_text(json.dumps({"gpu_monitor": gpu_monitor}), encoding="utf-8")
    return tmp_path


def test_the_magpie_block_yields_real_numbers():
    """The regression this started from: this exact shape read as all zeros."""
    out = gpu_metrics_from_report({"gpu_monitor": MAGPIE_BLOCK})

    assert out["avg_power_w"] == 476.5
    assert out["max_power_w"] == 943.0
    assert out["avg_temp_c"] == 58.6
    assert out["max_temp_c"] == 63.0
    assert out["avg_clock_mhz"] == 1030.1
    assert out["avg_mem_clock_mhz"] == 2000.0
    assert out["samples"] == 35
    assert out["blocks"] == 1


def test_means_are_weighted_by_sample_count():
    """A 35-sample block must not pull the round's mean as hard as a 126-sample one."""
    out = gpu_metrics_from_report({"gpu_monitor": [MAGPIE_BLOCK, MAGPIE_BLOCK_2]})

    # (476.5*35 + 471.6*126) / 161
    assert out["avg_power_w"] == 472.67
    assert out["avg_temp_c"] == 58.76
    assert out["max_power_w"] == 943.0
    assert out["samples"] == 161
    assert out["blocks"] == 2


def test_magpie_reports_no_utilization_or_vram():
    """Single-node reality, and the reason those fields are tri-state.

    Null, not 0.0: a 0% utilization reading asserts the GPU sat idle through the
    round, which is the opposite of what an unsampled metric means.
    """
    out = gpu_metrics_from_report({"gpu_monitor": MAGPIE_BLOCK})

    assert out["avg_gpu_util_pct"] is None
    assert out["max_gpu_util_pct"] is None
    assert out["avg_vram_pct"] is None
    assert out["max_vram_pct"] is None


def test_multi_node_flat_samples_carry_utilization_and_vram():
    """``_row_to_gpu_sample``'s own field names, on the path that has them."""
    flat = [
        {"power_w": 100.0, "temperature_c": 50.0, "clock_mhz": 1000.0, "gpu_util_pct": 80.0, "vram_pct": 40.0},
        {"power_w": 200.0, "temperature_c": 60.0, "clock_mhz": 2000.0, "gpu_util_pct": 90.0, "vram_pct": 50.0},
    ]
    out = gpu_metrics_from_report({"gpu_monitor": flat})

    assert out["avg_gpu_util_pct"] == 85.0
    assert out["max_gpu_util_pct"] == 90.0
    assert out["avg_vram_pct"] == 45.0
    assert out["max_vram_pct"] == 50.0
    # A flat scalar is that sample's mean and its peak alike.
    assert out["avg_power_w"] == 150.0
    assert out["max_power_w"] == 200.0
    assert out["samples"] == 2


def test_a_measured_zero_is_not_a_missing_reading():
    """The original defect in one line: ``_avg(a) or _avg(b)`` could not say this."""
    out = gpu_metrics_from_report({"gpu_monitor": {"power_w": 0.0, "power": 500.0}})

    assert out["avg_power_w"] == 0.0
    assert out["max_power_w"] == 0.0


def test_one_alias_wins_per_block_so_the_peak_cannot_fall_below_the_mean():
    """Resolving per statistic lets the mean come from one key and the peak from
    a stale sibling, which reports a maximum below the average."""
    out = gpu_metrics_from_report({"gpu_monitor": {"power_watts": {"avg": 300.0, "max": 316.0}, "power_w": 12.0}})

    assert out["avg_power_w"] == 300.0
    assert out["max_power_w"] == 316.0


def test_a_statistic_the_winning_alias_omits_stays_null():
    """Not borrowed from another key: the producer did not report it."""
    out = gpu_metrics_from_report({"gpu_monitor": {"power_watts": {"avg": 300.0}, "power_w": 12.0}})

    assert out["avg_power_w"] == 300.0
    assert out["max_power_w"] is None


def test_absolute_vram_is_not_folded_into_the_percent_field():
    """A MiB reading under a ``_pct`` name would be a units error, not a fallback."""
    out = gpu_metrics_from_report({"gpu_monitor": {"vram_used_mb": 81920.0, "memory_used_bytes": 8.6e10}})

    assert out["avg_vram_pct"] is None
    assert out["max_vram_pct"] is None
    # Nothing readable, so nothing is credited as measured.
    assert out["blocks"] == 1
    assert out["samples"] == 0


def test_a_block_that_measured_nothing_is_not_counted_as_a_sample():
    """``sample_count`` beside no recognised metric would put a large count next
    to a row of nulls."""
    out = gpu_metrics_from_report({"gpu_monitor": {"sample_count": 27000, "duration_sec": 10.0}})

    assert out["blocks"] == 1
    assert out["samples"] == 0
    assert out["avg_power_w"] is None


def test_zero_sample_count_is_not_promoted_to_one():
    """A monitor that started and sampled nothing reported zero, not one."""
    out = gpu_metrics_from_report({"gpu_monitor": {"sample_count": 0, "power_watts": {"avg": 300.0, "max": 300.0}}})

    assert out["samples"] == 0


def test_a_report_without_gpu_monitor_yields_nothing():
    """Absence of a block is a different answer from a block full of nulls."""
    assert gpu_metrics_from_report({"throughput": {"avg": 1.0}}) == {}
    assert gpu_metrics_from_report(None) == {}


def test_the_artifact_is_written_into_the_round(tmp_path):
    """Per round, beside the report it was read from."""
    ws = _round(tmp_path, MAGPIE_BLOCK)

    path = write_gpu_metrics(ws)

    assert path is not None
    payload = json.loads((ws / GPU_ARTIFACT_NAME).read_text(encoding="utf-8"))
    assert payload["avg_power_w"] == 476.5
    assert payload["schema_version"] == 1
    assert payload["source"] == "benchmark_report.json"


def test_no_report_writes_no_artifact(tmp_path):
    """Every round that never produced a benchmark report is this case."""
    assert write_gpu_metrics(tmp_path) is None
    assert not (tmp_path / GPU_ARTIFACT_NAME).exists()


def test_a_report_without_telemetry_writes_no_artifact(tmp_path):
    """An empty artifact would claim the question was asked and answered."""
    (tmp_path / "benchmark_report.json").write_text(json.dumps({"throughput": {}}), encoding="utf-8")

    assert write_gpu_metrics(tmp_path) is None
    assert not (tmp_path / GPU_ARTIFACT_NAME).exists()


def test_an_unreadable_report_does_not_raise(tmp_path):
    """Telemetry must never fail a round."""
    (tmp_path / "benchmark_report.json").write_text("{not json", encoding="utf-8")

    assert write_gpu_metrics(tmp_path) is None


def test_the_harvest_writes_it_for_every_round(tmp_path, monkeypatch):
    """The one hook: wherever a round's artifacts are finalised."""
    from hyperloom.orchestrator.actions.executors import benchmark_result as br

    monkeypatch.setattr(br, "harvest_mn_gpu_metrics", lambda *_a, **_k: {})
    ws = _round(tmp_path, MAGPIE_BLOCK)

    br.harvest_leaked_artifacts(ws)

    assert json.loads((ws / GPU_ARTIFACT_NAME).read_text(encoding="utf-8"))["avg_power_w"] == 476.5


def test_a_block_offering_only_min_contributes_nothing(tmp_path):
    """``min`` is not among the statistics reported, so a block with only ``min`` measured nothing reportable.

    Crediting its ``sample_count`` would put 27000 samples beside a row of nulls,
    which reads as a large, well-sampled round that somehow measured nothing.
    """
    out = gpu_metrics_from_report({"gpu_monitor": {"sample_count": 27000, "power_watts": {"min": 100.0}}})

    assert out["samples"] == 0
    assert out["avg_power_w"] is None
    assert out["max_power_w"] is None


def test_a_min_only_block_does_not_inflate_a_real_one(tmp_path):
    """The weighting must come from the blocks that actually reported."""
    out = gpu_metrics_from_report(
        {"gpu_monitor": [MAGPIE_BLOCK, {"sample_count": 27000, "power_watts": {"min": 100.0}}]}
    )

    assert out["samples"] == 35
    assert out["avg_power_w"] == 476.5


def test_a_failed_write_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    """Telemetry that was read but could not be written is a defect, and has to be audible.

    The bug this artifact exists to fix went unnoticed for 101 sessions because
    nothing said anything; a silent writer would repeat exactly that.
    """
    from hyperloom.common import io as common_io

    def explode(*_a, **_k):
        raise OSError("read-only workspace")

    monkeypatch.setattr(common_io, "atomic_write_json", explode)
    ws = _round(tmp_path, MAGPIE_BLOCK)

    with caplog.at_level(logging.WARNING):
        assert write_gpu_metrics(ws) is None
    assert "could not be written" in caplog.text


def test_a_round_without_telemetry_stays_quiet(tmp_path, caplog):
    """The ordinary case must not cry wolf, or the warning above stops meaning anything."""
    (tmp_path / "benchmark_report.json").write_text(json.dumps({"throughput": {}}), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert write_gpu_metrics(tmp_path) is None
    assert caplog.text == ""


def test_a_report_that_lands_after_the_harvest_is_still_written(tmp_path):
    """Harvest runs before the report is guaranteed to exist, so it cannot be the only chance.

    Magpie can finish writing ``benchmark_report.json`` after the subprocess is
    reaped; the round settles on a report the harvest never saw.
    """
    from hyperloom.orchestrator.actions.executors import benchmark_result as br
    from hyperloom.orchestrator.actions.executors._gpu_metrics import write_gpu_metrics_from_report

    # Harvest first, with no report on disk yet: nothing to write.
    br.harvest_leaked_artifacts(tmp_path)
    assert not (tmp_path / GPU_ARTIFACT_NAME).exists()

    # The report settles afterwards, which is where the second attempt reads it.
    report = {"gpu_monitor": MAGPIE_BLOCK}
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    write_gpu_metrics_from_report(tmp_path, report, source="benchmark_report.json")

    assert json.loads((tmp_path / GPU_ARTIFACT_NAME).read_text(encoding="utf-8"))["avg_power_w"] == 476.5


def test_the_artifact_is_in_the_package_globs():
    """It lives in the round workspace, which the bundle does not otherwise reach."""
    from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_GLOBS

    assert "runs/**/gpu_metrics.json" in PACKAGE_GLOBS
