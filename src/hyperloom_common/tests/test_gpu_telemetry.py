# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measured-phase GPU sampler: six metrics per card, three views of one batch of samples."""

from __future__ import annotations

import csv
import time
from typing import Any

import pytest

from hyperloom_common import gpu_telemetry
from hyperloom_common.gpu_telemetry import GpuPhase, GpuPowerRecorder, MetricStats, parse_gpu_sample

_TOTAL_MB = 294896.0


def _row(gpu: int, watts: float, used_mb: float, *, temp: float = 60.0, util: float = 95.0, sclk: float = 2100.0):
    """A card as ``amd-smi metric --power --mem-usage --clock --temperature --usage --json`` prints it on MI355X."""
    xcd = {"clk": {"value": sclk, "unit": "MHz"}, "min_clk": {"value": 500}, "max_clk": {"value": 2400}}
    asleep = {"clk": {"value": 90, "unit": "MHz"}, "min_clk": {"value": 500}, "max_clk": {"value": 2400}}
    return {
        "gpu": gpu,
        "power": {"socket_power": {"value": watts, "unit": "W"}, "gfx_voltage": "N/A"},
        "mem_usage": {"total_vram": {"value": _TOTAL_MB, "unit": "MB"}, "used_vram": {"value": used_mb, "unit": "MB"}},
        "clock": {"gfx_0": xcd, "gfx_1": asleep, "mem_0": {"clk": {"value": 2000, "unit": "MHz"}}},
        "temperature": {"edge": "N/A", "hotspot": {"value": temp, "unit": "C"}, "mem": {"value": 44, "unit": "C"}},
        "usage": {"gfx_activity": {"value": util, "unit": "%"}, "umc_activity": {"value": 30, "unit": "%"}},
    }


class _Replay:
    """Feeds sample payloads in order, repeating the last; a ``--violation`` read finds no counters."""

    def __init__(self, *payloads: Any) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def __call__(self, *metrics: str) -> Any:
        if "--violation" in metrics:
            return {}
        self.calls += 1
        return self._payloads[min(self.calls - 1, len(self._payloads) - 1)]


@pytest.fixture(autouse=True)
def _fast_and_no_npm(monkeypatch):
    monkeypatch.setattr(gpu_telemetry, "_read_node_power", lambda: None)
    monkeypatch.setattr(gpu_telemetry, "MIN_INTERVAL_SEC", 0.01)


def _measure(recorder: GpuPowerRecorder, samples: int) -> None:
    recorder.note_phase(GpuPhase.MEASURED, time.monotonic())
    deadline = time.monotonic() + 5.0
    while recorder.summary()["samples"] < samples and time.monotonic() < deadline:
        time.sleep(0.005)
    recorder.note_phase(GpuPhase.FINISHED, time.monotonic())


def test_a_sample_reads_six_metrics_per_card():
    sample = parse_gpu_sample({"gpu_data": [_row(3, 812.0, 147_448.0, temp=61.0, util=97.0)]}, 1000.0)[3]

    assert (sample.unix, sample.gpu, sample.power_w, sample.temperature_c) == (1000.0, 3, 812.0, 61.0)
    assert sample.gpu_clock_mhz == 2100.0, "the sleeping XCD is not the clock the card runs at"
    assert (sample.mem_clock_mhz, sample.gpu_util_pct, sample.vram_pct) == (2000.0, 97.0, 50.0)


def test_a_metric_amd_smi_did_not_report_is_none_not_zero():
    row = {"gpu": 0, "power": {"socket_power": "N/A"}, "temperature": {"edge": {"value": 41}, "hotspot": "N/A"}}
    sample = parse_gpu_sample({"gpu_data": [row]}, 0.0)[0]

    assert sample.temperature_c == 41.0, "edge stands in where hotspot is not read"
    assert (sample.power_w, sample.gpu_clock_mhz, sample.mem_clock_mhz) == (None, None, None)
    assert (sample.gpu_util_pct, sample.vram_pct) == (None, None)


def test_gpu_monitor_has_one_block_per_serving_card_with_every_metric():
    query = _Replay(
        [_row(0, 800.0, 250_000, temp=60.0, util=90.0), _row(1, 225.0, 284)],
        [_row(0, 900.0, 250_000, temp=64.0, util=100.0), _row(1, 225.0, 284)],
    )
    recorder = GpuPowerRecorder(output_path=None, gpus={0, 1}, interval_sec=0.01, query=query)
    _measure(recorder, samples=2)
    recorder.close()

    (block,) = recorder.gpu_monitor()
    n = recorder.summary()["samples"]
    assert (block.gpu, block.sample_count) == (0, n)
    assert block.power_watts == MetricStats(min=800.0, max=900.0, avg=round((800.0 + 900.0 * (n - 1)) / n, 2))
    assert block.temperature_c is not None and block.temperature_c.max == 64.0
    assert block.gpu_util_pct is not None and block.gpu_util_pct.min == 90.0
    assert block.gpu_clock_mhz == MetricStats(min=2100.0, max=2100.0, avg=2100.0)
    assert block.mem_clock_mhz == MetricStats(min=2000.0, max=2000.0, avg=2000.0)
    assert block.vram_pct is not None and block.vram_pct.avg == pytest.approx(84.78, abs=0.01)
    assert set(block.as_dict()["power_watts"]) == {"min", "max", "avg"}


def test_every_measured_sample_of_every_card_lands_in_the_csv(tmp_path):
    query = _Replay([_row(0, 800.0, 250_000), _row(1, 225.0, 284)])
    samples = tmp_path / "gpu_samples.csv"
    recorder = GpuPowerRecorder(output_path=None, interval_sec=0.01, query=query, samples_path=str(samples))
    recorder.note_phase(GpuPhase.WARMUP, time.monotonic())
    time.sleep(0.05)
    assert query.calls == 0, "nothing is sampled before the measured phase"
    _measure(recorder, samples=3)
    recorder.close()

    with samples.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "unix",
        "gpu",
        "power_w",
        "temperature_c",
        "gpu_clock_mhz",
        "mem_clock_mhz",
        "gpu_util_pct",
        "vram_pct",
    ]
    assert {row["gpu"] for row in rows} == {"0", "1"}
    assert len(rows) == 2 * recorder.summary()["samples"]


def test_a_recorder_that_never_measured_writes_no_monitor_block_and_a_header_only_csv(tmp_path):
    samples = tmp_path / "gpu_samples.csv"
    recorder = GpuPowerRecorder(output_path=None, query=_Replay({}), samples_path=str(samples))
    recorder.close()
    assert recorder.gpu_monitor() == ()
    assert samples.read_text().splitlines() == [
        "unix,gpu,power_w,temperature_c,gpu_clock_mhz,mem_clock_mhz,gpu_util_pct,vram_pct"
    ]


def test_the_v1_power_summary_keeps_its_keys():
    query = _Replay([_row(0, 800.0, 250_000)])
    recorder = GpuPowerRecorder(output_path=None, interval_sec=0.01, query=query)
    _measure(recorder, samples=2)
    out = recorder.close()

    assert list(out) == [
        "schema_version",
        "source",
        "phase",
        "started_unix",
        "interval_sec",
        "samples",
        "serving_gpus",
        "unread_gpus",
        "avg_power_w",
        "max_power_w",
        "per_gpu",
        "power_throttled_gpus",
        "query_errors",
        "last_error",
        "throttle_query_errors",
        "throttle_last_error",
        "node",
        "node_query_errors",
        "aborted",
    ]
    assert (out["schema_version"], out["phase"]) == (1, "measured")
    assert list(out["per_gpu"]["0"]) == [
        "avg_power_w",
        "max_power_w",
        "samples",
        "peak_vram_fraction",
        "sclk_mean_mhz",
        "sclk_max_mhz",
        "power_throttle_fraction",
    ]


def test_the_phase_vocabulary_is_closed_and_other_names_are_ignored():
    assert [phase.value for phase in GpuPhase] == ["boot", "warmup", "measured", "finished", "eval"]
    recorder = GpuPowerRecorder(output_path=None, query=_Replay({}))
    recorder.note_phase("ready", time.monotonic())
    assert recorder.phase is GpuPhase.BOOT
    recorder.note_phase("eval", time.monotonic())
    assert recorder.phase is GpuPhase.EVAL
    recorder.close()
