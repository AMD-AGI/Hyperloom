# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The measured-phase GPU sampler: six metrics per card, three views of one batch of samples."""

from __future__ import annotations

import csv
import time
from typing import Any

import pytest

from hyperloom_common import gpu_telemetry
from hyperloom_common.gpu_telemetry import GpuPhase, GpuPowerRecorder, MetricStats, RecorderOptions, parse_gpu_sample

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
    recorder = GpuPowerRecorder(
        output_path=None, interval_sec=0.01, query=query, options=RecorderOptions(samples_path=str(samples))
    )
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
    recorder = GpuPowerRecorder(output_path=None, query=_Replay({}), options=RecorderOptions(samples_path=str(samples)))
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
        "energy",
    ]
    assert (out["schema_version"], out["phase"]) == (1, "measured")
    assert list(out["energy"]) == [
        "enabled",
        "source",
        "boundary",
        "windows",
        "seconds",
        "total_energy_j",
        "total_power_w",
        "per_gpu",
        "query_errors",
        "last_error",
    ]
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


# Energy counters. A fake clock both the recorder and the fake amd-smi read keeps every window's length exact.

#: The serving card of an MI355X round in Hyperloom's CI smoke run (Qwen3-0.6B, TP1, vLLM): 606.48 W over 21 samples
#: two seconds apart, with 95.5% of its VRAM held.
_CI_WATTS = 606.48
_CI_USED_MB = 0.9552 * _TOTAL_MB
#: The counter does not start at zero: it holds everything the card drew since the driver loaded.
_SINCE_BOOT_J = 14_292_727.274
_ENERGY = RecorderOptions(energy=True)


def _energy(*cards: tuple[int, Any]) -> dict:
    """``amd-smi metric --energy --json``: each card's ``TOTAL_ENERGY_CONSUMPTION``, in J or as amd-smi's ``N/A``."""

    def reading(joules: Any) -> Any:
        return {"value": joules, "unit": "J"} if isinstance(joules, (int, float)) else joules

    return {"gpu_data": [{"gpu": gpu, "energy": {"total_energy_consumption": reading(j)}} for gpu, j in cards]}


class _Clock:
    """The time the recorder and the fake amd-smi both read, moved only by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_793_991_494.752 + self.now


def _steady(watts: float):
    return (lambda t: watts), (lambda t: watts * t)


def _bursts(high: float, low: float, period: float, on: float):
    """*high* W for the first *on* seconds of every *period*, *low* W the rest: power and its exact integral."""

    def joules(t: float) -> float:
        whole, rest = divmod(t, period)
        lit = min(rest, on)
        return whole * (high * on + low * (period - on)) + high * lit + low * (rest - lit)

    return (lambda t: high if t % period < on else low), joules


class _Meter:
    """amd-smi on cards drawing ``watts(t)``: a sample reads the power at the instant asked, and the energy counter
    holds its exact integral, as the firmware accumulator does. Every call made is kept in ``asked``."""

    def __init__(self, clock: _Clock, watts, joules, gpus=(0,)) -> None:
        self._clock, self._watts, self._joules, self._gpus = clock, watts, joules, gpus
        self.asked: list[tuple[str, ...]] = []

    def __call__(self, *metrics: str) -> Any:
        self.asked.append(metrics)
        t = self._clock.now
        if "--violation" in metrics:
            return {}
        if "--energy" in metrics:
            return _energy(*((gpu, _SINCE_BOOT_J + self._joules(t)) for gpu in self._gpus))
        return {"gpu_data": [_row(gpu, self._watts(t), _CI_USED_MB) for gpu in self._gpus]}


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(gpu_telemetry, "time", fake)
    return fake


def _window(recorder: GpuPowerRecorder, clock: _Clock, *, seconds: float, interval: float = 2.0) -> None:
    """One measured window driven as the sampler loop drives it: a sample, the window opened, a sample every
    *interval*, and the window closed when the phase moves on."""
    recorder._phase = GpuPhase.MEASURED
    recorder._sample_once()
    recorder._open_window()
    for _ in range(round(seconds / interval)):
        clock.now += interval
        recorder._sample_once()
    recorder._phase = GpuPhase.FINISHED
    recorder._close_window()


def test_a_steady_load_reads_the_same_power_from_its_energy_counter_as_from_its_samples(clock):
    recorder = GpuPowerRecorder(output_path=None, query=_Meter(clock, *_steady(_CI_WATTS)), options=_ENERGY)
    _window(recorder, clock, seconds=40.0)
    out = recorder.summary()

    energy = out["energy"]
    assert (out["samples"], out["avg_power_w"]) == (21, _CI_WATTS)
    assert (energy["enabled"], energy["boundary"], energy["windows"], energy["seconds"]) == (True, "gpu", 1, 40.0)
    assert energy["total_energy_j"] == pytest.approx(_CI_WATTS * 40.0, abs=0.1)
    assert energy["total_power_w"] == pytest.approx(_CI_WATTS, abs=0.01)
    assert energy["per_gpu"]["0"]["vs_sampled"] == pytest.approx(1.0, abs=1e-4)
    assert (energy["query_errors"], energy["last_error"]) == (0, None)


def test_a_bursty_load_sampled_in_its_bursts_reads_its_true_power_only_from_the_counter(clock):
    """Every two-second sample lands in a half-second burst: the samples say 1300 W, the card drew 550 W."""
    recorder = GpuPowerRecorder(
        output_path=None, query=_Meter(clock, *_bursts(1300.0, 300.0, 2.0, 0.5)), options=_ENERGY
    )
    _window(recorder, clock, seconds=40.0)
    out = recorder.summary()

    assert out["avg_power_w"] == 1300.0
    assert out["energy"]["total_power_w"] == pytest.approx(550.0)
    assert out["energy"]["per_gpu"]["0"]["vs_sampled"] == pytest.approx(550.0 / 1300.0, abs=1e-4)


def test_the_energy_of_every_measured_window_adds_up_and_the_boot_between_them_is_left_out(clock):
    """A GEAK replay boots a server per replica; the counter keeps counting through the boot, the energy does not."""
    recorder = GpuPowerRecorder(output_path=None, query=_Meter(clock, *_steady(600.0), gpus=(4, 5)), options=_ENERGY)
    _window(recorder, clock, seconds=20.0)
    clock.now += 30.0
    _window(recorder, clock, seconds=20.0)
    energy = recorder.summary()["energy"]

    assert (energy["windows"], energy["seconds"]) == (2, 40.0)
    assert energy["per_gpu"]["4"]["energy_j"] == pytest.approx(600.0 * 40.0)
    assert energy["total_energy_j"] == pytest.approx(2 * 600.0 * 40.0), "the serving cards' energy, summed"
    assert energy["total_power_w"] == pytest.approx(1200.0)


class _Scripted:
    """Two serving cards at 700 W; each ``--energy`` read answers the next of *reads*, raising it if an exception."""

    def __init__(self, *reads: Any) -> None:
        self._reads = iter(reads)
        self.energy_reads = 0

    def __call__(self, *metrics: str) -> Any:
        if "--violation" in metrics:
            return {}
        if "--energy" in metrics:
            self.energy_reads += 1
            read = next(self._reads)
            if isinstance(read, Exception):
                raise read
            return read
        return {"gpu_data": [_row(gpu, 700.0, 250_000) for gpu in (4, 5)]}


def _two_windows(clock: _Clock, *reads: Any) -> tuple[dict, _Scripted]:
    query = _Scripted(*reads)
    recorder = GpuPowerRecorder(output_path=None, query=query, options=_ENERGY)
    for _ in range(2):
        _window(recorder, clock, seconds=10.0)
    return recorder.summary(), query


def test_a_card_missing_from_one_window_has_no_energy_and_the_round_no_total(clock):
    out, _ = _two_windows(
        clock, _energy((4, 0.0), (5, 0.0)), _energy((4, 7e3), (5, 7e3)), _energy((4, 7e3), (5, 7e3)), _energy((4, 14e3))
    )
    assert list(out["energy"]["per_gpu"]) == ["4"], "card 5 was not read at the second window's close"
    assert out["energy"]["per_gpu"]["4"]["avg_power_w"] == pytest.approx(700.0)
    assert (out["energy"]["total_energy_j"], out["energy"]["total_power_w"]) == (None, None)
    assert out["avg_power_w"] == 700.0, "the sampled power is not touched"


def test_a_counter_that_went_back_leaves_its_card_without_energy(clock):
    """A driver reload or GPU reset restarts the counter; that window's difference is no energy."""
    out, _ = _two_windows(
        clock,
        _energy((4, 0.0), (5, 0.0)),
        _energy((4, 7e3), (5, 7e3)),
        _energy((4, 7e3), (5, 7e3)),
        _energy((4, 100.0), (5, 14e3)),
    )
    assert list(out["energy"]["per_gpu"]) == ["5"]
    assert out["energy"]["total_energy_j"] is None


def test_a_failed_energy_read_costs_the_round_its_energy_and_nothing_else(clock):
    unsupported = RuntimeError("amd-smi exited 2: amd-smi metric: error: unrecognized arguments: --energy")
    out, query = _two_windows(clock, unsupported, unsupported)

    assert query.energy_reads == 2, "a window whose open was not read is not read at its close either"
    assert out["energy"]["query_errors"] == 2 and "--energy" in out["energy"]["last_error"]
    assert (out["energy"]["per_gpu"], out["energy"]["total_energy_j"]) == ({}, None)
    assert (out["avg_power_w"], out["query_errors"], out["last_error"]) == (700.0, 0, None)
    assert out["throttle_query_errors"] == 0


def test_energy_off_asks_amd_smi_for_no_counter_and_leaves_every_other_field_as_it_was(clock):
    def summary(energy: bool) -> tuple[dict, _Meter]:
        clock.now = 0.0
        meter = _Meter(clock, *_bursts(1300.0, 300.0, 2.0, 0.5), gpus=(4, 5))
        recorder = GpuPowerRecorder(output_path=None, query=meter, options=RecorderOptions(energy=energy))
        _window(recorder, clock, seconds=20.0)
        return recorder.summary(), meter

    on, _ = summary(True)
    off, meter = summary(False)
    assert not any("--energy" in asked for asked in meter.asked)
    assert {k: v for k, v in on.items() if k != "energy"} == {k: v for k, v in off.items() if k != "energy"}
    assert off["energy"]["enabled"] is False
    assert (off["energy"]["per_gpu"], off["energy"]["total_energy_j"], off["energy"]["seconds"]) == ({}, None, None)


def test_an_energy_read_with_no_counter_says_what_amd_smi_printed():
    with pytest.raises(ValueError, match=r"card 0 energy=\{'total_energy_consumption': 'N/A'\}"):
        gpu_telemetry.parse_energy_sample(_energy((0, "N/A"), (1, "N/A")))
    with pytest.raises(ValueError, match="accumulated_energy"):
        gpu_telemetry.parse_energy_sample({"gpu_data": [{"gpu": 0, "energy": {"accumulated_energy": 5}}]})
    assert gpu_telemetry.parse_energy_sample(_energy((0, "N/A"), (1, 4.5))) == {1: 4.5}, "an N/A card is left out"
    assert gpu_telemetry.parse_energy_sample({"gpu_data": []}) == {}


def test_the_sampler_closes_its_window_at_the_phase_change_and_not_an_interval_later():
    """The counters' window ends when the measured phase does: a late close would count the start of what follows."""
    recorder = GpuPowerRecorder(output_path=None, query=_Replay([_row(0, 800.0, 250_000)]), interval_sec=30.0)
    recorder.note_phase(GpuPhase.MEASURED, time.monotonic())
    deadline = time.monotonic() + 5.0
    while recorder._window_open is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert recorder._window_open is not None

    recorder.note_phase(GpuPhase.FINISHED, time.monotonic())
    deadline = time.monotonic() + 5.0
    while recorder._window_open is not None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert recorder._window_open is None, "closed at the phase change, well before the 30 s interval"
    started = time.monotonic()
    recorder.close()
    assert time.monotonic() - started < 5.0, "close() wakes the sampler instead of waiting out its interval"


def test_a_phase_named_before_sampling_starts_does_not_hurry_the_first_interval():
    query = _Replay([_row(0, 800.0, 250_000)])
    recorder = GpuPowerRecorder(output_path=None, query=query, interval_sec=30.0)
    recorder.note_phase(GpuPhase.WARMUP, time.monotonic())
    recorder.note_phase(GpuPhase.MEASURED, time.monotonic())
    time.sleep(0.3)
    assert query.calls == 1, "one sample, then the sampler waits its interval"
    recorder.close()
