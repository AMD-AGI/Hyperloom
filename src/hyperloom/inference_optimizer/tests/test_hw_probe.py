# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``orchestrator.kernel.hw_probe``.

The probe layer replaces hand-maintained per-SKU tables with on-node
measurement, so what matters here is that it either produces a *correct* number
or produces nothing at all. These tests cover the reduction of raw probe output
into per-precision rates, the guards that stop a bad measurement from inflating
a ceiling, the cache round-trip, and the promise that every unsupported or
broken path returns ``None`` so callers fall back through the existing chain.

No GPU is required: the subprocess boundary is stubbed. A live counterpart runs
only when ``HYPERLOOM_GPU_PROBE_RUN_LIVE=1``.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from hyperloom.orchestrator.kernel import hw_probe
from hyperloom.orchestrator.kernel.hw_probe import (
    DISABLE_ENV,
    DeviceInfo,
    MfmaRate,
    ProbeResult,
    detect_arch,
    load_cached,
    normalize_arch,
    probe_compute_peak_tflops,
    probe_hbm_bandwidth_gb_per_sec,
    rocm_version,
)

#: Verbatim stdout from the gfx950 matrix-core probe. Both bf16 variants and
#: both fp8 variants are present, which is the case the max-wins reduction
#: exists to handle.
_MFMA_STDOUT = (
    '{"kind":"device","arch":"gfx950:sramecc+:xnack-","cus":256,"boost_mhz":2400}\n'
    '{"kind":"mfma","precision":"bf16","variant":"mfma_f32_16x16x32_bf16",'
    '"flops_per_sec":2.337600e+15}\n'
    '{"kind":"mfma","precision":"bf16","variant":"mfma_f32_16x16x16bf16_1k",'
    '"flops_per_sec":1.212149e+15}\n'
    '{"kind":"mfma","precision":"fp8","variant":"mfma_f32_16x16x32_fp8_fp8",'
    '"flops_per_sec":2.423697e+15}\n'
    '{"kind":"mfma","precision":"fp8","variant":"mfma_scale_f32_16x16x128_f8f6f4[e4m3]",'
    '"flops_per_sec":5.018500e+15}\n'
)


@pytest.fixture(autouse=True)
def _clear_probe_env(monkeypatch):
    """Keep probe env overrides and memoized lookups out of assertions."""
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(hw_probe.TIMEOUT_ENV, raising=False)
    hw_probe.clear_caches()
    yield
    hw_probe.clear_caches()


@pytest.fixture
def _isolated_cache(monkeypatch, tmp_path):
    """Point the probe cache at a scratch directory."""
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path))
    return tmp_path


def _result(**overrides) -> ProbeResult:
    """Build a probe result with sensible gfx950 defaults.

    Args:
        **overrides: Fields to replace.

    Returns:
        A ``ProbeResult`` for use in assertions.
    """
    base = {
        "schema": hw_probe._SCHEMA_VERSION,
        "device": DeviceInfo(arch="gfx950", cu_count=256, boost_sclk_mhz=2400.0),
        "rocm_version": "7.2.4",
        "mfma_rates": {
            "bf16": MfmaRate("bf16", "mfma_f32_16x16x32_bf16", 3805.0),
            "fp8": MfmaRate("fp8", "mfma_scale_f32_16x16x128_f8f6f4[e4m3]", 8168.0),
        },
        "bandwidth_gb_per_sec": {1: 7133.2, 8: 7123.9},
        "probe_sclk_mhz": 2400.0,
        "probed_at": 1.0,
    }
    base.update(overrides)
    return ProbeResult(**base)


class TestArchNormalization:
    """Target-feature suffixes must not fragment the cache key."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("gfx950:sramecc+:xnack-", "gfx950"),
            ("gfx942", "gfx942"),
            ("  GFX950  ", "gfx950"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_features_are_stripped(self, raw, expected) -> None:
        assert normalize_arch(raw) == expected


class TestMfmaReduction:
    """Reducing raw probe output to one rate per precision."""

    @pytest.fixture
    def _stub_probe(self, monkeypatch):
        """Return the recorded gfx950 stdout and a 2400 MHz clock sample."""
        monkeypatch.setattr(hw_probe, "_run", lambda *a, **k: _MFMA_STDOUT)
        monkeypatch.setattr(hw_probe, "_sample_sclk_once", lambda: 2400.0)

    def test_fastest_variant_wins_per_precision(self, _stub_probe) -> None:
        """A precision offering several opcodes is scored by its fastest.

        On gfx950 fp8 is reachable through both a 16x16x32 opcode and the
        ``f8f6f4`` path, and only the latter carries the doubled rate. Taking
        the first match instead of the maximum would understate fp8 by 2x.
        """
        device, rates, sclk = hw_probe._probe_mfma(hw_probe.Path("/probe"))

        assert device == DeviceInfo(arch="gfx950", cu_count=256, boost_sclk_mhz=2400.0)
        assert sclk == 2400.0
        assert rates["bf16"].variant == "mfma_f32_16x16x32_bf16"
        assert rates["fp8"].variant == "mfma_scale_f32_16x16x128_f8f6f4[e4m3]"

    def test_rates_land_near_the_architectural_rate(self, _stub_probe) -> None:
        """The measurement must reproduce the documented ISA rate.

        This is the whole premise of retiring the tables, so it is asserted
        against the published gfx950 figures rather than against itself.
        """
        _, rates, _ = hw_probe._probe_mfma(hw_probe.Path("/probe"))

        assert rates["bf16"].flops_per_clk_per_cu == pytest.approx(4096.0, rel=0.10)
        assert rates["fp8"].flops_per_clk_per_cu == pytest.approx(8192.0, rel=0.10)

    def test_empty_output_yields_no_rates(self, monkeypatch) -> None:
        monkeypatch.setattr(hw_probe, "_run", lambda *a, **k: "")
        assert hw_probe._probe_mfma(hw_probe.Path("/probe")) == (None, {}, 0.0)

    def test_device_line_without_cus_is_rejected(self, monkeypatch) -> None:
        """A device report with no CU count cannot produce a per-CU rate."""
        monkeypatch.setattr(
            hw_probe,
            "_run",
            lambda *a, **k: '{"kind":"device","arch":"gfx950","cus":0,"boost_mhz":2400}\n',
        )
        assert hw_probe._probe_mfma(hw_probe.Path("/probe")) == (None, {}, 0.0)


class TestClockSampleGuard:
    """Grossly implausible probe clocks are discarded in favour of boost.

    The sampled clock is the divisor, so reading it low scales every rate --
    and therefore the ceiling -- upward, which is the one direction this module
    must not fail in. The guard only catches *gross* contamination such as idle
    ticks; distinguishing the engine clock from the memory clock is the parser's
    job (see :class:`TestSclkParsing`), because at 2000 MHz against a 2400 MHz
    boost the memory clock is numerically indistinguishable from an ordinary
    throttled engine clock.
    """

    def test_idle_level_sample_is_rejected_in_favour_of_boost(self, monkeypatch) -> None:
        """An idle 95 MHz tick cannot describe a compute-saturating probe."""
        monkeypatch.setattr(hw_probe, "_run", lambda *a, **k: _MFMA_STDOUT)
        monkeypatch.setattr(hw_probe, "_sample_sclk_once", lambda: 95.0)

        _, rates, sclk = hw_probe._probe_mfma(hw_probe.Path("/probe"))

        assert sclk == 2400.0
        assert rates["bf16"].flops_per_clk_per_cu < 4096.0

    def test_missing_samples_fall_back_to_boost(self, monkeypatch) -> None:
        monkeypatch.setattr(hw_probe, "_run", lambda *a, **k: _MFMA_STDOUT)
        monkeypatch.setattr(hw_probe, "_sample_sclk_once", lambda: 0.0)

        _, _, sclk = hw_probe._probe_mfma(hw_probe.Path("/probe"))

        assert sclk == 2400.0

    @pytest.mark.parametrize("sampled", [2200.0, 2000.0, 1300.0])
    def test_a_genuinely_throttled_clock_is_honoured(self, monkeypatch, sampled) -> None:
        """A plausible sub-boost clock is real and must be used as-is.

        Substituting boost here would understate the rate and, once the roof is
        rebuilt at the workload's own clock, understate the ceiling.
        """
        monkeypatch.setattr(hw_probe, "_run", lambda *a, **k: _MFMA_STDOUT)
        monkeypatch.setattr(hw_probe, "_sample_sclk_once", lambda: sampled)

        _, _, sclk = hw_probe._probe_mfma(hw_probe.Path("/probe"))

        assert sclk == sampled


class TestSclkParsing:
    """Engine clock is read by column name, never by scanning the row."""

    def _stub_csv(self, monkeypatch, stdout: str) -> None:
        """Route ``rocm-smi`` through a canned CSV response."""
        monkeypatch.setattr(hw_probe.shutil, "which", lambda _: "/usr/bin/rocm-smi")
        monkeypatch.setattr(
            hw_probe.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout, ""),
        )

    def test_engine_clock_is_read_not_memory_clock(self, monkeypatch) -> None:
        """mclk sits in the same numeric range and must not be picked up."""
        self._stub_csv(
            monkeypatch,
            "device,fclk clock speed:,mclk clock speed:,sclk clock speed:\n"
            "card0,(1250Mhz),(2000Mhz),(1413Mhz)\n",
        )
        assert hw_probe._sample_sclk_once() == 1413.0

    def test_loaded_card_wins_over_idle_peers(self, monkeypatch) -> None:
        """One GPU under probe among idle peers must report the probe's clock."""
        self._stub_csv(
            monkeypatch,
            "device,mclk clock speed:,sclk clock speed:\n"
            "card0,(2000Mhz),(2394Mhz)\n"
            "card1,(2000Mhz),(95Mhz)\n"
            "card2,(2000Mhz),(95Mhz)\n",
        )
        assert hw_probe._sample_sclk_once() == 2394.0

    def test_missing_tool_reports_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr(hw_probe.shutil, "which", lambda _: None)
        assert hw_probe._sample_sclk_once() == 0.0


class TestCacheRoundTrip:
    """Cached results must survive JSON and be keyed by arch and toolchain."""

    def test_write_then_read_preserves_the_payload(self, _isolated_cache) -> None:
        original = _result()
        hw_probe._write_cache(original)

        restored = load_cached(arch="gfx950", rocm="7.2.4")

        assert restored is not None
        assert restored.device == original.device
        assert restored.mfma_rates["fp8"].variant == original.mfma_rates["fp8"].variant
        # JSON object keys are strings; the active-GPU keys must come back as ints.
        assert restored.bandwidth_gb_per_sec == {1: 7133.2, 8: 7123.9}

    def test_a_different_toolchain_is_a_cache_miss(self, _isolated_cache) -> None:
        """A ROCm upgrade can move the numbers, so it must not reuse the old ones."""
        hw_probe._write_cache(_result())
        assert load_cached(arch="gfx950", rocm="6.4.0") is None

    def test_a_stale_schema_is_ignored(self, _isolated_cache) -> None:
        hw_probe._write_cache(_result())
        path = next(_isolated_cache.glob("gpu_probes/*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema"] = hw_probe._SCHEMA_VERSION + 1
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert load_cached(arch="gfx950", rocm="7.2.4") is None

    def test_corrupt_cache_does_not_raise(self, _isolated_cache) -> None:
        path = _isolated_cache / "gpu_probes" / "probe-gfx950-rocm7.2.4.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert load_cached(arch="gfx950", rocm="7.2.4") is None

    def test_missing_cache_is_a_miss_not_an_error(self, _isolated_cache) -> None:
        assert load_cached(arch="gfx950", rocm="7.2.4") is None


class TestComputeRoof:
    """Turning a measured rate into a ceiling."""

    def test_roof_is_rate_times_cus_times_clock(self) -> None:
        result = _result()

        roof = probe_compute_peak_tflops("bf16", sustained_sclk_mhz=2400.0, result=result)

        assert roof == pytest.approx(3805.0 * 256 * 2400e6 / 1e12)

    def test_roof_stays_below_the_vendor_peak(self) -> None:
        """A measured roof above the vendor dense peak would be nonsense.

        The vendor peak is the architectural rate at boost, so a probe result
        that exceeded it would mean the measurement, not the hardware, is wrong.
        """
        roof = probe_compute_peak_tflops("bf16", sustained_sclk_mhz=2400.0, result=_result())

        assert roof < 2516.6

    def test_sustained_clock_scales_the_roof(self) -> None:
        """Compute throughput is linear in engine clock."""
        at_boost = probe_compute_peak_tflops("bf16", sustained_sclk_mhz=2400.0, result=_result())
        throttled = probe_compute_peak_tflops("bf16", sustained_sclk_mhz=1200.0, result=_result())

        assert throttled == pytest.approx(at_boost / 2.0)

    def test_absent_clock_falls_back_to_boost(self) -> None:
        assert probe_compute_peak_tflops("bf16", result=_result()) == pytest.approx(
            probe_compute_peak_tflops("bf16", sustained_sclk_mhz=2400.0, result=_result())
        )

    def test_unprobed_precision_returns_none(self) -> None:
        """fp4 was never probed here, so the caller must fall back."""
        assert probe_compute_peak_tflops("fp4", result=_result()) is None

    def test_unknown_precision_returns_none(self) -> None:
        assert probe_compute_peak_tflops("int3", result=_result()) is None


class TestBandwidthLookup:
    """Selecting a bandwidth measurement for a given load."""

    def test_closest_active_gpu_count_is_used(self) -> None:
        result = _result()
        assert probe_hbm_bandwidth_gb_per_sec(active_gpus=1, result=result) == 7133.2
        assert probe_hbm_bandwidth_gb_per_sec(active_gpus=8, result=result) == 7123.9
        assert probe_hbm_bandwidth_gb_per_sec(active_gpus=7, result=result) == 7123.9

    def test_per_gpu_bandwidth_is_flat_across_load_on_gfx950(self) -> None:
        """Per-GPU HBM is private, so peers streaming must not change it.

        Locks in a measurement that contradicted an earlier unsynchronized
        attempt: with all eight cards at 100% utilization the per-GPU figure is
        unchanged, so any future regression toward a load-dependent number is a
        measurement bug rather than a discovery.
        """
        result = _result()
        solo = probe_hbm_bandwidth_gb_per_sec(active_gpus=1, result=result)
        loaded = probe_hbm_bandwidth_gb_per_sec(active_gpus=8, result=result)

        assert loaded == pytest.approx(solo, rel=0.02)

    def test_no_bandwidth_measurement_returns_none(self) -> None:
        assert probe_hbm_bandwidth_gb_per_sec(result=_result(bandwidth_gb_per_sec={})) is None


class TestProbeAppliesOnlyToTheProbedPart:
    """A probe describes the local device and nothing else.

    Rooflines are routinely computed for a GPU other than the one running the
    code -- comparing parts, or replaying a recorded session from a different
    node. Caught in real-workload replay: an MI300X session evaluated on a
    gfx950 host picked up the local 7133 GB/s measurement in place of MI300X's
    5300 GB/s vendor peak and lifted that ceiling by 35%. Substituting one
    part's hardware for another's is worse than not measuring, because it moves
    the ceiling instead of falling back.
    """

    @pytest.fixture
    def _gfx950_cache(self, _isolated_cache, monkeypatch):
        """A cached gfx950 probe, as a gfx950 host would have."""
        hw_probe._write_cache(_result())
        monkeypatch.setattr(hw_probe, "detect_arch", lambda: "gfx950")
        return _isolated_cache

    def test_foreign_part_does_not_borrow_the_local_probe(self, _gfx950_cache) -> None:
        assert probe_hbm_bandwidth_gb_per_sec(gpu_type="mi300x") is None
        assert probe_compute_peak_tflops("bf16", gpu_type="mi300x") is None

    def test_matching_part_uses_the_probe(self, _gfx950_cache) -> None:
        assert probe_hbm_bandwidth_gb_per_sec(gpu_type="mi355x") == 7133.2
        assert probe_compute_peak_tflops("bf16", gpu_type="mi355x") is not None

    @pytest.mark.parametrize("gpu_type", ["mi300x", "mi308x", "mi325x"])
    def test_every_gfx942_part_is_excluded(self, _gfx950_cache, gpu_type) -> None:
        """MI300X, MI308X and MI325X are all gfx942, none of them gfx950."""
        assert probe_hbm_bandwidth_gb_per_sec(gpu_type=gpu_type) is None

    def test_unknown_gpu_type_does_not_block_the_probe(self, _gfx950_cache) -> None:
        """An unmappable key cannot contradict the cache, so it is not a veto."""
        assert probe_hbm_bandwidth_gb_per_sec(gpu_type="some-future-part") == 7133.2

    @pytest.mark.parametrize(
        ("gpu_type", "expected"),
        [("mi300x", "gfx942"), ("mi355x", "gfx950"), ("nonsense", ""), (None, "")],
    )
    def test_gpu_type_maps_to_architecture(self, gpu_type, expected) -> None:
        assert hw_probe.expected_arch_for(gpu_type) == expected


class TestFailsSoft:
    """Every unsupported path returns ``None`` so callers keep their fallback."""

    def test_disable_flag_suppresses_reads(self, monkeypatch, _isolated_cache) -> None:
        hw_probe._write_cache(_result())
        monkeypatch.setenv(DISABLE_ENV, "1")

        assert load_cached(arch="gfx950", rocm="7.2.4") is None
        assert hw_probe.probe_and_cache() is None

    def test_missing_hipcc_skips_compilation(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(hw_probe.shutil, "which", lambda _: None)

        assert hw_probe._compile("int main(){}", "probe", "gfx950", tmp_path) is None

    def test_compile_failure_returns_none(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(hw_probe.shutil, "which", lambda _: "/usr/bin/hipcc")
        monkeypatch.setattr(
            hw_probe.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "error: no such target"),
        )

        assert hw_probe._compile("int main(){}", "probe", "gfx950", tmp_path) is None

    def test_undetectable_arch_skips_probing(self, monkeypatch) -> None:
        monkeypatch.setattr(hw_probe, "detect_arch", lambda: "")
        assert hw_probe.probe_and_cache() is None

    def test_run_timeout_is_swallowed(self, monkeypatch, tmp_path) -> None:
        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="probe", timeout=1.0)

        monkeypatch.setattr(hw_probe.subprocess, "run", _timeout)

        assert hw_probe._run(tmp_path / "probe", []) == ""

    def test_readers_tolerate_a_totally_absent_cache(self, monkeypatch) -> None:
        monkeypatch.setattr(hw_probe, "load_cached", lambda **k: None)
        assert probe_compute_peak_tflops("bf16") is None
        assert probe_hbm_bandwidth_gb_per_sec() is None


class TestEnvironment:
    """Environment knobs."""

    def test_timeout_override_is_applied(self, monkeypatch) -> None:
        monkeypatch.setenv(hw_probe.TIMEOUT_ENV, "42")
        assert hw_probe._timeout_sec() == 42.0

    @pytest.mark.parametrize("raw", ["", "abc", "0", "-5"])
    def test_bad_timeout_falls_back_to_default(self, monkeypatch, raw) -> None:
        monkeypatch.setenv(hw_probe.TIMEOUT_ENV, raw)
        assert hw_probe._timeout_sec() == hw_probe._DEFAULT_TIMEOUT_SEC


@pytest.mark.skipif(
    os.environ.get("HYPERLOOM_GPU_PROBE_RUN_LIVE") != "1",
    reason="live GPU probe disabled; set HYPERLOOM_GPU_PROBE_RUN_LIVE=1 to enable",
)
class TestLiveProbe:
    """Runs the real probes against real hardware."""

    def test_probe_recovers_documented_rates(self, _isolated_cache) -> None:
        """Measured rates must land near the published architectural figures."""
        documented = {"bf16": 4096.0, "fp16": 4096.0, "fp8": 8192.0, "fp4": 16384.0}

        result = hw_probe.probe_and_cache(force=True)

        assert result is not None, "probe failed on a node that should support it"
        assert result.device.cu_count > 0
        for precision, rate in result.mfma_rates.items():
            expected = documented.get(precision)
            if expected is None:
                continue
            assert rate.flops_per_clk_per_cu == pytest.approx(expected, rel=0.15)

    def test_probe_result_is_cached_and_reused(self, _isolated_cache) -> None:
        first = hw_probe.probe_and_cache(force=True)
        assert first is not None

        reused = load_cached()

        assert reused is not None
        assert reused.mfma_rates.keys() == first.mfma_rates.keys()
