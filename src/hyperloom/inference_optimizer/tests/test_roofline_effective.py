# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``orchestrator.kernel.roofline_effective``.

Covers the effective-frequency compute derate, the achievable-bandwidth memory
derate, extraction of sustained clocks from telemetry, and the guarantee that
every unmeasured / malformed path degrades to the historical boost-anchored
ceiling rather than inventing one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.gpu_types import _AMD_GPU_DISPATCH_IDENTITIES
from hyperloom.orchestrator.kernel.roofline_ceiling import (
    HW_SPECS,
    ModelMeta,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_roofline_from_perfmodel,
    compute_theoretical_peak_output_tok_per_sec,
)
from hyperloom.orchestrator.kernel.roofline_effective import (
    _BW_EFFICIENCY_ENV,
    EffectiveClocks,
    GpuFreqSpec,
    effective_clock_provenance,
    effective_clocks_from_entry,
    effective_clocks_from_report,
    effective_clocks_from_samples,
    hbm_bw_efficiency,
    resolve_effective_clocks_from_state,
    resolve_freq_spec,
    sclk_derate_factor,
)


@pytest.fixture(autouse=True)
def _clear_bw_env(monkeypatch):
    """Keep the bandwidth-efficiency override out of unrelated assertions."""
    monkeypatch.delenv(_BW_EFFICIENCY_ENV, raising=False)


class TestBoostClocksAnchorTheVendorTable:
    """The vendor ``peak_tflops`` table is ``CUs * FLOPs_per_clk * f_boost``.

    Inverting the published TFLOPS against CU count and the recorded boost clock
    must land exactly on the architectural MFMA rate. This is the evidence that
    the vendor peaks are boost-anchored, and it guards the boost constants the
    derate divides by.
    """

    @pytest.mark.parametrize(
        ("gpu_type", "precision", "expected_flops_per_clk_per_cu"),
        [
            ("mi300x", "bf16", 2048.0),
            ("mi300x", "fp8", 4096.0),
            ("mi325x", "bf16", 2048.0),
            ("mi355x", "bf16", 4096.0),
            ("mi355x", "fp8", 8192.0),
            ("mi355x", "mxfp4", 16384.0),
        ],
    )
    def test_vendor_peak_inverts_to_architectural_rate(self, gpu_type, precision, expected_flops_per_clk_per_cu):
        peak_tflops = HW_SPECS[gpu_type]["peak_tflops"][precision]
        cus = _AMD_GPU_DISPATCH_IDENTITIES[gpu_type][1]
        boost_hz = resolve_freq_spec(gpu_type).boost_sclk_mhz * 1e6
        flops_per_clk_per_cu = peak_tflops * 1e12 / (cus * boost_hz)
        assert flops_per_clk_per_cu == pytest.approx(expected_flops_per_clk_per_cu, rel=1e-3)

    def test_every_hw_spec_gpu_has_a_frequency_spec(self):
        # A GPU with a compute peak but no boost clock would silently skip the
        # derate, so the tables must stay in step.
        assert set(HW_SPECS) == set(_AMD_GPU_DISPATCH_IDENTITIES)
        for gpu_type in HW_SPECS:
            assert resolve_freq_spec(gpu_type) is not None


class TestSclkDerateFactor:
    """Compute throughput is linear in engine clock; the factor is that ratio."""

    def test_at_boost_is_exactly_one(self):
        clocks = EffectiveClocks(sclk_mhz=2400.0, samples=10)
        assert sclk_derate_factor("mi355x", clocks) == 1.0

    def test_below_boost_scales_linearly(self):
        clocks = EffectiveClocks(sclk_mhz=1800.0, samples=10)
        assert sclk_derate_factor("mi355x", clocks) == pytest.approx(0.75)

    def test_above_reference_clamps_to_one(self):
        # A measurement above boost means the reference is wrong; inflating a
        # ceiling on bad telemetry is worse than leaving it alone.
        clocks = EffectiveClocks(sclk_mhz=3000.0, samples=10)
        assert sclk_derate_factor("mi355x", clocks) == 1.0

    def test_implausibly_low_clock_degrades_to_no_op(self):
        clocks = EffectiveClocks(sclk_mhz=50.0, samples=10)
        assert sclk_derate_factor("mi355x", clocks) == 1.0

    @pytest.mark.parametrize(
        "clocks",
        [
            None,
            EffectiveClocks(),
            EffectiveClocks(sclk_mhz=1800.0, samples=0),
            EffectiveClocks(sclk_mhz=0.0, samples=10),
        ],
        ids=["none", "empty", "no-samples", "no-clock"],
    )
    def test_unmeasured_degrades_to_no_op(self, clocks):
        assert sclk_derate_factor("mi355x", clocks) == 1.0

    def test_unknown_gpu_degrades_to_no_op(self):
        clocks = EffectiveClocks(sclk_mhz=1200.0, samples=10)
        assert sclk_derate_factor("mi999x", clocks) == 1.0

    def test_vendor_convention_uses_boost_reference(self):
        clocks = EffectiveClocks(sclk_mhz=1050.0, samples=10)
        assert sclk_derate_factor("mi300x", clocks, convention="vendor") == pytest.approx(0.5)

    def test_achievable_reference_overrides_boost_when_recorded(self):
        # ref_sclk_mhz exists so a sub-boost achievable measurement is not
        # double-counted once its true clock is known.
        spec = GpuFreqSpec(boost_sclk_mhz=2400.0, ref_sclk_mhz=2000.0)
        assert spec.reference_sclk("achievable") == 2000.0
        assert spec.reference_sclk("vendor") == 2400.0

    def test_unset_reference_falls_back_to_boost(self):
        spec = GpuFreqSpec(boost_sclk_mhz=2400.0)
        assert spec.reference_sclk("achievable") == 2400.0


class TestHbmBandwidthEfficiency:
    """Memory is derated by access efficiency, never by clock (mclk is pinned)."""

    def test_measured_part_uses_its_calibrated_figure(self):
        # Measured on an 8-GPU gfx950 node; see _GPU_FREQ_SPECS for methodology.
        assert hbm_bw_efficiency("mi355x") == pytest.approx(0.89)

    def test_unmeasured_part_stays_at_vendor_peak(self):
        # An unmeasured part must keep its historical ceiling rather than
        # inherit another part's efficiency.
        assert hbm_bw_efficiency("mi300x") == 1.0

    def test_unknown_gpu_is_no_op(self):
        assert hbm_bw_efficiency("mi999x") == 1.0

    def test_env_override_applies(self, monkeypatch):
        monkeypatch.setenv(_BW_EFFICIENCY_ENV, "0.72")
        assert hbm_bw_efficiency("mi355x") == pytest.approx(0.72)

    @pytest.mark.parametrize("raw", ["", "abc", "0", "-0.5", "1.5"])
    def test_invalid_or_out_of_range_override_ignored(self, monkeypatch, raw):
        monkeypatch.setenv(_BW_EFFICIENCY_ENV, raw)
        assert hbm_bw_efficiency("mi355x") == pytest.approx(0.89)

    def test_table_efficiency_clamped_to_valid_range(self):
        assert GpuFreqSpec(boost_sclk_mhz=2400.0, hbm_bw_efficiency=0.8).hbm_bw_efficiency == 0.8


class TestEffectiveClocksFromSamples:
    """Sustained clock is the mean over *active* samples only."""

    def test_idle_samples_are_excluded(self):
        # Idle ticks sit at a low DPM state; averaging them in would understate
        # the sustained clock and over-derate the ceiling.
        samples = [
            {"clock_mhz": 1413.0, "gpu_util_pct": 0.0},
            {"clock_mhz": 2000.0, "gpu_util_pct": 99.0},
            {"clock_mhz": 2100.0, "gpu_util_pct": 98.0},
        ]
        clocks = effective_clocks_from_samples(samples)
        assert clocks.sclk_mhz == pytest.approx(2050.0)
        assert clocks.samples == 2
        assert clocks.measured

    def test_single_busy_gpu_is_not_swamped_by_idle_peers(self):
        # Reproduces a real 8-GPU MI355X capture: one loaded card at ~2350 MHz
        # while seven idle peers sat at ~95 MHz. Averaging the whole node gave
        # 299.7 MHz, which would have derated the ceiling roughly 8x the wrong
        # way and inflated within% accordingly.
        samples = [{"clock_mhz": 2350.0, "gpu_util_pct": 100.0}]
        samples += [{"clock_mhz": 95.0, "gpu_util_pct": 0.0} for _ in range(7 * 10)]
        clocks = effective_clocks_from_samples(samples)
        assert clocks.sclk_mhz == pytest.approx(2350.0)
        assert sclk_derate_factor("mi355x", clocks) == pytest.approx(0.9792, abs=1e-3)

    def test_all_samples_kept_when_utilization_absent(self):
        samples = [{"clock_mhz": 1800.0}, {"clock_mhz": 2000.0}]
        clocks = effective_clocks_from_samples(samples)
        assert clocks.sclk_mhz == pytest.approx(1900.0)
        assert clocks.samples == 2

    def test_all_idle_yields_unmeasured(self):
        samples = [{"clock_mhz": 1413.0, "gpu_util_pct": 0.0}]
        assert not effective_clocks_from_samples(samples).measured

    def test_alternate_sclk_key_accepted(self):
        clocks = effective_clocks_from_samples([{"sclk_mhz": 1900.0}])
        assert clocks.sclk_mhz == pytest.approx(1900.0)

    def test_mclk_averaged_for_provenance(self):
        samples = [
            {"clock_mhz": 2000.0, "mclk_mhz": 2000.0, "gpu_util_pct": 90.0},
            {"clock_mhz": 2000.0, "mclk_mhz": 2000.0, "gpu_util_pct": 90.0},
        ]
        assert effective_clocks_from_samples(samples).mclk_mhz == pytest.approx(2000.0)

    @pytest.mark.parametrize(
        "samples",
        [None, [], "not-a-list", [None, 3], [{}], [{"clock_mhz": 0.0}], [{"clock_mhz": "x"}]],
        ids=["none", "empty", "string", "junk", "no-keys", "zero", "unparseable"],
    )
    def test_malformed_input_yields_unmeasured(self, samples):
        assert not effective_clocks_from_samples(samples).measured

    def test_from_report_accepts_list_and_dict_shapes(self):
        as_list = {"gpu_monitor": [{"clock_mhz": 1900.0}]}
        as_dict = {"gpu_monitor": {"clock_mhz": 1900.0}}
        assert effective_clocks_from_report(as_list).sclk_mhz == pytest.approx(1900.0)
        assert effective_clocks_from_report(as_dict).sclk_mhz == pytest.approx(1900.0)

    @pytest.mark.parametrize("report", [None, {}, "nope", {"gpu_monitor": None}])
    def test_from_report_degrades_on_missing_telemetry(self, report):
        assert not effective_clocks_from_report(report).measured


class TestCeilingsHonourTheDerates:
    """End-to-end: the derates must move the ceilings, and only when measured."""

    _CMP_KWARGS = dict(
        gpu_type="mi355x",
        num_gpus=8,
        precision_tag="fp8",
        active_weight_bytes=70_000_000_000,
        weight_bytes=140_000_000_000,
        weight_dtype_bytes=1.0,
    )

    def test_compute_ceiling_unchanged_without_telemetry(self):
        base = compute_compute_bound_ceiling_tok_per_sec(**self._CMP_KWARGS)
        at_boost = compute_compute_bound_ceiling_tok_per_sec(
            **self._CMP_KWARGS,
            clocks=EffectiveClocks(sclk_mhz=2400.0, samples=5),
        )
        assert at_boost == pytest.approx(base)

    def test_compute_ceiling_scales_with_sustained_clock(self):
        base = compute_compute_bound_ceiling_tok_per_sec(**self._CMP_KWARGS)
        derated = compute_compute_bound_ceiling_tok_per_sec(
            **self._CMP_KWARGS,
            clocks=EffectiveClocks(sclk_mhz=1800.0, samples=5),
        )
        assert derated == pytest.approx(base * 0.75)

    def test_memory_ceiling_scales_with_bandwidth_efficiency(self, monkeypatch):
        kwargs = dict(
            gpu_type="mi355x",
            num_gpus=8,
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=1024,
            osl=1024,
            concurrency=64,
        )
        # Baseline against an unmeasured part so the table efficiency does not
        # confound the ratio, then confirm the override scales it.
        monkeypatch.setenv(_BW_EFFICIENCY_ENV, "1.0")
        base = compute_theoretical_peak_output_tok_per_sec(**kwargs)
        monkeypatch.setenv(_BW_EFFICIENCY_ENV, "0.75")
        assert compute_theoretical_peak_output_tok_per_sec(**kwargs) == pytest.approx(base * 0.75)

    def test_measured_part_ceiling_reflects_calibration(self, monkeypatch):
        kwargs = dict(
            gpu_type="mi355x",
            num_gpus=8,
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=1024,
            osl=1024,
            concurrency=64,
        )
        monkeypatch.setenv(_BW_EFFICIENCY_ENV, "1.0")
        uncalibrated = compute_theoretical_peak_output_tok_per_sec(**kwargs)
        monkeypatch.delenv(_BW_EFFICIENCY_ENV, raising=False)
        assert compute_theoretical_peak_output_tok_per_sec(**kwargs) == pytest.approx(uncalibrated * 0.89)

    def test_perfmodel_path_scales_with_sustained_clock(self):
        meta = ModelMeta(
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype_bytes=2.0,
            hidden_size=8192,
            intermediate_size=28672,
            vocab_size=128256,
            num_attention_heads=64,
        )
        kwargs = dict(meta=meta, gpu_type="mi355x", concurrency=32, isl=1024, osl=1024, num_gpus=8)
        base = compute_roofline_from_perfmodel(**kwargs)
        derated = compute_roofline_from_perfmodel(**kwargs, clocks=EffectiveClocks(sclk_mhz=1200.0, samples=5))
        assert base is not None and derated is not None
        # Halving the clock halves the compute roof; the memory roof is
        # untouched, so the blended decode figure must fall without vanishing.
        assert derated.peak_achievable_tflops == pytest.approx(base.peak_achievable_tflops * 0.5)
        assert derated.decode_mem_tok_per_s == pytest.approx(base.decode_mem_tok_per_s)
        assert 0 < derated.decode_tok_per_s <= base.decode_tok_per_s


class TestResolvingClocksFromState:
    """The ceiling reads clocks off the measurement, not off disk."""

    def test_entry_round_trips_recorded_fields(self):
        clocks = effective_clocks_from_entry(
            {"effective_sclk_mhz": 1850.0, "effective_mclk_mhz": 2000.0, "effective_clock_samples": 12}
        )
        assert clocks.sclk_mhz == pytest.approx(1850.0)
        assert clocks.mclk_mhz == pytest.approx(2000.0)
        assert clocks.samples == 12

    def test_entry_without_sample_count_still_counts_as_measured(self):
        clocks = effective_clocks_from_entry({"effective_sclk_mhz": 1850.0})
        assert clocks.measured
        assert clocks.samples == 1

    @pytest.mark.parametrize(
        "entry",
        [None, {}, "nope", {"effective_sclk_mhz": 0}, {"effective_sclk_mhz": "x"}],
    )
    def test_entry_degrades_when_absent(self, entry):
        assert not effective_clocks_from_entry(entry).measured

    def test_state_prefers_current_best_then_baseline(self):
        state = SimpleNamespace(
            last_baseline={"effective_sclk_mhz": 1500.0, "effective_clock_samples": 5},
            current_best={"effective_sclk_mhz": 1900.0, "effective_clock_samples": 7},
        )
        assert resolve_effective_clocks_from_state(state).sclk_mhz == pytest.approx(1900.0)

    def test_state_falls_back_to_baseline_when_optimized_arm_has_none(self):
        state = SimpleNamespace(
            last_baseline={"effective_sclk_mhz": 1500.0, "effective_clock_samples": 5},
            current_best={},
        )
        assert resolve_effective_clocks_from_state(state).sclk_mhz == pytest.approx(1500.0)

    def test_baseline_arm_is_pinned_and_ignores_current_best(self):
        state = SimpleNamespace(
            last_baseline={"effective_sclk_mhz": 1500.0, "effective_clock_samples": 5},
            current_best={"effective_sclk_mhz": 1900.0, "effective_clock_samples": 7},
        )
        clocks = resolve_effective_clocks_from_state(state, arm="baseline")
        assert clocks.sclk_mhz == pytest.approx(1500.0)

    def test_state_without_arms_is_unmeasured(self):
        assert not resolve_effective_clocks_from_state(SimpleNamespace()).measured


class TestProvenance:
    """Provenance keeps a derated ``within%`` interpretable next to a raw one."""

    def test_measured_provenance_reports_applied_derate(self):
        prov = effective_clock_provenance(
            "mi355x",
            EffectiveClocks(sclk_mhz=1800.0, mclk_mhz=2000.0, samples=42),
        )
        assert prov["effective_sclk_mhz"] == pytest.approx(1800.0)
        assert prov["effective_mclk_mhz"] == pytest.approx(2000.0)
        assert prov["effective_clock_samples"] == 42
        assert prov["reference_sclk_mhz"] == 2400.0
        assert prov["sclk_derate_factor"] == pytest.approx(0.75)
        assert prov["hbm_bw_efficiency"] == pytest.approx(0.89)
        assert prov["effective_derate_source"] == "measured_telemetry"

    def test_unmeasured_provenance_marks_no_derate(self):
        prov = effective_clock_provenance("mi355x", None)
        assert prov["effective_sclk_mhz"] is None
        assert prov["effective_clock_samples"] == 0
        assert prov["sclk_derate_factor"] == 1.0
        assert prov["effective_derate_source"] == "unmeasured_no_derate"

    def test_unknown_gpu_reports_no_reference(self):
        prov = effective_clock_provenance("mi999x", EffectiveClocks(sclk_mhz=1800.0, samples=5))
        assert prov["reference_sclk_mhz"] is None
        assert prov["sclk_derate_factor"] == 1.0
