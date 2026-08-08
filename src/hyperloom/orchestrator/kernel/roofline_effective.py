# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Effective-frequency and achievable-bandwidth derating for the roofline ceiling.

The vendor compute peaks in ``roofline_ceiling.HW_SPECS`` are the architectural
product ``CUs x FLOPs_per_clk_per_CU x f_boost``. Both published tables invert
exactly at the peak *boost* engine clock::

    MI355X  2516.6e12 / (256 CU * 2.40 GHz) = 4096 FLOPs/clk/CU
    MI300X  1307.4e12 / (304 CU * 2.10 GHz) = 2048 FLOPs/clk/CU

A serving workload does not sustain boost: engine clock settles at whatever the
power/CAC/thermal controller allows. Compute throughput is linear in engine
clock, so a ceiling anchored at boost overstates the reachable compute roof by
exactly the ratio of sustained to boost clock.

Memory is a different mechanism and is deliberately NOT clock-scaled here. On
MI300-series parts mclk exposes a single DPM state (MI355X reports only
``2000Mhz``, and sampling confirms 2000 MHz with zero variance under load), so
memory clock does not droop under power constraints. The gap between vendor peak
HBM bandwidth and what a kernel actually attains is access efficiency -- row-
buffer locality, read/write turnaround, access pattern -- and is modelled as a
separate multiplicative efficiency.

How large each correction actually is, measured on an 8-GPU MI355X node:

* Engine clock holds near boost. Under 4 minutes of sustained 8-GPU bf16 GEMM
  the mean sclk was 2379 MHz (99.1% of the 2400 MHz boost) at 1277 W against a
  1400 W cap, and 2372 MHz (98.8%) under sustained memory-bound load. So on this
  part the compute derate is a small correction, not a large one.
* Bandwidth is the large error. Streaming reads reach ~89% of the vendor peak,
  and the decode ceiling is memory-bound, so this is the term that actually
  moves the reported roof.

The clock derate is kept regardless: it costs nothing when clocks are healthy,
and it is the only thing that will catch a genuinely power- or thermally-limited
node, where a boost-anchored ceiling would silently overstate the roof.

Both derates are no-ops until they have inputs: absent clock telemetry the
compute factor is ``1.0``, and the bandwidth efficiency defaults to ``1.0``
(uncalibrated) so the ceiling matches its historical value until a measured
figure is supplied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .hw_probe import probe_hbm_bandwidth_gb_per_sec

#: Environment override for the achievable-HBM-bandwidth efficiency, applied to
#: every GPU type. Accepts a fraction in ``(0, 1]``.
_BW_EFFICIENCY_ENV = "HYPERLOOM_ROOFLINE_HBM_BW_EFFICIENCY"

#: Lower bound on a believable sustained/reference clock ratio. A measured
#: effective clock below this fraction of the reference is treated as bad
#: telemetry (e.g. a sampler that caught only idle ticks) rather than a real
#: operating point, and the derate degrades to a no-op.
_MIN_SCLK_RATIO = 0.1


@dataclass(frozen=True)
class GpuFreqSpec:
    """Per-GPU clock references and achievable-bandwidth efficiency.

    Attributes:
        boost_sclk_mhz (float): Peak engine clock. This is the clock the vendor
            dense ``peak_tflops`` table is derived from (verified by inverting
            the published TFLOPS against CU count).
        ref_sclk_mhz (float): Engine clock the max-achievable TFLOPS table was
            measured at. Defaults to ``boost_sclk_mhz``: the measurement clock
            for the TraceLens arch figures is not recorded upstream, and
            assuming boost keeps the derate conservative (it can only reduce the
            ceiling, never inflate it). Correct this once the real measurement
            clock is known, otherwise a sub-boost measurement is double-counted.
        hbm_bw_efficiency (float): Achievable / peak HBM bandwidth. ``1.0``
            means uncalibrated -- the ceiling keeps its historical vendor-peak
            value. Override globally via ``HYPERLOOM_ROOFLINE_HBM_BW_EFFICIENCY``.
    """

    boost_sclk_mhz: float
    ref_sclk_mhz: float = 0.0
    hbm_bw_efficiency: float = 1.0

    def reference_sclk(self, convention: str) -> float:
        """Reference clock the compute peak for *convention* is anchored at.

        Args:
            convention: ``"achievable"`` for the sustained-TFLOPS table,
                anything else for the vendor dense peak.

        Returns:
            The reference engine clock in MHz.
        """
        if convention == "achievable" and self.ref_sclk_mhz > 0:
            return self.ref_sclk_mhz
        return self.boost_sclk_mhz


#: Boost clocks are the published peak engine clocks, cross-checked by
#: inverting the vendor ``peak_tflops`` against the CU counts in
#: ``inference_optimizer.gpu_types`` (both land exactly on the architectural
#: MFMA rate, confirming the tables are boost-anchored).
#:
#: ``hbm_bw_efficiency`` is measured, not estimated, and only for parts we have
#: run on. Unmeasured parts stay at 1.0 so their ceiling keeps its historical
#: value rather than inheriting another part's number.
#:
#: MI355X, measured on an 8-GPU gfx950 node (ROCm 7.2.4) with a fully coalesced
#: non-temporal streaming-read kernel over a 16 GiB buffer:
#:
#:     single GPU              7133 GB/s   89.2% of the 8000 GB/s vendor peak
#:     all 8 GPUs loaded  7102-7156 GB/s   88.8-89.5%
#:
#: The per-GPU figure does not degrade under full load: each GPU owns its HBM
#: stacks, so there is no shared path to contend for. Establishing that needed a
#: synchronized start across the eight processes -- allowed to free-run, they
#: finish allocation at different moments and the aggregate reads low, which is
#: what an earlier unsynchronized attempt here reported.
#:
#: The decode roofline counts read traffic (weights + KV), so the streaming-read
#: figure is the right one; copy/triad (~61%) model write-heavy traffic this
#: model does not have.
#:
#: Cross-checked against a real run rather than only a microbenchmark: the
#: 095726Z MoE decode session (see ``test_roofline_ceiling``) measured 6244 tok/s
#: at TP=1, which back-solves to 6096 GB/s of actual traffic -- 76.2% of vendor
#: peak, comfortably under the 89% attainable ceiling.
#:
#: Prefer erring loose over tight: a ceiling a real workload can exceed is worse
#: than a slightly generous one, since ``within%`` above 100% is meaningless and
#: would discredit the metric.
_GPU_FREQ_SPECS: dict[str, GpuFreqSpec] = {
    # CDNA3, 304 CU @ 2100 MHz -> 2048 FLOPs/clk/CU bf16. Bandwidth efficiency
    # not yet measured on this part.
    "mi300x": GpuFreqSpec(boost_sclk_mhz=2100.0),
    "mi308x": GpuFreqSpec(boost_sclk_mhz=2100.0),
    "mi325x": GpuFreqSpec(boost_sclk_mhz=2100.0),
    # CDNA4, 256 CU @ 2400 MHz -> 4096 FLOPs/clk/CU bf16.
    "mi355x": GpuFreqSpec(boost_sclk_mhz=2400.0, hbm_bw_efficiency=0.89),
}


@dataclass(frozen=True)
class EffectiveClocks:
    """Engine/memory clocks a benchmark actually ran at.

    Attributes:
        sclk_mhz (float): Mean sustained engine clock over the benchmark window;
            ``0`` when unmeasured.
        mclk_mhz (float): Mean memory clock; recorded for provenance only, since
            the memory roof is not clock-scaled.
        samples (int): Number of telemetry samples behind the means.
    """

    sclk_mhz: float = 0.0
    mclk_mhz: float = 0.0
    samples: int = 0

    @property
    def measured(self) -> bool:
        """Whether a usable engine-clock measurement is present.

        Returns:
            ``True`` when at least one sample yielded a positive engine clock.
        """
        return self.sclk_mhz > 0 and self.samples > 0


#: Minimum GPU utilization for a sample to count toward the effective clock.
#: The harvest window spans server start-up and inter-phase gaps, whose idle
#: ticks sit at a low DPM state (an idle MI355X reports ~95 MHz against a
#: 2400 MHz boost). Averaging those in would understate the sustained clock and derate
#: the ceiling too far, which inflates ``within%`` -- the opposite of the bug
#: this module exists to fix.
_MIN_ACTIVE_UTIL_PCT = 5.0


def _to_float(value: Any) -> float | None:
    """Coerce a telemetry field to ``float``.

    Args:
        value: Raw sample value.

    Returns:
        The float value, or ``None`` when absent or unparseable.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def effective_clocks_from_samples(
    samples: Any,
    *,
    min_util_pct: float = _MIN_ACTIVE_UTIL_PCT,
) -> EffectiveClocks:
    """Mean sustained clocks over the *active* samples of a benchmark window.

    Samples below *min_util_pct* utilization are excluded so start-up and idle
    ticks do not drag the mean down. When no sample carries utilization the
    filter is skipped rather than dropping everything, so older telemetry still
    yields a usable figure.

    Args:
        samples: Iterable of flat ``gpu_monitor`` sample dicts.
        min_util_pct: Utilization floor for a sample to count as active.

    Returns:
        The measured ``EffectiveClocks``; unmeasured when no usable sample
        carried an engine clock.
    """
    if not isinstance(samples, (list, tuple)):
        return EffectiveClocks()
    rows = [s for s in samples if isinstance(s, dict)]
    if not rows:
        return EffectiveClocks()

    def _sclk(sample: dict[str, Any]) -> float | None:
        """Engine clock of one sample, tolerating either key spelling."""
        return _to_float(sample.get("clock_mhz")) or _to_float(sample.get("sclk_mhz"))

    active = [s for s in rows if (_to_float(s.get("gpu_util_pct")) or 0.0) >= min_util_pct]
    # No utilization recorded anywhere -> keep every sample rather than none.
    if not active and not any(_to_float(s.get("gpu_util_pct")) is not None for s in rows):
        active = rows
    if not active:
        return EffectiveClocks()

    sclks = [v for v in (_sclk(s) for s in active) if v is not None and v > 0]
    if not sclks:
        return EffectiveClocks()
    mclks = [v for v in (_to_float(s.get("mclk_mhz")) for s in active) if v is not None and v > 0]
    return EffectiveClocks(
        sclk_mhz=sum(sclks) / len(sclks),
        mclk_mhz=(sum(mclks) / len(mclks)) if mclks else 0.0,
        samples=len(sclks),
    )


def effective_clocks_from_report(report: Any) -> EffectiveClocks:
    """Effective clocks from a ``benchmark_report.json`` mapping.

    Reads the ``gpu_monitor`` block written by the sampler harvest, accepting
    either the list-of-samples or single-sample shape.

    Args:
        report: Parsed benchmark report.

    Returns:
        The measured ``EffectiveClocks``, unmeasured when absent.
    """
    if not isinstance(report, dict):
        return EffectiveClocks()
    gm = report.get("gpu_monitor")
    if isinstance(gm, dict):
        gm = [gm]
    return effective_clocks_from_samples(gm)


def effective_clocks_from_entry(entry: Any) -> EffectiveClocks:
    """Effective clocks recorded on a measurement / state arm entry.

    Reads the fields stamped by the benchmark-result normalizer, so the ceiling
    does not have to rediscover ``benchmark_report.json`` from state.

    Args:
        entry: A measurement or state arm mapping (``last_baseline``,
            ``current_best``, ...).

    Returns:
        The recorded ``EffectiveClocks``, unmeasured when absent.
    """
    if not isinstance(entry, dict):
        return EffectiveClocks()
    sclk = _to_float(entry.get("effective_sclk_mhz"))
    if sclk is None or sclk <= 0:
        return EffectiveClocks()
    samples = _to_float(entry.get("effective_clock_samples")) or 0.0
    return EffectiveClocks(
        sclk_mhz=sclk,
        mclk_mhz=_to_float(entry.get("effective_mclk_mhz")) or 0.0,
        # A recorded clock with no sample count still describes a real run;
        # floor at 1 so it is not discarded as unmeasured.
        samples=max(int(samples), 1),
    )


def resolve_effective_clocks_from_state(state: Any, *, arm: str | None = None) -> EffectiveClocks:
    """Effective clocks for the arm a roofline ceiling is being built for.

    Mirrors the ceiling's own arm selection: the optimized arm when it carries
    clocks, otherwise baseline. Never raises; an unmeasured result leaves the
    ceiling boost-anchored exactly as before.

    Args:
        state: Shared run state carrying ``last_baseline`` / ``current_best``.
        arm: Pins the source arm; ``None`` prefers ``current_best``.

    Returns:
        The resolved ``EffectiveClocks``, unmeasured when no arm recorded any.
    """
    baseline = getattr(state, "last_baseline", None)
    if arm == "baseline":
        return effective_clocks_from_entry(baseline)
    current = effective_clocks_from_entry(getattr(state, "current_best", None))
    if current.measured:
        return current
    return effective_clocks_from_entry(baseline)


def resolve_freq_spec(gpu_type: str | None) -> GpuFreqSpec | None:
    """Look up the frequency spec for *gpu_type* (case-insensitive).

    Args:
        gpu_type: GPU type key.

    Returns:
        The ``GpuFreqSpec``, or ``None`` for an unknown GPU.
    """
    return _GPU_FREQ_SPECS.get((gpu_type or "").strip().lower())


def _env_bw_efficiency() -> float:
    """Read the bandwidth-efficiency override from the environment.

    Returns:
        The override in ``(0, 1]``, or ``0.0`` when unset or unparseable.
    """
    raw = os.environ.get(_BW_EFFICIENCY_ENV, "")
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return value if 0.0 < value <= 1.0 else 0.0


def hbm_bw_efficiency(gpu_type: str | None) -> float:
    """Achievable / peak HBM bandwidth for *gpu_type*.

    The environment override wins over the per-GPU table so a calibrated figure
    can be applied without a code change. Unknown GPUs and uncalibrated entries
    return ``1.0``, leaving the memory roof at its vendor-peak value.

    Args:
        gpu_type: GPU type key.

    Returns:
        A derating fraction in ``(0, 1]``.
    """
    override = _env_bw_efficiency()
    if override > 0:
        return override
    spec = resolve_freq_spec(gpu_type)
    if spec is None:
        return 1.0
    eff = spec.hbm_bw_efficiency
    return eff if 0.0 < eff <= 1.0 else 1.0


def effective_hbm_bw_gbps(
    gpu_type: str | None,
    peak_gbps: float,
    *,
    active_gpus: int = 1,
) -> tuple[float, str]:
    """Per-GPU achievable HBM bandwidth, preferring measurement over the table.

    Precedence: an on-node streaming-read probe, then the vendor peak scaled by
    the calibrated efficiency (itself ``1.0``, i.e. the raw vendor peak, on
    parts nobody has measured).

    The probe is preferred because it reports an absolute GB/s rather than a
    fraction of a theoretical peak, and the theoretical peak is the shakier
    input: it cannot be derived at runtime, since the DDR multiplier implied by
    ``hipDeviceProp_t`` is wrong for HBM3E by a factor of two.

    Args:
        gpu_type: GPU type key.
        peak_gbps: Vendor peak per-GPU bandwidth from the spec table.
        active_gpus: How many GPUs the workload loads concurrently.

    Returns:
        ``(gb_per_sec, source)`` where source is ``"probe"`` or
        ``"vendor_peak_x_efficiency"``.
    """
    probed = probe_hbm_bandwidth_gb_per_sec(gpu_type=gpu_type, active_gpus=active_gpus)
    if probed is not None and probed > 0:
        return probed, "probe"
    return peak_gbps * hbm_bw_efficiency(gpu_type), "vendor_peak_x_efficiency"


def sclk_derate_factor(
    gpu_type: str | None,
    clocks: EffectiveClocks | None,
    *,
    convention: str = "achievable",
) -> float:
    """Ratio of sustained to reference engine clock, for scaling a compute peak.

    Compute throughput is linear in engine clock, so the reachable compute roof
    scales by ``f_effective / f_reference``. The result is clamped to ``1.0``:
    the reference is a boost (or boost-assumed) clock, so a measurement above it
    means the reference is wrong, and inflating a ceiling on bad telemetry is
    worse than leaving it alone.

    Args:
        gpu_type: GPU type key, for the reference clock.
        clocks: Measured clocks; ``None`` or unmeasured yields ``1.0``.
        convention: Which compute-peak table the factor will scale --
            ``"achievable"`` or ``"vendor"``.

    Returns:
        A factor in ``(0, 1]``; exactly ``1.0`` when the derate cannot be
        applied, so callers degrade to their historical ceiling.
    """
    if clocks is None or not clocks.measured:
        return 1.0
    spec = resolve_freq_spec(gpu_type)
    if spec is None:
        return 1.0
    reference = spec.reference_sclk(convention)
    if reference <= 0:
        return 1.0
    ratio = clocks.sclk_mhz / reference
    if ratio < _MIN_SCLK_RATIO:
        # Implausibly low: treat as unusable telemetry rather than a real
        # operating point.
        return 1.0
    return min(ratio, 1.0)


def effective_clock_provenance(
    gpu_type: str | None,
    clocks: EffectiveClocks | None,
    *,
    convention: str = "achievable",
    peak_gbps: float = 0.0,
    active_gpus: int = 1,
) -> dict[str, Any]:
    """Provenance describing how (and whether) the effective derates applied.

    Surfacing the measured clock, its reference, and both factors keeps a
    derated ``within%`` interpretable next to an underated one. When a peak
    bandwidth is supplied, the resolved memory roof and the layer that produced
    it are reported too, since a probed roof and a table-derived one are not
    interchangeable.

    Args:
        gpu_type: GPU type key.
        clocks: Measured clocks, if any.
        convention: Compute-peak convention the factor is anchored to.
        peak_gbps: Vendor peak per-GPU bandwidth; ``0`` omits the memory fields.
        active_gpus: GPUs the workload loads, for selecting a probe measurement.

    Returns:
        A provenance mapping describing the applied derates.
    """
    spec = resolve_freq_spec(gpu_type)
    factor = sclk_derate_factor(gpu_type, clocks, convention=convention)
    bw_eff = hbm_bw_efficiency(gpu_type)
    measured = clocks is not None and clocks.measured
    memory: dict[str, Any] = {}
    if peak_gbps > 0:
        resolved_gbps, bw_source = effective_hbm_bw_gbps(
            gpu_type, peak_gbps, active_gpus=active_gpus
        )
        memory = {
            "hbm_bw_gbps_effective": round(resolved_gbps, 1),
            "hbm_bw_source": bw_source,
            "hbm_bw_active_gpus": active_gpus,
        }
    return {
        **memory,
        "effective_sclk_mhz": round(clocks.sclk_mhz, 1) if measured else None,
        "effective_mclk_mhz": (round(clocks.mclk_mhz, 1) if (clocks is not None and clocks.mclk_mhz > 0) else None),
        "effective_clock_samples": clocks.samples if clocks is not None else 0,
        "reference_sclk_mhz": (spec.reference_sclk(convention) if spec is not None else None),
        "sclk_derate_factor": round(factor, 4),
        "hbm_bw_efficiency": round(bw_eff, 4),
        "effective_derate_source": ("measured_telemetry" if measured else "unmeasured_no_derate"),
    }
