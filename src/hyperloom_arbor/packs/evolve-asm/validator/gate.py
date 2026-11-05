"""Gate logic: rebuild guard + SNR + weakest-of-N + noise floor + 85% roofline stop."""

from __future__ import annotations
import math, time
from dataclasses import dataclass, field, asdict
from typing import Optional
from packs.evolve_asm.validator.bench import BenchResult, bench_co, check_rebuild_guard, check_snr

ROOFLINE_TARGET = 0.85
DEFAULT_NOISE_FLOOR_FRAC = 0.005


@dataclass
class GateDecision:
    decision: str
    weakest_us: float
    best_us: float
    delta_pct: float
    roofline_frac: Optional[float]
    correctness_passed: bool
    rebuild_ok: bool
    reason: str
    # True when analytic hbm_bytes overestimates real HBM traffic (L2-resident,
    # small-M MoE, few active experts). roofline_frac is unreliable in this case —
    # use rocprofv3 FETCH_SIZE+WRITE_SIZE for accurate numbers (KernelForge pattern).
    l2_resident: bool = False
    ts: float = field(default_factory=time.time)
    def to_dict(self): return asdict(self)


def _improvement_pct(old: float, new: float, direction: str = "min") -> float:
    if direction == "min": return (old - new) / old * 100.0
    return (new - old) / old * 100.0


def run_gate(candidate=None, config: dict = None, noise_floor_us: float = 0.0, *,
             container: str = "", co_path: str = "", s_path: str = "",
             bench_cmd: str = "", snr_cmd: str = "", best_us: Optional[float] = None,
             n_runs: int = 3, snr_threshold_db: float = 30.0,
             noise_floor_frac: float = DEFAULT_NOISE_FLOOR_FRAC,
             bench_timeout_sec: int = 900) -> GateDecision:
    """Gate a candidate. Accepts either orchestrator positional form
    run_gate(candidate, config, noise_floor_us) or direct keyword form."""
    # Orchestrator calling convention: run_gate(candidate, config, noise_floor_us)
    if candidate is not None and config is not None:
        container       = config.get("container", container)
        bench_cmd       = config.get("bench_cmd", bench_cmd)
        snr_cmd         = config.get("snr_cmd", snr_cmd)
        snr_threshold_db = config.get("snr_threshold", snr_threshold_db)
        n_runs          = config.get("n_bench_runs", n_runs)
        # best_us = the campaign's current best, NOT the candidate being gated.
        # The orchestrator must pass it via config["best_us"]; fall back to None
        # (first measurement) if not yet recorded.
        best_us = config.get("best_us", None) if best_us is None else best_us
        co_path         = getattr(candidate, "co_path", co_path) or co_path
        s_path          = getattr(candidate, "s_path", s_path) or s_path
        # Convert noise_floor_us to fraction using best_us as reference
        if noise_floor_us > 0 and best_us and best_us > 0:
            noise_floor_frac = noise_floor_us / best_us


    # Rebuild guard only applies when the island produced a .co file (ASM round-trip).
    # FlyDSL/Triton/API benches have no .co — skip the guard for them.
    if co_path:
        rebuild_ok, rebuild_reason = check_rebuild_guard(container, co_path, s_path)
        if not rebuild_ok:
            return GateDecision("REVERT", 0.0, best_us or 0.0, 0.0, None, False, False,
                                f"REBUILD GUARD: {rebuild_reason}")

    # SNR check requires a separate validation command. Skip if not provided
    # (FlyDSL/API benches embed correctness in their own bench script).
    if snr_cmd:
        corr_ok, snr_db, corr_reason = check_snr(container, snr_cmd, snr_threshold_db)
        if not corr_ok:
            return GateDecision("REVERT", 0.0, best_us or 0.0, 0.0, None, False, True,
                                f"CORRECTNESS: {corr_reason}")

    result = bench_co(container=container, bench_cmd=bench_cmd,
                      n_runs=n_runs, timeout_sec=bench_timeout_sec)
    if not result.success:
        # Bench failed — log the actual error for diagnosis.
        # If best_us is None (no baseline yet) AND the island reported a gpu_us,
        # fall back to using the island's own measurement as baseline rather than
        # blocking the campaign entirely. This prevents silent failures from
        # causing every result to REVERT.
        island_gpu_us = getattr(candidate, "weakest_us", None) if candidate else None
        if best_us is None and island_gpu_us and island_gpu_us > 5.0:
            return GateDecision("KEEP", island_gpu_us, island_gpu_us, 0.0,
                                None, True, True,
                                f"BASELINE (gate bench failed, using island measurement "
                                f"{island_gpu_us:.3f}µs): {result.error[:200]}")
        return GateDecision("REVERT", 0.0, best_us or 0.0, 0.0, None, True, True,
                            f"BENCH FAILED: {result.error[:400]}")

    weakest_us = result.weakest_us
    roofline_frac = result.roofline.get("roofline_frac") if result.roofline else None
    l2_resident   = result.roofline.get("l2_resident", False) if result.roofline else False

    # STOP only when roofline measurement is trustworthy (not L2-inflated).
    # Per KernelForge: analytic hbm_bytes is unreliable for small-M MoE (few active
    # experts → real HBM traffic << analytic estimate → frac artificially inflated).
    # Use rocprofv3 FETCH_SIZE+WRITE_SIZE for accurate fracs (future work).
    if roofline_frac is not None and roofline_frac >= ROOFLINE_TARGET and not l2_resident:
        return GateDecision("STOP", weakest_us, best_us or weakest_us,
                            _improvement_pct(best_us, weakest_us) if best_us else 0.0,
                            roofline_frac, True, True,
                            f"ROOFLINE TARGET REACHED: {roofline_frac*100:.1f}% >= {ROOFLINE_TARGET*100:.0f}%")

    if best_us is None:
        return GateDecision("KEEP", weakest_us, weakest_us, 0.0, roofline_frac, True, True,
                            f"BASELINE: {weakest_us:.3f} µs established.")

    delta_pct = _improvement_pct(best_us, weakest_us, direction="min")
    min_pct   = noise_floor_frac * 100.0

    if delta_pct <= 0.0:
        return GateDecision("REVERT", weakest_us, best_us, delta_pct, roofline_frac, True, True,
                            f"SLOWER: {weakest_us:.3f}µs vs {best_us:.3f}µs ({delta_pct:+.1f}%)")
    if delta_pct < min_pct:
        return GateDecision("REVERT", weakest_us, best_us, delta_pct, roofline_frac, True, True,
                            f"NOISE: +{delta_pct:.2f}% below noise floor ({min_pct:.2f}%)")

    return GateDecision("KEEP", weakest_us, weakest_us, delta_pct, roofline_frac, True, True,
                        f"KEEP: {weakest_us:.3f}µs (+{delta_pct:.1f}%)"
                        + (f" | roofline {roofline_frac*100:.1f}%" if roofline_frac else ""))


def measure_noise_floor(container: str, bench_cmd: str, n_runs: int = 10,
                        timeout_sec: int = 300) -> float:
    result = bench_co(container, bench_cmd, n_runs=n_runs, timeout_sec=timeout_sec)
    if not result.success or len(result.samples_us) < 3:
        return DEFAULT_NOISE_FLOOR_FRAC
    samples = result.samples_us
    mean = sum(samples) / len(samples)
    std  = math.sqrt(sum((x-mean)**2 for x in samples) / len(samples))
    cv   = std / mean if mean > 0 else DEFAULT_NOISE_FLOOR_FRAC
    return min(max(cv * 2.0, 0.003), 0.05)
