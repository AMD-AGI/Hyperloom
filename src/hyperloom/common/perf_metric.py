# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Composite performance metric (input tput + intvty p90 + output tput).

Weighted improvement vs the baseline triple, gated behind
``HYPERLOOM_PERF_METRIC=composite_v1``. Noise floors default to 0 (raw
``Δ``); set ``HYPERLOOM_PERF_NOISE_PCT`` to subtract a band. Serving /
AgentX workloads only; scriptable frameworks keep output-tput grading.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.common.env import env_str

COMPOSITE_V1 = "composite_v1"
_DEFAULT_WEIGHTS = (0.55, 0.30, 0.15)
_DEFAULT_NOISE_PCT = (0.0, 0.0, 0.0)


def composite_metric_enabled() -> bool:
    """True when the composite grading flag is on."""
    return env_str("HYPERLOOM_PERF_METRIC").lower() == COMPOSITE_V1


def composite_grading_enabled(framework: str | None = None) -> bool:
    """Composite grading is opt-in and limited to non-scriptable serving runs."""
    if not composite_metric_enabled():
        return False
    if not framework:
        return True
    from hyperloom.inference_optimizer import framework_registry

    return not framework_registry.is_scriptable(framework)


def _parse_triple(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = env_str(name)
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    out: list[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError:
            return default
    return out[0], out[1], out[2]


def parse_weights() -> tuple[float, float, float]:
    """Return ``(w_in, w_intv, w_out)`` from ``HYPERLOOM_PERF_WEIGHTS``."""
    return _parse_triple("HYPERLOOM_PERF_WEIGHTS", _DEFAULT_WEIGHTS)


def parse_noise_pct() -> tuple[float, float, float]:
    """Return noise floors ``(in, intv, out)`` in percent from env (default 0)."""
    return _parse_triple("HYPERLOOM_PERF_NOISE_PCT", _DEFAULT_NOISE_PCT)


def perf_snapshot_from_mapping(source: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Extract the perf triple when all three axes are positive."""
    if not isinstance(source, Mapping):
        return None
    out = source.get("output_throughput", source.get("tput"))
    inp = source.get("input_throughput")
    intv = source.get("intvty_p90")
    if not all(isinstance(v, (int, float)) and float(v) > 0 for v in (out, inp, intv)):
        return None
    snap: dict[str, float] = {
        "output_throughput": float(out),
        "input_throughput": float(inp),
        "intvty_p90": float(intv),
    }
    tpot = source.get("tpot_p90_ms")
    if isinstance(tpot, (int, float)) and float(tpot) > 0:
        snap["tpot_p90_ms"] = float(tpot)
    return snap


def resolve_baseline_perf(state: Any) -> dict[str, float] | None:
    """Read the session baseline perf triple from shared state."""
    raw = getattr(state, "baseline_perf", None)
    if isinstance(raw, dict):
        return perf_snapshot_from_mapping(raw)
    return None


def delta_improvement(new: float, base: float) -> float:
    """Fractional improvement (0 when not strictly better)."""
    if base <= 0 or new <= 0:
        return 0.0
    return max(0.0, (float(new) - float(base)) / float(base))


def noise_adjusted_delta(delta: float, noise_pct: float) -> float:
    """Subtract a noise floor (percent points) from a fractional delta."""
    return max(0.0, float(delta) - float(noise_pct) / 100.0)


def composite_score(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    weights: tuple[float, float, float] | None = None,
    noise_pct: tuple[float, float, float] | None = None,
) -> float:
    """Weighted noise-adjusted improvement vs baseline on all three axes."""
    w_in, w_intv, w_out = weights or parse_weights()
    n_in, n_intv, n_out = noise_pct or parse_noise_pct()
    d_in = noise_adjusted_delta(
        delta_improvement(float(candidate["input_throughput"]), float(baseline["input_throughput"])),
        n_in,
    )
    d_intv = noise_adjusted_delta(
        delta_improvement(float(candidate["intvty_p90"]), float(baseline["intvty_p90"])),
        n_intv,
    )
    d_out = noise_adjusted_delta(
        delta_improvement(float(candidate["output_throughput"]), float(baseline["output_throughput"])),
        n_out,
    )
    return w_in * d_in + w_intv * d_intv + w_out * d_out


def passes_intvty_gate(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    *,
    noise_pct: float | None = None,
) -> bool:
    """Hard gate: intvty p90 must not regress beyond the noise band."""
    _n_in, n_intv, _n_out = parse_noise_pct()
    floor_pct = float(noise_pct if noise_pct is not None else n_intv)
    anchor_intv = float(anchor.get("intvty_p90") or 0.0)
    cand_intv = float(candidate.get("intvty_p90") or 0.0)
    if anchor_intv <= 0 or cand_intv <= 0:
        return True
    min_allowed = anchor_intv * (1.0 - floor_pct / 100.0)
    return cand_intv >= min_allowed


def score_gain_pct(
    candidate: Mapping[str, float],
    anchor: Mapping[str, float],
    baseline: Mapping[str, float],
) -> float | None:
    """Incremental composite-score gain of *candidate* over *anchor* (both vs *baseline*)."""
    anchor_score = composite_score(anchor, baseline)
    cand_score = composite_score(candidate, baseline)
    if cand_score <= 0:
        return None
    if anchor_score <= 0:
        return cand_score * 100.0
    return (cand_score - anchor_score) / anchor_score * 100.0


def resolve_grading_anchor_score(state: Any) -> float:
    """Composite score of the config candidates are composed on (0 before any lift)."""
    cb = getattr(state, "current_best", None)
    baseline = resolve_baseline_perf(state)
    if baseline and isinstance(cb, dict):
        anchor = perf_snapshot_from_mapping(cb)
        if anchor:
            return composite_score(anchor, baseline)
    return 0.0


__all__ = [
    "COMPOSITE_V1",
    "composite_grading_enabled",
    "composite_metric_enabled",
    "composite_score",
    "delta_improvement",
    "noise_adjusted_delta",
    "parse_noise_pct",
    "parse_weights",
    "passes_intvty_gate",
    "perf_snapshot_from_mapping",
    "resolve_baseline_perf",
    "resolve_grading_anchor_score",
    "score_gain_pct",
]
