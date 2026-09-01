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


def keep_gain_pct(
    candidate: Mapping[str, Any] | None,
    *,
    state: Any = None,
    framework: str | None = None,
    base_tput: float | None = None,
) -> tuple[float | None, bool]:
    """KEEP gain percent, using the composite score when the flag and triples are present.

    Returns:
        ``(gain_pct, used_composite)``. Composite ``gain_pct`` is ``None`` when
        *S* did not improve (same meaning as :func:`score_gain_pct`). Output-tput
        fallback uses :func:`hyperloom.common.gain_math.gain_pct`.
    """
    from hyperloom.common.gain_math import gain_pct as tput_gain_pct

    fw = framework or (getattr(state, "framework", None) if state is not None else None)
    cand_snap = perf_snapshot_from_mapping(candidate)
    baseline = resolve_baseline_perf(state)
    if composite_grading_enabled(fw) and cand_snap and baseline:
        anchor = perf_snapshot_from_mapping(getattr(state, "current_best", None) if state is not None else None)
        return score_gain_pct(cand_snap, anchor or baseline, baseline), True
    new_tput: float | None = None
    if isinstance(candidate, Mapping):
        raw = candidate.get("output_throughput", candidate.get("tput", candidate.get("new_tput")))
        if isinstance(raw, (int, float)):
            new_tput = float(raw)
    return tput_gain_pct(new_tput, float(base_tput or 0.0)), False


def session_gain_pct(
    candidate: Mapping[str, Any] | None,
    *,
    state: Any = None,
    framework: str | None = None,
    base_tput: float | None = None,
) -> tuple[float | None, bool]:
    """Session-total gain percent vs the session baseline (not vs ``current_best``).

    Composite ``gain_pct`` is ``S * 100`` (including ``0.0`` when no axis
    improved). KEEP incremental grading is :func:`keep_gain_pct`.
    """
    from hyperloom.common.gain_math import gain_pct as tput_gain_pct

    fw = framework or (getattr(state, "framework", None) if state is not None else None)
    cand_snap = perf_snapshot_from_mapping(candidate)
    baseline = resolve_baseline_perf(state)
    if composite_grading_enabled(fw) and cand_snap and baseline:
        return composite_score(cand_snap, baseline) * 100.0, True
    new_tput: float | None = None
    if isinstance(candidate, Mapping):
        raw = candidate.get("output_throughput", candidate.get("tput", candidate.get("new_tput")))
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            new_tput = float(raw)
    bt = base_tput
    if bt is None and state is not None:
        bt = getattr(state, "baseline_tput", None)
    return tput_gain_pct(new_tput, float(bt or 0.0)), False


def session_gain_from_measurement(
    new_tput: float,
    *,
    state: Any = None,
    candidate: Mapping[str, Any] | None = None,
    base_tput: float | None = None,
) -> tuple[float | None, bool]:
    """Session gain for a measured output tput, filling axes from *candidate* then ``current_best``."""
    mapping: dict[str, Any] = {}
    cb = getattr(state, "current_best", None) if state is not None else None
    if isinstance(cb, Mapping):
        mapping.update(cb)
    if isinstance(candidate, Mapping):
        mapping.update(candidate)
    mapping["tput"] = float(new_tput)
    mapping["output_throughput"] = float(new_tput)
    return session_gain_pct(mapping, state=state, base_tput=base_tput)


def perf_axes_from_mapping(source: Mapping[str, Any] | None) -> dict[str, float]:
    """Positive input / intvty / output / tpot fields for stack-lift payloads."""
    if not isinstance(source, Mapping):
        return {}
    out: dict[str, float] = {}
    for key in ("input_throughput", "intvty_p90", "tpot_p90_ms", "output_throughput"):
        val = source.get(key)
        if isinstance(val, (int, float)) and float(val) > 0:
            out[key] = float(val)
    if "output_throughput" not in out:
        tput = source.get("tput", source.get("new_tput"))
        if isinstance(tput, (int, float)) and float(tput) > 0:
            out["output_throughput"] = float(tput)
    if "output_throughput" in out:
        out["tput"] = out["output_throughput"]
    return out


def resolve_grading_anchor_score(state: Any) -> float:
    """Composite score of the config candidates are composed on (0 before any lift)."""
    cb = getattr(state, "current_best", None)
    baseline = resolve_baseline_perf(state)
    if baseline and isinstance(cb, dict):
        anchor = perf_snapshot_from_mapping(cb)
        if anchor:
            return composite_score(anchor, baseline)
    return 0.0


def session_composite_score(state: Any) -> float | None:
    """Composite score *S* of ``current_best`` vs the session baseline, or None."""
    fw = getattr(state, "framework", None) if state is not None else None
    if not composite_grading_enabled(fw):
        return None
    baseline = resolve_baseline_perf(state)
    snap = perf_snapshot_from_mapping(getattr(state, "current_best", None) if state is not None else None)
    if baseline is None or snap is None:
        return None
    return composite_score(snap, baseline)


def composite_watermark_levels(state: Any) -> tuple[float, float] | None:
    """``(1+S_now, 1+S_last_snapshot)`` for the 10% roofline watermark, or None."""
    score = session_composite_score(state)
    if score is None:
        return None
    last_s = getattr(state, "last_roofline_score", None) if state is not None else None
    if not isinstance(last_s, (int, float)) or isinstance(last_s, bool):
        last_s = 0.0
    return 1.0 + float(score), 1.0 + float(last_s)


__all__ = [
    "COMPOSITE_V1",
    "composite_grading_enabled",
    "composite_metric_enabled",
    "composite_score",
    "composite_watermark_levels",
    "delta_improvement",
    "keep_gain_pct",
    "noise_adjusted_delta",
    "parse_noise_pct",
    "parse_weights",
    "perf_axes_from_mapping",
    "perf_snapshot_from_mapping",
    "resolve_baseline_perf",
    "resolve_grading_anchor_score",
    "score_gain_pct",
    "session_composite_score",
    "session_gain_from_measurement",
    "session_gain_pct",
]
