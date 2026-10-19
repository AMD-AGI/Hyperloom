# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Throughput percentage-gain helpers (``gain_math``). Stdlib-only."""

from __future__ import annotations

from typing import Any, NamedTuple

from hyperloom.common.coerce import to_float
from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_TOTAL, graded_axes_of, holds_within_band


def gain_pct(new: float | None, base: float) -> float | None:
    """``(new-base)/base*100``; None when *new* is not a positive finite number or *base*<=0."""
    coerced = to_float(new)
    if coerced is None or coerced <= 0 or base <= 0:
        return None
    return (coerced - base) / base * 100.0


def gain_pct_or_zero(new: float, base: float) -> float:
    """``(new-base)/base*100`` when *base*>0 else 0.0 (negative on regression)."""
    if base <= 0:
        return 0.0
    return (new - base) / base * 100.0


def incremental_gain_pct(new: float, ref: float) -> float | None:
    """``(new-ref)/ref*100`` when *ref*>0 else None (*ref* e.g. current_best)."""
    if ref <= 0:
        return None
    return (new - ref) / ref * 100.0


class _GuardReading(NamedTuple):
    """The guard axis as each side of one concurrency pair measured it, and whether the optimized side held it."""

    baseline: float | None
    optimized: float | None
    holds: bool | None


def _speedup(baseline: float | None, optimized: float | None) -> float | None:
    """``optimized / baseline`` when both sides measured a positive value, else None."""
    if baseline is not None and baseline > 0 and optimized is not None and optimized > 0:
        return optimized / baseline
    return None


def _read_guard(
    baseline: dict[str, Any], optimized: dict[str, Any], guard_axis: str, noise_pct: float | None
) -> _GuardReading:
    """Read *guard_axis* off both points of a pair; nothing is read when there is no guard axis."""
    if not guard_axis:
        return _GuardReading(None, None, None)
    # ``graded_axes_of`` is what normalises the sweep's ``total_token_throughput`` onto GRADED_TOTAL, so the
    # guard cannot read a differently-named axis as an absent one.
    base_axes = graded_axes_of(baseline)
    opt_axes = graded_axes_of(optimized)
    holds: bool | None = None
    if base_axes.get(guard_axis) and opt_axes.get(guard_axis):
        holds = holds_within_band(opt_axes, base_axes, guard_axis, noise_pct=noise_pct)
    return _GuardReading(base_axes.get(guard_axis), opt_axes.get(guard_axis), holds)


def _speedup_stats(rows: list[dict[str, Any]], speedups: list[float]) -> dict[str, Any]:
    """The best rung, and the median and mean of the speedups the pairs produced."""
    best_idx, best_val = max(
        ((i, r["speedup"]) for i, r in enumerate(rows) if isinstance(r.get("speedup"), float)),
        key=lambda x: x[1],
    )
    sorted_sp = sorted(speedups)
    n = len(sorted_sp)
    median = sorted_sp[n // 2] if n % 2 == 1 else 0.5 * (sorted_sp[n // 2 - 1] + sorted_sp[n // 2])
    return {
        "best_conc": rows[best_idx]["conc"],
        "best_speedup": round(best_val, 4),
        # The headline rung is the best on the objective alone. Whether the session's own KEEP rule would
        # have accepted it is a second fact, and the two disagreeing is worth seeing rather than resolving.
        "best_conc_guard_holds": rows[best_idx]["guard_holds"],
        "median_speedup": round(median, 4),
        "mean_speedup": round(sum(speedups) / len(speedups), 4),
    }


def conc_pair_comparison(
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
    *,
    metric_key: str = "output_throughput",
    guard_noise_pct: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair curve points by CONC (outer join), compute per-conc speedup on *metric_key*, and aggregate.

    Under the interactivity objective each pair also reports whether throughput held within the noise band the
    session grades under. It is reported, not enforced: InferenceX publishes a 2-D frontier with no fixed
    interactivity target, so a rung that traded throughput for interactivity moved along that frontier rather than
    violating a constraint -- and a sweep exists to draw the frontier. Gating on the guard here would drop half the
    curve. The KEEP path enforces it because a stack promotion at one concurrency is a different question.
    """
    # The guard belongs to the interactivity objective; on the output axis there is no second axis to hold.
    guard_axis = GRADED_TOTAL if metric_key == GRADED_INTVTY else ""

    def _norm_conc(p: dict[str, Any]) -> int | float | str:
        raw = p.get("conc")
        if isinstance(raw, bool):
            return int(raw)
        return raw  # type: ignore[return-value]

    by_conc_b = {_norm_conc(p): p for p in baseline_points}
    by_conc_o = {_norm_conc(p): p for p in optimized_points}
    rows: list[dict[str, Any]] = []
    speedups: list[float] = []
    for c in sorted(
        set(by_conc_b) | set(by_conc_o), key=lambda x: (0, x) if isinstance(x, (int, float)) else (1, str(x))
    ):
        b = by_conc_b.get(c) or {}
        o = by_conc_o.get(c) or {}
        bt = to_float(b.get(metric_key))
        ot = to_float(o.get(metric_key))
        speedup = _speedup(bt, ot)
        delta_pct: float | None = None
        if speedup is not None:
            delta_pct = (speedup - 1.0) * 100.0
            speedups.append(speedup)
        guard = _read_guard(b, o, guard_axis, guard_noise_pct)
        rows.append(
            {
                "conc": c,
                # Named for the axis rather than for throughput: under the interactivity objective these hold a
                # slow-tail percentile, and ``summary.metric`` is what says which.
                "baseline_value": bt,
                "optimized_value": ot,
                "speedup": speedup,
                "delta_pct": delta_pct,
                "baseline_status": b.get("status"),
                "optimized_status": o.get("status"),
                # The guard axis beside the objective, so the frontier this rung sits on is readable rather than
                # only the one number it was ranked by. Null off the interactivity objective, and null when a side
                # did not measure the axis -- which is not the same as a rung that measured it and fell outside.
                "baseline_guard": guard.baseline,
                "optimized_guard": guard.optimized,
                "guard_holds": guard.holds,
            }
        )
    summary: dict[str, Any] = {
        "metric": metric_key,
        # The axis held beside the objective, empty off the interactivity objective. Named here so a reader of
        # ``best_conc_guard_holds`` does not have to infer which axis the verdict is about.
        "guard_axis": guard_axis,
        "successful_pairs": len(speedups),
        "failed_pairs": len(rows) - len(speedups),
        "best_conc": None,
        "best_speedup": None,
        "best_conc_guard_holds": None,
        "median_speedup": None,
        "mean_speedup": None,
    }
    if speedups:
        summary.update(_speedup_stats(rows, speedups))
    return rows, summary


__all__ = [
    "conc_pair_comparison",
    "gain_pct",
    "gain_pct_or_zero",
    "incremental_gain_pct",
]
