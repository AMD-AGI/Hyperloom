# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Throughput percentage-gain helpers (``gain_math``). Stdlib-only."""

from __future__ import annotations

import statistics
from typing import Any

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
    by_conc_b = {p.get("conc"): p for p in baseline_points}
    by_conc_o = {p.get("conc"): p for p in optimized_points}
    rows: list[dict[str, Any]] = []
    for c in sorted(
        set(by_conc_b) | set(by_conc_o), key=lambda x: (0, x) if isinstance(x, (int, float)) else (1, str(x))
    ):
        b = by_conc_b.get(c) or {}
        o = by_conc_o.get(c) or {}
        bt = to_float(b.get(metric_key))
        ot = to_float(o.get(metric_key))
        speedup = ot / bt if bt is not None and bt > 0 and ot is not None and ot > 0 else None
        # ``graded_axes_of`` is what normalises the sweep's ``total_token_throughput`` onto GRADED_TOTAL, so the
        # guard cannot read a differently-named axis as an absent one.
        base_axes = graded_axes_of(b)
        opt_axes = graded_axes_of(o)
        guard_holds: bool | None = None
        if base_axes.get(guard_axis) and opt_axes.get(guard_axis):
            guard_holds = holds_within_band(opt_axes, base_axes, guard_axis, noise_pct=guard_noise_pct)
        rows.append(
            {
                "conc": c,
                # Named for the axis rather than for throughput: under the interactivity objective these hold a
                # slow-tail percentile, and ``summary.metric`` is what says which.
                "baseline_value": bt,
                "optimized_value": ot,
                "speedup": speedup,
                "delta_pct": None if speedup is None else (speedup - 1.0) * 100.0,
                "baseline_status": b.get("status"),
                "optimized_status": o.get("status"),
                # The guard axis beside the objective, so the frontier this rung sits on is readable rather than
                # only the one number it was ranked by. Null off the interactivity objective, and null when a side
                # did not measure the axis -- which is not the same as a rung that measured it and fell outside.
                "baseline_guard": base_axes.get(guard_axis),
                "optimized_guard": opt_axes.get(guard_axis),
                "guard_holds": guard_holds,
            }
        )
    paired = [r for r in rows if r["speedup"] is not None]
    summary: dict[str, Any] = {
        "metric": metric_key,
        # The axis held beside the objective, empty off the interactivity objective. Named here so a reader of
        # ``best_conc_guard_holds`` does not have to infer which axis the verdict is about.
        "guard_axis": guard_axis,
        "successful_pairs": len(paired),
        "failed_pairs": len(rows) - len(paired),
        **dict.fromkeys(("best_conc", "best_speedup", "best_conc_guard_holds", "median_speedup", "mean_speedup")),
    }
    if paired:
        best = max(paired, key=lambda r: r["speedup"])
        speedups = [r["speedup"] for r in paired]
        summary.update(
            {
                "best_conc": best["conc"],
                "best_speedup": round(best["speedup"], 4),
                # The headline rung is the best on the objective alone. Whether the session's own KEEP rule would
                # have accepted it is a second fact, and the two disagreeing is worth seeing rather than resolving.
                "best_conc_guard_holds": best["guard_holds"],
                "median_speedup": round(statistics.median(speedups), 4),
                "mean_speedup": round(sum(speedups) / len(speedups), 4),
            }
        )
    return rows, summary


__all__ = [
    "conc_pair_comparison",
    "gain_pct",
    "gain_pct_or_zero",
    "incremental_gain_pct",
]
