# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Throughput percentage-gain helpers (``gain_math``).

Default paths stay stdlib-only so the report-time collector does not import
Magpie/torch. The optional ``use_composite`` branch of
:func:`conc_pair_comparison` lazily imports :mod:`hyperloom.common.perf_metric`.
"""

from __future__ import annotations

from typing import Any

from hyperloom.common.coerce import to_float


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
    use_composite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair curve points by CONC (outer join), compute per-conc speedup, and aggregate.

    Shared by the conc-sweep post-hook and the breakdown collector, which must
    produce byte-identical rows/summary from the same curves. Default ranking
    is output-tput ratio and stays stdlib-only. When *use_composite* is true
    and both arms have a full perf triple, speedup is ``1+S`` (``delta_pct``
    is ``S*100``) with per-pair output-tput fallback.

    Args:
        baseline_points: Curve rows for the baseline arm.
        optimized_points: Curve rows for the optimized arm.
        use_composite: Rank on composite score *S* when both points have a
            full triple. Default ``False`` (output-tput ratio).

    Returns:
        A tuple of ``(per_conc_rows, summary_dict)``.
    """

    def _norm_conc(p: dict[str, Any]) -> int | float | str:
        raw = p.get("conc")
        if isinstance(raw, bool):
            return int(raw)
        return raw  # type: ignore[return-value]

    score_fn = None
    snap_fn = None
    if use_composite:
        from hyperloom.common.perf_metric import composite_score, perf_snapshot_from_mapping

        score_fn = composite_score
        snap_fn = perf_snapshot_from_mapping

    by_conc_b = {_norm_conc(p): p for p in baseline_points}
    by_conc_o = {_norm_conc(p): p for p in optimized_points}
    rows: list[dict[str, Any]] = []
    speedups: list[float] = []
    successful_pairs = 0
    failed_pairs = 0
    for c in sorted(
        set(by_conc_b) | set(by_conc_o), key=lambda x: (0, x) if isinstance(x, (int, float)) else (1, str(x))
    ):
        b = by_conc_b.get(c) or {}
        o = by_conc_o.get(c) or {}
        bt = to_float(b.get("output_throughput"))
        ot = to_float(o.get("output_throughput"))
        speedup: float | None = None
        delta_pct: float | None = None
        used_composite = False
        if score_fn is not None and snap_fn is not None:
            b_snap = snap_fn(b)
            o_snap = snap_fn(o)
            if b_snap is not None and o_snap is not None:
                score = score_fn(o_snap, b_snap)
                speedup = 1.0 + score
                delta_pct = score * 100.0
                used_composite = True
                speedups.append(speedup)
                successful_pairs += 1
        if speedup is None:
            if bt is not None and bt > 0 and ot is not None and ot > 0:
                speedup = ot / bt
                delta_pct = (speedup - 1.0) * 100.0
                speedups.append(speedup)
                successful_pairs += 1
            else:
                failed_pairs += 1
        rows.append(
            {
                "conc": c,
                "baseline_tput": bt,
                "optimized_tput": ot,
                "speedup": speedup,
                "delta_pct": delta_pct,
                "baseline_status": b.get("status"),
                "optimized_status": o.get("status"),
                "used_composite": used_composite,
            }
        )
    summary: dict[str, Any] = {
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs,
        "best_conc": None,
        "best_speedup": None,
        "median_speedup": None,
        "mean_speedup": None,
        "metric": "composite_v1" if use_composite else "output_throughput",
    }
    if speedups:
        best_idx, best_val = max(
            ((i, r["speedup"]) for i, r in enumerate(rows) if isinstance(r.get("speedup"), float)),
            key=lambda x: x[1],
        )
        sorted_sp = sorted(speedups)
        n = len(sorted_sp)
        median = sorted_sp[n // 2] if n % 2 == 1 else 0.5 * (sorted_sp[n // 2 - 1] + sorted_sp[n // 2])
        summary.update(
            {
                "best_conc": rows[best_idx]["conc"],
                "best_speedup": round(best_val, 4),
                "median_speedup": round(median, 4),
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
