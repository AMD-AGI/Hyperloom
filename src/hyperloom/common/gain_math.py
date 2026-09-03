# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Throughput percentage-gain helpers (``gain_math``). Stdlib-only."""

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
    metric_key: str = "output_throughput",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair curve points by CONC (outer join), compute per-conc speedup, and aggregate.

    Shared by the conc-sweep post-hook and the breakdown collector, which must
    produce byte-identical rows/summary from the same curves on the same
    ``metric_key`` -- the collector reads its key off the report it recovers
    for. Stdlib-only so the collector never drags in Magpie/torch at report
    time.

    Args:
        baseline_points: Curve rows for the baseline arm.
        optimized_points: Curve rows for the optimized arm.
        metric_key: The point field the speedup is measured on. It has to be
            the axis the session is ranked by, or the summary reports a
            different quantity from the curve drawn beside it.

    Returns:
        A tuple of ``(per_conc_rows, summary_dict)``; the summary names the
        metric it used.
    """

    def _norm_conc(p: dict[str, Any]) -> int | float | str:
        raw = p.get("conc")
        if isinstance(raw, bool):
            return int(raw)
        return raw  # type: ignore[return-value]

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
        bt = to_float(b.get(metric_key))
        ot = to_float(o.get(metric_key))
        speedup: float | None = None
        delta_pct: float | None = None
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
            }
        )
    summary: dict[str, Any] = {
        "metric": metric_key,
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs,
        "best_conc": None,
        "best_speedup": None,
        "median_speedup": None,
        "mean_speedup": None,
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
