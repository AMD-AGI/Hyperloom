# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Throughput percentage-gain helpers (tree-reform.MD §7 — ``gain_math``).

Relocated from ``inference_optimizer.orchestrator.gain_math`` (which now
re-exports from here). One home per edge contract; stdlib-only.
"""

from __future__ import annotations


def gain_pct(new: float | None, base: float) -> float | None:
    """``(new-base)/base*100``; None when *new* is not positive-numeric or *base*<=0."""
    if not isinstance(new, (int, float)) or new <= 0 or base <= 0:
        return None
    return (float(new) - base) / base * 100.0


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


__all__ = ["gain_pct", "gain_pct_or_zero", "incremental_gain_pct"]
