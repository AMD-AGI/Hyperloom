# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Predict the CUDA-graph-ON e2e gain from a CUDA-graph-DISABLED launch-bound share.

Why this exists (review P0): the diagnosis trace is captured with CUDA graphs
DISABLED so per-kernel launches are visible, but production decode runs with CUDA
graph ON, which already removes most launch/dispatch overhead. So a cgnone
``launch_bound_share`` (e.g. 0.35) massively OVERSTATES the real cg-ON headroom.
Every decode fusion measured this week landed at only +1.7% .. +5.3% e2e with
CUDA graph ON. This module converts the (upper-bound) share into a conservative
PREDICTED cg-ON gain that the candidate gate uses instead of the raw share.

The default is an intentionally conservative PRIOR. The predictor is pluggable and
learnable: pass measured ``(share, gain)`` points (or point the
``FORGE_FUSION_CALIBRATION`` env var at a JSON list) and prediction switches to
monotone interpolation over history, so it "gets smarter every campaign".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Conservative discount mapping a cgnone launch-bound share to a predicted cg-ON
# e2e gain. Calibrated against this week's ground truth: cg-ON serving gains sat in
# the low single digits while cgnone shares were ~0.25-0.45, i.e. only ~10-13% of
# the raw share survives CUDA-graph capture. This is a PRIOR; measured points, when
# provided, override it.
DEFAULT_SHARE_TO_GAIN_DISCOUNT = 0.13

# Memory channel (complements the flat discount above). Under CUDA-graph-ON the
# per-launch overhead is already removed, so the surviving fusion headroom is the
# HBM round-trips a fused kernel eliminates. A chain of tiny ops materializes each
# intermediate to HBM; fusing collapses those intermediate write+read round-trips.
# ``MEM_SAVED_FRACTION`` is the (conservative) fraction of the chain's MEASURED
# memory-traffic share that fusion removes -- unlike the 0.13 launch-share prior,
# it is applied to a REAL bytes signal, not a launch-time proxy.
DEFAULT_MEM_SAVED_FRACTION = 0.5

# Default acceptance bar for a fusion candidate (fraction). The user's bar is 3%.
DEFAULT_MIN_PREDICTED_GAIN = 0.03

_CALIBRATION_ENV = "FORGE_FUSION_CALIBRATION"


def _batch_factor(decode_batch: int) -> float:
    """Gain shrinks at larger decode batch (elementwise tail is a smaller share of
    the more GEMM-bound large-batch decode). ~1.0 at batch<=16, ~0.5 at batch 64,
    matching the measured GraniteMoE +4.7/3.5/2.4% and dense +2.2/1.9/0.7% trend.
    """
    b = max(1, int(decode_batch or 16))
    if b <= 16:
        return 1.0
    return max(0.35, (16.0 / b) ** 0.5)


def load_calibration_points(source: Optional[str] = None) -> list[tuple[float, float]]:
    """Load measured ``(share, gain)`` calibration points.

    Args:
        source: JSON file path; defaults to ``$FORGE_FUSION_CALIBRATION``. The file
            is a list of ``{"share": float, "gain": float}`` (or ``[share, gain]``)
            entries.

    Returns:
        Sorted ``[(share, gain), ...]`` (by share); empty when unavailable/invalid.
    """
    path = source or os.environ.get(_CALIBRATION_ENV, "").strip()
    if not path:
        return []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    points: list[tuple[float, float]] = []
    for row in data if isinstance(data, list) else []:
        try:
            if isinstance(row, dict):
                s, g = float(row["share"]), float(row["gain"])
            else:
                s, g = float(row[0]), float(row[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if s >= 0 and g >= 0:
            points.append((s, g))
    points.sort(key=lambda sg: sg[0])
    return points


def _interp(points: list[tuple[float, float]], share: float) -> float:
    """Monotone piecewise-linear interpolation of gain at ``share`` (clamped)."""
    if not points:
        return 0.0
    if share <= points[0][0]:
        return points[0][1]
    if share >= points[-1][0]:
        return points[-1][1]
    for (s0, g0), (s1, g1) in zip(points, points[1:]):
        if s0 <= share <= s1:
            t = 0.0 if s1 == s0 else (share - s0) / (s1 - s0)
            return g0 + t * (g1 - g0)
    return points[-1][1]


def predict_cuda_graph_on_gain(
    launch_bound_share: float,
    *,
    decode_batch: int = 16,
    calibration: Optional[list[tuple[float, float]]] = None,
    discount: float = DEFAULT_SHARE_TO_GAIN_DISCOUNT,
    mem_share: Optional[float] = None,
    mem_saved_fraction: float = DEFAULT_MEM_SAVED_FRACTION,
) -> float:
    """Predict the CUDA-graph-ON e2e gain (fraction) for a launch-bound share.

    Priority of signals (highest first):

    1. Measured ``(share, gain)`` calibration points (monotone interpolation) --
       ground truth, used as-is.
    2. The MEASURED memory channel (``mem_share``): when provided, the gain is
       grounded in the real fraction of GPU memory traffic the fused chain carries
       times ``mem_saved_fraction`` (the round-trips fusion removes). This replaces
       the flat 0.13 launch-share discount, which only crudely approximated the
       surviving cg-ON headroom.
    3. The conservative launch-share ``discount`` prior (legacy default).

    All routes are scaled by the decode-batch factor and never exceed the raw
    launch-bound share (the theoretical upper bound). Backward compatible: with
    ``mem_share=None`` and no calibration points, behavior is unchanged.

    Args:
        launch_bound_share: The cgnone launch-bound share (upper bound).
        decode_batch: Representative decode batch size.
        calibration: Optional measured ``(share, gain)`` points; when ``None`` the
            ``$FORGE_FUSION_CALIBRATION`` file is consulted.
        discount: Prior discount used when no calibration/memory signal exists.
        mem_share: MEASURED share of GPU memory traffic in the fused chain; when
            provided (not ``None``), grounds the prediction in memory saved.
        mem_saved_fraction: Fraction of that memory share fusion removes.

    Returns:
        Predicted cg-ON e2e gain as a fraction (e.g. ``0.04`` == +4%).
    """
    share = max(0.0, float(launch_bound_share))
    points = calibration if calibration is not None else load_calibration_points()
    if points:
        # Measured points are used as-is (they already encode the batch they were
        # captured at); do NOT re-apply the batch factor or we double-discount.
        return min(share, _interp(points, share))
    if mem_share is not None:
        m = max(0.0, float(mem_share))
        gain = m * max(0.0, mem_saved_fraction) * _batch_factor(decode_batch)
        # Cap by BOTH upper bounds: the launch-bound share (fusing cannot yield
        # more e2e than the fraction of time those ops occupy -- the documented
        # invariant) AND the chain's own measured memory traffic m (cannot save
        # more than it moves; guards mem_saved_fraction > 1).
        return min(share, m, gain)
    return min(share, share * discount * _batch_factor(decode_batch))
