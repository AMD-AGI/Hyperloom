###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-fusion opportunity analysis for the bypass analysis backend.

The candidate/roofline artifacts answer *what* kernels exist (name-aggregated).
Fusion needs *relationships*: which small kernels run back-to-back and could be
fused into one launch. This module derives that from the time-ordered per-launch
device sequence (which the reader otherwise aggregates away):

* ``fusable_clusters``: maximal runs of CONSECUTIVE launches in point-wise /
  memory-bound categories (Elementwise / Normalization / Quantization /
  KVCacheStore) — the classic fusion targets (elementwise chains, norm+quant).
  Each cluster reports its members, launch count, aggregate device time, and the
  inter-kernel gap it would remove.
* ``adjacent_pairs``: recurring (categoryA -> categoryB) transitions, surfacing
  systematic fusion patterns across the whole run.

Independent + GPU-free; consumes only the trace's per-launch (name, op_name,
category, ts, dur) tuples.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# Point-wise / memory-bound categories that fuse well (chain -> one launch).
_FUSABLE_CATEGORIES = frozenset({"Elementwise", "Normalization", "Quantization", "KVCacheStore"})


def _summarize_cluster(run: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one consecutive fusable run into a cluster record."""
    names = [str(r.get("op_name") or r.get("name") or "") for r in run]
    cats = [str(r.get("category") or "") for r in run]
    total_dur = sum(float(r.get("dur") or 0.0) for r in run)
    start = min(float(r.get("ts") or 0.0) for r in run)
    end = max(float(r.get("ts") or 0.0) + float(r.get("dur") or 0.0) for r in run)
    span = max(0.0, end - start)
    # Inter-kernel time a fused launch would remove.
    gap = max(0.0, span - total_dur)
    return {
        "launch_count": len(run),
        "categories": sorted(set(cats)),
        "members": [{"name": names[i], "category": cats[i], "dur_us": round(float(run[i].get("dur") or 0.0), 3)} for i in range(len(run))],
        "distinct_kernels": sorted(set(names)),
        "aggregate_dur_us": round(total_dur, 3),
        "span_us": round(span, 3),
        "inter_kernel_gap_us": round(gap, 3),
        "dominant_category": Counter(cats).most_common(1)[0][0] if cats else "",
    }


def fusable_clusters(launches: list[dict[str, Any]], *, min_len: int = 2, top_k: int = 0) -> list[dict[str, Any]]:
    """Find maximal runs of consecutive fusable-category launches.

    Args:
        launches: Time-ordered per-launch dicts (``name``/``op_name``/``category``
            /``ts``/``dur``). MUST already be sorted by ``ts``.
        min_len: Minimum consecutive launches to count as a cluster (>=2).
        top_k: Cap on returned clusters (0 = all), ranked by aggregate time.

    Returns:
        Cluster records ranked by ``aggregate_dur_us`` descending.
    """
    clusters: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    for lc in launches:
        if str(lc.get("category") or "") in _FUSABLE_CATEGORIES:
            run.append(lc)
        else:
            if len(run) >= min_len:
                clusters.append(_summarize_cluster(run))
            run = []
    if len(run) >= min_len:
        clusters.append(_summarize_cluster(run))
    clusters.sort(key=lambda c: c["aggregate_dur_us"], reverse=True)
    return clusters[:top_k] if top_k and top_k > 0 else clusters


def adjacent_pairs(launches: list[dict[str, Any]], *, top_n: int = 20) -> list[dict[str, Any]]:
    """Count recurring (categoryA -> categoryB) consecutive transitions.

    Args:
        launches: Time-ordered per-launch dicts (see :func:`fusable_clusters`).
        top_n: Max transitions to return, most frequent first.

    Returns:
        ``[{from, to, count}]`` ranked by count descending.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for a, b in zip(launches, launches[1:]):
        ca = str(a.get("category") or "")
        cb = str(b.get("category") or "")
        if ca and cb:
            pairs[(ca, cb)] += 1
    ranked = pairs.most_common(top_n if top_n and top_n > 0 else None)
    return [{"from": a, "to": b, "count": n} for (a, b), n in ranked]


def analyze_fusion(launches: list[dict[str, Any]], *, top_k_clusters: int = 20) -> dict[str, Any]:
    """Build the fusion analysis payload from a time-ordered launch sequence.

    Args:
        launches: Time-ordered per-launch dicts (``name``/``op_name``/``category``
            /``ts``/``dur``).
        top_k_clusters: Cap on returned fusable clusters.

    Returns:
        ``{launch_count, fusable_clusters, fusable_cluster_count,
        fusable_time_us, adjacent_pairs}``. ``fusable_cluster_count`` /
        ``fusable_time_us`` are totals over ALL clusters; ``fusable_clusters`` is
        capped to the ``top_k_clusters`` largest (list-size bound only).
    """
    # Totals over ALL clusters; the returned LIST is capped to top_k.
    all_clusters = fusable_clusters(launches)
    fusable_time = sum(c["aggregate_dur_us"] for c in all_clusters)
    listed = all_clusters[:top_k_clusters] if top_k_clusters and top_k_clusters > 0 else all_clusters
    return {
        "launch_count": len(launches),
        "fusable_cluster_count": len(all_clusters),
        "fusable_time_us": round(fusable_time, 3),
        "fusable_clusters": listed,
        "adjacent_pairs": adjacent_pairs(launches),
    }


__all__ = ["analyze_fusion", "fusable_clusters", "adjacent_pairs"]
