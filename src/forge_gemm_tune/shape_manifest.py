# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consume a TraceShapeManifest (Hyperloom WP-1) as a weighted GEMM-shape source.

The manifest is the model-agnostic, variant-discriminating, replay-weighted
artifact produced by Hyperloom's bypass trace analysis. This module turns it
into the ``M,N,K`` (+ optional ``q_dtype_w``) untuned CSV the aiter dense tuners
already consume, selecting the tuner-addressable (``is_target_gemm``) rows and
ordering them by steady-state GPU-time weight so the highest-impact shapes are
tuned first.

It is intentionally additive: nothing here runs unless ``--shapes-manifest`` is
supplied. Pure stdlib; no GPU or aiter dependency, so it is unit-testable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_KIND = "trace_shape_manifest"


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a TraceShapeManifest JSON file.

    Raises ``ValueError`` when the file is not a trace shape manifest so a
    caller does not silently tune off an unrelated JSON blob.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError(
            f"{path} is not a {MANIFEST_KIND} (manifest_kind={data.get('manifest_kind') if isinstance(data, dict) else type(data).__name__!r})"
        )
    return data


def _q_dtype_w(in_dtype: str | None) -> str:
    """Map a manifest input dtype to aiter's ``q_dtype_w`` weight-quant token."""
    t = (in_dtype or "").lower()
    if "e5m2" in t:
        return "torch.float8_e5m2fnuz"
    return "torch.float8_e4m3fnuz"


def _row_weight(row: dict[str, Any], variant_steady_replay: dict[str, Any]) -> float:
    """Steady-state GPU-time weight for a manifest row.

    ``cum_gpu_us`` is the per-window time. For ``capture_only`` rows (structure
    recovered from a CUDA-graph capture shard, single-shot capture-time cost) we
    scale by the variant's steady replay count when it is known; when it is not
    (``variant_steady_replay`` null, e.g. multi-variant unresolved) we keep the
    capture-time cost as a relative-ranking proxy and never fabricate a steady
    number. Eager rows are already steady per-iteration (replay 1).
    """
    w = float(row.get("cum_gpu_us", 0.0) or 0.0)
    if row.get("capture_only"):
        r = variant_steady_replay.get(row.get("graph_variant"))
        if isinstance(r, (int, float)) and r > 0:
            w *= float(r)
    return w


def manifest_to_shapes(
    manifest: dict[str, Any],
    *,
    target_only: bool = True,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return GEMM shapes from a manifest, deduped by (M,N,K), weight-ordered.

    Args:
        manifest: A loaded TraceShapeManifest dict.
        target_only: Keep only tuner-addressable rows (``is_target_gemm``).
        top_k: Optional cap on the number of shapes (highest weight first). The
            caller is responsible for logging when it truncates.

    Returns:
        A list of ``{"M","N","K","weight","quant","in_dtype"}`` dicts, sorted by
        descending steady-state weight. Rows without a full integer (M,N,K) are
        dropped (a GEMM cannot be tuned without its dims).
    """
    rows = manifest.get("rows") or []
    workload = manifest.get("workload") or {}
    vsr = workload.get("variant_steady_replay") or {}
    agg: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        if target_only and not row.get("is_target_gemm"):
            continue
        dims = row.get("dims") or {}
        m, n, k = dims.get("M"), dims.get("N"), dims.get("K")
        if not (isinstance(m, int) and isinstance(n, int) and isinstance(k, int)):
            continue
        if m <= 0 or n <= 0 or k <= 0:
            continue
        key = (m, n, k)
        w = _row_weight(row, vsr)
        existing = agg.get(key)
        if existing is None:
            agg[key] = {
                "M": m,
                "N": n,
                "K": k,
                "weight": round(w, 3),
                "quant": row.get("quant", "") or "",
                "in_dtype": row.get("in_dtype", "") or "",
            }
        else:
            existing["weight"] = round(existing["weight"] + w, 3)
    shapes = sorted(agg.values(), key=lambda s: s["weight"], reverse=True)
    if top_k and top_k > 0 and len(shapes) > top_k:
        shapes = shapes[:top_k]
    return shapes


def write_manifest_untuned_csv(
    path: str | Path,
    work_dir: str | Path,
    *,
    needs_q_dtype_w: bool = False,
    target_only: bool = True,
    top_k: int | None = None,
) -> Path | None:
    """Load a manifest and write an aiter-compatible untuned CSV.

    Output columns match the existing dense-tuner contract (``M,N,K`` or
    ``M,N,K,q_dtype_w``), rows ordered by descending weight. Returns the CSV
    path, or ``None`` when the manifest yields no usable target GEMM shapes.
    """
    manifest = load_manifest(path)
    shapes = manifest_to_shapes(manifest, target_only=target_only, top_k=top_k)
    if not shapes:
        log.warning("shape_manifest: %s yielded no tunable target GEMM shapes", path)
        return None
    out = Path(work_dir) / "untuned_manifest.csv"
    with out.open("w", encoding="utf-8") as f:
        if needs_q_dtype_w:
            f.write("M,N,K,q_dtype_w\n")
            for s in shapes:
                f.write(f"{s['M']},{s['N']},{s['K']},{_q_dtype_w(s.get('in_dtype'))}\n")
        else:
            f.write("M,N,K\n")
            for s in shapes:
                f.write(f"{s['M']},{s['N']},{s['K']}\n")
    log.info(
        "shape_manifest: wrote %d target GEMM shape(s) from %s -> %s (weight-ordered)",
        len(shapes),
        path,
        out,
    )
    return out


__all__ = [
    "MANIFEST_KIND",
    "load_manifest",
    "manifest_to_shapes",
    "write_manifest_untuned_csv",
]
