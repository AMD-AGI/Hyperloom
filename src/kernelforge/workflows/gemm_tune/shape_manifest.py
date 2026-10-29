# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consume a TraceShapeManifest (Hyperloom WP-1) as a weighted GEMM-shape source."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_KIND = "trace_shape_manifest"


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a TraceShapeManifest JSON file."""
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
    """Steady-state GPU-time weight for a manifest row."""
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
    """Return GEMM shapes from a manifest, deduped by (M,N,K), weight-ordered."""
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
    """Load a manifest and write an aiter-compatible untuned CSV."""
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
