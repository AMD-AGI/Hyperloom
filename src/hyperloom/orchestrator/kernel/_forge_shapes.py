# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shapes JSON and token list a forge tuning run is pointed at."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .gemm_shape_coverage import canonical_dtype, traced_gemm_shapes

log = logging.getLogger(__name__)


def _is_forge_compatible_shapes_json(path: Path) -> bool:
    """Validate that a shapes JSON file matches forge's expected format.

    Forge expects: [{"M": int, "N": int, "K": int}, ...]
    or {"shapes": [{"M": int, "N": int, "K": int}, ...]}
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("shapes", [])
        if not isinstance(data, list) or not data:
            return False
        sample = data[0]
        if not isinstance(sample, dict):
            return False
        # Must have M/N/K keys (case-insensitive).
        keys = {k.upper() for k in sample}
        return {"M", "N", "K"}.issubset(keys)
    except (json.JSONDecodeError, OSError, TypeError):
        return False


def _resolve_forge_shapes(state, *, precision: str = "") -> str:
    """Find TraceLens shapes JSON if available and in forge-compatible format.

    Forge dense tuners expect: [{"M": int, "N": int, "K": int}, ...]
    Only passes files that match this schema; incompatible formats are
    silently skipped so forge falls back to config.json shape derivation.

    ``precision`` scopes the traced shapes to the dtype the tuner will actually
    tune (a trace carries every GEMM dtype the model runs, e.g. FP8 projections
    alongside BF16 router heads). Empty means "no dtype scoping".

    When scoping is requested the candidate extraction runs first, because it is
    the only source whose dtype can be checked: a pre-rendered shapes artifact is
    a bare ``[{M,N,K}]`` list carrying no dtype or provenance, so an artifact
    recorded for another dtype would otherwise be handed to the tuner ahead of
    correctly-scoped shapes. Artifacts stay the fallback for the unscoped case
    and for when the trace yields nothing for this precision. A profile that does
    not match the active workload/config contributes nothing.
    """
    if not state.profile_trace_matches_workload():
        log.info(
            "Forge GEMM shapes: latest TraceLens profile does not match the "
            "active workload/config; ignoring its shape artifacts"
        )
        return ""
    last_trace = getattr(state, "last_trace_analyze", None) or {}
    if not isinstance(last_trace, dict):
        return ""

    candidates: list[str] = []

    # Prefer explicit artifact fields when TraceLens exposes them.
    artifact_paths = last_trace.get("artifact_paths")
    for source, keys in (
        (last_trace, ("shapes_json", "shapes_path")),
        (artifact_paths if isinstance(artifact_paths, dict) else {}, ("shapes_json", "shapes", "gemm_shapes_json")),
    ):
        for key in keys:
            raw = str(source.get(key) or "").strip()
            if raw:
                candidates.append(raw)
    # Fallback: check beside candidates_path.
    candidates_path_str = last_trace.get("candidates_path") or ""
    if candidates_path_str:
        cand_file = Path(candidates_path_str)
        if cand_file.is_file():
            shapes_file = cand_file.parent / "shapes.json"
            candidates.append(str(shapes_file))

    # Extract the GEMM shapes observed by the latest TraceLens analysis. Older
    # traces can describe a backend that is no longer active.
    def _extracted() -> str:
        return _extract_gemm_shapes_from_candidates(str(last_trace.get("candidates_path") or ""), precision=precision)

    if canonical_dtype(precision):
        scoped = _extracted()
        if scoped:
            return scoped

    for candidate in candidates:
        p = Path(candidate)
        if p.is_file() and _is_forge_compatible_shapes_json(p):
            return str(p)

    return _extracted()


def _extract_gemm_shapes_from_candidates(candidates_path_str: str, *, precision: str = "") -> str:
    """Extract M,N,K from kernel_candidates.json hot_kernels input_shapes.

    Derives the GEMM dimensions actually observed during serving and writes a
    forge-compatible shapes JSON beside the candidates file, returning its path.

    ``precision`` scopes the result to one traced dtype. A trace records every
    GEMM the model runs, so an FP8 tuner would otherwise also receive the BF16
    router/head shapes; those rows are never looked up at serve time and they
    displace real FP8 shapes in the call-count ordering below. Empty keeps every
    dtype (historical behaviour).
    """
    import json as _json

    if not candidates_path_str:
        return ""
    cand_file = Path(candidates_path_str)
    if not cand_file.is_file():
        return ""

    try:
        data = _json.loads(cand_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    hot_kernels = data.get("hot_kernels", [])
    if not isinstance(hot_kernels, list):
        return ""

    # ``weight`` is the observed call count: decode GEMMs are invoked far more
    # often than prefill ones, so ordering by it puts the throughput-dominant
    # shapes first and they still get tuned when the tuner runs out of budget.
    # The same (M,N,K) can be reported by several kernels; keep the largest
    # count, otherwise a rare first sighting would outrank the hot one.
    weights: dict[tuple[int, int, int], int] = {}
    order: dict[tuple[int, int, int], int] = {}

    for kernel in hot_kernels:
        for key, weight in traced_gemm_shapes(kernel, precision=precision):
            if key not in order:
                order[key] = len(order)
            weights[key] = max(weights.get(key, 0), weight)

    if not weights:
        return ""

    # Most-called first; ties keep discovery order so output stays deterministic.
    ranked = sorted(weights, key=lambda key: (-weights[key], order[key]))
    shapes = [{"M": M, "N": N, "K": K} for M, N, K in ranked]

    out_path = cand_file.parent / "traced_gemm_shapes.json"
    try:
        out_path.write_text(_json.dumps(shapes, indent=2), encoding="utf-8")
    except OSError:
        return ""

    log.info(
        "extracted %d unique GEMM shapes from kernel_candidates.json -> %s",
        len(shapes),
        out_path,
    )
    return str(out_path)
