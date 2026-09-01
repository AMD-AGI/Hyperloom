# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TuningArtifactManifest: the provenance + coverage record shipped with a
tuned CSV (P0-A / WP-4).

A tuned CSV is only reusable under the exact conditions it was produced. This
manifest pins those conditions so a downstream consumer (Hyperloom engagement /
E2E gate) can decide reuse-vs-stale and prove how much of the target GEMM time
the artifact actually covers, instead of applying a bare CSV by model name.

Records: tool/version + generation time, tuning provenance (gpu/dtype/quant/tp/
lib), the source TraceShapeManifest linkage (trace/capture hashes, graph
variants, manifest hash), per-tuner micro results + CSV sha256, a weighted
ShapeCoverageFactor, and invalidation keys. Pure stdlib; unit-testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__, shape_manifest as _sm
from .utils import sha256_file

TUNING_ARTIFACT_SCHEMA_VERSION = 1


def _source_manifest_block(shape_manifest_path: str | Path | None) -> tuple[dict[str, Any], dict | None]:
    """Return (source_manifest_block, loaded_manifest_or_None).

    Loads the input TraceShapeManifest (if supplied) for trace provenance and
    coverage denominators; degrades to ``{"present": False}`` on absence/error.
    """
    if not shape_manifest_path:
        return {"present": False}, None
    try:
        m = _sm.load_manifest(shape_manifest_path)
    except (OSError, ValueError):
        return {"present": False, "path": str(shape_manifest_path), "error": "unreadable_or_invalid"}, None
    gen = m.get("generated_from") or {}
    workload = m.get("workload") or {}
    block = {
        "present": True,
        "path": str(shape_manifest_path),
        "manifest_hash": m.get("manifest_hash", ""),
        "tracelens_revision": gen.get("tracelens_revision"),
        "main_trace_hash": gen.get("main_trace_hash", ""),
        "capture_trace_hashes": gen.get("capture_trace_hashes", {}),
        "graph_variants": sorted((workload.get("variant_steady_replay") or {}).keys()),
        "total_target_gemm_us": workload.get("total_target_gemm_us"),
    }
    return block, m


def _coverage_block(manifest: dict | None, results: list) -> dict[str, Any]:
    """Weighted ShapeCoverageFactor = improved-target GEMM weight / total target
    GEMM weight, using the source manifest's per-(M,N,K) steady-state weights.

    "Covered" = a shape the tuner actually improved (i.e. produced an applicable
    tuned config); a no-improvement shape keeps the default and is not counted.
    Returns nulls (not a fabricated number) when no source manifest is present.
    """
    if manifest is None:
        return {"shape_coverage_factor": None, "note": "no source manifest supplied"}
    shapes = _sm.manifest_to_shapes(manifest, target_only=True)
    weight_by_key = {(s["M"], s["N"], s["K"]): float(s["weight"]) for s in shapes}
    total_weight = sum(weight_by_key.values())
    improved_keys: set[tuple[int, int, int]] = set()
    for r in results:
        for sr in getattr(r, "shape_results", None) or []:
            if sr.get("improved"):
                improved_keys.add((sr.get("M"), sr.get("N"), sr.get("K")))
    covered = sum(w for k, w in weight_by_key.items() if k in improved_keys)
    return {
        "shape_coverage_factor": round(covered / total_weight, 4) if total_weight > 0 else None,
        "covered_target_weight": round(covered, 3),
        "total_target_weight": round(total_weight, 3),
        "target_shape_count": len(weight_by_key),
        "improved_shape_count": len(improved_keys & set(weight_by_key)),
    }


def build_artifact_manifest(
    report: Any,
    results: list,
    *,
    shape_manifest_path: str | Path | None = None,
    gpu_type: str = "",
    framework: str = "",
    precision: str = "",
    quant_type: str = "",
    tp: int = 1,
    tuner_lib_version: str = "",
    generated_at: str = "",
) -> dict[str, Any]:
    """Assemble the TuningArtifactManifest dict from a TuneReport + results."""
    source_block, manifest = _source_manifest_block(shape_manifest_path)

    tuners: list[dict[str, Any]] = []
    for r in results:
        tuners.append(
            {
                "tuner": r.tuner_name,
                "status": r.status,
                "backend_env": r.env_var,
                "artifact_path": r.artifact_path,
                "csv_sha256": sha256_file(r.artifact_path),
                "total_shapes": r.total_shapes,
                "improved_shapes": r.improved_shapes,
                "best_micro_speedup": round(r.best_micro_speedup, 4),
                "avg_micro_speedup": round(r.avg_micro_speedup, 4),
                "shape_results": r.shape_results,
            }
        )

    return {
        "schema_version": TUNING_ARTIFACT_SCHEMA_VERSION,
        # Stable artifact identifier, deliberately not renamed when the tuner
        # folded into the one forge CLI: it keys already-written manifests, and
        # the invocation it once named is recorded by "version" + schema_version.
        "tool": "forge-gemm-tune",
        "version": __version__,
        "generated_at": generated_at,
        "micro_decision": getattr(report, "micro_decision", ""),
        "requires_e2e_validation": getattr(report, "requires_e2e_validation", True),
        "provenance": {
            "gpu_type": gpu_type or None,
            "framework": framework or None,
            "precision": precision or None,
            "quant_type": quant_type or None,
            "tp": tp,
            "tuner_lib_version": tuner_lib_version or None,
        },
        "recommended_env": dict(getattr(report, "recommended_env", {}) or {}),
        "source_manifest": source_block,
        "coverage": _coverage_block(manifest, results),
        "tuners": tuners,
        "invalidation": {
            "note": "Reuse the tuned CSV only when all of these match the target run.",
            "keys": [
                "provenance.gpu_type",
                "provenance.precision",
                "provenance.quant_type",
                "provenance.tp",
                "provenance.tuner_lib_version",
                "source_manifest.manifest_hash",
            ],
        },
    }


def write_artifact_manifest(
    report: Any,
    results: list,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Write the TuningArtifactManifest to ``<output_dir>/tuning_artifact_manifest.json``."""
    manifest = build_artifact_manifest(report, results, **kwargs)
    out = Path(output_dir) / "tuning_artifact_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return out


__all__ = [
    "TUNING_ARTIFACT_SCHEMA_VERSION",
    "build_artifact_manifest",
    "write_artifact_manifest",
]
