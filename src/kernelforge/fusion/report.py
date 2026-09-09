# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 5: assemble the fixed JSON manifest (the Hyperloom handoff contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .models import CompilePassOutcome, Diagnosis, FusionArtifacts, Recipe, ValidationResult
from kernelforge.durable_io import atomic_write_text

# v2 widens the ``verdict`` enum and adds the ``error`` block.
FUSION_MANIFEST_SCHEMA_VERSION = 2

# Third verdict: the run could not ask the model, so it has no opinion about this kernel at all.
LLM_UNAVAILABLE_VERDICT = "llm_unavailable"


def build_manifest(
    *,
    framework: str,
    model_path: str,
    model_type: str,
    diagnosis: Diagnosis,
    recipe: Optional[Recipe],
    candidates: Optional[list[Recipe]] = None,
    validation: Optional[ValidationResult] = None,
    artifacts: Optional[FusionArtifacts] = None,
    loop: Optional[dict[str, Any]] = None,
    verdict_override: str = "",
    compile_pass: Optional[CompilePassOutcome] = None,
    error: Optional[dict[str, Any]] = None,
    patches: Optional[list[dict[str, Any]]] = None,
    nomination: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the JSON manifest dict."""
    verdict = verdict_override or ("candidate" if (diagnosis.is_candidate and recipe is not None) else "no_opportunity")
    return {
        "schema_version": FUSION_MANIFEST_SCHEMA_VERSION,
        # Manifest consumers key off this name; it stays even though the command is now `kernelforge forge-fuse`.
        "tool": "forge-fusion",
        "version": __version__,
        "verdict": verdict,
        "framework": framework,
        "model": {"path": model_path, "model_type": model_type},
        "diagnosis": diagnosis.to_dict(),
        "fusion": recipe.to_dict() if recipe is not None else None,
        "fusion_candidates": [c.to_dict() for c in (candidates or [])],
        "validation": validation.to_dict() if validation is not None else None,
        # A compile_pass claim is validated by a config + serving A/B, not by the kernel-level gates, so it carries
        # its own verdict.
        "compile_pass": compile_pass.to_dict() if compile_pass is not None else None,
        "fusion_loop": loop,
        "artifacts": artifacts.to_dict() if artifacts is not None else None,
        "error": dict(error) if error else None,
        # The nomination contract: N independent sibling patches.
        "patches": [dict(p) for p in patches] if patches is not None else None,
        "nomination": dict(nomination) if nomination else None,
    }


def write_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    """Write the manifest to ``<output_dir>/fusion_manifest.json``; return the path."""
    path = Path(output_dir) / "fusion_manifest.json"
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
