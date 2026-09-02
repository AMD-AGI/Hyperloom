# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 5: assemble the fixed JSON manifest (the Hyperloom handoff contract).

The manifest is the stable machine-readable output of a forge-fuse run. In
dry-run (Phase 1) ``validation`` and ``artifacts`` are null; a full run fills them
with the kernel-level parity/speedup and the emitted kernel + framework-wiring
patch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .models import CompilePassOutcome, Diagnosis, FusionArtifacts, Recipe, ValidationResult
from kernelforge.durable_io import atomic_write_text

# v2 widens the ``verdict`` enum and adds the ``error`` block. Adding a value to
# an enum is not additive for a consumer that switches exhaustively on it, so the
# version moves even though every v1 field kept its name, type and meaning:
#
#   v1 -> v2
#     verdict: {candidate, no_opportunity} -> + llm_unavailable
#     error:   (absent)                    -> object | null, null on every
#                                            verdict except llm_unavailable
#
# A v2 reader handles v1 payloads unchanged (a missing ``error`` reads as null).
# A v1 reader that only reads known keys and treats an unrecognized verdict as
# "not a KEEP" is unaffected; one that asserts ``verdict in {...}`` must be
# updated. The only in-tree consumer, Hyperloom's ``agents/kernel/tools/
# kernelforge.fusion.py`` wrapper, ignores ``schema_version``, derives ``kept`` from
# ``fusion_loop``/``validation`` rather than from the verdict, and passes the
# verdict through verbatim — so it reads v2 without changes; it needs a change
# only to STOP mapping an outage onto ``no_improvement``/REVERT, which is the
# point of the new verdict.
FUSION_MANIFEST_SCHEMA_VERSION = 2

# Third verdict: the run could not ask the model, so it has no opinion about
# this kernel at all. Kept distinct from ``no_opportunity`` because a consumer
# that cannot tell them apart reports an outage as an optimization result.
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
    """Assemble the JSON manifest dict.

    ``verdict`` is ``candidate`` when the trace is launch-bound AND a fusion
    pattern was localized into a recipe, and ``no_opportunity`` when the run
    looked and found none.

    A run that could not look — discovery never reached the model — must pass
    ``verdict_override=LLM_UNAVAILABLE_VERDICT`` together with ``error``. The
    two-state expression below cannot express that case: with
    ``diagnosis.is_candidate`` true and ``recipe`` None it falls to
    ``no_opportunity``, which reads as "this model has no fusion opportunity"
    when what actually happened is that the gateway was down.

    ``fusion`` is the top (selected) recipe; ``fusion_candidates`` lists every
    localized recipe (ranked) for caller visibility. ``error`` is null on every
    normal run.

    ``patches`` is the nomination envelope: one independent sibling patch per kept
    recipe (design §3.3), each a dict with ``kernel_name`` / ``patch_path`` /
    ``target_file`` / ``kernel_repo`` / ``snapshot_dir`` / ``base_commit`` /
    ``micro_speedup``. It is present (possibly empty) on the multi-patch path and
    omitted (``None``) on the single-combined-patch (combine) path so a legacy
    consumer reading only ``artifacts`` is byte-unaffected. ``nomination`` is the
    round's summary counts (``candidates_seen`` / ``resolved`` / ``selected``).
    """
    verdict = verdict_override or ("candidate" if (diagnosis.is_candidate and recipe is not None) else "no_opportunity")
    return {
        "schema_version": FUSION_MANIFEST_SCHEMA_VERSION,
        # Manifest consumers key off this name; it stays even though the command
        # is now `kernelforge forge-fuse`.
        "tool": "forge-fusion",
        "version": __version__,
        "verdict": verdict,
        "framework": framework,
        "model": {"path": model_path, "model_type": model_type},
        "diagnosis": diagnosis.to_dict(),
        "fusion": recipe.to_dict() if recipe is not None else None,
        "fusion_candidates": [c.to_dict() for c in (candidates or [])],
        "validation": validation.to_dict() if validation is not None else None,
        # A compile_pass claim is validated by a config + serving A/B, not by the
        # kernel-level gates, so it carries its own verdict. Consumers must read
        # this rather than inferring "unvalidated" from a null ``validation``.
        "compile_pass": compile_pass.to_dict() if compile_pass is not None else None,
        "fusion_loop": loop,
        "artifacts": artifacts.to_dict() if artifacts is not None else None,
        "error": dict(error) if error else None,
        # The nomination contract: N independent sibling patches. None on the
        # combine path keeps the legacy single-``artifacts`` shape byte-identical.
        "patches": [dict(p) for p in patches] if patches is not None else None,
        "nomination": dict(nomination) if nomination else None,
    }


def write_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    """Write the manifest to ``<output_dir>/fusion_manifest.json``; return the path."""
    path = Path(output_dir) / "fusion_manifest.json"
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
