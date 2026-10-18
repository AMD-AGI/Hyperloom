# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serving smoke gate that confirms or demotes a kernel-level KEEP with CUDA graphs on."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.io import WriteOptions, atomic_write_json

from .campaign import _safe_artifact_id
from .emit import FusionFiles, export_artifacts
from .validate import (
    KERNEL_KEEP_CHECKPOINT,
    DecodeLoad,
    SmokeServer,
    fused_symbol_invocation_evidence,
    serving_smoke_verdict,
)

log = logging.getLogger("forge_fusion")


@dataclass(frozen=True)
class ServingSession:
    """The serving deployment a fusion is smoke-tested against; 0 block size or model length leaves the default."""

    framework: str
    model_path: str
    gpu: str
    isl: int
    osl: int
    server_extra: str = ""
    tp: int = 1
    block_size: int = 0
    max_model_len: int = 0


def apply_serving_gate(
    result,
    session: ServingSession,
    out: Path,
    *,
    repo_root: str = "",
    pristine_dir: str = "",
) -> None:
    """Boot the real server once; only a fused-kernel fault demotes a KEEP."""
    if not (_serving_check_enabled() and session.model_path and result.best_recipe):
        return
    recipe = result.best_recipe
    vr = result.best
    # Export BEFORE the smoke, so a forge-fuse killed while serving still leaves an applicable patch.
    exported = _export_salvage_patch(
        out,
        getattr(recipe, "source_file", ""),
        repo_root=repo_root,
        pristine_dir=pristine_dir,
    )
    if exported:
        _write_kernel_keep_checkpoint(out, recipe, vr, repo_root=repo_root)
    else:
        log.warning(
            "no fusion patch could be exported for %s; a killed run cannot be salvaged",
            recipe.pattern_id,
        )
    disposition, note, termination = _run_serving_smoke(recipe, vr.note, session, out)
    vr.note = note
    if termination:
        result.termination_reason = termination
    # A fusion nothing calls (not_wired) or one that faults CUDA-graph decode (serving_crash) is not a real KEEP:
    # demote it so Hyperloom never boots it.
    if disposition in ("not_wired", "serving_crash"):
        result.kept = False
        vr.kept = False
        vr.kernel_speedup = None
        if disposition == "serving_crash":
            vr.correctness_passed = False
        _clear_kernel_keep_checkpoint(out)


def _run_serving_smoke(recipe, base_note: str, session: ServingSession, out: Path) -> tuple[str, str, str]:
    """Wiring-check + CUDA-graph-ON serving smoke for ONE recipe."""
    flags = dict.fromkeys(recipe.env_flag.split(), "1")
    safe_id = _safe_artifact_id(recipe.pattern_id)
    smoke_block = int(session.block_size) if int(session.block_size or 0) > 0 else None
    smoke_mml = int(session.max_model_len) if int(session.max_model_len or 0) > 0 else 4096
    # Cheapest gate first, and the only one that catches a fusion nothing calls: the smoke would boot, decode and
    # PASS, because stock code is what ran.
    wiring = fused_symbol_invocation_evidence(
        getattr(recipe, "source_file", ""),
        getattr(recipe, "extra_files", ()),
    )
    if wiring.verdict == "not_wired":
        log.warning("fusion not wired into %s: %s", recipe.pattern_id, wiring.reason)
        note = (
            f"KERNEL OK but NOT WIRED IN: {wiring.reason}. The microbench measured the fused "
            f"entry point directly, so its speedup says nothing about the served model, "
            f"whose end-to-end gain is exactly zero. | LESSON: authoring the fused module "
            f"is half the deliverable -- replace the ORIGINAL call site in the framework's "
            f"forward path with a call to the fused entry point, under the same env gate, "
            f"and leave the unfused code as the fallback branch."
        )
        return "not_wired", note, "not_wired"
    if wiring.verdict == "wired":
        log.info("fusion wiring confirmed for %s: %s", recipe.pattern_id, wiring.reason)
        wiring_note = ""
    else:
        log.info("fusion wiring NOT CHECKED for %s: %s", recipe.pattern_id, wiring.reason)
        wiring_note = f" | WIRING UNCHECKED: {wiring.reason}"
    verdict = serving_smoke_verdict(
        SmokeServer(
            session.model_path,
            flags,
            framework=session.framework,
            gpu=session.gpu,
            server_extra=session.server_extra,
            tp=session.tp,
            block_size=smoke_block,
            max_model_len=smoke_mml,
        ),
        DecodeLoad(isl=session.isl, osl=session.osl),
        log_path=str(out / f"serving_smoke_{safe_id}.log"),
    )
    reason = verdict.reason
    if verdict.ok:
        log.info("serving smoke OK for %s", recipe.pattern_id)
        return "ok", f"{base_note} | SERVING SMOKE OK{wiring_note}", ""
    if verdict.blames_kernel:
        log.warning("serving smoke FAILED for %s: %s", recipe.pattern_id, reason)
        note = (
            f"KERNEL OK but SERVING CRASHED (CUDA-graph-ON decode): {reason} "
            f"| LESSON: the kernel is NOT CUDA-graph-capture safe. Use a STATIC "
            f"launch grid (no data-dependent grid size), pre-allocate every "
            f"scratch/output tensor ONCE outside the fused path (no per-call "
            f"torch.empty/zeros/cat), avoid host<->device syncs, and index "
            f"strictly in bounds for every token count. Re-author CUDA-graph safe."
        )
        return "serving_crash", note, "serving_crash"
    log.warning(
        "serving smoke unconfirmed for %s at stage %s (keeping micro KEEP): %s",
        recipe.pattern_id,
        verdict.stage,
        reason,
    )
    note = (
        f"{base_note} | SERVING SMOKE UNCONFIRMED at stage {verdict.stage} "
        f"(defer e2e): {reason}{wiring_note} | LESSON: the GPU did not fault, so nothing here "
        f"is evidence against the kernel. Do not re-author to fix it; Hyperloom "
        f"e2e is the KEEP/REVERT gate."
    )
    return "unconfirmed", note, "serving_unconfirmed"


def _serving_check_enabled() -> bool:
    """Serving smoke is ON by default; ``FORGE_FUSION_SERVING_CHECK=0`` disables it."""
    return os.environ.get("FORGE_FUSION_SERVING_CHECK", "1") != "0"


def _export_salvage_patch(
    out: Path,
    source_file: str,
    *,
    repo_root: str = "",
    pristine_dir: str = "",
) -> bool:
    """Write ``fusion.patch`` for the edits made so far; report whether one exists."""
    if not repo_root or not source_file:
        return False
    # This output directory may be reused.
    _clear_kernel_keep_checkpoint(out)
    artifacts = export_artifacts(
        repo_root,
        FusionFiles(source_file),
        out / "fusion.patch",
        pristine_dir=pristine_dir or None,
    )
    if not artifacts.patch:
        return False
    patch = Path(artifacts.patch)
    return patch.is_file() and patch.stat().st_size > 0


def _write_kernel_keep_checkpoint(out: Path, recipe, vr, *, repo_root: str = "") -> None:
    """Persist a micro KEEP so a killed forge-fuse process can still be salvaged."""
    payload = {
        "kept": True,
        "kernel_speedup": getattr(vr, "kernel_speedup", None),
        "eager_us": getattr(vr, "eager_us", None),
        "fused_us": getattr(vr, "fused_us", None),
        "env_flag": getattr(recipe, "env_flag", ""),
        "pattern_id": getattr(recipe, "pattern_id", ""),
        "source_file": getattr(recipe, "source_file", ""),
        "repo_root": repo_root,
        "note": getattr(vr, "note", ""),
    }
    with contextlib.suppress(OSError):
        atomic_write_json(out / KERNEL_KEEP_CHECKPOINT, payload, options=WriteOptions())


def _clear_kernel_keep_checkpoint(out: Path) -> None:
    """Drop salvage artifacts after a real fused-kernel serving crash."""
    for name in (KERNEL_KEEP_CHECKPOINT, "fusion.patch"):
        with contextlib.suppress(OSError):
            (out / name).unlink()
