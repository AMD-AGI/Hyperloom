# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recipe loop for forge-fuse: one forge-loop campaign per ranked recipe."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .models import Recipe, ValidationResult
from kernelforge.experience_distillation import (
    ConstraintMemory,
    extract_signature,
    render_ledger,
)
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

from .validate import DEFAULT_TARGET_SPEEDUP

log = logging.getLogger("forge_fusion")


class FusionAbort(Exception):
    """Abort the whole run from inside a campaign, recording no verdict."""


# ─────────────────────────── experience ledger ────────────────────────────── Mirrors
# kernelforge.loop.experience.ExperienceLedger, but with fusion / ROCm-specific constraint rules instead of the
# forge-loop's FlyDSL rules.

# Known error-signature -> crisp, reusable constraint.
_CONSTRAINT_RULES: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"serving crashed|scheduler crashed|cuda-?graph|hsa_status|"
            r"hardware exception|illegal memory access|memory access fault|"
            r"device-side assert|not cuda-graph",
            re.IGNORECASE,
        ),
        "The kernel passed kernel-level parity but CRASHED real sglang serving inside "
        "the decode CUDA graph. Make it CUDA-graph-capture safe: use a STATIC launch "
        "grid (never size the grid from a runtime/host value), pre-allocate every "
        "scratch/output tensor ONCE outside the fused path (no per-call "
        "torch.empty/zeros/cat), never read .item()/dynamic .shape into host control "
        "flow, avoid host<->device syncs, and index strictly in bounds for every token "
        "count so graph replay over varying batch sizes never goes out of bounds.",
    ),
    (
        re.compile(r"cuda[_-]?bf16|cuda[_-]?fp16|cuda-only|fused_qk_norm_rope|nvcc|cutlass", re.IGNORECASE),
        "Do NOT reuse a framework CUDA-only fused op (e.g. fused_qk_norm_rope pulls "
        "in cuda_bf16.h): it will not build on ROCm. Author a ROCm-native Triton kernel.",
    ),
    (
        re.compile(r"out of resource|shared memory|triton.*(compil|jit)|tl\.constexpr", re.IGNORECASE),
        "Keep the Triton kernel within gfx942 limits: bound BLOCK size and "
        "shared-memory usage and fix tl.constexpr shapes so the kernel JIT-compiles.",
    ),
    (
        re.compile(r"mamba|causal_conv1d|selective_scan|\bssm\b|hybrid", re.IGNORECASE),
        "bench_one_batch cannot init the Mamba/SSM backend on ROCm — for hybrid "
        "models the decode microbench is unavailable; gate on kernel parity and do "
        "not treat a skipped microbench as a failure.",
    ),
    (
        re.compile(r"parity failed|snr|allclose|max_abs_err", re.IGNORECASE),
        "bf16 + fp32-accum is not bit-exact: accumulate in fp32 inside the fused "
        "kernel and compare with an SNR (>= "
        f"{DEFAULT_SNR_THRESHOLD_DB:g} dB) gate, not strict allclose.",
    ),
]

# Words that promote an agent LESSON into a soft (advisory) constraint.
_RULE_WORDS = ("avoid", "must", "do not", "don't", "never", "author", "keep")

# Heuristic markers for the single most informative line in an error blob.
_ERR_MARKERS = (
    "error",
    "failed",
    "compile",
    "parity",
    "snr",
    "speedup",
    "skipped",
    "not fast",
    "cuda",
    "triton",
    "mamba",
    "serving",
    "crash",
    "hsa",
)


# How much of a signature line, or of a promoted agent lesson, survives.
_SIGNATURE_CHARS = 200


def _extract_signature(text: str) -> str:
    """Pull one normalized, informative line out of an error/outcome blob."""
    return extract_signature(text, markers=_ERR_MARKERS, limit=_SIGNATURE_CHARS)


@dataclass
class ExperienceEntry:
    """One attempt's compressed record (no full transcript)."""

    label: str  # e.g. "recipe 1 / attempt 2 (residual_add_rmsnorm)"
    outcome: str  # KEPT / PARITY FAILED / COMPILE FAILED / ...
    error_sig: str = ""
    lesson: str = ""
    best_so_far: str = ""


class FusionExperienceLedger:
    """Per-run experience store injected into each next author attempt's prompt."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        *,
        keep_recent: int = 6,
        max_constraints: int = 12,
    ):
        self.path = Path(output_dir) / "fusion_experience.md" if output_dir else None
        self.keep_recent = keep_recent
        self.memory = ConstraintMemory(_CONSTRAINT_RULES, max_constraints=max_constraints)
        self.entries: list[ExperienceEntry] = []

    @property
    def constraints(self) -> list[str]:
        """The distilled constraints carried into the next attempt's prompt."""
        return self.memory.constraints

    def record(
        self,
        *,
        label: str,
        outcome: str,
        error_text: str = "",
        lesson: str = "",
        best_so_far: str = "",
    ) -> None:
        """Record one attempt and refresh the distilled constraints."""
        self.memory.distill(error_text, outcome)
        lesson = (lesson or "").strip()
        if lesson and any(w in lesson.lower() for w in _RULE_WORDS):
            self.memory.add(f"(agent) {lesson[:_SIGNATURE_CHARS]}")
        self.entries.append(
            ExperienceEntry(
                label=label,
                outcome=(outcome or "").strip(),
                error_sig=_extract_signature(error_text),
                lesson=lesson,
                best_so_far=(best_so_far or "").strip(),
            )
        )
        self.flush()

    @staticmethod
    def _entry_lines(entry: ExperienceEntry) -> list[str]:
        lines = [f"- {entry.label}: {entry.outcome}"]
        if entry.error_sig:
            lines.append(f"    signature: {entry.error_sig}")
        if entry.best_so_far:
            lines.append(f"    best-so-far: {entry.best_so_far}")
        if entry.lesson:
            lines.append(f"    LESSON: {entry.lesson}")
        return lines

    def _render(self, entries: list[ExperienceEntry]) -> str:
        return render_ledger(
            constraints_heading="## Known constraints (do NOT repeat these mistakes)",
            constraints=self.constraints,
            entries_heading="## Recent attempts",
            entry_lines=[self._entry_lines(entry) for entry in entries],
        )

    def render_for_prompt(self, include_recent: bool = True) -> str:
        """Bounded experience text for the next author attempt's prompt."""
        entries = self.entries[-self.keep_recent :] if include_recent else []
        return self._render(entries)

    def flush(self) -> None:
        """Persist the FULL history to disk (best-effort; never breaks the loop)."""
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = "# Forge-fusion experience ledger\n\n"
            self.path.write_text(header + self._render(self.entries) + "\n", encoding="utf-8")
        except OSError as e:
            log.debug("could not flush fusion experience ledger: %s", e)


# ─────────────────────────── loop config + I/O types ────────────────────────


@dataclass
class LoopConfig:
    """Tunables for :func:`run_fusion_loop`."""

    max_recipes: int = 3  # how many ranked recipes to try
    target_speedup: float = DEFAULT_TARGET_SPEEDUP  # the campaign's KEEP gate
    output_dir: Optional[str] = None  # where fusion_experience.md is persisted


@dataclass
class LoopIteration:
    """One recipe campaign's record, surfaced in the manifest history."""

    recipe_index: int
    attempt: int
    pattern_id: str
    env_flag: str
    kept: bool
    correctness_passed: bool
    kernel_speedup: Optional[float]
    max_abs_err: Optional[float]
    note: str
    lesson: str = ""
    # The forge-loop run behind this attempt.
    experiment_id: str = ""

    def to_dict(self) -> dict:
        return {
            "recipe_index": self.recipe_index,
            "attempt": self.attempt,
            "pattern": self.pattern_id,
            "env_flag": self.env_flag,
            "kept": self.kept,
            "correctness_passed": self.correctness_passed,
            "kernel_speedup": self.kernel_speedup,
            "max_abs_err": self.max_abs_err,
            "note": self.note,
            "lesson": self.lesson,
            "experiment_id": self.experiment_id,
        }


@dataclass(frozen=True)
class RecipePatch:
    """One kept recipe's independently-emitted patch (a nomination sibling)."""

    kernel_name: str
    patch_path: str
    source_file: str
    micro_speedup: Optional[float] = None
    snapshot_dir: str = ""
    base_commit: str = ""
    env_flag: str = ""


@dataclass
class LoopResult:
    """Outcome of the whole validate-driven loop."""

    kept: bool
    best: Optional[ValidationResult]
    best_recipe: Optional[Recipe]
    history: list[LoopIteration] = field(default_factory=list)
    experience_path: Optional[str] = None
    termination_reason: str = ""
    # One patch per kept recipe, strongest first.
    patches: list["RecipePatch"] = field(default_factory=list)

    def to_dict(self) -> dict:
        best_experiment_id = ""
        if self.best_recipe is not None:
            best_experiment_id = next(
                (it.experiment_id for it in reversed(self.history) if it.pattern_id == self.best_recipe.pattern_id),
                "",
            )
        return {
            "kept": self.kept,
            "termination_reason": self.termination_reason,
            "best": self.best.to_dict() if self.best is not None else None,
            "best_pattern": self.best_recipe.pattern_id if self.best_recipe else None,
            "best_env_flag": self.best_recipe.env_flag if self.best_recipe else None,
            "attempts": len(self.history),
            "history": [it.to_dict() for it in self.history],
            "experience_ledger": self.experience_path,
            "best_experiment_id": best_experiment_id,
        }


# Injectable callable signature (documented for callers / tests). (recipe, experience) -> the campaign's verdict for
# that recipe.
CampaignFn = Callable[[Recipe, str], ValidationResult]

# Per-keeper export hook (documented for callers / tests). (kept recipe, its verdict) -> the sibling patch that recipe
# emitted, or None when the caller could not export one (which drops that keeper from patches[] without aborting the
# loop).
OnKeepFn = Callable[[Recipe, ValidationResult], Optional[RecipePatch]]


def _outcome_label(vr: ValidationResult) -> str:
    """Compact, objective outcome tag for the ledger (ground truth)."""
    if not vr.correctness_passed:
        head = (vr.note or "").split(":", 1)[0].strip() or "CORRECTNESS FAILED"
        return head
    if vr.kept:
        return f"KEPT (speedup={vr.kernel_speedup}x)"
    if vr.kernel_speedup is None:
        return "PARITY OK; speedup unverified"
    return f"PARITY OK; speedup={vr.kernel_speedup}x (< target)"


def _default_lesson(vr: ValidationResult) -> str:
    """Synthesize a one-line LESSON from a failed/weak validation result."""
    note = vr.note or ""
    marker = note.split("LESSON:", 1)
    if len(marker) == 2:
        return marker[1].strip()[:200]
    if not vr.correctness_passed:
        return (
            "Fix correctness first: compile a ROCm-native kernel and match the "
            f"eager op (SNR >= {DEFAULT_SNR_THRESHOLD_DB:g} dB)."
        )
    if vr.kernel_speedup is not None and vr.kernel_speedup < 1.0:
        return "The fused path is slower than eager — reduce launches / memory traffic before retrying."
    return "Correct but not fast enough; try a cheaper fused schedule to clear the speedup target."


def _is_better_fallback(cand: ValidationResult, best: Optional[ValidationResult]) -> bool:
    """Rank non-kept results so the loop can still report its best near-miss."""
    if best is None:
        return True
    if cand.correctness_passed != best.correctness_passed:
        return cand.correctness_passed
    cs = cand.kernel_speedup if cand.kernel_speedup is not None else -1.0
    bs = best.kernel_speedup if best.kernel_speedup is not None else -1.0
    return cs > bs


def run_fusion_loop(
    recipes: list[Recipe],
    *,
    framework: str,
    campaign_fn: CampaignFn,
    config: Optional[LoopConfig] = None,
    ledger: Optional[FusionExperienceLedger] = None,
    on_keep: Optional[OnKeepFn] = None,
) -> LoopResult:
    """Try each ranked recipe as one forge-loop campaign, best first."""
    cfg = config or LoopConfig()
    ledger = ledger or FusionExperienceLedger(cfg.output_dir)

    history: list[LoopIteration] = []
    best_result: Optional[ValidationResult] = None
    best_recipe: Optional[Recipe] = None
    global_best_speedup: Optional[float] = None
    kept_any = False
    # Keepers accumulate as (speedup-key, patch) so they can be ordered strongest-first once the whole budget has run.
    kept_patches: list[tuple[float, RecipePatch]] = []

    considered = [r for r in recipes if not getattr(r, "already_satisfied", False)]
    for ri, recipe in enumerate(considered[: cfg.max_recipes]):
        label = f"recipe {ri + 1}/{min(len(considered), cfg.max_recipes)} ({recipe.pattern_id})"
        best_ctx = (
            f"best kept speedup so far = {global_best_speedup}x"
            if global_best_speedup is not None
            else "no kept fusion yet"
        )

        experience = ledger.render_for_prompt()
        try:
            vr = campaign_fn(recipe, experience)
        except FusionAbort:
            raise
        except Exception as e:  # noqa: BLE001 — a campaign crash costs one recipe.
            log.error(
                "campaign for %s raised %s: %s",
                recipe.pattern_id,
                type(e).__name__,
                e,
            )
            vr = ValidationResult(
                correctness_passed=False,
                max_abs_err=None,
                rtol=None,
                kernel_speedup=None,
                eager_us=None,
                fused_us=None,
                kept=False,
                note=f"CAMPAIGN FAILED: {type(e).__name__}: {e}",
            )

        # Export + gate the keeper BEFORE recording it, so a hook that demotes the sibling (e.g. a serving-smoke
        # crash) is reflected in ``history`` and in the ``best``/``kept_any`` decisions below rather than leaving
        # stale KEEP state.
        patch: Optional[RecipePatch] = None
        if vr.kept and on_keep is not None:
            patch = on_keep(recipe, vr)

        lesson = _default_lesson(vr)
        outcome = _outcome_label(vr)
        ledger.record(label=label, outcome=outcome, error_text=vr.note, lesson=lesson, best_so_far=best_ctx)
        history.append(
            LoopIteration(
                recipe_index=ri,
                attempt=1,
                pattern_id=recipe.pattern_id,
                env_flag=recipe.env_flag,
                kept=vr.kept,
                correctness_passed=vr.correctness_passed,
                kernel_speedup=vr.kernel_speedup,
                max_abs_err=vr.max_abs_err,
                note=vr.note,
                lesson=lesson,
            )
        )

        if vr.kernel_speedup is not None and (global_best_speedup is None or vr.kernel_speedup > global_best_speedup):
            global_best_speedup = vr.kernel_speedup

        if vr.kept:
            kept_any = True
            log.info("fusion loop KEPT at %s: speedup=%s", label, vr.kernel_speedup)
            # The strongest keeper is also the loop's reported best, so combine-path callers (which ignore patches[])
            # still see it.
            if _is_better_fallback(vr, best_result):
                best_result, best_recipe = vr, recipe
            if patch is not None:
                # None speedup sorts weakest so a measured keeper always outranks an unmeasured one.
                sort_key = vr.kernel_speedup if vr.kernel_speedup is not None else -1.0
                kept_patches.append((sort_key, patch))
            # Do NOT early-exit: the remaining recipes are independent siblings.
            continue

        # A near-miss only becomes the reported best while nothing has been kept; once any keeper exists it owns
        # ``best``/``best_recipe``.
        if not kept_any and _is_better_fallback(vr, best_result):
            best_result, best_recipe = vr, recipe

    ledger.flush()
    kept_patches.sort(key=lambda item: item[0], reverse=True)
    return LoopResult(
        kept=kept_any,
        best=best_result,
        best_recipe=best_recipe,
        history=history,
        experience_path=str(ledger.path) if ledger.path else None,
        termination_reason="kept" if kept_any else "exhausted",
        patches=[patch for _key, patch in kept_patches],
    )
