# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dataclasses shared across the fusion pipeline.

These are the stable in-memory contracts between stages (diagnose -> locate ->
author -> validate -> emit) and mirror the fields of the emitted JSON manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Diagnosis:
    """Result of stage 1 (trace diagnosis).

    Attributes:
        launch_bound_share: Combined GPU-busy-time share of the launch-bound op
            categories (elementwise/rmsnorm/rope/add/... ). This is measured on a
            CUDA-graph-DISABLED trace so it is an UPPER BOUND on the real
            CUDA-graph-ON headroom, not the expected gain.
        busy_fraction_of_wall: Fraction of wall time the GPU was busy (low =>
            dispatch/host bound => fusion is high value). ``None`` if unknown.
        predicted_e2e_gain: Predicted CUDA-graph-ON end-to-end gain (fraction),
            derived from ``launch_bound_share`` via the calibration model. This is
            what the candidate gate uses, NOT the raw launch-bound share.
        dominant_categories: Launch-bound categories ordered by descending share.
        kernels_per_step: Mean GPU kernels launched per decode step.
        category_shares: Full category -> busy-time-share map.
        is_candidate: Whether the decode path is a fusion candidate.
        reason: Human-readable verdict reason.
        category_bytes_share: Per-category share of GPU memory traffic (fraction of
            summed input+output tensor bytes), MEASURED from the trace's op shapes.
            Empty when the trace carries no shape/dtype info -> memory signal
            unavailable, callers fall back to the launch-share discount.
    """

    launch_bound_share: float
    busy_fraction_of_wall: Optional[float]
    dominant_categories: list[str]
    kernels_per_step: float
    category_shares: dict[str, float]
    is_candidate: bool
    reason: str
    predicted_e2e_gain: float = 0.0
    category_bytes_share: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "launch_bound_share": round(self.launch_bound_share, 4),
            "launch_bound_share_note": "upper bound (cuda-graph-disabled trace); see predicted_e2e_gain",
            "predicted_e2e_gain": round(self.predicted_e2e_gain, 4),
            "category_bytes_share": {k: round(v, 4) for k, v in self.category_bytes_share.items()},
            "busy_fraction_of_wall": (
                round(self.busy_fraction_of_wall, 4) if self.busy_fraction_of_wall is not None else None
            ),
            "dominant_categories": list(self.dominant_categories),
            "kernels_per_step": round(self.kernels_per_step, 2),
            "category_shares": {k: round(v, 4) for k, v in self.category_shares.items()},
            "is_candidate": self.is_candidate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FusionPattern:
    """A model-agnostic template describing one fusible op chain.

    The pattern library is the "hybrid" half of discovery: the launch-bound
    categories a trace shows map to a fusion HYPOTHESIS (this template), which the
    locate stage then confirms/localizes against the real model source. Templates
    carry NO per-model literals.

    Attributes:
        id: Stable pattern id (e.g. ``residual_add_rmsnorm``).
        trigger_categories: Launch-bound categories whose presence suggests this
            pattern.
        min_trigger_share: Minimum combined share of ``trigger_categories`` (of
            GPU busy time) for the pattern to be proposed.
        description: One-line human description.
        source_hints: Symbols/opnames to grep for in the model source to localize
            the chain (e.g. ``["+ residual", "RMSNorm"]``).
        fusion_math: Sketch of the fused computation, handed to the author LLM.
        eager_reference_hint: How to build the correctness reference by IMPORTING
            the real eager ops (never re-implemented by the LLM).
        env_flag: Suggested env-gate flag name for the fused path.
        frameworks: Frameworks this pattern applies to.
        rocm_native: When True, the author MUST write a ROCm-native (Triton/aiter)
            kernel and must NOT reuse a framework CUDA-only fused op (e.g. sglang's
            ``fused_qk_norm_rope``), which fails to build on ROCm.
        fused_markers: Regexes whose presence in the model source indicates the
            fusion is ALREADY implemented there (a framework already fuses this) ->
            the pattern is already-satisfied and should be skipped (no-op recipe).
    """

    id: str
    trigger_categories: frozenset[str]
    min_trigger_share: float
    description: str
    source_hints: tuple[str, ...]
    fusion_math: str
    eager_reference_hint: str
    env_flag: str
    frameworks: frozenset[str]
    rocm_native: bool = True
    fused_markers: tuple[str, ...] = ()


@dataclass
class Recipe:
    """A concrete, localized fusion plan produced by the locate stage.

    This is the pattern instantiated for a specific model/framework: which source
    file carries the chain, the representative decode shapes to validate against,
    and the matched categories that justified it.
    """

    pattern_id: str
    description: str
    env_flag: str
    source_file: str
    source_hints: list[str]
    fusion_math: str
    eager_reference_hint: str
    shapes: dict[str, Any]
    matched_categories: list[str]
    trigger_share: float
    rocm_native: bool = True
    source_confirmed: Optional[bool] = None
    already_satisfied: bool = False
    predicted_gain: float = 0.0
    # MEASURED share of GPU memory traffic flowing through this candidate's op
    # chain (0.0 when the trace carried no shape/dtype info). This is the memory
    # channel that grounds ``predicted_gain`` under CUDA-graph-ON.
    mem_share: float = 0.0
    # "new_fusion" authors a kernel from scratch; "integration" must first
    # benchmark and wire ``existing_operator``, a retrieved ROCm-native op;
    # "compile_pass" authors NOTHING -- the framework already implements this
    # fusion and merely ships it disabled, so the change is enabling
    # ``compile_pass_flag`` and the win is the framework's own kernel.
    candidate_kind: str = "new_fusion"
    existing_operator: str = ""
    compile_pass_flag: str = ""
    # Why a matched framework compile pass was NOT claimed (absent / undecidable /
    # pinned off by an optimization level). Empty when nothing was matched or the
    # pass was claimed. Keeps "we could not decide" distinguishable from "the
    # framework already does it" in the manifest.
    compile_pass_note: str = ""
    # Which mechanism located ``source_file``. Distinguishes the registry
    # answering from the path convention answering after it did not.
    source_resolution_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern_id,
            "description": self.description,
            "env_flag": self.env_flag,
            "source_file": self.source_file,
            "source_hints": list(self.source_hints),
            "fusion_math": self.fusion_math,
            "eager_reference_hint": self.eager_reference_hint,
            "shapes": dict(self.shapes),
            "matched_categories": list(self.matched_categories),
            "trigger_share": round(self.trigger_share, 4),
            "predicted_gain": round(self.predicted_gain, 4),
            "mem_share": round(self.mem_share, 4),
            "rocm_native": self.rocm_native,
            "source_confirmed": self.source_confirmed,
            "already_satisfied": self.already_satisfied,
            "candidate_kind": self.candidate_kind,
            "existing_operator": self.existing_operator,
            "compile_pass_flag": self.compile_pass_flag,
            "compile_pass_note": self.compile_pass_note,
            "source_resolution_note": self.source_resolution_note,
        }


@dataclass
class ValidationResult:
    """Kernel-level validation outcome (stage 4). e2e is out of scope.

    On the forge-loop path these fields have MIXED provenance: ``kernel_speedup``
    is the loop's mean over repeated benchmarks, while ``max_abs_err``,
    ``eager_us`` and ``fused_us`` come from the single harness report behind that
    decision. ``fused_us / eager_us`` therefore does not reproduce
    ``kernel_speedup`` and must not be used to check it, and ``rtol`` stays None
    because the harness reports SNR and absolute error, never a relative one.
    """

    correctness_passed: bool
    max_abs_err: Optional[float]
    rtol: Optional[float]
    kernel_speedup: Optional[float]
    eager_us: Optional[float]
    fused_us: Optional[float]
    kept: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": {
                "passed": self.correctness_passed,
                "max_abs_err": self.max_abs_err,
                "rtol": self.rtol,
            },
            "kernel_speedup": self.kernel_speedup,
            "eager_us": self.eager_us,
            "fused_us": self.fused_us,
            "kept": self.kept,
            "note": self.note,
        }


@dataclass
class CompilePassOutcome:
    """Outcome of claiming a framework compile pass that shipped switched off.

    A compile_pass run has no authored kernel, so the kernel-level
    :class:`ValidationResult` gates (SNR parity, microbench) do not apply. It needs
    its own structured verdict instead: that the edit actually changed the RESOLVED
    config, and that a same-shape disabled/enabled serving A/B measured a real
    gain. Without both, "the server booted" would be enough to ship a no-op or even
    a regression.
    """

    flag: str
    config_file: str = ""
    source: str = ""
    enabled_after_edit: Optional[bool] = None
    baseline_tok_s: Optional[float] = None
    enabled_tok_s: Optional[float] = None
    speedup: Optional[float] = None
    target_speedup: float = 0.0
    pass_activated: Optional[bool] = None
    activation_evidence: list[str] = field(default_factory=list)
    validated: bool = False
    kept: bool = False
    reverted: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "config_file": self.config_file,
            "source": self.source,
            "enabled_after_edit": self.enabled_after_edit,
            "baseline_tok_s": self.baseline_tok_s,
            "enabled_tok_s": self.enabled_tok_s,
            "speedup": round(self.speedup, 4) if self.speedup is not None else None,
            "target_speedup": self.target_speedup,
            "pass_activated": self.pass_activated,
            "activation_evidence": list(self.activation_evidence),
            "validated": self.validated,
            "kept": self.kept,
            "reverted": self.reverted,
            "note": self.note,
        }


@dataclass
class FusionArtifacts:
    """Emitted artifacts (stage 5): the Hyperloom handoff contract."""

    changes: list[dict[str, str]] = field(default_factory=list)
    patch: Optional[str] = None
    harness: Optional[str] = None
    # Repo/package root the patch paths are relative to. Hyperloom must apply the
    # patch against THIS root (may be a site-packages dir, not a git toplevel).
    repo_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": list(self.changes),
            "patch": self.patch,
            "harness": self.harness,
            "repo_root": self.repo_root,
        }
