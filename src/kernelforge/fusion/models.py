# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dataclasses shared across the fusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Diagnosis:
    """Result of stage 1 (trace diagnosis)."""

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
    """A model-agnostic template describing one fusible op chain."""

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
    """A concrete, localized fusion plan produced by the locate stage."""

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
    # MEASURED share of GPU memory traffic flowing through this candidate's op chain (0.0 when the trace carried no
    # shape/dtype info).
    mem_share: float = 0.0
    # "new_fusion" authors a kernel from scratch; "integration" must first benchmark and wire ``existing_operator``, a
    # retrieved ROCm-native op; "compile_pass" authors NOTHING -- the framework already implements this fusion and
    # merely ships it disabled, so the change is enabling ``compile_pass_flag`` and the win is the framework's own
    # kernel.
    candidate_kind: str = "new_fusion"
    existing_operator: str = ""
    compile_pass_flag: str = ""
    # Why a matched framework compile pass was NOT claimed (absent / undecidable / pinned off by an optimization
    # level).
    compile_pass_note: str = ""
    # Which mechanism located ``source_file``.
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
    """Kernel-level validation outcome (stage 4). e2e is out of scope."""

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
    """Outcome of claiming a framework compile pass that shipped switched off."""

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
    # Repo/package root the patch paths are relative to.
    repo_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": list(self.changes),
            "patch": self.patch,
            "harness": self.harness,
            "repo_root": self.repo_root,
        }
