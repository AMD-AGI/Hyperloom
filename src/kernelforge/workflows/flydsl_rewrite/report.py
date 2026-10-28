# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Assemble the final forge-rewrite result."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from kernelforge.workflows.flydsl_rewrite import protocol
from kernelforge.workflows.flydsl_rewrite.budget import DEFAULT_REWRITE_BUDGET, RewriteBudgetPolicy
from kernelforge.utils.durable_io import atomic_write_text

# The nested forge-loop sentinel is suppressed while its stdout is streamed by workflows.flydsl_rewrite.optimize, so this is
# the only one a caller sees.
SENTINEL = protocol.RESULT_SENTINEL


@dataclass
class RewriteResult:
    logical_op_name: str
    # Reported so a consumer can name the produced factory without reproducing KernelForge's normalization rule.
    operator_slug: str
    builder_symbol: str
    # Always "flydsl" — this layer only rewrites into FlyDSL.
    target_language: str
    port_ok: bool
    compiled: bool
    correct: bool
    source_ms: float | None
    flydsl_best_ms: float | None
    speedup: float | None
    experiment_id: str | None
    llm_usage: dict
    port_attempts: int
    # Forge-loop-compatible result view consumed by Hyperloom.
    success: bool
    baseline_ms: float | None
    best_ms: float | None
    improved: bool
    total_speedup: float | None
    base_commit: str
    best_commit: str
    flydsl_best_commit: str
    applyback_commit_ref: str
    patch_path: str
    artifacts: list[str]
    artifact_kind: str
    artifact_schema_version: int
    canonical_manifest: str
    canonical_patch_path: str
    canonical_files_root: str
    canonical_result_path: str
    forge_workspace: str
    changed_files: list[str]
    applyback_required: bool
    applyback_ok: bool
    applyback_error: str
    terminated_for_deadline: bool
    # Names which contract or stage rejected the run.
    failure_class: str
    failure_detail: str
    # Workspace-relative producer-owned paths the consumer may reclaim.
    temporary_paths: list[str]
    kb_experience: dict
    budget_policy: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResultScope:
    """What every result of one run reports alike: the logical op, the producer's scratch paths, the budget applied.

    ``budget_policy`` is reported rather than assumed because a caller that declines apply-back holds no reserve for
    it, and a reader comparing the search window against a default would otherwise be told about time never set aside.
    """

    op_name: str
    temporary_paths: list[str] = field(default_factory=list)
    budget_policy: RewriteBudgetPolicy = DEFAULT_REWRITE_BUDGET


@dataclass(frozen=True)
class RewriteOutcome:
    """How far one run got: PORT, the source timing, OPTIMIZE and apply-back, its spend, KB traffic and any failure."""

    port_ok: bool
    port_attempts: int
    source_ms: float | None
    optimize_result: dict
    applyback_result: dict | None = None
    applyback_required: bool = False
    llm_usage: dict = field(default_factory=dict)
    kb_experience: dict = field(default_factory=dict)
    failure_class: str = ""
    failure_detail: str = ""


def build_result(scope: ResultScope, outcome: RewriteOutcome) -> RewriteResult:
    """Combine the port + preflight + optimize outcomes into one result."""
    port_ok = outcome.port_ok
    optimize_result = outcome.optimize_result
    applyback_required = outcome.applyback_required
    flydsl_best_ms = optimize_result.get("best_ms")
    applyback = outcome.applyback_result or {}
    applyback_ok = bool(applyback.get("ok"))
    flydsl_best_commit = str(optimize_result.get("best_commit") or "")
    # With apply-back required this key means the apply-back commit and nothing else; the standalone best is reported
    # only as flydsl_best_commit.
    best_commit = applyback.get("best_commit") or ("" if applyback_required else flydsl_best_commit)
    speedup = optimize_result.get("mean_case_speedup") if port_ok else None
    # Only a published apply-back bundle carries an artifact kind; an interim or failed run must not name one.
    artifact_kind, artifact_schema_version = (
        (protocol.ARTIFACT_KIND_FRAMEWORK_APPLYBACK, protocol.ARTIFACT_SCHEMA_VERSION) if applyback_ok else ("", 0)
    )

    return RewriteResult(
        logical_op_name=scope.op_name,
        operator_slug=protocol.operator_slug(scope.op_name),
        builder_symbol=protocol.builder_symbol(scope.op_name),
        target_language="flydsl",
        # MVP mapping: a successful port yields a building, correct FlyDSL kernel (forge-loop only keeps
        # correctness-passing versions), so compiled and correct track port_ok.
        port_ok=port_ok,
        compiled=port_ok,
        correct=port_ok,
        source_ms=outcome.source_ms,
        flydsl_best_ms=flydsl_best_ms,
        speedup=speedup,
        experiment_id=optimize_result.get("experiment_id"),
        llm_usage=dict(outcome.llm_usage),
        port_attempts=outcome.port_attempts,
        success=bool(port_ok and (not applyback_required or applyback_ok)),
        baseline_ms=outcome.source_ms,
        best_ms=flydsl_best_ms,
        improved=bool(speedup and speedup > 1.0),
        total_speedup=speedup,
        base_commit=str(applyback.get("base_commit") or ""),
        best_commit=best_commit,
        flydsl_best_commit=flydsl_best_commit,
        applyback_commit_ref=str(applyback.get("commit_ref") or ""),
        patch_path=str(applyback.get("patch_path") or ""),
        artifacts=list(applyback.get("artifacts", [])),
        artifact_kind=artifact_kind,
        artifact_schema_version=artifact_schema_version,
        canonical_manifest=str(applyback.get("manifest_path") or ""),
        canonical_patch_path=str(applyback.get("canonical_patch_path") or applyback.get("patch_path") or ""),
        canonical_files_root=str(applyback.get("canonical_files_root") or ""),
        canonical_result_path=str(applyback.get("canonical_result_path") or ""),
        forge_workspace=str(applyback.get("forge_workspace") or ""),
        changed_files=list(applyback.get("changed_files", [])),
        applyback_required=applyback_required,
        applyback_ok=applyback_ok,
        applyback_error=str(applyback.get("error") or ""),
        terminated_for_deadline=bool(optimize_result.get("terminated_for_deadline")),
        failure_class=outcome.failure_class,
        failure_detail=outcome.failure_detail,
        temporary_paths=list(scope.temporary_paths),
        kb_experience=dict(outcome.kb_experience),
        budget_policy=scope.budget_policy.to_dict(),
    )


def emit_result(result: RewriteResult, result_json: str | None = None) -> str:
    """Write (optional) + sentinel-wrap the result JSON, returning the payload."""
    document = result.to_dict()
    if result.applyback_ok and result.applyback_required:
        protocol.validate_applyback_outer_result(document)
    payload = json.dumps(document)
    if result_json:
        atomic_write_text(result_json, payload)
    return payload
