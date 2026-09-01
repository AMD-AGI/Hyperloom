# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report generation: structured JSON output for Hyperloom consumption."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_analyzer import ModelProfile
from .tuners.base import TuneResult


@dataclass
class TuneReport:
    """Complete tuning session report."""

    status: str  # "ok", "no_improvement", "empty_output", "skipped", "failed"
    # "candidate", "no_improvement", "empty_output", "partial_output",
    # "partial_failure", "skipped", "failed"
    micro_decision: str
    requires_e2e_validation: bool = True

    # Input context
    model_path: str = ""
    framework: str = ""
    precision: str = ""
    quant_type: str = ""
    gpu_type: str = ""
    tp: int = 1
    conc: int = 0
    tokens: list[int] = field(default_factory=list)

    # Results
    tuners_run: list[dict[str, Any]] = field(default_factory=list)
    tuners_skipped: list[dict[str, Any]] = field(default_factory=list)
    # Every tuner that failed, listed regardless of the overall decision. A
    # sibling tuner succeeding must not make a crash invisible.
    failed_tuners: list[dict[str, Any]] = field(default_factory=list)
    recommended_env: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    # Timing
    total_elapsed_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    # Errors (if overall failure)
    error: str = ""
    error_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "micro_decision": self.micro_decision,
            "requires_e2e_validation": self.requires_e2e_validation,
            "model_path": self.model_path,
            "framework": self.framework,
            "precision": self.precision,
            "quant_type": self.quant_type,
            "gpu_type": self.gpu_type,
            "tp": self.tp,
            "conc": self.conc,
            "tokens": self.tokens,
            "tuners_run": self.tuners_run,
            "recommended_env": self.recommended_env,
            "artifacts": self.artifacts,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.tuners_skipped:
            d["tuners_skipped"] = self.tuners_skipped
        if self.failed_tuners:
            d["failed_tuners"] = self.failed_tuners
        if self.error:
            d["error"] = self.error
            d["error_class"] = self.error_class
        return d


def build_report(
    results: list[TuneResult],
    skipped: list[tuple[str, str]],  # (tuner_name, skip_reason)
    *,
    profile: ModelProfile,
    framework: str,
    precision: str,
    quant_type: str,
    gpu_type: str,
    tp: int,
    conc: int,
    tokens: list[int],
    started_at: str,
    total_elapsed_s: float,
) -> TuneReport:
    """Build a TuneReport from individual tuner results.

    Args:
        results: Results from tuners that actually ran.
        skipped: List of (name, reason) for tuners that were skipped.
        profile: Model profile.
        framework: Target framework.
        precision: Target precision.
        quant_type: Resolved quant type.
        gpu_type: GPU type.
        tp: Tensor parallel degree.
        conc: Target concurrency.
        tokens: Token coverage used.
        started_at: ISO timestamp of session start.
        total_elapsed_s: Total elapsed time.
    """
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    tuners_run = [r.to_dict() for r in results]
    tuners_skipped_list = [{"tuner": name, "skip_reason": reason} for name, reason in skipped]

    # Determine overall status and recommended_env
    recommended_env: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    has_candidate = False

    for r in results:
        # An explicitly forced candidate (r.candidate) is promoted regardless of
        # micro status: split-K tuning yields a valid deployable artifact whose
        # benefit is e2e-only, so the tuner reports status="no_improvement" /
        # best_micro==1.0 yet the CSV delivers several % e2e. Gating solely on
        # status=="ok" would silently drop it; requires_e2e_validation stays True
        # so it is still confirmed at e2e before final deploy.
        # partial_output counts alongside ok: the rows the tuner did write are a
        # valid deployable artifact, the shortfall is reported separately via
        # expected_shapes/missing_shapes rather than by discarding the result.
        if (r.status in ("ok", "partial_output") and r.has_improvement) or (r.candidate and r.status != "failed"):
            has_candidate = True
            if r.env_var and r.env_value:
                recommended_env[r.env_var] = r.env_value
            if r.env_vars:
                recommended_env.update(r.env_vars)
            if r.artifact_path:
                artifacts[r.tuner_name] = r.artifact_path

    # Overall decision
    failed_results = [r for r in results if r.status == "failed"]
    failed_tuners = [
        {
            "tuner": r.tuner_name,
            "error_class": r.error_class,
            "error": r.error,
        }
        for r in failed_results
    ]
    all_failed = bool(results) and len(failed_results) == len(results)
    all_skipped = len(results) == 0

    # Strict status (A2b): distinguish a genuine "compared, no shape improved"
    # (no_improvement) from "tuner ran but produced nothing parseable"
    # (empty_output) so an empty/unparsed run is never silently reported as a
    # real no-improvement result.
    non_failed_statuses = [r.status for r in results if r.status != "failed"]
    all_empty_non_failed = bool(non_failed_statuses) and all(s == "empty_output" for s in non_failed_statuses)

    if all_skipped:
        status = "skipped"
        micro_decision = "skipped"
    elif has_candidate:
        status = "ok"
        micro_decision = "candidate"
    elif all_failed:
        status = "failed"
        micro_decision = "failed"
    elif failed_results:
        # Some tuners crashed while others merely found nothing. Reporting the
        # batch as no_improvement here is what let 14 hard failures read as
        # "this model has no headroom" for a week. Note this branch is below
        # has_candidate on purpose: a usable artifact is still deployed, and the
        # crash stays visible through failed_tuners either way.
        status = "ok"
        micro_decision = "partial_failure"
    elif all_empty_non_failed:
        # Every tuner that did not fail produced no parseable output.
        status = "ok"
        micro_decision = "empty_output"
    elif any(r.status == "partial_output" for r in results):
        # Some shapes were lost. Without a candidate to validate there is nothing
        # to deploy, but this is not the same as "compared and nothing won" --
        # surface it so a truncated run is not read as a real no-improvement.
        status = "ok"
        micro_decision = "partial_output"
    else:
        # At least one tuner genuinely compared shapes without a win (possibly
        # alongside failures/empties) -> no_improvement.
        status = "ok"
        micro_decision = "no_improvement"

    return TuneReport(
        status=status,
        micro_decision=micro_decision,
        requires_e2e_validation=has_candidate,
        model_path=profile.model_path,
        framework=framework,
        precision=precision,
        quant_type=quant_type,
        gpu_type=gpu_type,
        tp=tp,
        conc=conc,
        tokens=tokens,
        tuners_run=tuners_run,
        tuners_skipped=tuners_skipped_list,
        failed_tuners=failed_tuners,
        recommended_env=recommended_env,
        artifacts=artifacts,
        total_elapsed_s=total_elapsed_s,
        started_at=started_at,
        finished_at=finished_at,
    )


def write_report(report: TuneReport, output_dir: Path) -> Path:
    """Write the report JSON to output_dir/result.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return path
