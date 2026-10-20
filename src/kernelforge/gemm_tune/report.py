# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report generation: structured JSON output for Hyperloom consumption."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import _tuner_env, failed_tuner_records, is_candidate, per_tuner_candidates
from .tuners.base import TuneContext, TuneResult


@dataclass
class TuneReport:
    """Complete tuning session report."""

    status: str  # "ok", "no_improvement", "empty_output", "skipped", "failed"
    # "candidate", "no_improvement", "empty_output", "partial_output", "partial_failure", "skipped", "failed"
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
    # Every tuner that failed, listed regardless of the overall decision.
    failed_tuners: list[dict[str, Any]] = field(default_factory=list)
    recommended_env: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    # Per-tuner deployable results, each landable on its own.
    candidates: list[dict[str, Any]] = field(default_factory=list)

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
            "candidates": self.candidates,
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
    ctx: TuneContext,
    started_at: str,
    total_elapsed_s: float,
) -> TuneReport:
    """Build a TuneReport from individual tuner results, recording the context they were tuned in."""
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # An explicitly forced candidate (r.candidate) is promoted regardless of micro status: the aiter dense tuner's
    # split-K cap yields a valid deployable artifact whose benefit is e2e-only, so it reports
    # status="no_improvement" with a best_micro of at most 1.0 -- null when no shape had a baseline -- yet the CSV
    # delivers several % e2e.
    winners = [r for r in results if is_candidate(r)]
    failed_tuners = failed_tuner_records(results)

    if not results:
        micro_decision = "skipped"
    elif winners:
        micro_decision = "candidate"
    elif len(failed_tuners) == len(results):
        micro_decision = "failed"
    elif failed_tuners:
        # Some tuners crashed while others merely found nothing.
        micro_decision = "partial_failure"
    elif all(r.status == "empty_output" for r in results):
        # Strict status (A2b): "tuner ran but produced nothing parseable" is never silently reported as a genuine
        # "compared, no shape improved" result.
        micro_decision = "empty_output"
    elif any(r.status == "partial_output" for r in results):
        # Some shapes were lost.
        micro_decision = "partial_output"
    else:
        # At least one tuner genuinely compared shapes without a win.
        micro_decision = "no_improvement"

    return TuneReport(
        status=micro_decision if micro_decision in ("skipped", "failed") else "ok",
        micro_decision=micro_decision,
        requires_e2e_validation=bool(winners),
        model_path=ctx.profile.model_path,
        framework=ctx.framework,
        precision=ctx.precision,
        quant_type=ctx.quant_type,
        gpu_type=ctx.gpu_type,
        tp=ctx.tp,
        conc=ctx.conc,
        tokens=ctx.tokens,
        tuners_run=[r.to_dict() for r in results],
        tuners_skipped=[{"tuner": name, "skip_reason": reason} for name, reason in skipped],
        failed_tuners=failed_tuners,
        recommended_env={name: value for r in winners for name, value in _tuner_env(r).items()},
        artifacts={r.tuner_name: r.artifact_path for r in winners if r.artifact_path},
        # Per-tuner projection: each deployable tuner as its own landable candidate, so a caller can KEEP one and
        # REVERT another instead of all-or-nothing.
        candidates=[c.to_dict() for c in per_tuner_candidates(results)],
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
