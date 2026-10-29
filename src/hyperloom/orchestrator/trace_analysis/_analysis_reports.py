# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The reports one analysis run writes, and the audit summary beside them."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.io import JsonFormat, atomic_write_json
from hyperloom.common.timeutil import now_iso

from ._kernel_partition import build_kernel_candidates_document
from ._kernel_roofline import (
    build_kernel_roofline_payload,
    enrich_candidates_with_runtime_metadata,
    kernel_roofline_path_for_run,
)
from ._trace_inputs import TraceInputs
from .tracelens_skill_runner import aggregate_by_source_function
from hyperloom.inference_optimizer.framework_registry import tracelens_analysis_mode


def build_task_groups(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate reusable candidates by AST-resolved source function.

    A wrapper over :func:`aggregate_by_source_function`; only
    ``reusable_native_kernel`` candidates are grouped.

    Args:
        candidates: The hot-kernel candidate rows.

    Returns:
        The task-group dicts, or ``[]`` when none carries a parseable launcher
        path (callers fall through to per-kernel dispatch).
    """
    reusable = [c for c in candidates if isinstance(c, dict) and c.get("reusable_native_kernel")]
    if not reusable:
        return []
    return aggregate_by_source_function(reusable)


@dataclass(frozen=True)
class AuditSubject:
    """The trace and platform an audit summary describes."""

    trace_input: str
    framework: str = ""
    target_platform: str = ""


def build_audit_summary(
    candidates: list[dict[str, Any]],
    subject: AuditSubject,
    *,
    task_groups: list[dict[str, Any]] | None = None,
    trace_health_warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``tracelens/summary.json`` payload.

    Splits candidates into routable ``tasks`` and ``skipped`` (each with a
    ``skip_reason``), preserving priority order. Pure function.

    Args:
        candidates: The hot-kernel candidate rows.
        subject: The trace input path, framework and target platform recorded in the summary.
        task_groups: Optional task-group projections to include.
        trace_health_warnings: Optional trace-health warnings to include.

    Returns:
        The ``summary.json`` payload dict.
    """
    tasks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        reusable = bool(cand.get("reusable_native_kernel"))
        compact = {
            "kernel_id": cand.get("kernel_id"),
            "name": cand.get("name"),
            "source_file": cand.get("source_file") or "",
            "source_type": cand.get("source_type") or "",
            "kernel_category": cand.get("kernel_category") or "",
            "gpu_pct": cand.get("gpu_pct"),
            "duration_us": cand.get("duration_us"),
            "call_count": cand.get("call_count"),
            "tracelens_pitem_rank": cand.get("tracelens_pitem_rank"),
            "tracelens_pitem_title": cand.get("tracelens_pitem_title"),
            "bound_type": cand.get("bound_type") or "",
        }
        if reusable:
            compact["recommended_backends"] = list(cand.get("recommended_backends") or [])
            tasks.append(compact)
        else:
            compact["skip_reason"] = cand.get("skip_reason") or "unknown"
            skipped.append(compact)
    # Compact task_group projections for the audit view.
    group_entries: list[dict[str, Any]] = []
    for group in task_groups or []:
        if not isinstance(group, dict):
            continue
        group_entries.append(
            {
                "task_group_id": group.get("task_group_id"),
                "source_path": group.get("source_path"),
                "definition_line": group.get("definition_line"),
                "function_name": group.get("function_name"),
                "ast_resolved": bool(group.get("ast_resolved")),
                "primary_kernel_id": group.get("primary_kernel_id"),
                "kernel_ids": list(group.get("kernel_ids") or []),
                "row_count": len(group.get("rows") or []),
                "aggregate_duration_us": group.get("aggregate_duration_us"),
                "aggregate_call_count": group.get("aggregate_call_count"),
                "aggregate_gpu_pct": group.get("aggregate_gpu_pct"),
            }
        )
    return {
        "generated_at": now_iso(timespec="auto"),
        "trace_input": subject.trace_input,
        "framework": subject.framework,
        "target_platform": subject.target_platform,
        "task_count": len(tasks),
        "skipped_count": len(skipped),
        "task_group_count": len(group_entries),
        "tasks": tasks,
        "skipped": skipped,
        "task_groups": group_entries,
        # Trace-quality findings (empty = healthy; non-empty explains an empty ``tasks``).
        "trace_health_warnings": list(trace_health_warnings or []),
    }


@dataclass(frozen=True)
class OrchestratorOutcome:
    """What the analysis route left for the report writer.

    Attributes:
        report_path: The report it produced (``analysis.md`` or ``analysis.json``), if any.
        health_warnings: Trace-health warnings accumulated before and during the run.
    """

    report_path: Path | None = None
    health_warnings: list[dict[str, Any]] | None = None


def write_reports(
    run_dir: Path,
    trace_inputs: TraceInputs,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    outcome: OrchestratorOutcome = OrchestratorOutcome(),
) -> dict[str, str]:
    """Write Hyperloom-owned sidecar JSONs and surface the report the analysis route produced.

    The report (``analysis.md`` or ``analysis.json``) is owned by the route and not copied/aliased.

    Args:
        run_dir: The per-run output directory to write sidecars into.
        trace_inputs: Whether the trace input is a file or capture dir, and the
            resolved trace files for the manifest.
        candidates: The finalized hot-kernel candidate rows.
        args: Parsed CLI args carrying model/framework/platform settings.
        outcome: The pre-existing report path to surface and the trace-health
            warnings to record.

    Returns:
        A mapping of report name to its written/surfaced path.

    Raises:
        RuntimeError: When the route did not produce its report
            (rather than fabricating one).
    """
    existing_report_path = outcome.report_path
    tracelens_dir = run_dir / "tracelens"
    (tracelens_dir / "system_findings").mkdir(parents=True, exist_ok=True)
    (tracelens_dir / "category_findings").mkdir(parents=True, exist_ok=True)
    enrich_candidates_with_runtime_metadata(candidates, args)

    manifest = {
        "trace_input": str(Path(args.trace_input).resolve()),
        "trace_input_type": trace_inputs.input_type,
        "trace_files": [str(p) for p in trace_inputs.files],
        "created_at": now_iso(timespec="auto"),
    }
    # Aggregate reusable candidates into source-function task groups.
    task_groups = build_task_groups(candidates)
    report = {
        "model_name": args.model_name,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "analysis_mode": tracelens_analysis_mode(args.framework),
        "trace_input_type": trace_inputs.input_type,
        "hot_kernels": candidates,
        "task_groups": task_groups,
        "source": "tracelens_analysis",
        "dry_run": args.dry_run,
    }
    atomic_write_json(run_dir / "trace_input_manifest.json", manifest, fmt=JsonFormat(trailing_newline=True))
    atomic_write_json(tracelens_dir / "tracelens_report.json", report, fmt=JsonFormat(trailing_newline=True))
    kernel_candidates_path = run_dir / "kernel_candidates.json"
    # ``hot_kernels`` is always the full ranked set; the reusable dispatch subset
    # is exposed as ``routable_kernels`` and non-routable dicts as ``skipped_kernels``.
    # This site's routability is the coarse reusability predicate; the shared
    # builder guarantees the two lists partition ``candidates`` by construction.
    atomic_write_json(
        kernel_candidates_path,
        build_kernel_candidates_document(
            report,
            candidates,
            task_groups,
            is_routable=lambda c: c.get("reusable_native_kernel") is True,
        ),
        fmt=JsonFormat(trailing_newline=True),
    )

    # Per-run audit sidecar (tasks routed vs skipped w/ reason).
    summary = build_audit_summary(
        candidates,
        AuditSubject(
            trace_input=str(Path(args.trace_input).resolve()),
            framework=args.framework,
            target_platform=args.target_platform,
        ),
        task_groups=task_groups,
        trace_health_warnings=outcome.health_warnings,
    )
    summary_path = tracelens_dir / "summary.json"
    atomic_write_json(summary_path, summary, fmt=JsonFormat(trailing_newline=True))

    missing_trace_report = existing_report_path is None or not existing_report_path.exists()
    trace_quality_blocked = any(w.get("code") == "trace_split_no_steady_state" for w in (outcome.health_warnings or []))
    if missing_trace_report:
        if not getattr(args, "dry_run", False):
            if trace_quality_blocked:
                # Refused to run on a raw/non-steady trace; leave trace_report_path empty.
                existing_report_path = None
            else:
                raise RuntimeError(
                    "TraceLens SDK orchestrator did not produce analysis.md "
                    f"(expected at {existing_report_path}); refusing to "
                    "fabricate a Markdown report. Inspect the TraceLens skill "
                    "log and report upstream if this is reproducible."
                )
        else:
            # ``--dry-run``: synthesize a tiny stub so existence checks pass.
            stub_md = tracelens_dir / "analysis.md"
            stub_md.write_text(
                "# TraceLens dry-run stub (no SDK orchestrator output)\n",
                encoding="utf-8",
            )
            existing_report_path = stub_md

    kernel_roofline_path = kernel_roofline_path_for_run(
        run_dir,
        filename=getattr(args, "roofline_output_name", "kernel_roofline.json") or "kernel_roofline.json",
    )
    kernel_roofline_payload = build_kernel_roofline_payload(
        trace_input=str(Path(args.trace_input).resolve()),
        trace_input_type=trace_inputs.input_type,
        analysis_md_path=(str(existing_report_path) if existing_report_path else ""),
        kernel_candidates_path=str(kernel_candidates_path),
        candidates=candidates,
    )
    atomic_write_json(kernel_roofline_path, kernel_roofline_payload, fmt=JsonFormat(trailing_newline=True))

    return {
        "trace_input_manifest": str(run_dir / "trace_input_manifest.json"),
        "kernel_candidates": str(kernel_candidates_path),
        "kernel_roofline": str(kernel_roofline_path),
        "tracelens_report_json": str(tracelens_dir / "tracelens_report.json"),
        # Canonical Markdown exit is the orchestrator's analysis.md (surfaced, not aliased).
        "trace_report_path": str(existing_report_path) if existing_report_path else "",
        "tracelens_summary": str(summary_path),
    }
