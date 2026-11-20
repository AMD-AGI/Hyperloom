# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The artifact catalog that maps every usable Analysis artifact for downstream agents."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from hyperloom_common.io import DURABLE, JsonFormat, atomic_write_json
from kernelforge.workflows.forge_loop.roles.analysis_bundle import AnalysisCase
from kernelforge.workflows.forge_loop.roles.analysis_session import AnalysisSessionJournal


_GLOBAL_ARTIFACTS = {
    "request.json": (
        "analysis_request",
        "Immutable commit, input digests, objective, and expected cases.",
        ["analysis commit", "driver/source provenance", "expected case IDs"],
    ),
    "incremental_diff.patch": (
        "keep_diff",
        "Source changes between the previous analyzed commit and this KEEP.",
        ["changed source", "incremental analysis scope"],
    ),
    "workflow.json": (
        "analysis_workflow",
        "Durable status, attempts, outputs, and errors for the Analysis session.",
        ["session status", "attempt history", "resume point"],
    ),
    "workflow_events.jsonl": (
        "analysis_workflow_events",
        "Append-only timeline of Analysis session transitions.",
        ["session timing", "attempt history", "failure chronology"],
    ),
    "source_map.md": (
        "source_map",
        "Target call chain, source ownership, and performance-relevant regions.",
        ["call graph", "target regions", "editable source map"],
    ),
    "report.md": (
        "analysis_summary",
        "Primary Markdown Analysis report for downstream agents.",
        ["case findings", "evidence interpretation", "optimization directions"],
    ),
    "case_inventory.json": (
        "case_inventory",
        "Expected cases and their COMPLETE/FAILED coverage status.",
        ["case IDs", "shape coverage", "missing or failed cases"],
    ),
    "progress.json": (
        "analysis_progress",
        "Framework-owned phase and per-case durability checkpoint.",
        ["completed phases", "completed cases", "resume point"],
    ),
    "commands.jsonl": (
        "analysis_commands",
        "Executed commands with timeout, exit status, signal, and output paths.",
        ["command provenance", "profiler failures", "artifact locations"],
    ),
    "manifest.json": (
        "analysis_manifest",
        "Final READY/PARTIAL/FAILED bundle status and case accounting.",
        ["bundle status", "completed cases", "failed cases"],
    ),
}

_CASE_ARTIFACTS = {
    "case.json": (
        "case_definition",
        "Case identity, shape, dtype, and baseline/current latency context.",
        ["shape", "dtype", "latency context"],
    ),
    "profile": (
        "raw_profile",
        "Raw per-case profiler output and preserved workload files.",
        ["raw counters", "kernel traces", "profiler logs"],
    ),
    "normalized_metrics.json": (
        "normalized_metrics",
        "Validated target-kernel metrics normalized from raw profiler artifacts.",
        ["utilization", "occupancy", "memory and compute metrics"],
    ),
    "bottleneck.json": (
        "case_bottleneck",
        "Structured bottleneck classification, confidence, and evidence links.",
        ["primary bottleneck", "confidence", "flags"],
    ),
    "analysis.md": (
        "case_analysis",
        "Per-case explanation separating measured facts from interpretation.",
        ["profile interpretation", "limiting mechanisms"],
    ),
    "directions.md": (
        "case_directions",
        "Markdown optimization directions for this case.",
        ["strategy families", "candidate variants", "risks"],
    ),
    "failure.md": (
        "case_failure",
        "Markdown record of why this case could not be completed.",
        ["failed step", "error reason", "degradation cause"],
    ),
}


_CATALOG_ARTIFACT = (
    "analysis_artifact_catalog",
    "Index of Analysis artifact paths, contents, status, and available information for downstream agents.",
    ["artifact discovery", "partial checkpoint status", "per-file semantics"],
)


def _manifest_or_empty(work_root: Path) -> dict:
    """The work root's manifest.json, or an empty dict when it is absent, unreadable or not an object."""
    manifest = {}
    manifest_path = work_root / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text())
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            manifest = {}
    return manifest


def _case_artifact_entries(
    work_root: Path,
    cases: tuple[AnalysisCase, ...],
    *,
    output_path: Callable[[Path], str],
    case_status: Callable[[str], str],
) -> list[dict]:
    """Catalog entries for every non-empty per-case artifact present under the work root."""
    artifacts = []
    for case in cases:
        case_root = work_root / "cases" / case.directory
        for name, (
            kind,
            description,
            information,
        ) in _CASE_ARTIFACTS.items():
            path = case_root / name
            if not path.exists():
                continue
            if path.is_dir() and not any(path.iterdir()):
                continue
            artifacts.append(
                {
                    "kind": kind,
                    "scope": "case",
                    "case_id": case.case_id,
                    "status": case_status(case.case_id),
                    "path": output_path(path),
                    "description": description,
                    "available_information": information,
                }
            )
    return artifacts


def _write_artifact_catalog(
    work_root: Path,
    workflow: AnalysisSessionJournal,
    cases: tuple[AnalysisCase, ...],
    *,
    artifact_root: Path | None = None,
) -> Path:
    """Publish a compact map of every currently usable Analysis artifact."""
    target_root = (artifact_root or work_root).resolve()
    session_complete = workflow.status == "COMPLETE"
    manifest = _manifest_or_empty(work_root)
    completed = set(manifest.get("completed_case_ids") or [])
    failed = set(manifest.get("failed_case_ids") or [])
    skipped = set(manifest.get("skipped_case_ids") or [])

    def output_path(path: Path) -> str:
        relative = path.resolve().relative_to(work_root.resolve())
        return str(target_root / relative)

    def case_status(case_id: str) -> str:
        if not session_complete:
            return "AVAILABLE"
        if case_id in completed:
            return "COMPLETE"
        if case_id in failed:
            return "FAILED"
        if case_id in skipped:
            return "SKIPPED"
        return "AVAILABLE"

    artifacts = []
    for name, (kind, description, information) in _GLOBAL_ARTIFACTS.items():
        path = work_root / name
        if not path.exists():
            continue
        artifacts.append(
            {
                "kind": kind,
                "scope": "global",
                "case_id": None,
                "status": ("COMPLETE" if session_complete else "AVAILABLE"),
                "path": output_path(path),
                "description": description,
                "available_information": information,
            }
        )
    artifacts.extend(_case_artifact_entries(work_root, cases, output_path=output_path, case_status=case_status))
    catalog_path = work_root / "artifact_catalog.json"
    kind, description, information = _CATALOG_ARTIFACT
    artifacts.append(
        {
            "kind": kind,
            "scope": "global",
            "case_id": None,
            "status": ("COMPLETE" if session_complete else "AVAILABLE"),
            "path": output_path(catalog_path),
            "description": description,
            "available_information": information,
        }
    )
    atomic_write_json(
        catalog_path,
        {
            "analysis_commit": workflow.analysis_commit,
            "workflow_status": workflow.state["status"],
            "analysis_session_status": workflow.status,
            "artifacts": artifacts,
        },
        fmt=JsonFormat(trailing_newline=True),
        options=DURABLE,
    )
    return catalog_path
