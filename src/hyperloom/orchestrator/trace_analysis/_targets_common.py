# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a projection-based kernel-target tool writes in TraceLens' place: the run's subject, its sidecar files, and
GEMM speedups from a gemm-tune result.

A tool that projects the kernels instead of reading a trace (:mod:`.primus_targets`) writes the files a TraceLens
run would: ``kernel_candidates.json``, the TraceLens report and summary, and the workspace's kernel roofline, so the
rest of the pipeline reads it as it reads a trace's analysis.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from ._kernel_partition import build_kernel_candidates_document
from ._kernel_roofline import build_kernel_roofline_payload

_UTF8 = "utf-8"
ROOFLINE_FILE = "kernel_roofline.json"


@dataclasses.dataclass
class Subject:
    """What the run is about, as TraceLens' arguments name it."""

    workspace_path: str
    session_id: str = ""
    model_name: str = ""
    framework: str = ""
    target_platform: str = ""
    roofline_output_name: str = ROOFLINE_FILE


@dataclasses.dataclass
class Run:
    """One run directory being written: where, for what, and what it holds so far."""

    subject: Subject
    doc: dict[str, Any]
    run_dir: Path
    trace_input: str
    candidates: list[dict[str, Any]]
    artifacts: dict[str, str]


def write_json(path: Path, payload: Any) -> str:
    """Write ``payload`` to ``path`` atomically; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding=_UTF8)
    os.replace(tmp, path)
    return str(path)


def write_sidecars(
    run: Run, md_path: Path, warnings: list[dict[str, Any]], *, tool: str, trace_input_type: str
) -> list[dict[str, Any]]:
    """The candidates, report, audit summary and roofline TraceLens writes, as ``tool``; returns the task groups."""
    # Imported here: it needs TraceLens, which a tool's other commands do not.
    from ._analysis_reports import AuditSubject, build_audit_summary, build_task_groups

    subject, doc, candidates, artifacts = run.subject, run.doc, run.candidates, run.artifacts
    task_groups = build_task_groups(candidates)
    header = {
        "model_name": subject.model_name,
        "framework": subject.framework,
        "target_platform": subject.target_platform,
        "analysis_mode": tool,
        "trace_input_type": trace_input_type,
        "hot_kernels": candidates,
        "task_groups": task_groups,
        "source": tool,
        "dry_run": False,
        "deployment": doc.get("deployment"),
        "provenance": doc.get("provenance"),
        "e2e": doc.get("e2e"),
    }
    tracelens_dir = run.run_dir / "tracelens"
    artifacts["tracelens_report_json"] = write_json(tracelens_dir / "tracelens_report.json", header)
    artifacts["kernel_candidates"] = write_json(
        run.run_dir / "kernel_candidates.json",
        build_kernel_candidates_document(
            header, candidates, task_groups, is_routable=lambda c: c.get("reusable_native_kernel") is True
        ),
    )
    audit = AuditSubject(
        trace_input=run.trace_input, framework=subject.framework, target_platform=subject.target_platform
    )
    artifacts["tracelens_summary"] = write_json(
        tracelens_dir / "summary.json",
        build_audit_summary(candidates, audit, task_groups=task_groups, trace_health_warnings=warnings),
    )
    roofline = build_kernel_roofline_payload(
        trace_input=run.trace_input,
        trace_input_type=trace_input_type,
        analysis_md_path=str(md_path),
        kernel_candidates_path=artifacts["kernel_candidates"],
        candidates=candidates,
    )
    roofline["source"] = tool
    reports = Path(subject.workspace_path) / "reports"
    artifacts["kernel_roofline"] = write_json(reports / (subject.roofline_output_name or ROOFLINE_FILE), roofline)
    return task_groups


def _tuner_dtype(name: str) -> str | None:
    n = str(name or "").lower()
    if "a4w4" in n or "fp4" in n:
        return "mxfp4"
    if "a8w8" in n or "fp8" in n:
        return "fp8"
    if "bf16" in n or "hipblaslt" in n:
        return "bf16"
    return None


def speedups_from_gemm_tune(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One exact-shape entry per GEMM shape a gemm-tune run made faster."""
    out = []
    for tuner in result.get("tuners_run") or []:
        if not isinstance(tuner, dict):
            continue
        dtype = _tuner_dtype(tuner.get("tuner_name") or tuner.get("name") or "")
        for row in tuner.get("shape_results") or []:
            base, tuned, speedup = row.get("default_us"), row.get("tuned_us"), row.get("speedup")
            if not speedup and base and tuned:
                speedup = float(base) / float(tuned)
            if not speedup or float(speedup) <= 1.0:
                continue  # a tuner keeps the default when it found nothing faster
            entry = {
                "op": "*",
                "kind": "gemm",
                "dims": {"m": int(row["M"]), "n": int(row["N"]), "k": int(row["K"])},
                "source": f"gemm-tune:{tuner.get('tuner_name') or 'tuner'}",
            }
            if base and tuned:
                entry |= {"baseline_us": float(base), "tuned_us": float(tuned)}
            else:
                entry["speedup"] = float(speedup)
            if dtype:
                entry["dtype"] = dtype
            out.append(entry)
    return out
