# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark report writer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float, to_int


def create_workspace(base_dir: Path, framework: str) -> Path:
    """Create a Magpie-compatible benchmark workspace."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = (base_dir / f"benchmark_{framework}_{timestamp}").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "torch_trace").mkdir(exist_ok=True)
    (workspace / "system_profile").mkdir(exist_ok=True)
    return workspace


@dataclass(frozen=True)
class ReportSubject:
    """What a benchmark report describes: the framework, the model and whether the run was profiled."""

    framework: str
    model: str
    profiling_enabled: bool = False


@dataclass(frozen=True)
class RunOutcome:
    """How one benchmark run ended: its verdict, errors and raw client result."""

    success: bool
    errors: list[str] | None = None
    raw_result: dict[str, Any] | None = None


def build_report(
    subject: ReportSubject,
    outcome: RunOutcome,
    *,
    workspace_dir: str,
    execution_time: float,
) -> dict[str, Any]:
    """Build a Magpie-compatible ``benchmark_report.json`` dict."""
    raw = outcome.raw_result or {}
    report: dict[str, Any] = {
        "success": bool(outcome.success),
        "framework": subject.framework,
        "model": subject.model or raw.get("model_id", ""),
        "throughput": None,
        "latency": None,
        "kernel_summary": [],
        "top_bottlenecks": [],
        "tracelens_analysis": None,
        "gap_analysis": None,
        "gpu_monitor": None,
        "workspace_dir": workspace_dir,
        "execution_time": to_float(execution_time, 0.0),
        "errors": list(outcome.errors or []),
        "profiling_enabled": bool(subject.profiling_enabled),
    }
    if raw:
        report["throughput"] = {
            "request_throughput": to_float(raw.get("request_throughput"), 0.0),
            "output_throughput": to_float(raw.get("output_throughput"), 0.0),
            "total_token_throughput": to_float(raw.get("total_token_throughput"), 0.0),
            "completed_requests": to_int(raw.get("completed"), 0),
            "total_input_tokens": to_int(raw.get("total_input_tokens"), 0),
            "total_output_tokens": to_int(raw.get("total_output_tokens"), 0),
            "duration_seconds": to_float(raw.get("duration"), 0.0),
        }
        report["latency"] = {
            metric: {
                f"{stat}_ms": to_float(raw.get(f"{stat}_{metric}_ms"), 0.0) for stat in ("mean", "median", "p99", "std")
            }
            for metric in ("ttft", "tpot", "itl", "e2el")
        }
        # Scriptable extras are carried verbatim so downstream gates can branch; only emitted when present to keep the
        # serving schema unchanged.
        for key in (
            "workload_kind",
            "throughput_unit",
            "quality_gate",
            "latency_s",
            "bench_summary",
            "bench_config",
            "frames_per_run",
            "precision_locked",
        ):
            if raw.get(key) is not None:
                report[key] = raw[key]
    return report


def format_summary_text(report: dict[str, Any]) -> str:
    """Build a Magpie-compatible human-readable ``summary.txt`` body."""
    lines = [
        f"success: {report.get('success')}",
        f"framework: {report.get('framework')}",
        f"model: {report.get('model')}",
        f"profiling_enabled: {report.get('profiling_enabled', False)}",
        f"execution_time_s: {report.get('execution_time')}",
    ]
    throughput = report.get("throughput")
    if isinstance(throughput, dict):
        lines.extend(
            [
                f"output_throughput: {throughput.get('output_throughput')}",
                f"request_throughput: {throughput.get('request_throughput')}",
                f"total_token_throughput: {throughput.get('total_token_throughput')}",
                f"completed_requests: {throughput.get('completed_requests')}",
                f"duration_seconds: {throughput.get('duration_seconds')}",
            ]
        )
    latency = report.get("latency")
    if isinstance(latency, dict):
        ttft = latency.get("ttft") or {}
        tpot = latency.get("tpot") or {}
        lines.extend(
            [
                f"mean_ttft_ms: {ttft.get('mean_ms')}",
                f"mean_tpot_ms: {tpot.get('mean_ms')}",
            ]
        )
    errors = report.get("errors") or []
    if errors:
        lines.append("errors:")
        lines.extend(f"  - {err}" for err in errors)
    return "\n".join(lines) + "\n"


def write_log_aliases(workspace: Path) -> None:
    """Write Magpie-compatible ``benchmark_stdout/stderr.log`` aliases (best-effort)."""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for tag in ("client", "eval", "scriptable"):
        for stream, parts in (("stdout", stdout_parts), ("stderr", stderr_parts)):
            path = workspace / f"{tag}_{stream}.log"
            if not path.exists():
                continue
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    try:
        if stdout_parts:
            (workspace / "benchmark_stdout.log").write_text(
                "\n".join(stdout_parts),
                encoding="utf-8",
            )
        if stderr_parts:
            (workspace / "benchmark_stderr.log").write_text(
                "\n".join(stderr_parts),
                encoding="utf-8",
            )
    except OSError:
        pass


def write_report(workspace: Path, report: dict[str, Any]) -> Path:
    """Write Magpie-compatible report artifacts into the workspace."""
    report_path = workspace / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        (workspace / "summary.txt").write_text(format_summary_text(report), encoding="utf-8")
    except OSError:
        pass
    write_log_aliases(workspace)
    return report_path
