# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Bypass benchmark report writer.

Normalizes an InferenceX-style raw result (``inferencex_result.json``) into a
Magpie-compatible ``benchmark_report.json`` so Hyperloom's existing
``extract_benchmark_measurement`` / breakdown collectors consume bypass runs
unchanged. Also owns the bypass workspace layout, which mirrors Magpie's
``benchmark_<framework>_<timestamp>/`` structure.

Only the fields Hyperloom actually reads are emitted; the schema is kept
byte-compatible with Magpie's ``BenchmarkResult.to_dict()`` for the serving
path (throughput + latency + success/framework/model/errors).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def create_workspace(base_dir: Path, framework: str) -> Path:
    """Create a Magpie-compatible benchmark workspace.

    Args:
        base_dir: Output root the workspace is created under.
        framework: Framework name used in the workspace directory name.

    Returns:
        The created workspace directory (absolute).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = (base_dir / f"benchmark_{framework}_{timestamp}").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "torch_trace").mkdir(exist_ok=True)
    (workspace / "system_profile").mkdir(exist_ok=True)
    return workspace


def _f(value: Any, default: float = 0.0) -> float:
    """Coerce to float, tolerating None/str; return default on failure."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    """Coerce to int, tolerating None/str; return default on failure."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_report(
    raw_result: dict[str, Any] | None,
    *,
    framework: str,
    model: str,
    success: bool,
    workspace_dir: str,
    execution_time: float,
    errors: list[str] | None = None,
    analysis: dict[str, Any] | None = None,
    profiling_enabled: bool = False,
) -> dict[str, Any]:
    """Build a Magpie-compatible ``benchmark_report.json`` dict.

    The raw result is the flat InferenceX ``inferencex_result.json`` mapping
    (same keys Magpie's ``ResultParser.parse_inferencex_result`` reads).

    Args:
        raw_result: Flat InferenceX result mapping, or None when absent.
        framework: Framework name.
        model: Model id/path.
        success: Whether the benchmark process succeeded.
        workspace_dir: Absolute workspace path.
        execution_time: Wall-clock seconds of the run.
        errors: Optional list of error strings.
        analysis: Optional bypass-specific analysis block; emitted under
            ``report["bypass_analysis"]`` only when provided so the
            serving schema stays unchanged.
        profiling_enabled: Whether torch_profiler was enabled for this run.

    Returns:
        A report dict matching the fields Hyperloom consumes.
    """
    raw = raw_result or {}
    report: dict[str, Any] = {
        "success": bool(success),
        "framework": framework,
        "model": model or raw.get("model_id", ""),
        "throughput": None,
        "latency": None,
        "kernel_summary": [],
        "top_bottlenecks": [],
        "tracelens_analysis": None,
        "gap_analysis": None,
        "gpu_monitor": None,
        "workspace_dir": workspace_dir,
        "execution_time": _f(execution_time),
        "errors": list(errors or []),
        "profiling_enabled": bool(profiling_enabled),
    }
    if raw:
        report["throughput"] = {
            "request_throughput": _f(raw.get("request_throughput")),
            "output_throughput": _f(raw.get("output_throughput")),
            "total_token_throughput": _f(raw.get("total_token_throughput")),
            "completed_requests": _i(raw.get("completed")),
            "total_input_tokens": _i(raw.get("total_input_tokens")),
            "total_output_tokens": _i(raw.get("total_output_tokens")),
            "duration_seconds": _f(raw.get("duration")),
        }
        report["latency"] = {
            "ttft": {
                "mean_ms": _f(raw.get("mean_ttft_ms")),
                "median_ms": _f(raw.get("median_ttft_ms")),
                "p99_ms": _f(raw.get("p99_ttft_ms")),
                "std_ms": _f(raw.get("std_ttft_ms")),
            },
            "tpot": {
                "mean_ms": _f(raw.get("mean_tpot_ms")),
                "median_ms": _f(raw.get("median_tpot_ms")),
                "p99_ms": _f(raw.get("p99_tpot_ms")),
                "std_ms": _f(raw.get("std_tpot_ms")),
            },
            "itl": {
                "mean_ms": _f(raw.get("mean_itl_ms")),
                "median_ms": _f(raw.get("median_itl_ms")),
                "p99_ms": _f(raw.get("p99_itl_ms")),
                "std_ms": _f(raw.get("std_itl_ms")),
            },
            "e2el": {
                "mean_ms": _f(raw.get("mean_e2el_ms")),
                "median_ms": _f(raw.get("median_e2el_ms")),
                "p99_ms": _f(raw.get("p99_e2el_ms")),
                "std_ms": _f(raw.get("std_e2el_ms")),
            },
        }
        # Scriptable extras are carried verbatim so downstream gates can branch;
        # only emitted when present to keep the serving schema unchanged.
        for key in ("workload_kind", "throughput_unit", "quality_gate", "latency_s"):
            if raw.get(key) is not None:
                report[key] = raw[key]
    if analysis:
        report["bypass_analysis"] = analysis
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
        lines.extend([
            f"output_throughput: {throughput.get('output_throughput')}",
            f"request_throughput: {throughput.get('request_throughput')}",
            f"total_token_throughput: {throughput.get('total_token_throughput')}",
            f"completed_requests: {throughput.get('completed_requests')}",
            f"duration_seconds: {throughput.get('duration_seconds')}",
        ])
    latency = report.get("latency")
    if isinstance(latency, dict):
        ttft = latency.get("ttft") or {}
        tpot = latency.get("tpot") or {}
        lines.extend([
            f"mean_ttft_ms: {ttft.get('mean_ms')}",
            f"mean_tpot_ms: {tpot.get('mean_ms')}",
        ])
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
                "\n".join(stdout_parts), encoding="utf-8",
            )
        if stderr_parts:
            (workspace / "benchmark_stderr.log").write_text(
                "\n".join(stderr_parts), encoding="utf-8",
            )
    except OSError:
        pass


def write_report(workspace: Path, report: dict[str, Any]) -> Path:
    """Write Magpie-compatible report artifacts into the workspace.

    Emits ``benchmark_report.json``, ``summary.txt``, and aggregated
    ``benchmark_stdout.log`` / ``benchmark_stderr.log`` aliases.

    Args:
        workspace: Benchmark workspace directory.
        report: Report dict to serialize.

    Returns:
        Path to the written report file.
    """
    report_path = workspace / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        (workspace / "summary.txt").write_text(format_summary_text(report), encoding="utf-8")
    except OSError:
        pass
    write_log_aliases(workspace)
    return report_path