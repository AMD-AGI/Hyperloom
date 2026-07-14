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


def write_report(workspace: Path, report: dict[str, Any]) -> Path:
    """Write ``benchmark_report.json`` into the workspace.

    Args:
        workspace: Benchmark workspace directory.
        report: Report dict to serialize.

    Returns:
        Path to the written report file.
    """
    report_path = workspace / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path