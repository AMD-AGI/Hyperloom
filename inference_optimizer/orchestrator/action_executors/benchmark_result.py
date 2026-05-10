"""Benchmark result parsing shared by Magpie-backed executors.

Magpie and shell wrappers can report failure after InferenceX has already
written valid throughput numbers (for example a post-benchmark cleanup error).
The optimizer should treat the measurement as usable whenever the benchmark
completed requests and produced positive throughput, while preserving the
wrapper status as diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            return parsed
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _candidate_raw_jsons(workspace: Path) -> list[Path]:
    """Return likely InferenceX result files, preferring baseline over profile."""
    paths = [
        p for p in workspace.rglob("*.json")
        if p.name != "benchmark_report.json"
    ]
    return sorted(
        paths,
        key=lambda p: (
            "profile" in p.name.lower(),
            "eval" in str(p).lower(),
            str(p),
        ),
    )


def _merge_raw_result(
    measurement: dict[str, Any],
    raw: dict[str, Any],
    *,
    source_path: Path,
) -> None:
    if measurement.get("output_throughput") is None:
        measurement["output_throughput"] = _to_float(raw.get("output_throughput"))
    if measurement.get("request_throughput") is None:
        measurement["request_throughput"] = _to_float(raw.get("request_throughput"))
    if measurement.get("total_token_throughput") is None:
        measurement["total_token_throughput"] = _to_float(
            raw.get("total_token_throughput")
        )
    if measurement.get("completed_requests") is None:
        measurement["completed_requests"] = _first_int(
            raw.get("completed_requests"),
            raw.get("completed"),
        )
    if measurement.get("duration_seconds") is None:
        measurement["duration_seconds"] = _first_float(
            raw.get("duration_seconds"),
            raw.get("duration"),
        )
    if measurement.get("ttft_mean_ms") is None:
        measurement["ttft_mean_ms"] = _to_float(raw.get("mean_ttft_ms"))
    if measurement.get("ttft_p99_ms") is None:
        measurement["ttft_p99_ms"] = _to_float(raw.get("p99_ttft_ms"))
    if measurement.get("e2el_mean_ms") is None:
        measurement["e2el_mean_ms"] = _first_float(
            raw.get("mean_e2el_ms"),
            raw.get("mean_latency_ms"),
        )
    if measurement.get("e2el_p99_ms") is None:
        measurement["e2el_p99_ms"] = _first_float(
            raw.get("p99_e2el_ms"),
            raw.get("p99_latency_ms"),
        )
    if measurement.get("raw_result_path") is None:
        measurement["raw_result_path"] = str(source_path)


def extract_benchmark_measurement(
    report: dict[str, Any] | None,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Extract a normalized measurement from Magpie and InferenceX outputs."""
    report = report or {}
    throughput = report.get("throughput") or {}
    latency = report.get("latency") or {}
    ttft = latency.get("ttft") or {}
    e2el = latency.get("e2el") or {}

    measurement: dict[str, Any] = {
        "reported_success": report.get("success") if report else None,
        "framework": report.get("framework"),
        "model": report.get("model"),
        "request_throughput": _to_float(throughput.get("request_throughput")),
        "output_throughput": _to_float(throughput.get("output_throughput")),
        "total_token_throughput": _to_float(
            throughput.get("total_token_throughput")
        ),
        "completed_requests": _first_int(
            throughput.get("completed_requests"),
            throughput.get("completed"),
        ),
        "duration_seconds": _to_float(throughput.get("duration_seconds")),
        "ttft_mean_ms": _to_float(ttft.get("mean_ms")),
        "ttft_p99_ms": _to_float(ttft.get("p99_ms")),
        "e2el_mean_ms": _to_float(e2el.get("mean_ms")),
        "e2el_p99_ms": _to_float(e2el.get("p99_ms")),
        "raw_result_path": None,
        "nonfatal_warnings": [],
    }

    if workspace is not None:
        for raw_path in _candidate_raw_jsons(workspace):
            raw = _load_json(raw_path)
            if not raw or _to_float(raw.get("output_throughput")) is None:
                continue
            _merge_raw_result(measurement, raw, source_path=raw_path)
            if is_valid_measurement(measurement):
                break

    warnings = measurement["nonfatal_warnings"]
    if report and report.get("success") is not True:
        warnings.append("benchmark_report_success_false")
    if workspace is not None and measurement.get("raw_result_path"):
        warnings.append("raw_inferencex_result_used")

    measurement["valid_measurement"] = is_valid_measurement(measurement)
    return measurement


def is_valid_measurement(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    output_tput = _to_float(result.get("output_throughput"))
    completed = _to_int(result.get("completed_requests"))
    return (
        output_tput is not None
        and output_tput > 0
        and completed is not None
        and completed > 0
    )


__all__ = [
    "extract_benchmark_measurement",
    "is_valid_measurement",
]
