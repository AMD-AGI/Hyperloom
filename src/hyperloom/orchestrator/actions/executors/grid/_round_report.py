# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading what a grid round left on disk: its benchmark report, its server log, its error tails."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from .._round_folder import RoundFolder, is_round_folder
from ..benchmark_result import extract_benchmark_measurement, harvest_mn_gpu_metrics, round_succeeded
from ..gpu._gpu_metrics import write_gpu_metrics_from_report
from ._grid_base import WARMUP_ROUND_DIR

log = logging.getLogger(__name__)


def _parse_report(workspace: Path) -> dict[str, Any] | None:
    """Load ``benchmark_report.json`` from a benchmark workspace."""
    report = workspace / "benchmark_report.json"
    if not report.exists():
        return None
    try:
        with report.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# How long to keep re-reading ``benchmark_report.json`` when the report does not yet parse into a valid measurement.
REPORT_SETTLE_SECONDS = 30.0
REPORT_SETTLE_POLL_SECONDS = 1.0


async def _settled_measurement(
    workspace: Path,
    *,
    subprocess_started_unix: float | None,
    settle_seconds: float = REPORT_SETTLE_SECONDS,
    poll_seconds: float = REPORT_SETTLE_POLL_SECONDS,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read the benchmark report, re-reading briefly while it is still settling; a finished run_dir never settles.

    A multi-node round's pod GPU samples are first merged into ``workspace`` as ``gpu_metrics.csv``.
    """
    try:
        harvest_mn_gpu_metrics(workspace, subprocess_started_unix=subprocess_started_unix)
    except (OSError, ValueError, csv.Error) as exc:
        log.warning("grid_runner: MN GPU-metrics harvest failed: %s", exc)
    if is_round_folder(workspace):
        RoundFolder.read(workspace).write_gpu_metrics(workspace.parent)
        settle_seconds = 0.0
    deadline = time.monotonic() + max(0.0, float(settle_seconds))
    attempts = 0
    while True:
        report = _parse_report(workspace)
        measurement = extract_benchmark_measurement(
            report,
            workspace=workspace,
            subprocess_started_unix=subprocess_started_unix,
        )
        attempts += 1
        succeeded = round_succeeded(measurement)
        if succeeded or time.monotonic() >= deadline:
            if attempts > 1 and succeeded:
                log.info(
                    "grid_runner: benchmark_report became valid after %d read(s); "
                    "the report was still being written when the subprocess was reaped",
                    attempts,
                )
            if report is not None:
                write_gpu_metrics_from_report(workspace, report, source="benchmark_report.json")
            return report, measurement
        await asyncio.sleep(max(0.01, float(poll_seconds)))


def _existing_log_path(path: Path) -> str | None:
    """Return ``path`` as a string when it exists, else ``None``."""
    return str(path) if path.exists() else None


def measurement_server_log_path(
    server_log: Path,
    workspace: Path | None = None,
    *,
    slot: Path | None = None,
) -> str | None:
    """Return the server log attributable to this variant's measurement."""
    owning_slot = slot or server_log.parent
    direct = _existing_log_path(server_log)
    if not direct and workspace is not None:
        direct = _existing_log_path(workspace / "server.log")
    if direct:
        return direct
    warmup_dir = owning_slot / WARMUP_ROUND_DIR
    try:
        candidates = [p for d in (owning_slot, warmup_dir) for p in d.glob("*/server.log") if p.is_file()]
    except OSError:
        candidates = []
    if not candidates:
        return None
    try:
        return str(max(candidates, key=lambda path: path.stat().st_mtime))
    except OSError:
        return None


def _report_errors_summary(report: dict[str, Any] | None, limit: int = 2000) -> str:
    """Join ``benchmark_report.json`` ``errors`` into a single diagnostic."""
    if not isinstance(report, dict):
        return ""
    errors = report.get("errors")
    if not isinstance(errors, list):
        return ""
    text = "; ".join(str(item).strip() for item in errors if str(item).strip())
    return text[-limit:] if text else ""


def _on_disk_stderr_tail(*dirs: Path, limit: int = 2000) -> str:
    """Return the tail of the first non-empty on-disk benchmark log."""
    for d in dirs:
        if not d:
            continue
        for name in ("benchmark_stderr.log", "benchmark_stdout.log"):
            try:
                p = d / name
                if not p.is_file():
                    continue
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return f"[{name}] {text[-limit:]}"
            except OSError:
                continue
    return ""
