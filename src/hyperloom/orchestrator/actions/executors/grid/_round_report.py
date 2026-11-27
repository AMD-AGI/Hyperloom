# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading what a grid round left on disk: its measurement, its server log, its error tails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._round_folder import RoundFolder
from ..benchmark_result import extract_benchmark_measurement, harvest_mn_gpu_metrics
from ._grid_base import WARMUP_ROUND_DIR


def record_round(run_dir: Path, *, started_unix: float) -> dict[str, Any]:
    """Write what the host derives from the round in ``run_dir`` and return the round's measurement.

    A multi-node round's pod GPU samples are merged into the run_dir as ``gpu_metrics.csv``; ``gpu_metrics.json`` goes
    into the round directory holding the run_dir.
    """
    harvest_mn_gpu_metrics(run_dir, subprocess_started_unix=started_unix)
    RoundFolder.read(run_dir).write_gpu_metrics(run_dir.parent)
    return extract_benchmark_measurement(run_dir)


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
