# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only ``amd-smi`` primitives shared by GPU telemetry and the power-settings checks."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed amd-smi invocations.
from typing import Any, Sequence

__all__ = ["AmdSmiError", "gpu_rows", "metric_value", "read_amd_smi_json", "run_amd_smi"]

_READ_TIMEOUT_SEC = 30.0


class AmdSmiError(RuntimeError):
    """``amd-smi`` could not be run or its output could not be read."""


def gpu_rows(payload: Any) -> list[dict[str, Any]]:
    """The per-card rows of an ``amd-smi --json`` payload, each keyed by its integer ``gpu`` index."""
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("gpu"), int)]


def metric_value(block: Any) -> float | None:
    """The number in an ``amd-smi`` reading, bare or as ``{"value": n, "unit": ...}``; ``None`` for anything else."""
    if isinstance(block, dict):
        block = block.get("value")
    return float(block) if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def run_amd_smi(args: Sequence[str], timeout: float = _READ_TIMEOUT_SEC) -> str:
    """Run ``amd-smi`` with ``args`` and return its stdout; raise :class:`AmdSmiError` when it fails."""
    try:
        done = subprocess.run(["amd-smi", *args], capture_output=True, text=True, timeout=timeout, check=False)  # nosec B603 B607
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AmdSmiError(f"amd-smi {' '.join(args)} could not run: {exc}") from exc
    if done.returncode != 0:
        # amd-smi prints a traceback on failure; its last line is the one that says why.
        lines = [line.strip() for line in (done.stderr or done.stdout or "").splitlines() if line.strip()]
        detail = (lines[-1] if lines else "")[:200]
        raise AmdSmiError(f"amd-smi {' '.join(args)} exited {done.returncode}: {detail}")
    return str(done.stdout or "")


def read_amd_smi_json(*args: str) -> Any:
    """Parse the ``--json`` output of a read-only ``amd-smi`` subcommand; raise when it fails or prints no JSON."""
    try:
        return json.loads(run_amd_smi([*args, "--json"]))
    except ValueError as exc:
        raise AmdSmiError(f"amd-smi {' '.join(args)} printed no JSON") from exc
