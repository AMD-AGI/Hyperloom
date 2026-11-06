# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only ``amd-smi`` primitives shared by GPU telemetry and the power-settings checks."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed amd-smi invocations.
from typing import Any, Iterable, Sequence

__all__ = [
    "NPM_ENABLED",
    "NPM_STATES",
    "AmdSmiError",
    "gfx_clock_limits",
    "gpu_rows",
    "metric_value",
    "read_amd_smi_json",
    "run_amd_smi",
    "single_clock_limit",
]

_READ_TIMEOUT_SEC = 30.0
#: The NPM states ``amd-smi node -p`` reports, lowercased.
NPM_ENABLED = "enabled"
NPM_STATES = (NPM_ENABLED, "disabled")


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


def gfx_clock_limits(clock: Any) -> frozenset[float]:
    """The graphics clock's maximum limits in one card's ``amd-smi metric --clock`` block.

    amd-smi prints all eight ``gfx_<n>`` slots, fills only those with an XCD behind them (a partitioned card has fewer)
    and copies the card's one limit into each, so the limits are those the filled slots report: empty when none is
    filled. ``max_clk`` there is the limit in force: an ``amd-smi set -L sclk max`` shows in ``metric --clock``, not in
    ``static --clock``.
    """
    if not isinstance(clock, dict):
        return frozenset()
    return frozenset(
        limit
        for name, block in clock.items()
        if str(name).startswith("gfx_")
        and isinstance(block, dict)
        and metric_value(block.get("clk")) is not None
        and (limit := metric_value(block.get("max_clk"))) is not None
    )


def single_clock_limit(limits: Iterable[float]) -> float | None:
    """The one graphics-clock limit the readings agree on; ``None`` when there is none, or more than one."""
    distinct = set(limits)
    return next(iter(distinct)) if len(distinct) == 1 else None


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
