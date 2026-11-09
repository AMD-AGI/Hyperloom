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
    "energy_joules",
    "gfx_clock_limits",
    "gpu_rows",
    "gpus_in_rocm_order",
    "metric_value",
    "read_amd_smi_json",
    "run_amd_smi",
    "single_clock_limit",
]

_READ_TIMEOUT_SEC = 30.0
#: The NPM states ``amd-smi node -p`` reports, lowercased.
NPM_ENABLED = "enabled"
NPM_STATES = (NPM_ENABLED, "disabled")
#: Joules per unit of an energy reading. amd-smi prints ``TOTAL_ENERGY_CONSUMPTION`` in J; the micro units are spelt
#: with the micro sign, the Greek mu or a plain ``u``, depending on who wrote the reading.
_JOULES_PER_MICROJOULE = 1e-6
_JOULES_PER_UNIT = {
    "J": 1.0,
    "mJ": 1e-3,
    "uJ": _JOULES_PER_MICROJOULE,
    "\u00b5J": _JOULES_PER_MICROJOULE,
    "\u03bcJ": _JOULES_PER_MICROJOULE,
}


class AmdSmiError(RuntimeError):
    """``amd-smi`` could not be run or its output could not be read."""


def gpu_rows(payload: Any) -> list[dict[str, Any]]:
    """The per-card rows of an ``amd-smi --json`` payload, each keyed by its integer ``gpu`` index."""
    rows = payload.get("gpu_data") if isinstance(payload, dict) else payload
    return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get("gpu"), int)]


def gpus_in_rocm_order() -> list[int]:
    """The amd-smi index of every card, in the order ROCm numbers them for ``ROCR_VISIBLE_DEVICES`` and
    ``HIP_VISIBLE_DEVICES``.

    ROCm numbers cards in KFD node order and amd-smi in PCI order, and the two differ on MI300X/MI355X boards, so a
    visible-device number is never an amd-smi index: card ``n`` to ROCm is ``gpus_in_rocm_order()[n]`` to amd-smi.

    Raises:
        AmdSmiError: ``amd-smi list`` failed or reported a card without its KFD node.
    """
    rows = gpu_rows(read_amd_smi_json("list"))
    nodes = {row["gpu"]: row["node_id"] for row in rows if isinstance(row.get("node_id"), int)}
    if len(nodes) != len(rows):
        raise AmdSmiError("amd-smi list reports a card without its KFD node, so its ROCm number is unknown")
    return sorted(nodes, key=nodes.__getitem__)


def metric_value(block: Any) -> float | None:
    """The number in an ``amd-smi`` reading, bare or as ``{"value": n, "unit": ...}``; ``None`` for anything else."""
    if isinstance(block, dict):
        block = block.get("value")
    return float(block) if isinstance(block, (int, float)) and not isinstance(block, bool) else None


def energy_joules(block: Any) -> float | None:
    """An ``amd-smi`` energy reading in joules: bare, which is J, or ``{"value": n, "unit": ...}`` in J, mJ or uJ.

    ``None`` for anything else, a unit it does not know included: a reading in a unit it cannot convert is not one.
    """
    value = metric_value(block)
    if value is None:
        return None
    unit = block.get("unit", "J") if isinstance(block, dict) else "J"
    scale = _JOULES_PER_UNIT.get(str(unit).strip())
    return None if scale is None else value * scale


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
