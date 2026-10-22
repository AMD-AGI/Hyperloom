# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a kernel attempt's gain comes from, and the axis it is stated on.

``e2e_gain_pct`` on an attempt's ``e2e`` block is Hyperloom-measured. GEAK's
own figure is ``self_reported_e2e_gain_pct``. An attempt's ``gain_axis`` names
the instrument behind its ``gain_pct``: a lane's isolated kernel timing, a
rebench's output throughput, or the axis the integrate gate graded on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .event_fields import float_or_none, text_or_none

GAIN_AXIS_KERNEL_MICRO = "kernel_micro"
GAIN_AXIS_OUTPUT_THROUGHPUT = "output_throughput"


def claimed_gain(e2e: Mapping[str, Any]) -> float | None:
    """GEAK's own e2e gain: a journey states it as ``e2e_gain_pct``, and a row
    already read back from the spool carries it as ``self_reported_e2e_gain_pct``."""
    return float_or_none(e2e.get("self_reported_e2e_gain_pct", e2e.get("e2e_gain_pct")))


def stamp_rebench_gain(row: dict[str, Any], delta_pct: Any) -> None:
    """Put the rebench's measured delta on the attempt and on its e2e block.

    A rebench that measured no delta leaves both as they were.
    """
    gain = float_or_none(delta_pct)
    if gain is None:
        return
    row["gain_pct"] = gain
    row["gain_axis"] = GAIN_AXIS_OUTPUT_THROUGHPUT
    if row.get("e2e"):
        row["e2e"] = {**row["e2e"], "e2e_gain_pct": gain}


def stamp_gate_gain(row: dict[str, Any], measured: Any, axis: Any) -> None:
    """Put the integrate gate's gain and the axis it graded on on the attempt.

    A gate that ruled without a number leaves an earlier instrument's gain,
    and its axis, in place: the lane's own timing may be the only measurement.
    """
    gain = float_or_none(measured)
    if gain is not None:
        row["gain_pct"] = gain
        row["gain_axis"] = text_or_none(axis)


def lane_gain_axis(row: Mapping[str, Any], gain: float | None) -> str | None:
    """The axis a forge lane's ``gain``, if it measured one, is stated on.

    A GEMM tuning run that was applied carries the axis its campaign graded
    on; every other lane timed the kernel in isolation.
    """
    if gain is None:
        return None
    return text_or_none(row.get("graded_objective")) or GAIN_AXIS_KERNEL_MICRO


def lane_gain_pct(row: Mapping[str, Any]) -> float | None:
    """The gain a forge lane measured on its own candidate, as a percentage.

    A rewrite reports a ratio and a fusion or GEMM table a percentage. Both
    are the same fact about the same kind of candidate, so the attempt carries
    one axis and the reader is never asked which lane wrote the row before it
    can compare two of them.

    Returns:
        float | None: The gain, or ``None`` when the lane measured none.
    """
    stated = float_or_none(row.get("gain_pct"))
    if stated is not None:
        return stated
    speedup = float_or_none(row.get("speedup"))
    return round((speedup - 1.0) * 100.0, 4) if speedup is not None else None
