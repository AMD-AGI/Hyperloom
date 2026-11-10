# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a tuner's CSVs hold any timing at all.

A tuner whose kernels never ran on the GPU (an aiter built for another image, a device it cannot see) still writes its
rows, each with ``us`` 0 and ``tflops`` inf. Every candidate then ties at zero, the pick is arbitrary, and a speedup
over a zero baseline means nothing, so such a run is a failure, not a table.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _timed(row: dict[str, str]) -> bool:
    us, tflops = _number(row.get("us")), _number(row.get("tflops"))
    return us is not None and math.isfinite(us) and us > 0 and (tflops is None or math.isfinite(tflops))


def _timing_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            return [row for row in csv.DictReader(fh) if "us" in row]
    except (OSError, csv.Error):
        return []


def unmeasured_timings(*paths: Path) -> str:
    """Why the timings in ``paths`` cannot be used, or ``""`` when every CSV with rows has at least one timed row."""
    for path in paths:
        rows = _timing_rows(path)
        if rows and not any(_timed(row) for row in rows):
            return (
                f"all {len(rows)} rows of {path} have us <= 0 or tflops = inf: the candidates were never timed on "
                "the GPU, so neither the picks nor the speedups mean anything"
            )
    return ""
