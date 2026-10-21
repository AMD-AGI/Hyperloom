# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a GEAK candidate's e2e gain comes from.

``e2e_gain_pct`` on an attempt's ``e2e`` block is Hyperloom-measured. GEAK's
own figure is ``self_reported_e2e_gain_pct``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .event_fields import float_or_none


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
    if row.get("e2e"):
        row["e2e"] = {**row["e2e"], "e2e_gain_pct": gain}
