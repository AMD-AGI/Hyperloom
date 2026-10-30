# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one rule for how much wall-clock budget a session has used, over its persisted fields."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperloom.common.coerce import recorded_float, to_unix


def elapsed_sec_of(record: Mapping[str, Any], *, now_unix: float) -> float:
    """Seconds of budget consumed, from ``state.json`` or a live state's fields.

    The charged total plus the live leg since its anchor; a session with neither falls back to the time since
    ``start_ts``.
    """
    charged = max(0.0, recorded_float(record.get("elapsed_charged_sec")))
    anchor = recorded_float(record.get("leg_anchor_unix"))
    if anchor > 0.0:
        return charged + max(0.0, now_unix - anchor)
    if charged > 0.0:
        return charged
    started = to_unix(str(record.get("start_ts") or "").strip())
    return 0.0 if started is None else max(0.0, now_unix - started)
