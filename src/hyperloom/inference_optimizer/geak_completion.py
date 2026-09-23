# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read GEAK's search termination independently of its performance verdict."""

from __future__ import annotations

import math
from typing import Any, Mapping


def search_termination(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read the structured producer observation; legacy results imply no exhaustion."""
    raw = result.get("search_termination")
    if not isinstance(raw, dict) or raw.get("reason") not in ("dispatch_cutoff", "completed"):
        return {"reason": "unknown"}
    out: dict[str, Any] = {"reason": raw["reason"]}
    for name in ("budget_s", "dispatch_cutoff_s", "elapsed_s", "remaining_s"):
        value = raw.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            out[name] = value
    return out
