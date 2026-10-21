# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign-length policy: the long-horizon gate and one Implementer session's wall-clock budget."""

from __future__ import annotations

LONG_HORIZON_THRESHOLD_HOURS = 2.0
# One implementer session's wall-clock budget is a FUNCTION of the campaign, not a fixed number.
FORGE_SESSION_BUDGET_FRACTION = 0.15
FORGE_SESSION_BUDGET_MIN_MINUTES = 90
FORGE_SESSION_BUDGET_MAX_MINUTES = 210


def _forge_session_timeout_sec(max_hours: float, override_sec: int | None) -> int:
    """Wall-clock budget for one implementer session, in seconds."""
    if override_sec is not None:
        return int(override_sec)
    total_min = float(max_hours) * 60.0
    session_min = min(
        FORGE_SESSION_BUDGET_MAX_MINUTES,
        max(
            FORGE_SESSION_BUDGET_MIN_MINUTES,
            FORGE_SESSION_BUDGET_FRACTION * total_min,
        ),
    )
    return int(round(session_min * 60.0))


def _is_long_horizon(max_hours: float) -> bool:
    """Whether one campaign session enables expensive long-horizon agents."""
    return float(max_hours) > LONG_HORIZON_THRESHOLD_HOURS
