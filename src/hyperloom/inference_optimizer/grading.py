# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a session grades on the interactivity objective, and under what noise band."""

from __future__ import annotations

from typing import Any

from hyperloom.common.perf_metric import GRADED_INTVTY, intvty_serving_grading_enabled

from . import framework_registry

__all__ = ["resolved_grading"]


def resolved_grading(state: Any) -> tuple[bool, float | None]:
    """Whether the interactivity objective applies to *state*, and the noise band it grades under.

    Prefers what the session recorded at seed over re-deriving it. The derivation reads the environment, and every
    later reader of it is somewhere the environment is not evidence: a resumed process, a re-baseline subprocess, an
    export driven from CLOSE. Sessions seeded before ``SharedState.grading`` existed carry nothing and only those
    derive, reporting a null band because the band they actually applied was never recorded.
    """
    recorded = getattr(state, "grading", None)
    recorded = recorded if isinstance(recorded, dict) else {}
    objective = str(recorded.get("objective") or "").strip()
    if objective:
        noise_pct = recorded.get("noise_pct")
        return objective == GRADED_INTVTY, (float(noise_pct) if isinstance(noise_pct, (int, float)) else None)
    return (
        intvty_serving_grading_enabled(
            scriptable=framework_registry.is_scriptable(getattr(state, "framework", None)),
            benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""),
        ),
        None,
    )
