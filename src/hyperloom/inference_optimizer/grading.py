# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a session grades on the interactivity objective, and under what noise band."""

from __future__ import annotations

from typing import Any

from hyperloom.common.perf_metric import GRADED_INTVTY, intvty_serving_grading_enabled

from . import framework_registry

__all__ = ["recorded_grading", "resolved_grading"]


def recorded_grading(recorded: Any) -> tuple[bool, float | None] | None:
    """The interactivity flag and noise band a recorded ``grading`` block states; ``None`` when it names no objective."""
    recorded = recorded if isinstance(recorded, dict) else {}
    objective = str(recorded.get("objective") or "").strip()
    if not objective:
        return None
    noise_pct = recorded.get("noise_pct")
    return objective == GRADED_INTVTY, (float(noise_pct) if isinstance(noise_pct, (int, float)) else None)


def resolved_grading(state: Any) -> tuple[bool, float | None]:
    """Whether the interactivity objective applies to *state*, and the noise band it grades under.

    Prefers what the session recorded at seed over re-deriving it. The derivation reads the environment, and every
    later reader of it is somewhere the environment is not evidence: a resumed process, a re-baseline subprocess, an
    export driven from CLOSE. Sessions seeded before ``SharedState.grading`` existed carry nothing and only those
    derive, reporting a null band because the band they actually applied was never recorded.
    """
    recorded = recorded_grading(getattr(state, "grading", None))
    if recorded is not None:
        return recorded
    return (
        intvty_serving_grading_enabled(
            scriptable=framework_registry.is_scriptable(getattr(state, "framework", None)),
            benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""),
        ),
        None,
    )
