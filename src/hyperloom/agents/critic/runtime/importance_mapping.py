# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map Critic confidence / verdict signals to KB ``importance`` floats."""

from __future__ import annotations

CRITIC_IMPORTANCE_CEILING = 0.84  # Anything above is alchemist-only.

_HIGH_VERDICT_WITH_MEASUREMENT = 0.7
_HIGH_VERDICT_WITHOUT_MEASUREMENT = 0.4
_MEDIUM_VERDICT = 0.5
_DEFAULT_VERDICT = 0.4
_LOW_VERDICT = 0.4


def importance_for_verdict(
    *,
    verdict: str,
    confidence: str | None = None,
    has_measurement: bool = False,
) -> float:
    """Choose KB ``importance`` for a review_verdict-derived KB write."""
    confidence_label = (confidence or "medium").lower()
    # ``advise`` / ``needs_review`` are informational — keep them low.
    if verdict in ("advise", "needs_review"):
        return _LOW_VERDICT
    if confidence_label == "high":
        return _HIGH_VERDICT_WITH_MEASUREMENT if has_measurement else _HIGH_VERDICT_WITHOUT_MEASUREMENT
    if confidence_label == "low":
        return _LOW_VERDICT
    return _MEDIUM_VERDICT if has_measurement else _DEFAULT_VERDICT


def cap_importance(value: float) -> float:
    """Clamp ``value`` to Critic's allowed write range."""
    return min(max(0.0, float(value)), CRITIC_IMPORTANCE_CEILING)


__all__ = [
    "CRITIC_IMPORTANCE_CEILING",
    "cap_importance",
    "importance_for_verdict",
]
