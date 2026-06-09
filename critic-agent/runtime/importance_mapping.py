# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Map Critic confidence / verdict signals to KB ``importance`` floats.

Critic may not write the top tier (``>= 0.85``), reserved for Alchemist
promotion (contract §2.3); the service guards downgrades (G-2
``max(existing, incoming)``) so we just emit honest in-range values via
:func:`importance_for_verdict` (Trigger A) and :func:`importance_for_kb_draft`
(Trigger B).
"""

from __future__ import annotations

CRITIC_IMPORTANCE_CEILING = 0.84  # Anything above is alchemist-only.

_HIGH_VERDICT_WITH_MEASUREMENT = 0.7
_HIGH_VERDICT_WITHOUT_MEASUREMENT = 0.4
_MEDIUM_VERDICT = 0.5
_DEFAULT_VERDICT = 0.4
_LOW_VERDICT = 0.4

_DRAFT_DEFAULT = 0.5
_DRAFT_HIGH_CONFIDENCE = 0.6


def importance_for_verdict(
    *,
    verdict: str,
    confidence: str | None = None,
    has_measurement: bool = False,
) -> float:
    """Choose KB ``importance`` for a review_verdict-derived KB write.

    Args:
        verdict: One of the contract verdicts (``approve`` / ``reject`` / ...).
        confidence: ``high`` / ``medium`` / ``low`` (Critic schema). ``None``
            falls back to ``medium``.
        has_measurement: True when the packet carries a comparable
            before/after benchmark or reproducer evidence.
    """
    confidence_label = (confidence or "medium").lower()
    # ``advise`` and ``needs_review`` are usually informational — keep them
    # low so they don't crowd out higher-quality entries.
    if verdict in ("advise", "needs_review"):
        return _LOW_VERDICT
    if confidence_label == "high":
        return (
            _HIGH_VERDICT_WITH_MEASUREMENT
            if has_measurement
            else _HIGH_VERDICT_WITHOUT_MEASUREMENT
        )
    if confidence_label == "low":
        return _LOW_VERDICT
    return _MEDIUM_VERDICT if has_measurement else _DEFAULT_VERDICT


def importance_for_kb_draft(*, confidence: float | None) -> float:
    """Choose KB ``importance`` for a Critic kb_draft entry.

    The Critic SKILL emits ``confidence`` as a float in ``[0.0, 1.0]``; we
    promote drafts that pass ``0.8`` to ``0.6`` and otherwise default to
    ``0.5`` (contract §2.3 Critic default).
    """
    if confidence is None:
        return _DRAFT_DEFAULT
    if confidence >= 0.8:
        return _DRAFT_HIGH_CONFIDENCE
    return _DRAFT_DEFAULT


def cap_importance(value: float) -> float:
    """Clamp ``value`` to Critic's allowed write range."""
    return min(max(0.0, float(value)), CRITIC_IMPORTANCE_CEILING)


__all__ = [
    "CRITIC_IMPORTANCE_CEILING",
    "cap_importance",
    "importance_for_kb_draft",
    "importance_for_verdict",
]
