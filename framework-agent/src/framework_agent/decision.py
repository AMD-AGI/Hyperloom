# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Winner-decision gates (throughput / accuracy / completed) split out of ``explorer.py`` for isolated testing and reuse by ``runtime/tools_api.evaluate_candidate_outcome``."""

from __future__ import annotations

from .models import ExploreRequest


def winner_decision(
    req: ExploreRequest,
    throughput: float | None,
    accuracy: float | None,
    completed: str,
) -> tuple[bool, str]:
    """Apply throughput / accuracy / completed gates for a candidate.

    Short-circuits on the first failing gate: (1) throughput > 0 and ratio
    >= ``min_throughput_ratio``; (2) when baseline accuracy is set, accuracy
    present and drop <= ``max_accuracy_drop``; (3) ``completed`` "K/N" must
    have K == N.

    Args:
        req: The explore request carrying baseline and thresholds.
        throughput: Candidate throughput, or ``None`` if unmeasured.
        accuracy: Candidate accuracy, or ``None`` if unmeasured.
        completed: Benchmark completion marker in ``"K/N"`` form.

    Returns:
        A ``(is_winner, reason)`` tuple; ``reason`` is always set for audit.
    """
    if throughput is None or throughput <= 0:
        return False, "missing throughput"
    if req.baseline.throughput <= 0:
        return False, "baseline throughput is 0 — cannot compute ratio"
    ratio = throughput / req.baseline.throughput
    if ratio < req.thresholds.min_throughput_ratio:
        return (
            False,
            f"throughput ratio {ratio:.4f} below required "
            f"{req.thresholds.min_throughput_ratio:.4f}",
        )
    if req.baseline.accuracy is not None:
        if accuracy is None:
            return False, "missing accuracy while baseline accuracy is set"
        drop = req.baseline.accuracy - accuracy
        if drop > req.thresholds.max_accuracy_drop:
            return (
                False,
                f"accuracy drop {drop:.4f} exceeds max "
                f"{req.thresholds.max_accuracy_drop:.4f}",
            )
    if completed and "/" in completed:
        left, _, right = completed.partition("/")
        if left.strip() != right.strip():
            return False, f"benchmark completed={completed} is incomplete"
    return True, "throughput and accuracy gates passed"


def candidate_score(
    req: ExploreRequest,
    throughput: float | None,
    accuracy: float | None,
) -> float:
    """Compute a sortable ranking score for a candidate.

    The score is ``throughput_ratio - accuracy_drop_penalty`` (higher is
    better); missing throughput scores ``0.0`` so failed candidates sort to
    the tail.

    Args:
        req: The explore request carrying baseline and thresholds.
        throughput: Candidate throughput, or ``None`` if unmeasured.
        accuracy: Candidate accuracy, or ``None`` if unmeasured.

    Returns:
        The ranking score as a float.
    """
    if throughput is None or throughput <= 0 or req.baseline.throughput <= 0:
        return 0.0
    ratio = throughput / req.baseline.throughput
    if req.baseline.accuracy is not None and accuracy is not None:
        drop = max(0.0, req.baseline.accuracy - accuracy)
        # Penalty keeps accuracy-degraded candidates below clean ones.
        max_drop = max(req.thresholds.max_accuracy_drop, 1e-6)
        penalty = drop / max_drop
        return ratio - penalty
    return ratio


__all__ = ["candidate_score", "winner_decision"]
