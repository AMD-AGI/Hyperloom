"""Winner-decision gates split out of ``explorer.py``.

Per merged-design §2.2 / §4.6, the 3-gate logic
(throughput / accuracy / completed) lives in its own module so it can
be tested in isolation and reused by callers that integrate fa as a
library (e.g. ``runtime/tools_api.evaluate_candidate_outcome``).

The behaviour is byte-equivalent to the previous ``_winner_decision``
in explorer.py; only the module boundary moved.
"""

from __future__ import annotations

from .models import ExploreRequest


def winner_decision(
    req: ExploreRequest,
    throughput: float | None,
    accuracy: float | None,
    completed: str,
) -> tuple[bool, str]:
    """Apply throughput / accuracy / completed gates.

    Returns ``(is_winner, reason)``. ``reason`` is always populated so
    the explore summary can audit why a candidate was rejected.

    Gate order (short-circuit on first miss):

      1. **Throughput presence + ratio gate**: throughput must be > 0 and
         ``throughput / req.baseline.throughput >= thresholds.min_throughput_ratio``.
      2. **Accuracy drop gate** (only when baseline accuracy is set): the
         candidate must produce an accuracy reading and the drop must not
         exceed ``thresholds.max_accuracy_drop``.
      3. **Completed N/N gate**: when the benchmark exposes a
         ``completed`` field of the form ``"K/N"`` the gate fails unless
         ``K == N`` (partial benchmark runs cannot be promoted).
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
    """Compute a sortable score for ranking mode (merged-design §4.4.1).

    Score = ``throughput_ratio - accuracy_drop_penalty``. Used by
    ``explore()`` to sort the result list when no early-stop ``winner``
    short-circuits the loop. Higher = better. Missing throughput is
    treated as 0 so failed candidates sort to the tail without crashing
    the comparator.
    """
    if throughput is None or throughput <= 0 or req.baseline.throughput <= 0:
        return 0.0
    ratio = throughput / req.baseline.throughput
    if req.baseline.accuracy is not None and accuracy is not None:
        drop = max(0.0, req.baseline.accuracy - accuracy)
        # Convert "drop above the allowed max" into a positive penalty so
        # an accuracy-degraded candidate still scores below a clean one.
        max_drop = max(req.thresholds.max_accuracy_drop, 1e-6)
        penalty = drop / max_drop
        return ratio - penalty
    return ratio


__all__ = ["candidate_score", "winner_decision"]
