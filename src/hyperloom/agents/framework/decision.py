# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Winner-decision gates (throughput / accuracy / completed) for candidate evaluation."""

from __future__ import annotations

from typing import Any, Iterable

from hyperloom.common.coerce import to_float

from .models import Candidate, ExploreRequest


def _field(obj: Candidate | dict[str, Any], name: str, default: Any = "") -> Any:
    """Read ``name`` from a Candidate-like object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _keywords(values: Iterable[str] | None) -> set[str]:
    """Normalize keyword strings for prior-score association."""
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def _jaccard(left: set[str], right: set[str]) -> float:
    """Return Jaccard overlap for two keyword sets."""
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def prior_score(
    candidate: Candidate | dict[str, Any],
    *,
    gap_canonical_id: str = "",
    gap_keywords: Iterable[str] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    min_samples: int = 1,
) -> float:
    """Compute a KB-derived pre-benchmark priority for a PR candidate.

    The score is advisory only: callers should use it to sort candidates
    before expensive benchmarking, never to bypass throughput/accuracy gates.
    Cold starts (no matching history) return ``0.0``.

    Args:
        candidate: Candidate object or candidate dict.
        gap_canonical_id: Current canonical gap id; falls back to the candidate.
        gap_keywords: Current gap keywords; falls back to the candidate.
        ledger: Historical ``lessons.jsonl`` records.
        min_samples: Minimum associated records needed to emit a non-zero score.

    Returns:
        A score in ``[0, 1]`` combining exact/fuzzy association, historical
        throughput gain, and historical integration success.
    """
    records = ledger or []
    if not records:
        return 0.0

    cand_framework = str(_field(candidate, "framework", "") or "").strip().lower()
    current_gap = str(gap_canonical_id or _field(candidate, "gap_canonical_id", "") or "").strip()
    current_keywords = _keywords(gap_keywords or _field(candidate, "gap_keywords", []) or [])
    param_fields = ("model_class", "gpu_type", "precision")
    candidate_params = {
        field: str(_field(candidate, field, "") or "").strip().lower()
        for field in param_fields
        if str(_field(candidate, field, "") or "").strip()
    }
    candidate_urls = {
        str(v).strip()
        for v in (
            _field(candidate, "pr_url", ""),
            _field(candidate, "html_url", ""),
        )
        if str(v).strip()
    }
    candidate_shas = {
        str(v).strip()
        for v in (
            _field(candidate, "pr_sha", ""),
            _field(candidate, "head_sha", ""),
        )
        if str(v).strip()
    }

    exact_pr_records: list[tuple[float, dict[str, Any]]] = []
    associated: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        rec_framework = str(rec.get("framework") or "").strip().lower()
        if cand_framework and rec_framework and cand_framework != rec_framework:
            continue
        rec_url = str(rec.get("pr_url") or "").strip()
        rec_sha = str(rec.get("pr_sha") or "").strip()
        exact_pr = bool((rec_url and rec_url in candidate_urls) or (rec_sha and rec_sha in candidate_shas))
        rec_gap = str(rec.get("gap_canonical_id") or "").strip()
        exact = bool(current_gap and rec_gap and current_gap == rec_gap)
        fuzzy = _jaccard(current_keywords, _keywords(rec.get("gap_keywords") or []))
        if exact_pr:
            exact_pr_records.append((1.0, rec))
            continue
        if not exact and fuzzy <= 0.0:
            continue
        association = 1.0 if exact else fuzzy
        associated.append((association, rec))

    if exact_pr_records:
        associated = exact_pr_records
    if len(associated) < max(1, min_samples):
        return 0.0

    weights = [score for score, _ in associated]
    weight_sum = sum(weights) or 1.0
    avg_association = sum(weights) / len(weights)
    gain = sum(to_float(rec.get("tps_delta_pct"), default=0.0) * w for w, rec in associated) / weight_sum
    gain_score = max(0.0, min(1.0, gain / 20.0))
    param_score = 0.0
    if candidate_params:
        param_hits = 0.0
        for weight, rec in associated:
            matches = 0
            for field, wanted in candidate_params.items():
                if str(rec.get(field) or "").strip().lower() == wanted:
                    matches += 1
            if matches == len(candidate_params):
                param_hits += weight
        param_score = param_hits / weight_sum

    def _success_value(outcome: str) -> float:
        if outcome == "integrated":
            return 1.0
        if outcome == "already_present":
            return 0.5
        return 0.0

    apply_score = (
        sum(_success_value(str(rec.get("outcome") or "")) * w for w, rec in associated) / weight_sum
    )
    quality = 0.45 * apply_score + 0.35 * gain_score + 0.20 * param_score
    score = min(1.0, avg_association) * quality
    return round(max(0.0, min(1.0, score)), 4)


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
            f"throughput ratio {ratio:.4f} below required {req.thresholds.min_throughput_ratio:.4f}",
        )
    if req.baseline.accuracy is not None:
        if accuracy is None:
            return False, "missing accuracy while baseline accuracy is set"
        drop = req.baseline.accuracy - accuracy
        if drop > req.thresholds.max_accuracy_drop:
            return (
                False,
                f"accuracy drop {drop:.4f} exceeds max {req.thresholds.max_accuracy_drop:.4f}",
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


__all__ = ["candidate_score", "prior_score", "winner_decision"]
