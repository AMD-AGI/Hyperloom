# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decide an A/B comparison from interleaved pairs rather than two blocks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median

log = logging.getLogger(__name__)

# Same order as the KEEP margin: a difference the measurement cannot resolve is not a difference.
DEFAULT_THRESHOLD_PCT = 3.0

# One pair is a coincidence.
MIN_PAIRS = 2


@dataclass(frozen=True)
class PairedVerdict:
    """The outcome of a paired A/B comparison."""

    decisive: bool
    reason: str
    # Positive => B (the candidate) is faster than A (the baseline).
    median_delta_pct: float | None

    @property
    def candidate_wins(self) -> bool:
        return self.decisive and self.reason == "candidate_faster"


def interleaved_plan(n_pairs: int) -> list[str]:
    """The measurement order for ``n_pairs`` pairs: A, B, A, B, ..."""
    return [side for _ in range(max(int(n_pairs), 0)) for side in ("A", "B")]


def assess_paired(pairs: list[tuple[float, float]]) -> PairedVerdict:
    """Judge interleaved ``(baseline, candidate)`` throughput pairs."""
    usable = [
        (float(a), float(b))
        for a, b in pairs
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0
    ]
    if len(usable) < MIN_PAIRS:
        return PairedVerdict(False, "insufficient_pairs", None)

    deltas = [round((b - a) / a * 100.0, 4) for a, b in usable]
    signs = {d > 0 for d in deltas if d != 0}
    med = round(median(deltas), 4)

    if len(signs) > 1:
        # The pairs disagree about which side is faster.
        log.info("paired A/B inconclusive: deltas disagree in sign %s", deltas)
        return PairedVerdict(False, "sign_disagreement", med)

    if med > DEFAULT_THRESHOLD_PCT:
        return PairedVerdict(True, "candidate_faster", med)
    if med < -DEFAULT_THRESHOLD_PCT:
        return PairedVerdict(True, "candidate_slower", med)
    return PairedVerdict(True, "within_noise", med)
