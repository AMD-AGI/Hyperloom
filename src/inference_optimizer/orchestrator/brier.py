"""BrierTracker — DESIGN §9.4 / §G11.

Per-agent rolling Brier score for predicted-vs-actual gain. Used by the
parliament to weight votes (mature data only — DESIGN §9.4 footnote) and
by ``early_stop.signal_brier_plateau`` to detect a Critic that has
stopped improving.

A *Brier score* here is the squared error between predicted and actual
gain percentage, normalized to [0, 1] via ``min(1, |delta| / 100)``. A
LOWER score is better.

STATUS (v0.7):
    Pure-Python implementation. Storage is in-memory by default; the
    Conductor can persist via ``snapshot()`` / ``restore()`` for
    resumability.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


__all__ = [
    "BrierEntry",
    "BrierTracker",
    "weight_for_score",
]


_DEFAULT_WINDOW: int = 50
_DEFAULT_SHRINKAGE_PRIOR: float = 0.25
_DEFAULT_SHRINKAGE_K: int = 5


@dataclass
class BrierEntry:
    agent: str
    predicted_gain_pct: float
    actual_gain_pct: float
    score: float

    @classmethod
    def make(cls, agent: str, predicted: float, actual: float) -> "BrierEntry":
        delta = abs(float(predicted) - float(actual))
        # normalize: a 100 percentage-point miss → 1.0
        score = max(0.0, min(1.0, delta / 100.0))
        return cls(agent=agent, predicted_gain_pct=float(predicted),
                   actual_gain_pct=float(actual), score=score)


class BrierTracker:
    """Rolling per-agent Brier window (default 50 most-recent samples)."""

    def __init__(
        self,
        *,
        window: int = _DEFAULT_WINDOW,
        prior: float = _DEFAULT_SHRINKAGE_PRIOR,
        prior_k: int = _DEFAULT_SHRINKAGE_K,
    ) -> None:
        self.window = max(1, int(window))
        self.prior = float(prior)
        self.prior_k = max(1, int(prior_k))
        self._per_agent: dict[str, deque[BrierEntry]] = {}

    # ------------------------------------------------------------------
    def record(
        self, agent: str, *, predicted_gain_pct: float, actual_gain_pct: float
    ) -> BrierEntry:
        entry = BrierEntry.make(agent, predicted_gain_pct, actual_gain_pct)
        bucket = self._per_agent.setdefault(agent, deque(maxlen=self.window))
        bucket.append(entry)
        return entry

    def score_for(self, agent: str) -> float:
        """Smoothed mean Brier score for ``agent``.

        Uses additive shrinkage with ``prior`` weighted by ``prior_k`` so a
        Critic with only a couple of samples cannot dominate the parliament
        before they have a track record.
        """
        bucket = self._per_agent.get(agent)
        if not bucket:
            return self.prior
        observed_sum = sum(e.score for e in bucket)
        n = len(bucket)
        smoothed = (observed_sum + self.prior * self.prior_k) / (n + self.prior_k)
        return float(smoothed)

    def weight_for(self, agent: str) -> float:
        """Vote weight = ``1 / (1 + score)`` clamped to [0.1, 1.0]."""
        return weight_for_score(self.score_for(agent))

    def history(self, agent: str) -> tuple[BrierEntry, ...]:
        return tuple(self._per_agent.get(agent, ()))

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "prior": self.prior,
            "prior_k": self.prior_k,
            "per_agent": {
                agent: [
                    {
                        "agent": e.agent,
                        "predicted_gain_pct": e.predicted_gain_pct,
                        "actual_gain_pct": e.actual_gain_pct,
                        "score": e.score,
                    }
                    for e in bucket
                ]
                for agent, bucket in self._per_agent.items()
            },
        }

    @classmethod
    def restore(cls, snap: dict[str, Any]) -> "BrierTracker":
        t = cls(
            window=int(snap.get("window", _DEFAULT_WINDOW)),
            prior=float(snap.get("prior", _DEFAULT_SHRINKAGE_PRIOR)),
            prior_k=int(snap.get("prior_k", _DEFAULT_SHRINKAGE_K)),
        )
        for agent, entries in (snap.get("per_agent") or {}).items():
            bucket = t._per_agent.setdefault(agent, deque(maxlen=t.window))
            for e in entries:
                bucket.append(
                    BrierEntry(
                        agent=str(e.get("agent", agent)),
                        predicted_gain_pct=float(e.get("predicted_gain_pct", 0.0)),
                        actual_gain_pct=float(e.get("actual_gain_pct", 0.0)),
                        score=float(e.get("score", 0.0)),
                    )
                )
        return t


def weight_for_score(score: float) -> float:
    """Map a Brier score (0 perfect → 1 awful) onto a vote weight."""
    s = max(0.0, min(1.0, float(score)))
    raw = 1.0 / (1.0 + s)
    return max(0.1, min(1.0, raw))
