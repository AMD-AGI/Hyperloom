# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""In-process metrics shim.

In production the Critic agent will export the metrics enumerated in
``kb-critic-integration-contract`` Appendix E via Prometheus. The runtime
here keeps a process-local registry so unit tests can assert that the
right counters fire without requiring a Prometheus client dependency.

If Prometheus support is later wired in, replace
:func:`get_registry` with a real client and keep the API surface
(``inc`` / ``observe``) identical.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class _Counter:
    """A monotonically increasing counter keyed by label sets.

    Attributes:
        name (str): Metric name.
        values (dict[tuple[tuple[str, str], ...], float]): Accumulated totals
            keyed by a sorted tuple of label ``(key, value)`` pairs.
    """

    name: str
    values: dict[tuple[tuple[str, str], ...], float] = field(default_factory=dict)

    def inc(self, labels: dict[str, str] | None = None, by: float = 1.0) -> None:
        """Increment the counter for a given label set.

        Args:
            labels (dict[str, str] | None): Label key/value pairs identifying
                the series; ``None`` targets the empty label set.
            by (float): Amount to add.
        """
        key = tuple(sorted((labels or {}).items()))
        self.values[key] = self.values.get(key, 0.0) + by


@dataclass
class _Histogram:
    """A sample-collecting histogram keyed by label sets.

    Attributes:
        name (str): Metric name.
        samples (dict[tuple[tuple[str, str], ...], list[float]]): Observed
            values keyed by a sorted tuple of label ``(key, value)`` pairs.
    """

    name: str
    samples: dict[tuple[tuple[str, str], ...], list[float]] = field(default_factory=lambda: defaultdict(list))

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record one observed value.

        Args:
            value (float): The sample value to record.
            labels (dict[str, str] | None): Label key/value pairs identifying
                the series; ``None`` targets the empty label set.
        """
        key = tuple(sorted((labels or {}).items()))
        self.samples[key].append(float(value))


class MetricsRegistry:
    """Tiny in-memory metrics registry."""

    def __init__(self) -> None:
        """Initialise empty counter and histogram maps with a lock."""
        self._lock = Lock()
        self._counters: dict[str, _Counter] = {}
        self._histograms: dict[str, _Histogram] = {}

    def counter(self, name: str) -> _Counter:
        """Get or lazily create a counter by name.

        Args:
            name (str): Metric name.

        Returns:
            _Counter: The existing or newly created counter for ``name``.
        """
        with self._lock:
            counter = self._counters.get(name)
            if counter is None:
                counter = _Counter(name=name)
                self._counters[name] = counter
            return counter

    def histogram(self, name: str) -> _Histogram:
        """Get or lazily create a histogram by name.

        Args:
            name (str): Metric name.

        Returns:
            _Histogram: The existing or newly created histogram for ``name``.
        """
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = _Histogram(name=name)
                self._histograms[name] = hist
            return hist

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copied view of all metric values.

        Returns:
            dict[str, Any]: A mapping with ``counters`` and ``histograms``
            keys holding copies of the current values per series.
        """
        return {
            "counters": {n: dict(c.values) for n, c in self._counters.items()},
            "histograms": {n: {k: list(v) for k, v in h.samples.items()} for n, h in self._histograms.items()},
        }

    def reset(self) -> None:
        """Clear all registered counters and histograms."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry singleton.

    Returns:
        MetricsRegistry: The shared registry instance.
    """
    return _registry


# Conventional metric names — keep stable so dashboards don't churn.
CRITIC_KB_WRITE_TOTAL = "critic_kb_write_total"
CRITIC_KB_WRITE_DURATION_SECONDS = "critic_kb_write_duration_seconds"
CRITIC_KB_DEAD_LETTER_COUNT = "critic_kb_dead_letter_count"
CRITIC_KB_DISTILL_TOKENS_TOTAL = "critic_kb_distill_tokens_total"
CRITIC_KB_DISTILL_COST_TOTAL = "critic_kb_distill_cost_total"
CRITIC_KB_PRIOR_CACHE_HIT = "critic_kb_prior_cache_hit"
CRITIC_KB_PRIOR_CACHE_MISS = "critic_kb_prior_cache_miss"
CRITIC_KB_UNREACHABLE_TOTAL = "critic_kb_unreachable_total"
CRITIC_KB_BREAKER_OPEN_TOTAL = "critic_kb_breaker_open_total"
CRITIC_REVIEW_VERDICT_TOTAL = "critic_review_verdict_total"


__all__ = [
    "CRITIC_KB_BREAKER_OPEN_TOTAL",
    "CRITIC_KB_DEAD_LETTER_COUNT",
    "CRITIC_KB_DISTILL_COST_TOTAL",
    "CRITIC_KB_DISTILL_TOKENS_TOTAL",
    "CRITIC_KB_PRIOR_CACHE_HIT",
    "CRITIC_KB_PRIOR_CACHE_MISS",
    "CRITIC_KB_UNREACHABLE_TOTAL",
    "CRITIC_KB_WRITE_DURATION_SECONDS",
    "CRITIC_KB_WRITE_TOTAL",
    "CRITIC_REVIEW_VERDICT_TOTAL",
    "MetricsRegistry",
    "get_registry",
]
