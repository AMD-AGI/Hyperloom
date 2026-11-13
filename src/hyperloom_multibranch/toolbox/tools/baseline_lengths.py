# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prompt and output lengths of a request on the session's baseline, which weight the phase windows of a profile whose
point names none (:mod:`~.phase_mix`).

An AgentX point replays a corpus of multi-turn conversations: it names a concurrency but no ``ISL`` or ``OSL``, and most
of a request's prompt is a prefix the server's prefix cache already holds. A prefill step computes only the rest, so the
prompt length that weights the prefill window is what the server computed of a request:

    computed prompt tokens per request = mean prompt tokens per request x (1 - prefix-cache hit rate)

The mean prompt tokens per request are the baseline's input throughput x duration / completed requests, its output tokens
per request the same of its output throughput (each figure the mean over its measured rounds, as the baseline records
it). The hit rate is the share of the prompt tokens the server looked up in its prefix cache that it found there, summed
over the measured phase of each measuring round the baseline tests landed (``kv_metrics.json``, vLLM's
``prefix_cache_queries`` and ``prefix_cache_hits`` counters). A baseline whose rounds record no such split (SGLang counts
the tokens it found, not those it looked up) is weighted by the mean prompt tokens per request, which counts the cached
prefix as prefill work, and the source says so. A baseline without these figures gives no lengths, and the mix stays
degraded: a length is never guessed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_DURATION, GRADED_OUTPUT
from hyperloom_multibranch.toolbox.tools.phase_mix import MEASURED_ISL_KEY, MEASURED_OSL_KEY, MEASURED_SOURCE_KEY
from inference_suite.contracts.layout import KV_METRICS

#: The KV-cache record of each measuring round in a test's landed directory; a warm-up round's lies a level further down.
ROUND_KV_METRICS_GLOB = f"variant_*/{KV_METRICS}"
#: The baseline's mean prompt tokens per second and completed requests, as its measured rounds report them.
INPUT_THROUGHPUT, COMPLETED_REQUESTS = "input_throughput", "completed_requests"
#: Where ``kv_metrics.json`` keeps the prefix-cache counters' increments over the round's measured phase.
_PREFIX_CACHE, _QUERIED, _HIT = "prefix_cache", "prefix_cache_queries_delta", "prefix_cache_hits_delta"
_KV_MAX_BYTES = 64 << 20
_ENCODING = "utf-8"


@dataclass(frozen=True)
class PrefixCacheReading:
    """The prompt tokens a baseline's server looked up in its prefix cache over its measured rounds, and found there."""

    queried: float
    hit: float

    @property
    def hit_rate(self) -> float:
        return self.hit / self.queried


def _round_counts(path: Path) -> PrefixCacheReading | None:
    """The prefix-cache counts of one round's ``kv_metrics.json``; ``None`` when it records no lookup and hit both."""
    try:
        if path.stat().st_size > _KV_MAX_BYTES:
            return None
        window = json.loads(path.read_text(encoding=_ENCODING)).get(_PREFIX_CACHE)
    except (OSError, ValueError, AttributeError):
        return None
    counts = [window.get(key) if isinstance(window, dict) else None for key in (_QUERIED, _HIT)]
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in counts):
        return None
    return PrefixCacheReading(float(counts[0]), float(counts[1]))


def prefix_cache_reading(landed: Iterable[Path]) -> PrefixCacheReading | None:
    """The prefix-cache counts of every measuring round below the ``landed`` test directories that records them, summed;
    ``None`` when none does, or when the sums are no split of the prompt (no lookup, a hit past it, no token computed)."""
    rounds = [
        counts
        for directory in landed
        for path in sorted(directory.glob(ROUND_KV_METRICS_GLOB))
        if (counts := _round_counts(path)) is not None
    ]
    queried, hit = math.fsum(c.queried for c in rounds), math.fsum(c.hit for c in rounds)
    return PrefixCacheReading(queried, hit) if 0 <= hit < queried else None


def _per_request(values: Mapping[str, float], throughput: str) -> float | None:
    """Tokens per completed request of the baseline's ``throughput``; ``None`` when a figure is missing or not positive."""
    figures = [values.get(key) for key in (throughput, GRADED_DURATION, COMPLETED_REQUESTS)]
    if not all(isinstance(f, (int, float)) and math.isfinite(f) and f > 0 for f in figures):
        return None
    rate, duration, completed = figures
    return rate * duration / completed


def baseline_lengths(values: Mapping[str, float], cache: PrefixCacheReading | None) -> dict[str, str]:
    """The request keys of the prompt and output lengths per request the baseline's reading ``values`` measured, the
    prompt's computed share when ``cache`` splits it, and their source; empty when the reading lacks a figure."""
    isl, osl = _per_request(values, INPUT_THROUGHPUT), _per_request(values, GRADED_OUTPUT)
    if isl is None or osl is None:
        return {}
    if cache is None:
        weighted = isl
        source = (
            f"the AgentX baseline's mean prompt and output tokens per request, {isl:.0f} and {osl:.0f}; its rounds "
            "record no prefix-cache split, so the cached prefix counts as prefill work"
        )
    else:
        weighted = isl * (1 - cache.hit_rate)
        source = (
            f"the AgentX baseline's computed prompt tokens per request, {weighted:.0f} ({isl:.0f} less the "
            f"{cache.hit_rate:.1%} its prefix cache served), and its {osl:.0f} output tokens per request"
        )
    return {MEASURED_ISL_KEY: f"{weighted:.6g}", MEASURED_OSL_KEY: f"{osl:.6g}", MEASURED_SOURCE_KEY: source}


__all__ = [
    "PrefixCacheReading",
    "baseline_lengths",
    "prefix_cache_reading",
]
