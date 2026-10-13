# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How far a graded interactivity comparison is from being indistinguishable, read off the requests behind it.

A graded axis is one percentile of one run's requests, and a round reports it as a single number, so two rounds that
differ by less than their own request-to-request spread read exactly like two that differ by more. The requests are
already on disk: ``aiperf`` writes one record per request, and the sample those records form is what says whether a
reported delta is separable at all.

The interval is advisory. It covers request sampling only -- not the run-to-run spread, which on this workload is
larger -- so a separable interval is a floor on the uncertainty, never a licence to treat one round as a result.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_INTVTY_P50

log = logging.getLogger(__name__)

#: The per-request metric a graded interactivity axis is a percentile of.
_SAMPLE_METRIC: str = "e2e_output_token_throughput"

#: Only the scored phase counts; aiperf writes warmup records to the same file.
_SCORED_PHASE: str = "profiling"

#: Quantile of the per-request distribution each graded axis names. The slow-tail axis is called ``p90`` because it
#: is the 90th percentile of *latency*, which is the 10th of the throughput these records carry.
_AXIS_QUANTILE: dict[str, float] = {GRADED_INTVTY_P50: 0.50, GRADED_INTVTY: 0.10}

#: Resamples behind an interval. Enough to place a 95% interval to within about a percent of its width, and cheap
#: enough to run while rendering a report.
_RESAMPLES: int = 500

#: Fixed so a report re-rendered from the same runs states the same interval.
_SEED: int = 20260101

#: A percentile of fewer requests than this says more about which requests landed than about the configuration.
_MIN_SAMPLES: int = 30


def profiling_samples(workspace: str | Path | None) -> list[float]:
    """Per-request interactivity for the scored phase of the run under *workspace*, newest export wins."""
    root = Path(str(workspace or "")).expanduser()
    if not str(workspace or "").strip() or not root.is_dir():
        return []
    exports = sorted(root.glob("**/aiperf_artifacts/profile_export.jsonl"), key=lambda p: p.stat().st_mtime)
    if not exports:
        return []
    samples: list[float] = []
    try:
        with exports[-1].open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = json.loads(line) if line.strip() else {}
                if not isinstance(record, dict):
                    continue
                if (record.get("metadata") or {}).get("benchmark_phase") != _SCORED_PHASE:
                    continue
                value = ((record.get("metrics") or {}).get(_SAMPLE_METRIC) or {}).get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    samples.append(float(value))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("intvty_samples: %s unusable: %s", exports[-1], exc)
        return []
    return samples


def quantile(sorted_samples: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted, non-empty sample."""
    index = (len(sorted_samples) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return sorted_samples[low]
    return sorted_samples[low] + (sorted_samples[high] - sorted_samples[low]) * (index - low)


def graded_quantile(samples: list[float], objective: str) -> float | None:
    """The graded axis *objective* names, recomputed from *samples*; ``None`` when the axis is not one of them."""
    q = _AXIS_QUANTILE.get(objective)
    if q is None or not samples:
        return None
    return quantile(sorted(samples), q)


def bootstrap_delta_ci(
    candidate: list[float],
    reference: list[float],
    objective: str,
    *,
    confidence: float = 0.95,
) -> dict[str, float | bool | int] | None:
    """Interval on the percent delta between two runs' graded axis, or ``None`` when the samples cannot carry one."""
    q = _AXIS_QUANTILE.get(objective)
    if q is None or len(candidate) < _MIN_SAMPLES or len(reference) < _MIN_SAMPLES:
        return None
    point = graded_quantile(candidate, objective)
    base = graded_quantile(reference, objective)
    if not point or not base:
        return None
    rng = random.Random(_SEED)
    deltas: list[float] = []
    for _ in range(_RESAMPLES):
        boot_c = quantile(sorted(rng.choices(candidate, k=len(candidate))), q)
        boot_r = quantile(sorted(rng.choices(reference, k=len(reference))), q)
        if boot_r > 0:
            deltas.append((boot_c - boot_r) / boot_r * 100.0)
    if len(deltas) < _RESAMPLES // 2:
        return None
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    low, high = quantile(deltas, tail), quantile(deltas, 1.0 - tail)
    return {
        "delta_pct": round((point - base) / base * 100.0, 3),
        "ci_low_pct": round(low, 3),
        "ci_high_pct": round(high, 3),
        "crosses_zero": low <= 0.0 <= high,
        "n_candidate": len(candidate),
        "n_reference": len(reference),
        "confidence": confidence,
    }


def comparison_interval(
    candidate_workspace: str | Path | None,
    reference_workspace: str | Path | None,
    objective: str,
) -> dict[str, float | bool | int] | None:
    """Interval for a reported comparison, read from the two runs' request exports."""
    return bootstrap_delta_ci(
        profiling_samples(candidate_workspace),
        profiling_samples(reference_workspace),
        objective,
    )


__all__ = [
    "bootstrap_delta_ci",
    "comparison_interval",
    "graded_quantile",
    "profiling_samples",
    "quantile",
]
