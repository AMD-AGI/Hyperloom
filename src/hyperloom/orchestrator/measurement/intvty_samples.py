# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much a graded comparison can carry, read off the requests behind it.

A graded axis is one percentile of one run's requests, and a round reports it as a single number, so two rounds that
differ by less than their own request-to-request spread read exactly like two that differ by more. The requests are
already on disk: ``aiperf`` writes one record per request, and those records answer two questions a pair of
aggregates cannot -- whether a reported delta is separable at all, and whether the two runs were even in the same
state while they measured it.

Both answers are advisory. The interval covers request sampling only -- not the run-to-run spread, which on this
workload is larger -- so a separable interval is a floor on the uncertainty, never a licence to treat one round as
a result.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_INTVTY_P50

log = logging.getLogger(__name__)

#: Decimal places every ratio and percentage in this module is reported to.
_REPORTED_PLACES: int = 3

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

#: How far two runs' TTFT inflation may diverge before the pair is called cross-regime. Set between the spread
#: within one state on the GLM-5.3 host (1.6-1.7, and 3.4-5.4) and the gap between the two states.
CROSS_REGIME_RATIO: float = 1.8


def profiling_records(workspace: str | Path | None) -> list[dict[str, Any]]:
    """Per-request metric blocks for the scored phase of the run under *workspace*, newest export wins."""
    root = Path(str(workspace or "")).expanduser()
    if not str(workspace or "").strip() or not root.is_dir():
        return []
    exports = sorted(root.glob("**/aiperf_artifacts/profile_export.jsonl"), key=lambda p: p.stat().st_mtime)
    if not exports:
        return []
    records: list[dict[str, Any]] = []
    try:
        with exports[-1].open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = json.loads(line) if line.strip() else {}
                if not isinstance(record, dict):
                    continue
                if (record.get("metadata") or {}).get("benchmark_phase") != _SCORED_PHASE:
                    continue
                metrics = record.get("metrics")
                if isinstance(metrics, dict):
                    records.append(metrics)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("intvty_samples: %s unusable: %s", exports[-1], exc)
        return []
    return records


def metric_values(records: list[dict[str, Any]], name: str) -> list[float]:
    """Positive values of one per-request metric, skipping records that did not report it."""
    out: list[float] = []
    for metrics in records:
        value = (metrics.get(name) or {}).get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            out.append(float(value))
    return out


def profiling_samples(workspace: str | Path | None) -> list[float]:
    """Per-request interactivity for the scored phase of the run under *workspace*."""
    return metric_values(profiling_records(workspace), _SAMPLE_METRIC)


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
        "delta_pct": round((point - base) / base * 100.0, _REPORTED_PLACES),
        "ci_low_pct": round(low, _REPORTED_PLACES),
        "ci_high_pct": round(high, _REPORTED_PLACES),
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


def regime_fingerprint(records: list[dict[str, Any]]) -> dict[str, float] | None:
    """How stalled a run was while it measured, as the inflation of its own latency means over their medians.

    An engine stall lengthens every request in flight, so it lands in the mean long before the median moves. On the
    GLM-5.3 AgentX workload the two states a host sat in were cleanly separated by this one ratio: runs in the slow
    state carried TTFT mean/median of 3.4-5.4, runs in the fast state 1.6-1.7, with medians that barely differed.
    A pair of aggregates cannot show that; these records can.
    """
    out: dict[str, float] = {}
    for key, metric in (("ttft", "time_to_first_token"), ("itl", "inter_token_latency")):
        values = sorted(metric_values(records, metric))
        if len(values) < _MIN_SAMPLES:
            continue
        median = quantile(values, 0.50)
        if median <= 0:
            continue
        out[f"{key}_p50_ms"] = round(median, 2)
        out[f"{key}_mean_ms"] = round(sum(values) / len(values), 2)
        out[f"{key}_inflation"] = round(out[f"{key}_mean_ms"] / median, _REPORTED_PLACES)
    return out or None


def regimes_differ(candidate: dict[str, float] | None, reference: dict[str, float] | None) -> float | None:
    """Ratio between two runs' TTFT inflation, larger side over smaller; ``None`` when either is unmeasured."""
    if not candidate or not reference:
        return None
    left, right = candidate.get("ttft_inflation"), reference.get("ttft_inflation")
    if not left or not right:
        return None
    return round(max(left, right) / min(left, right), _REPORTED_PLACES)


def comparison_regimes(
    candidate_workspace: str | Path | None,
    reference_workspace: str | Path | None,
) -> dict[str, Any] | None:
    """Both runs' stall fingerprints and how far apart they are, or ``None`` when neither can be read."""
    candidate = regime_fingerprint(profiling_records(candidate_workspace))
    reference = regime_fingerprint(profiling_records(reference_workspace))
    spread = regimes_differ(candidate, reference)
    if candidate is None and reference is None:
        return None
    return {
        "candidate": candidate,
        "reference": reference,
        "inflation_ratio": spread,
        "cross_regime": bool(spread is not None and spread >= CROSS_REGIME_RATIO),
    }


__all__ = [
    "CROSS_REGIME_RATIO",
    "bootstrap_delta_ci",
    "comparison_interval",
    "comparison_regimes",
    "graded_quantile",
    "metric_values",
    "profiling_records",
    "profiling_samples",
    "quantile",
    "regime_fingerprint",
    "regimes_differ",
]
