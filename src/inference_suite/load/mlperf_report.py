# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map the MLPerf agentic harness's ``result_summary.json`` and ``scores.json`` into the InferenceX result schema.

This module imports nothing beyond the standard library: Hyperloom deploys it beside the harness's mapping script.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class MlperfReportError(ValueError):
    """A harness file does not have the upstream shape :func:`map_mlperf` reads."""


_AVG = "avg"
_STD_DEV = "std_dev"
_PERCENTILES = "percentiles"
_P50, _P75, _P90, _P99 = "p50", "p75", "p90", "p99"
_OSL = "output_sequence_lengths"
#: Each result key prefix, by the ``Report`` series it is read from; all are in nanoseconds.
_LATENCY_SERIES = (("ttft", "ttft"), ("tpot", "tpot"), ("e2el", "latency"))
#: The ``inference_endpoint.metrics.report.Report`` fields :func:`map_mlperf` reads.
_REPORT_FIELDS = (
    "n_samples_issued",
    "n_samples_completed",
    "n_samples_failed",
    "duration_ns",
    "state",
    "complete",
    "qps",
    "tps",
    *(name for _, name in _LATENCY_SERIES),
    _OSL,
)
#: The registry keys percentiles by ``str(float)``.
_PERCENTILE_KEYS = {_P50: "50.0", _P75: "75.0", _P90: "90.0", _P99: "99.0"}
#: Each latency key's statistic, by the series figure it is read from.
_LATENCY_STATS = (("mean", _AVG), ("std", _STD_DEV), ("median", _P50), (_P90, _P90), (_P99, _P99))
_OSL_STATS = (_AVG, _P50, _P75, _P90, _P99)
_NS_PER_MS = 1e6
_NS_PER_SEC = 1e9


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MlperfReportError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _series(report: Mapping[str, Any], name: str) -> dict[str, float] | None:
    """The mean, standard deviation and percentiles of one series; ``None`` when it recorded no samples (``{}``)."""
    block = report[name]
    if not isinstance(block, Mapping):
        raise MlperfReportError(f"{name}: expected a series object, got {type(block).__name__}")
    if not block:
        return None
    missing = [key for key in (_AVG, _STD_DEV, _PERCENTILES) if key not in block]
    if missing:
        raise MlperfReportError(f"{name}: series is missing {missing}")
    percentiles = block[_PERCENTILES]
    if not isinstance(percentiles, Mapping):
        raise MlperfReportError(f"{name}.{_PERCENTILES}: expected an object")
    out = {key: _finite(block[key], f"{name}.{key}") for key in (_AVG, _STD_DEV)}
    for label, key in _PERCENTILE_KEYS.items():
        if key not in percentiles:
            raise MlperfReportError(f"{name}.{_PERCENTILES} has no {key!r} (has {sorted(percentiles)})")
        out[label] = _finite(percentiles[key], f"{name}.{_PERCENTILES}[{key!r}]")
    return out


def _inline_accuracy(scores: Mapping[str, Any]) -> dict[str, Any]:
    """The harness's inline accuracy score and the turns it could not score."""
    if not isinstance(scores, Mapping) or "score" not in scores:
        raise MlperfReportError("scores.json: no 'score'")
    turns = scores.get("turns")
    if not isinstance(turns, Mapping) or "missing" not in turns:
        raise MlperfReportError("scores.json: no 'turns.missing'")
    return {
        "accuracy_score": _finite(scores["score"], "scores.score"),
        "accuracy_missing_turns": int(_finite(turns["missing"], "scores.turns.missing")),
    }


def _counts(report: Mapping[str, Any], corpus: str) -> dict[str, Any]:
    issued = int(_finite(report["n_samples_issued"], "n_samples_issued"))
    failed = int(_finite(report["n_samples_failed"], "n_samples_failed"))
    reasons = ["interrupted"] if report["state"] == "interrupted" else []
    if report["complete"] is not True:
        reasons.append("incomplete_run")
    return {
        "completed": int(_finite(report["n_samples_completed"], "n_samples_completed")),
        "request_error_rate": 100.0 * failed / issued if issued > 0 else None,
        "submission_valid": not reasons,
        "submission_invalid_reasons": reasons,
        "corpus_loader": corpus,
    }


def _rates(report: Mapping[str, Any]) -> dict[str, float]:
    rates = {}
    if report["duration_ns"] is not None:
        rates["duration"] = _finite(report["duration_ns"], "duration_ns") / _NS_PER_SEC
    if report["qps"] is not None:
        rates["request_throughput"] = _finite(report["qps"], "qps")
    if report["tps"] is not None:
        rates["output_throughput"] = _finite(report["tps"], "tps")
    return rates


def _latencies(report: Mapping[str, Any]) -> dict[str, Any]:
    latencies: dict[str, Any] = {}
    for prefix, name in _LATENCY_SERIES:
        series = _series(report, name)
        if series is not None:
            latencies.update({f"{stat}_{prefix}_ms": series[key] / _NS_PER_MS for stat, key in _LATENCY_STATS})
    osl = _series(report, _OSL)
    if osl is not None:
        latencies["total_output_tokens"] = int(_finite(report[_OSL].get("total"), "osl.total"))
        latencies["osl_distribution"] = {key: int(osl[key]) for key in _OSL_STATS}
    return latencies


def map_mlperf(report: Mapping[str, Any], *, corpus: str, scores: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Convert ``result_summary.json`` (and the inline ``scores.json``, when the run wrote one) into the InferenceX
    result schema, recording ``corpus`` as the dataset the trajectories came from.

    The harness publishes no per-request OSL or E2EL series, so no ``e2e_norm_intvty_*`` key is written. A field the
    upstream ``Report`` does not carry is absent, never 0.0.

    Raises:
        MlperfReportError: A file does not match the upstream schema.
    """
    if not isinstance(report, Mapping):
        raise MlperfReportError("result_summary.json: expected an object")
    missing = [key for key in _REPORT_FIELDS if key not in report]
    if missing:
        raise MlperfReportError(f"result_summary.json is missing {missing}")
    mapped = {**_counts(report, corpus), **_rates(report), **_latencies(report)}
    return mapped if scores is None else {**mapped, **_inline_accuracy(scores)}


__all__ = ["MlperfReportError", "map_mlperf"]
