# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map the MLCommons inference-endpoint report into Hyperloom metrics."""

from __future__ import annotations

import math
from typing import Any, Mapping


class MlperfReportError(ValueError):
    """An MLPerf harness file does not have the upstream shape :func:`map_mlperf` reads."""


# ``inference_endpoint.metrics.report.Report`` fields :func:`map_mlperf` reads.
_REPORT_FIELDS = (
    "n_samples_issued",
    "n_samples_completed",
    "n_samples_failed",
    "duration_ns",
    "state",
    "complete",
    "qps",
    "tps",
    "ttft",
    "tpot",
    "latency",
    "output_sequence_lengths",
)

# One series as ``_series_to_metric_dict`` writes it. A series that recorded no
# samples is written as ``{}``.
_SERIES_FIELDS = ("avg", "std_dev", "percentiles")

# The registry keys percentiles by ``str(float)``.
_PERCENTILE_KEYS = {"p50": "50.0", "p75": "75.0", "p90": "90.0", "p99": "99.0"}

# TTFT, TPOT and sample latency are recorded in nanoseconds.
_NS_PER_MS = 1e6

# Result key prefix -> ``Report`` series, all in nanoseconds.
_LATENCY_SERIES = (("ttft", "ttft"), ("tpot", "tpot"), ("e2el", "latency"))


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MlperfReportError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _series(report: Mapping[str, Any], name: str) -> dict[str, float] | None:
    """``avg``, ``std_dev`` and the :data:`_PERCENTILE_KEYS` of one series; ``None`` when it has no samples."""
    block = report[name]
    if not isinstance(block, dict):
        raise MlperfReportError(f"{name}: expected a series object, got {type(block).__name__}")
    if not block:
        return None
    missing = [key for key in _SERIES_FIELDS if key not in block]
    if missing:
        raise MlperfReportError(f"{name}: series is missing {missing}")
    percentiles = block["percentiles"]
    if not isinstance(percentiles, dict):
        raise MlperfReportError(f"{name}.percentiles: expected an object")
    out = {"avg": _finite(block["avg"], f"{name}.avg"), "std_dev": _finite(block["std_dev"], f"{name}.std_dev")}
    for label, key in _PERCENTILE_KEYS.items():
        if key not in percentiles:
            raise MlperfReportError(f"{name}.percentiles has no {key!r} (has {sorted(percentiles)})")
        out[label] = _finite(percentiles[key], f"{name}.percentiles[{key!r}]")
    return out


def _inline_accuracy(scores: Mapping[str, Any]) -> tuple[float, int]:
    """The harness's inline accuracy score and the turns it could not score."""
    if not isinstance(scores, Mapping) or "score" not in scores:
        raise MlperfReportError("scores.json: no 'score'")
    turns = scores.get("turns")
    if not isinstance(turns, Mapping) or "missing" not in turns:
        raise MlperfReportError("scores.json: no 'turns.missing'")
    return _finite(scores["score"], "scores.score"), int(_finite(turns["missing"], "scores.turns.missing"))


def map_mlperf(
    report: Mapping[str, Any],
    *,
    corpus: str,
    scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an ``inference-endpoint`` ``result_summary.json`` into the InferenceX result schema.

    The harness publishes no per-request OSL/E2EL series, so no ``e2e_norm_intvty_*``
    key is written. A field the upstream ``Report`` does not carry is absent, never
    0.0; a file that does not match that schema raises :class:`MlperfReportError`.

    Args:
        report: The parsed ``result_summary.json``.
        corpus: The dataset the trajectories came from.
        scores: The parsed inline ``scores.json``, when the run produced one.
    """
    if not isinstance(report, Mapping):
        raise MlperfReportError("result_summary.json: expected an object")
    missing = [key for key in _REPORT_FIELDS if key not in report]
    if missing:
        raise MlperfReportError(f"result_summary.json is missing {missing}")
    issued = int(_finite(report["n_samples_issued"], "n_samples_issued"))
    completed = int(_finite(report["n_samples_completed"], "n_samples_completed"))
    failed = int(_finite(report["n_samples_failed"], "n_samples_failed"))
    reasons = []
    if report["state"] == "interrupted":
        reasons.append("interrupted")
    if report["complete"] is not True:
        reasons.append("incomplete_run")
    mapped: dict[str, Any] = {
        "completed": completed,
        "request_error_rate": 100.0 * failed / issued if issued > 0 else None,
        "submission_valid": not reasons,
        "submission_invalid_reasons": reasons,
        "corpus_loader": str(corpus),
    }
    if report["duration_ns"] is not None:
        mapped["duration"] = _finite(report["duration_ns"], "duration_ns") / 1e9
    if report["qps"] is not None:
        mapped["request_throughput"] = _finite(report["qps"], "qps")
    if report["tps"] is not None:
        mapped["output_throughput"] = _finite(report["tps"], "tps")
    for prefix, name in _LATENCY_SERIES:
        series = _series(report, name)
        if series is None:
            continue
        mapped[f"mean_{prefix}_ms"] = series["avg"] / _NS_PER_MS
        mapped[f"std_{prefix}_ms"] = series["std_dev"] / _NS_PER_MS
        mapped[f"median_{prefix}_ms"] = series["p50"] / _NS_PER_MS
        mapped[f"p90_{prefix}_ms"] = series["p90"] / _NS_PER_MS
        mapped[f"p99_{prefix}_ms"] = series["p99"] / _NS_PER_MS
    osl = _series(report, "output_sequence_lengths")
    if osl is not None:
        mapped["total_output_tokens"] = int(_finite(report["output_sequence_lengths"].get("total"), "osl.total"))
        mapped["osl_distribution"] = {key: int(osl[key]) for key in ("avg", "p50", "p75", "p90", "p99")}
    if scores is not None:
        mapped["accuracy_score"], mapped["accuracy_missing_turns"] = _inline_accuracy(scores)
    return mapped
