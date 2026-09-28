# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map an AgentX client's export to the InferenceX result schema (``inferencex_result.json``).

Two clients, two readers: :func:`map_aiperf` for aiperf's ``profile_export_aiperf.json``
and :func:`map_mlperf` for the MLCommons ``inference-endpoint`` ``result_summary.json``
plus its inline ``scores.json``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# The canonical corpus, measured from Kimi-K3 session 20260831T124523Z (825
# requests over the 3600s window). Seeds ``SharedState.agentx_corpus_shape``
# so semantic consumers have a shape before the first measurement replaces it.
CANONICAL_CORPUS_LOADER = "semianalysis_cc_traces_weka_062126"
CANONICAL_CORPUS_ENTRIES = 393
CANONICAL_CORPUS_DURATION_S = 3600
CANONICAL_ISL = {"avg": 113814, "p50": 94821, "p75": 119126, "p90": 163328, "p99": 506158}
CANONICAL_OSL = {"avg": 806, "p50": 333, "p75": 801, "p90": 1874, "p99": 6386}
CANONICAL_PREFIX_CACHE_HIT = 0.975

# Percentiles carried forward from the aiperf sequence-length distributions.
_SHAPE_PERCENTILES = ("avg", "p50", "p75", "p90", "p99")


def stat(m: Mapping[str, Any], key: str, sub: str = "avg", default: float = 0.0) -> Any:
    """Read ``m[key][sub]`` with graceful fallbacks (avg, then ``default``)."""
    v = m.get(key)
    if isinstance(v, dict):
        # Coalesce explicit None: a present-but-null sub-key (or avg) must fall back to avg then the numeric default,
        # never emit None downstream.
        sv = v.get(sub)
        if sv is not None:
            return sv
        av = v.get("avg")
        return av if av is not None else default
    return v if v is not None else default


def pct(m: Mapping[str, Any], key: str, sub: str, default: float = 0.0) -> Any:
    """Read ``m[key][sub]`` with no ``avg`` fallback."""
    v = m.get(key)
    if isinstance(v, dict):
        sv = v.get(sub)
        return sv if sv is not None else default
    return default


def _valid_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _request_error_rate(metrics: Mapping[str, Any], export: Mapping[str, Any]) -> float | None:
    rate = stat(metrics, "request_error_rate", default=None)
    if rate is not None:
        return float(rate) if _valid_count(rate) and rate <= 100 else None

    success = stat(metrics, "request_count", default=None)
    errors = stat(metrics, "error_request_count", default=None)
    completed = stat(metrics, "completed_request_count", default=None)
    if any(value is not None and not _valid_count(value) for value in (success, errors, completed)):
        return None
    # AIPerf omits zero-error metrics; accounting counters also include warmup.
    if errors is None:
        if success is None or success <= 0 or export.get("error_summary") != []:
            return None
        errors = 0
    if completed is None:
        if success is None:
            return None
        completed = success + errors
    elif success is not None and completed != success + errors:
        return None
    if not math.isfinite(completed) or completed <= 0 or errors > completed:
        return None
    return 100.0 * (errors / completed)


def submission_outcome(export: Mapping[str, Any]) -> tuple[bool | None, list[str]]:
    """Read the scenario's submission verdict from an aiperf export."""
    md = export.get("metadata")
    if not isinstance(md, dict) or "submission_valid" not in md:
        return None, []
    reasons = md.get("submission_invalid_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return bool(md.get("submission_valid")), [str(r) for r in reasons]


def map_aiperf(
    export: Mapping[str, Any],
    *,
    noncanonical_reasons: "Sequence[str] | None" = None,
) -> dict[str, Any]:
    """Convert an aiperf export dict into the InferenceX result schema."""
    d = export
    verdict, reasons = submission_outcome(d)
    extra = [str(r) for r in (noncanonical_reasons or []) if str(r).strip()]
    if extra:
        verdict = False
        reasons = [*reasons, *extra]
    # aiperf may nest metrics under "metrics"; accept both shapes.
    m = d if ("time_to_first_token" in d or "output_token_throughput" in d) else d.get("metrics", d)

    out_tput = stat(m, "output_token_throughput")
    in_tput = stat(m, "input_token_throughput")
    total_tput = stat(m, "total_token_throughput") or ((in_tput or 0) + (out_tput or 0))
    success = stat(m, "request_count", default=None)
    rc = int(success) if _valid_count(success) else 0
    isl = stat(m, "input_sequence_length")

    # Scoring and comparison use aiperf's summary P10 of the per-request rate OSL/E2EL_s.
    intvty_p90 = pct(m, "e2e_output_token_throughput", "p10")
    # The median needs no slow-tail inversion: a monotone 1/x maps P50 of the ratio onto P50 of the rate.
    intvty_p50 = pct(m, "e2e_output_token_throughput", "p50")

    return {
        "request_throughput": stat(m, "request_throughput"),
        "output_throughput": out_tput,
        "input_throughput": in_tput,
        "total_token_throughput": total_tput,
        "completed": rc,
        "total_input_tokens": int(stat(m, "total_isl") or (isl * max(1, rc)) or 0),
        "total_output_tokens": int(stat(m, "total_output_tokens") or stat(m, "total_osl") or 0),
        "duration": stat(m, "benchmark_duration"),
        "mean_ttft_ms": stat(m, "time_to_first_token", "avg"),
        "median_ttft_ms": stat(m, "time_to_first_token", "p50"),
        "p90_ttft_ms": stat(m, "time_to_first_token", "p90"),
        "p99_ttft_ms": stat(m, "time_to_first_token", "p99"),
        "std_ttft_ms": stat(m, "time_to_first_token", "std"),
        "mean_tpot_ms": stat(m, "inter_token_latency", "avg"),
        "median_tpot_ms": stat(m, "inter_token_latency", "p50"),
        "p90_tpot_ms": stat(m, "inter_token_latency", "p90"),
        "p99_tpot_ms": stat(m, "inter_token_latency", "p99"),
        "std_tpot_ms": stat(m, "inter_token_latency", "std"),
        "e2e_norm_intvty_p90": intvty_p90,
        "e2e_norm_intvty_p50": intvty_p50,
        "mean_itl_ms": stat(m, "inter_token_latency", "avg"),
        "median_itl_ms": stat(m, "inter_token_latency", "p50"),
        "p99_itl_ms": stat(m, "inter_token_latency", "p99"),
        "std_itl_ms": stat(m, "inter_token_latency", "std"),
        "mean_e2el_ms": stat(m, "request_latency", "avg"),
        "median_e2el_ms": stat(m, "request_latency", "p50"),
        "p99_e2el_ms": stat(m, "request_latency", "p99"),
        "std_e2el_ms": stat(m, "request_latency", "std"),
        "theoretical_prefix_cache_hit": stat(m, "theoretical_prefix_cache_hit"),
        # Tri-state on purpose: True / False / None(unknown).
        "submission_valid": verdict,
        "submission_invalid_reasons": reasons,
        # A percentage, or unknown when profiling provides insufficient evidence.
        "request_error_rate": _request_error_rate(m, d),
        # Corpus shape. A single ISL/OSL scalar cannot describe this workload
        # (p50 95k, p99 506k), so the distributions travel instead.
        "corpus_loader": _corpus_loader(d),
        "isl_distribution": _distribution(m.get("input_sequence_length")),
        "osl_distribution": _distribution(m.get("output_sequence_length")),
    }


def _corpus_loader(export: Mapping[str, Any]) -> str:
    """The dataset loader aiperf replayed, from ``metadata.dataset.loader``."""
    dataset = (export.get("metadata") or {}).get("dataset")
    return str((dataset or {}).get("loader") or "")


def _distribution(metric: Any) -> dict[str, int]:
    """Project an aiperf sequence-length metric onto :data:`_SHAPE_PERCENTILES`."""
    if not isinstance(metric, dict):
        return {}
    return {key: int(metric[key]) for key in _SHAPE_PERCENTILES if isinstance(metric.get(key), (int, float))}


def map_corpus_shape(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a ``SharedState.agentx_corpus_shape`` record from a :func:`map_aiperf` result."""
    return {
        "corpus_loader": str(result.get("corpus_loader") or ""),
        "isl": dict(result.get("isl_distribution") or {}),
        "osl": dict(result.get("osl_distribution") or {}),
        "completed_requests": int(result.get("completed") or 0),
        "duration_s": float(result.get("duration") or 0.0),
        "prefix_cache_hit": float(result.get("theoretical_prefix_cache_hit") or 0.0),
        "request_error_rate": float(result.get("request_error_rate") or 0.0),
        "source": "measured",
    }


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
    issued_trajectories: int,
    corpus: str,
    scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an ``inference-endpoint`` ``result_summary.json`` into the InferenceX result schema.

    The harness publishes no per-request OSL/E2EL series, so no ``e2e_norm_intvty_*``
    key is written. A field the upstream ``Report`` does not carry is absent, never
    0.0; a file that does not match that schema raises :class:`MlperfReportError`.

    Args:
        report: The parsed ``result_summary.json``.
        issued_trajectories: Trajectories the run was configured to replay. The
            harness does not record it, and it is what two fixed-work rounds must
            share to be comparable.
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
        "issued_trajectories": int(issued_trajectories),
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
