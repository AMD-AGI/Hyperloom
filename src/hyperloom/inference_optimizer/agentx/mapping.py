# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map aiperf ``profile_export_aiperf.json`` metrics to the InferenceX result schema (``inferencex_result.json``)."""

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

# MLPerf agentic v6 corpus (HYPERLOOM_AGENTIC_BACKEND=mlperf). Search uses 150
# trajectories; KEEP validation uses the full 613-trajectory online set.
CANONICAL_MLPERF_CORPUS_LOADER = "agentic_combined_v6"
CANONICAL_MLPERF_CORPUS_ENTRIES = 613
CANONICAL_MLPERF_SMOKE_ENTRIES = 150
CANONICAL_MLPERF_CORPUS_DURATION_S = 0
CANONICAL_MLPERF_ISL: dict[str, int] = {}
CANONICAL_MLPERF_OSL: dict[str, int] = {}
CANONICAL_MLPERF_PREFIX_CACHE_HIT = 0.0

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


def _ns_to_ms(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return 0.0
    if value >= 1_000_000:
        return float(value) / 1_000_000.0
    return float(value)


def _series_ms(block: Any, percentile: int | None = None, *, avg: bool = False) -> float:
    if not isinstance(block, dict):
        if avg and isinstance(block, (int, float)) and not isinstance(block, bool):
            return _ns_to_ms(block)
        return 0.0
    if avg:
        for key in ("avg", "mean", "average"):
            if block.get(key) is not None:
                return _ns_to_ms(block.get(key))
    perc = block.get("percentiles") if isinstance(block.get("percentiles"), dict) else block
    if percentile is not None and isinstance(perc, dict):
        for key in (str(percentile), percentile, f"p{percentile}"):
            if perc.get(key) is not None:
                return _ns_to_ms(perc.get(key))
    return 0.0


def _seq_total(block: Any) -> int:
    if isinstance(block, dict):
        total = block.get("total")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            return int(total)
    if isinstance(block, (int, float)) and not isinstance(block, bool):
        return int(block)
    return 0


def _seq_distribution(block: Any) -> dict[str, int]:
    if not isinstance(block, dict):
        return {}
    out: dict[str, int] = {}
    avg = block.get("avg") or block.get("mean")
    if isinstance(avg, (int, float)) and not isinstance(avg, bool):
        out["avg"] = int(avg)
    perc = block.get("percentiles") if isinstance(block.get("percentiles"), dict) else block
    if isinstance(perc, dict):
        for name, key in (("p50", 50), ("p75", 75), ("p90", 90), ("p99", 99)):
            raw = perc.get(str(key), perc.get(key, perc.get(name)))
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                out[name] = int(raw)
    return out


def _accuracy_score(accuracy: Mapping[str, Any] | None) -> float | None:
    if not isinstance(accuracy, dict) or not accuracy:
        return None
    for key in ("score", "accuracy", "overall_score", "pass_rate", "inline_accuracy"):
        value = accuracy.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    metrics = accuracy.get("metrics")
    if isinstance(metrics, dict):
        for key in ("score", "accuracy", "pass_rate"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return float(value)
    return None


def _target_concurrency(summary: Mapping[str, Any]) -> float:
    """Served concurrency, from the harness's own record of the run.

    ``run_config`` is what the harness actually ran with, so it outranks the
    flat aliases; those remain for summaries that predate it.
    """
    run_config = summary.get("run_config")
    if isinstance(run_config, dict):
        load_pattern = run_config.get("load_pattern")
        if isinstance(load_pattern, dict):
            value = load_pattern.get("target_concurrency")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return float(value)
    for key in ("target_concurrency", "concurrency"):
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return 0.0


def map_mlperf(
    summary: Mapping[str, Any],
    *,
    accuracy: Mapping[str, Any] | None = None,
    noncanonical_reasons: "Sequence[str] | None" = None,
) -> dict[str, Any]:
    """Convert an MLPerf harness ``result_summary.json`` into the InferenceX result schema."""
    extra = [str(r) for r in (noncanonical_reasons or []) if str(r).strip()]
    completed = int(summary.get("n_samples_completed") or summary.get("completed") or 0)
    failed = int(summary.get("n_samples_failed") or summary.get("failed") or 0)
    issued = int(summary.get("n_samples_issued") or summary.get("issued") or (completed + failed))
    duration_ns = summary.get("duration_ns")
    if isinstance(duration_ns, (int, float)) and duration_ns > 0:
        duration_s = float(duration_ns) / 1e9
    else:
        duration_s = float(summary.get("duration") or 0.0)
    osl_total = _seq_total(summary.get("output_sequence_lengths"))
    isl_total = _seq_total(summary.get("input_sequence_lengths"))
    tps = summary.get("tps")
    out_tput = (
        float(tps)
        if isinstance(tps, (int, float)) and tps
        else ((osl_total / duration_s) if duration_s > 0 and osl_total else 0.0)
    )
    in_tput = (isl_total / duration_s) if duration_s > 0 and isl_total else 0.0
    req_tput = (completed / duration_s) if duration_s > 0 and completed else 0.0
    denom = issued if issued > 0 else (completed + failed)
    error_rate = (100.0 * failed / denom) if denom > 0 else (None if failed else 0.0)
    complete = bool(summary.get("complete"))
    interrupted = bool(summary.get("interrupted") or summary.get("error"))
    # Interactivity is system throughput per concurrent user, the axis AgentX
    # grades on. The harness publishes no such field -- measured: a v6 summary
    # carries tps/ttft/tpot/latency/qps and nothing else -- so it is derived,
    # matching utility/sweep.py's own definition. A run whose concurrency cannot
    # be read leaves it 0.0, which the graded comparison treats as incomparable
    # rather than as a perfect score.
    intvty = summary.get("e2e_avg_interactivity")
    if not isinstance(intvty, (int, float)) or isinstance(intvty, bool):
        intvty = summary.get("interactivity")
    if not isinstance(intvty, (int, float)) or isinstance(intvty, bool):
        conc = _target_concurrency(summary)
        intvty = (float(out_tput) / conc) if (conc and out_tput) else 0.0
    ttft = summary.get("ttft") or {}
    tpot = summary.get("tpot") or summary.get("itl") or {}
    latency = summary.get("latency") or summary.get("e2e") or {}
    acc_score = _accuracy_score(accuracy)
    reasons = list(extra)
    if not complete:
        reasons.append("incomplete_run")
    if interrupted:
        reasons.append("interrupted")
    verdict = complete and not interrupted and not extra
    return {
        "request_throughput": req_tput,
        "output_throughput": float(out_tput or 0.0),
        "input_throughput": float(in_tput or 0.0),
        "total_token_throughput": float(out_tput or 0.0) + float(in_tput or 0.0),
        "completed": completed,
        "total_input_tokens": isl_total,
        "total_output_tokens": osl_total,
        "duration": duration_s,
        "mean_ttft_ms": _series_ms(ttft, avg=True),
        "median_ttft_ms": _series_ms(ttft, 50),
        "p99_ttft_ms": _series_ms(ttft, 99),
        "std_ttft_ms": _ns_to_ms((ttft or {}).get("std") if isinstance(ttft, dict) else 0.0),
        "mean_tpot_ms": _series_ms(tpot, avg=True),
        "median_tpot_ms": _series_ms(tpot, 50),
        "p90_tpot_ms": _series_ms(tpot, 90),
        "p99_tpot_ms": _series_ms(tpot, 99),
        "std_tpot_ms": _ns_to_ms((tpot or {}).get("std") if isinstance(tpot, dict) else 0.0),
        "e2e_norm_intvty_p90": float(intvty or 0.0),
        "mean_itl_ms": _series_ms(tpot, avg=True),
        "median_itl_ms": _series_ms(tpot, 50),
        "p99_itl_ms": _series_ms(tpot, 99),
        "std_itl_ms": _ns_to_ms((tpot or {}).get("std") if isinstance(tpot, dict) else 0.0),
        "mean_e2el_ms": _series_ms(latency, avg=True),
        "median_e2el_ms": _series_ms(latency, 50),
        "p99_e2el_ms": _series_ms(latency, 99),
        "std_e2el_ms": _ns_to_ms((latency or {}).get("std") if isinstance(latency, dict) else 0.0),
        "theoretical_prefix_cache_hit": float(summary.get("prefix_cache_hit") or 0.0),
        "submission_valid": verdict,
        "submission_invalid_reasons": reasons,
        "request_error_rate": error_rate,
        "corpus_loader": str(summary.get("corpus_loader") or CANONICAL_MLPERF_CORPUS_LOADER),
        "isl_distribution": _seq_distribution(summary.get("input_sequence_lengths")),
        "osl_distribution": _seq_distribution(summary.get("output_sequence_lengths")),
        "accuracy_score": acc_score,
        "mlperf_complete": complete,
    }
