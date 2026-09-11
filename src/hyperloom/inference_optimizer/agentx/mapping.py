# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map aiperf ``profile_export_aiperf.json`` metrics to the InferenceX result schema (``inferencex_result.json``)."""

from __future__ import annotations

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
    rc = int(stat(m, "request_count") or 0)
    isl = stat(m, "input_sequence_length")

    # Scoring and comparison use aiperf's summary P10 of the per-request rate OSL/E2EL_s.
    intvty_p90 = pct(m, "e2e_output_token_throughput", "p10")

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
        "p99_ttft_ms": stat(m, "time_to_first_token", "p99"),
        "std_ttft_ms": stat(m, "time_to_first_token", "std"),
        "mean_tpot_ms": stat(m, "inter_token_latency", "avg"),
        "median_tpot_ms": stat(m, "inter_token_latency", "p50"),
        "p90_tpot_ms": stat(m, "inter_token_latency", "p90"),
        "p99_tpot_ms": stat(m, "inter_token_latency", "p99"),
        "std_tpot_ms": stat(m, "inter_token_latency", "std"),
        "e2e_norm_intvty_p90": intvty_p90,
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
        # Upstream's hard validity gate, as a percentage (aiperf declares this
        # metric PERCENT). ``default=None`` rather than 0.0: aiperf omits the
        # metric when no request completed, and coalescing that to zero would
        # report a perfect error rate for a run that measured nothing.
        "request_error_rate": stat(m, "request_error_rate", default=None),
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
