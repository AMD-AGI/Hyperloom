# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map aiperf ``profile_export_aiperf.json`` metrics to the InferenceX result
schema (``inferencex_result.json``).

Emits exactly the keys Magpie's ``ResultParser.parse_inferencex_result`` reads.
Each aiperf metric is a dict carrying at least ``avg``; latency metrics also
carry ``p50``/``p99``/``std``. ``stat`` reads a sub-key, falling back to ``avg``
then a numeric default so a missing metric never raises.

``pct`` is the strict variant: no ``avg`` fallback. Use it for any axis where
the percentile and the mean differ and grading depends on the result.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Canonical corpus constants derived from the full 393-entry, 3600s corpus
# (semianalysis_cc_traces_weka_062126).  These are used to seed
# SharedState.agentx_corpus_shape before the first measurement so semantic
# consumers have something to render immediately.  Measured values from a
# real aiperf export overwrite them after each run.
#
# Sources: measured from Kimi-K3 session 20260831T124523Z (825 requests,
# 3600 s window) and MiniMax-M3 sessions (1033-1097 requests, 3600 s).
CANONICAL_CORPUS_LOADER = "semianalysis_cc_traces_weka_062126"
CANONICAL_CORPUS_ENTRIES = 393
CANONICAL_CORPUS_DURATION_S = 3600
CANONICAL_ISL = {
    "avg": 113814,
    "p50": 94821,
    "p75": 119126,
    "p90": 163328,
    "p99": 506158,
}
CANONICAL_OSL = {
    "avg": 806,
    "p50": 333,
    "p75": 801,
    "p90": 1874,
    "p99": 6386,
}
CANONICAL_PREFIX_CACHE_HIT = 0.975


def stat(m: Mapping[str, Any], key: str, sub: str = "avg", default: float = 0.0) -> Any:
    """Read ``m[key][sub]`` with graceful fallbacks (avg, then ``default``)."""
    v = m.get(key)
    if isinstance(v, dict):
        # Coalesce explicit None: a present-but-null sub-key (or avg) must fall
        # back to avg then the numeric default, never emit None downstream.
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
    """Read the scenario's submission verdict from an aiperf export.

    aiperf stamps ``metadata.submission_valid`` (and, only when non-empty,
    ``metadata.submission_invalid_reasons``) whenever ``--scenario`` is set. It
    goes False for a scenario-invariant violation, a cancelled run, or a
    context-overflow rate above the scenario's limit.

    Returns:
        ``(verdict, reasons)`` where verdict is True/False, or **None when the
        field is absent** -- which is NOT the same as valid: it means either no
        scenario was requested or the aiperf build predates the field, and in
        both cases the run's comparability is unknown.
    """
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
    """Convert an aiperf export dict into the InferenceX result schema.

    Also carries the scenario submission verdict through as
    ``submission_valid`` / ``submission_invalid_reasons``. The *presence* of
    ``submission_valid`` is what marks a result as AgentX-produced downstream;
    synthetic results never carry it.

    Args:
        export: The parsed aiperf ``profile_export_aiperf.json``.
        noncanonical_reasons: Workload deviations the *client* detected, which
            the scenario cannot see. aiperf only judges what it was told to
            enforce -- it has no concept of corpus size, and it stamps a verdict
            of False only when ``--unsafe-override`` actually suppressed a
            violation -- so a shrunken corpus, or the override forced at the
            canonical duration, would otherwise come back submission_valid=True
            on a workload nothing on the leaderboard ran. Any reason here forces
            the verdict to False so ``is_valid_measurement`` refuses it.
    """
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

    # E2E Normalized Interactivity slow tail.
    #
    # InferenceX defines: r_i = E2EL_i / OSL_i (seconds per output token),
    # then interactivity_P90 = 1 / P90({r_i}).  In aiperf's export
    # ``e2e_output_token_throughput`` = OSL / E2EL_s is LARGER_IS_BETTER, so
    # its P10 corresponds to the slow-tail users (highest latency).  P10(rate)
    # = 1 / P90(ratio) — mathematically identical to the upstream formula.
    #
    # pct() is used (not stat()) because avg and P10 differ by an order of
    # magnitude on this corpus and grading against avg would miss latency outliers.
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
        # Renamed from intvty_p90_tok_s_user to make the slow-tail semantics
        # unambiguous.  ``e2e_norm_intvty_p90`` matches the field name the
        # grading layer reads from perf snapshots.
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
        # Tri-state on purpose: True / False / None(unknown). Never coerce the
        # unknown case to True -- that is exactly how an incomparable run would
        # slip into the leaderboard-comparable set.
        "submission_valid": verdict,
        "submission_invalid_reasons": reasons,
        # Corpus shape from this run — used by map_corpus_shape.
        "_corpus_shape_raw": {
            "isl": m.get("input_sequence_length"),
            "osl": m.get("output_sequence_length"),
            "completed": rc,
            "duration_s": stat(m, "benchmark_duration"),
            "theoretical_prefix_cache_hit": stat(m, "theoretical_prefix_cache_hit"),
            "error_request_count": stat(m, "error_request_count"),
            "request_error_rate": stat(m, "request_error_rate"),
        },
    }


def map_corpus_shape(
    result: Mapping[str, Any],
    *,
    corpus_loader: str = CANONICAL_CORPUS_LOADER,
) -> dict[str, Any]:
    """Extract a corpus-shape summary from a mapped aiperf result.

    The raw distribution dicts from aiperf carry avg/p50/p75/p90/p99/p99/std;
    we forward the subset that semantic consumers need.

    Args:
        result: The dict returned by :func:`map_aiperf`.
        corpus_loader: Corpus loader name; defaults to the canonical full corpus.

    Returns:
        A dict suitable for ``SharedState.agentx_corpus_shape``.
    """
    raw = result.get("_corpus_shape_raw") or {}

    def _dist(v: Any) -> dict[str, Any] | None:
        if not isinstance(v, dict):
            return None
        out: dict[str, Any] = {}
        for k in ("avg", "p50", "p75", "p90", "p99"):
            val = v.get(k)
            if val is not None:
                out[k] = int(val) if isinstance(val, (int, float)) else val
        return out or None

    shape: dict[str, Any] = {"corpus_loader": corpus_loader}
    isl_dist = _dist(raw.get("isl"))
    if isl_dist:
        shape["isl"] = isl_dist
    osl_dist = _dist(raw.get("osl"))
    if osl_dist:
        shape["osl"] = osl_dist
    completed = raw.get("completed")
    if completed:
        shape["completed_requests"] = int(completed)
    duration = raw.get("duration_s")
    if duration:
        shape["duration_s"] = float(duration)
    pch = raw.get("theoretical_prefix_cache_hit")
    if pch is not None:
        shape["prefix_cache_hit"] = float(pch)
    err_rate = raw.get("request_error_rate")
    if err_rate is not None:
        shape["request_error_rate"] = float(err_rate)
    return shape
