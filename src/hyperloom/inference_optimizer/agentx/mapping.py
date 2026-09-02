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

    # E2E Normalized Interactivity (OSL/E2EL), the axis InferenceX reports at
    # p90. ``output_token_throughput_per_user`` is 1/ITL and drops TTFT from the
    # denominator, which on a ~114k-prompt replay is most of what a user waits
    # for -- a candidate could double TTFT and leave that number untouched.
    # pct() is used here (not stat()) because avg and p90 differ by >2x on
    # this metric and grading against avg would make the veto gate meaningless.
    intvty_p90 = pct(m, "e2e_output_token_throughput", "p90")

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
        "intvty_p90_tok_s_user": intvty_p90,
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
    }
