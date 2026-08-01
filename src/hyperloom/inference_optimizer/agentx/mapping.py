# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map aiperf ``profile_export_aiperf.json`` metrics to the InferenceX result
schema (``inferencex_result.json``).

Emits exactly the keys Magpie's ``ResultParser.parse_inferencex_result`` reads.
Each aiperf metric is a dict carrying at least ``avg``; latency metrics also
carry ``p50``/``p99``/``std``. ``stat`` reads a sub-key, falling back to ``avg``
then a numeric default so a missing metric never raises.
"""

from __future__ import annotations

from typing import Any, Mapping


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


def map_aiperf(export: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an aiperf export dict into the InferenceX result schema."""
    d = export
    # aiperf may nest metrics under "metrics"; accept both shapes.
    m = d if ("time_to_first_token" in d or "output_token_throughput" in d) else d.get("metrics", d)

    out_tput = stat(m, "output_token_throughput")
    in_tput = stat(m, "input_token_throughput")
    total_tput = stat(m, "total_token_throughput") or ((in_tput or 0) + (out_tput or 0))
    rc = int(stat(m, "request_count") or 0)
    isl = stat(m, "input_sequence_length")

    return {
        "request_throughput": stat(m, "request_throughput"),
        "output_throughput": out_tput,
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
        "p99_tpot_ms": stat(m, "inter_token_latency", "p99"),
        "std_tpot_ms": stat(m, "inter_token_latency", "std"),
        "mean_itl_ms": stat(m, "inter_token_latency", "avg"),
        "median_itl_ms": stat(m, "inter_token_latency", "p50"),
        "p99_itl_ms": stat(m, "inter_token_latency", "p99"),
        "std_itl_ms": stat(m, "inter_token_latency", "std"),
        "mean_e2el_ms": stat(m, "request_latency", "avg"),
        "median_e2el_ms": stat(m, "request_latency", "p50"),
        "p99_e2el_ms": stat(m, "request_latency", "p99"),
        "std_e2el_ms": stat(m, "request_latency", "std"),
        "theoretical_prefix_cache_hit": stat(m, "theoretical_prefix_cache_hit"),
    }
