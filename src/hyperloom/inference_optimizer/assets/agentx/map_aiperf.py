#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: aiperf ``profile_export_aiperf.json`` -> ``inferencex_result.json``.

Thin shim over ``hyperloom.inference_optimizer.agentx.mapping.map_aiperf`` (runs
in the same venv as Hyperloom). Falls back to a vendored copy of the mapping if
the package is not importable, so the deployed script is self-sufficient.
"""

import json
import sys

try:
    from hyperloom.inference_optimizer.agentx.mapping import map_aiperf
except Exception:  # noqa: BLE001 — self-sufficient fallback when pkg not on path

    def _stat(m, key, sub="avg", default=0.0):
        v = m.get(key)
        if isinstance(v, dict):
            return v.get(sub, v.get("avg", default))
        return v if v is not None else default

    def map_aiperf(export):
        d = export
        m = d if ("time_to_first_token" in d or "output_token_throughput" in d) else d.get("metrics", d)
        out_tput = _stat(m, "output_token_throughput")
        in_tput = _stat(m, "input_token_throughput")
        total_tput = _stat(m, "total_token_throughput") or ((in_tput or 0) + (out_tput or 0))
        rc = int(_stat(m, "request_count") or 0)
        isl = _stat(m, "input_sequence_length")
        return {
            "request_throughput": _stat(m, "request_throughput"),
            "output_throughput": out_tput,
            "total_token_throughput": total_tput,
            "completed": rc,
            "total_input_tokens": int(_stat(m, "total_isl") or (isl * max(1, rc)) or 0),
            "total_output_tokens": int(_stat(m, "total_output_tokens") or _stat(m, "total_osl") or 0),
            "duration": _stat(m, "benchmark_duration"),
            "mean_ttft_ms": _stat(m, "time_to_first_token", "avg"),
            "median_ttft_ms": _stat(m, "time_to_first_token", "p50"),
            "p99_ttft_ms": _stat(m, "time_to_first_token", "p99"),
            "std_ttft_ms": _stat(m, "time_to_first_token", "std"),
            "mean_tpot_ms": _stat(m, "inter_token_latency", "avg"),
            "median_tpot_ms": _stat(m, "inter_token_latency", "p50"),
            "p99_tpot_ms": _stat(m, "inter_token_latency", "p99"),
            "std_tpot_ms": _stat(m, "inter_token_latency", "std"),
            "mean_itl_ms": _stat(m, "inter_token_latency", "avg"),
            "median_itl_ms": _stat(m, "inter_token_latency", "p50"),
            "p99_itl_ms": _stat(m, "inter_token_latency", "p99"),
            "std_itl_ms": _stat(m, "inter_token_latency", "std"),
            "mean_e2el_ms": _stat(m, "request_latency", "avg"),
            "median_e2el_ms": _stat(m, "request_latency", "p50"),
            "p99_e2el_ms": _stat(m, "request_latency", "p99"),
            "std_e2el_ms": _stat(m, "request_latency", "std"),
            "theoretical_prefix_cache_hit": _stat(m, "theoretical_prefix_cache_hit"),
        }


def main(src, dst):
    with open(src) as f:
        data = json.load(f)
    res = map_aiperf(data)
    with open(dst, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
