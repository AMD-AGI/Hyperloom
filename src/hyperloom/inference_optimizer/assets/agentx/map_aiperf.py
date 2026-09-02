#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: aiperf ``profile_export_aiperf.json`` -> ``inferencex_result.json``.

Thin shim over ``hyperloom.inference_optimizer.agentx.mapping.map_aiperf`` (runs
in the same venv as Hyperloom). Falls back to a vendored copy of the mapping if
the package is not importable, so the deployed script is self-sufficient.
"""

import json
import os
import sys


def _noncanonical_reasons():
    """Workload deviations the client detected; see aiperf_client.sh.

    aiperf cannot judge these -- it has no concept of corpus size, and it only
    stamps a False verdict when --unsafe-override actually suppressed a
    violation -- so the client passes them here and the verdict is forced.
    """
    raw = (os.environ.get("AGENTX_NONCANONICAL_REASONS") or "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else []


try:
    from hyperloom.inference_optimizer.agentx.mapping import map_aiperf
except Exception:  # noqa: BLE001 — self-sufficient fallback when pkg not on path

    def _stat(m, key, sub="avg", default=0.0):
        v = m.get(key)
        if isinstance(v, dict):
            return v.get(sub, v.get("avg", default))
        return v if v is not None else default

    def _pct(m, key, sub, default=0.0):
        v = m.get(key)
        if isinstance(v, dict):
            sv = v.get(sub)
            return sv if sv is not None else default
        return default

    def _submission_outcome(export):
        # Tri-state: True / False / None(absent). Absent is NOT valid -- it means
        # no --scenario was requested or the aiperf build predates the field.
        md = export.get("metadata")
        if not isinstance(md, dict) or "submission_valid" not in md:
            return None, []
        reasons = md.get("submission_invalid_reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return bool(md.get("submission_valid")), [str(r) for r in reasons]

    def map_aiperf(export, *, noncanonical_reasons=None):
        d = export
        _verdict, _reasons = _submission_outcome(d)
        _extra = [str(r) for r in (noncanonical_reasons or []) if str(r).strip()]
        if _extra:
            _verdict = False
            _reasons = [*_reasons, *_extra]
        m = d if ("time_to_first_token" in d or "output_token_throughput" in d) else d.get("metrics", d)
        out_tput = _stat(m, "output_token_throughput")
        in_tput = _stat(m, "input_token_throughput")
        total_tput = _stat(m, "total_token_throughput") or ((in_tput or 0) + (out_tput or 0))
        rc = int(_stat(m, "request_count") or 0)
        isl = _stat(m, "input_sequence_length")
        # E2E Normalized Interactivity (OSL/E2EL), the axis InferenceX reports
        # at p90; the per-user variant is 1/ITL and omits TTFT.
        intvty_p90 = _pct(m, "e2e_output_token_throughput", "p90")
        return {
            "request_throughput": _stat(m, "request_throughput"),
            "output_throughput": out_tput,
            "input_throughput": in_tput,
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
            "p90_tpot_ms": _stat(m, "inter_token_latency", "p90"),
            "p99_tpot_ms": _stat(m, "inter_token_latency", "p99"),
            "std_tpot_ms": _stat(m, "inter_token_latency", "std"),
            "intvty_p90_tok_s_user": intvty_p90,
            "mean_itl_ms": _stat(m, "inter_token_latency", "avg"),
            "median_itl_ms": _stat(m, "inter_token_latency", "p50"),
            "p99_itl_ms": _stat(m, "inter_token_latency", "p99"),
            "std_itl_ms": _stat(m, "inter_token_latency", "std"),
            "mean_e2el_ms": _stat(m, "request_latency", "avg"),
            "median_e2el_ms": _stat(m, "request_latency", "p50"),
            "p99_e2el_ms": _stat(m, "request_latency", "p99"),
            "std_e2el_ms": _stat(m, "request_latency", "std"),
            "theoretical_prefix_cache_hit": _stat(m, "theoretical_prefix_cache_hit"),
            "submission_valid": _verdict,
            "submission_invalid_reasons": _reasons,
        }


def main(src, dst):
    with open(src) as f:
        data = json.load(f)
    res = map_aiperf(data, noncanonical_reasons=_noncanonical_reasons())
    with open(dst, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
