#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI wrapper: aiperf ``profile_export_aiperf.json`` -> ``inferencex_result.json``."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path


def _noncanonical_reasons():
    """Workload deviations the client detected; see aiperf_client.sh."""
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
        # Tri-state: True / False / None(absent).
        md = export.get("metadata")
        if not isinstance(md, dict) or "submission_valid" not in md:
            return None, []
        reasons = md.get("submission_invalid_reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return bool(md.get("submission_valid")), [str(r) for r in reasons]

    _SHAPE_PERCENTILES = ("avg", "p50", "p75", "p90", "p99")

    def _distribution(metric):
        if not isinstance(metric, dict):
            return {}
        return {k: int(metric[k]) for k in _SHAPE_PERCENTILES if isinstance(metric.get(k), (int, float))}

    def _corpus_loader(export):
        dataset = (export.get("metadata") or {}).get("dataset")
        return str((dataset or {}).get("loader") or "")

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
        # E2E normalised interactivity slow tail: P10 of the per-request rate OSL/E2EL_s equals 1/P90 of the
        # E2EL/OSL ratio, which is the definition upstream uses (MODELS.md:78).
        intvty_p90 = _pct(m, "e2e_output_token_throughput", "p10")
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
            "e2e_norm_intvty_p90": intvty_p90,
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
            "request_error_rate": _stat(m, "request_error_rate", default=None),
            "corpus_loader": _corpus_loader(d),
            "isl_distribution": _distribution(m.get("input_sequence_length")),
            "osl_distribution": _distribution(m.get("output_sequence_length")),
        }


def _request_metric(metrics, name, unit):
    value = metrics.get(name)
    if isinstance(value, dict):
        if value.get("unit", unit) != unit:
            raise ValueError(f"unexpected unit for {name}")
        value = value.get("value")
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value > 0 else None


def comparison_metrics(src):
    """Read the summary's request records once without changing the optimization metric."""
    records_path = Path(src).with_name("profile_export.jsonl")
    result = {
        "status": "unavailable",
        "reason": "",
        "metric_basis": "inverse_linear_p90_e2el_per_output_token",
        "unit": "tok/s/user",
        "e2e_norm_intvty_p90": None,
        "sample_count": 0,
        "source": records_path.name,
    }
    digest = hashlib.sha256()
    ratios = []
    try:
        with records_path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for line in handle:
                digest.update(line)
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("request record must be an object")
                metadata = record.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError("request metadata must be an object")
                if metadata.get("benchmark_phase") and metadata["benchmark_phase"] != "profiling":
                    continue
                metrics = record.get("metrics", {})
                if not isinstance(metrics, dict):
                    raise ValueError("request metrics must be an object")
                latency = _request_metric(metrics, "request_latency", "ms")
                ttft = _request_metric(metrics, "time_to_first_token", "ms")
                isl = _request_metric(metrics, "input_sequence_length", "tokens")
                osl = _request_metric(metrics, "output_sequence_length", "tokens")
                # Match InferenceX's turn eligibility, including the two fields outside the ratio.
                if any(value is None for value in (latency, ttft, isl, osl)):
                    continue
                ratio = latency / 1000.0 / osl
                if math.isfinite(ratio) and ratio > 0:
                    ratios.append(ratio)
            after = os.fstat(handle.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            result["reason"] = "request_records_changed"
            return result
    except FileNotFoundError:
        result["reason"] = "request_records_missing"
        return result
    except OSError:
        result["reason"] = "request_records_unreadable"
        return result
    except (ValueError, OverflowError):
        result["reason"] = "request_records_invalid"
        return result
    result["source_sha256"] = digest.hexdigest()
    result["sample_count"] = len(ratios)
    if not ratios:
        result["reason"] = "no_eligible_requests"
        return result
    ratios.sort()
    position = 0.9 * (len(ratios) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ratios) - 1)
    p90 = ratios[lower] + (ratios[upper] - ratios[lower]) * (position - lower)
    result["e2e_norm_intvty_p90"] = 1.0 / p90
    result["status"] = "ok"
    return result


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    res = map_aiperf(data, noncanonical_reasons=_noncanonical_reasons())
    res["comparison_metrics"] = comparison_metrics(src)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
