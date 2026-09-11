# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read external-comparison inputs from the accepted run, without changing grading."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.inference_optimizer.framework_registry import server_args_env_name


_COMPARISON_BASIS = "inverse_linear_p90_e2el_per_output_token"
_TP_FLAGS = {"--tp", "--tp-size", "--tensor-parallel-size", "-tp"}
_SINGLETON_FLAGS = {
    "--pp-size",
    "--pipeline-parallel-size",
    "-pp",
    "--dp-size",
    "--data-parallel-size",
    "-dp",
    "--prefill-context-parallel-size",
    "-pcp",
    "--nnodes",
    "--num-nodes",
}


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: Any) -> int | None:
    number = _positive(value)
    return int(number) if number is not None and number.is_integer() else None


def _measurement_recipe(evidence: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = evidence.get("materialized_config_path")
    digest = evidence.get("recipe_digest")
    if not path or not digest:
        return {}, "recipe_missing"
    try:
        source = Path(path).read_bytes()
        if f"sha256:{hashlib.sha256(source).hexdigest()}" != digest:
            return {}, "recipe_mismatch"
        config = yaml.safe_load(source)
    except (OSError, ValueError, yaml.YAMLError):
        return {}, "recipe_unreadable"
    benchmark = config.get("benchmark") if isinstance(config, dict) else None
    if not isinstance(benchmark, dict):
        return {}, "recipe_invalid"
    return benchmark, ""


def _recipe_scope(benchmark: dict[str, Any], evidence: Mapping[str, Any]) -> tuple[int | None, int | None, str]:
    envs = benchmark.get("envs")
    if not isinstance(envs, dict):
        return None, None, "gpu_count_missing"
    conc = _positive_int(envs.get("CONC"))
    tp = _positive_int(envs.get("TP"))
    if tp is None:
        return None, conc, "gpu_count_missing"
    if str(benchmark.get("framework") or "").lower() not in {"sglang", "vllm"}:
        return None, conc, "unsupported_topology"
    if benchmark.get("run_mode", "local") != "local" or benchmark.get("pd"):
        return None, conc, "unsupported_topology"
    for key in ("PP", "DP", "PCP", "NNODES", "NUM_NODES", "INFERENCE_OPTIMIZER_NODES"):
        if key in envs and _positive_int(envs[key]) != 1:
            return None, conc, "unsupported_topology"
    if str(envs.get("PD_MODE") or "aggregated").lower() not in {"aggregated", "none"}:
        return None, conc, "unsupported_topology"
    warm_reuse = evidence.get("warm_reuse") or {}
    if isinstance(warm_reuse, dict) and warm_reuse.get("provenance") == "caller_ready_server":
        return None, conc, "topology_unverified"
    requested_env = evidence.get("requested_server_env") or {}
    if isinstance(requested_env, dict):
        for key, expected in (("TP", tp), ("CONC", conc)):
            if key in requested_env and _positive_int(requested_env[key]) != expected:
                return None, conc, "topology_mismatch"
    observed = evidence.get("observed_server_identity") or {}
    if isinstance(observed, dict):
        if "tp_size" in observed and _positive_int(observed["tp_size"]) != tp:
            return None, conc, "topology_mismatch"
        for key in ("pp_size", "dp_size", "pcp_size", "nnodes"):
            if key in observed and _positive_int(observed[key]) != 1:
                return None, conc, "unsupported_topology"
    arguments = [str(evidence.get("requested_server_args") or "")]
    arguments.append(str(envs.get(server_args_env_name(benchmark.get("framework"))) or ""))
    try:
        for argument in arguments:
            tokens = shlex.split(argument)
            for index, token in enumerate(tokens):
                flag, sep, inline = token.partition("=")
                if flag == "--disaggregation-mode":
                    return None, conc, "unsupported_topology"
                if flag not in _TP_FLAGS | _SINGLETON_FLAGS:
                    continue
                value = inline if sep else tokens[index + 1] if index + 1 < len(tokens) else None
                expected = tp if flag in _TP_FLAGS else 1
                if _positive_int(value) != expected:
                    return None, conc, "unsupported_topology"
    except ValueError:
        return None, conc, "topology_unverified"
    return tp, conc, ""


def load_local_measurement(current_best: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the winner's explicit raw result; unavailable axes stay absent."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": "",
        "total_tput_per_gpu": None,
        "e2e_norm_intvty_p90": None,
        "gpu_count": None,
        "conc": None,
        "throughput_reason": "",
        "interactivity_reason": "",
    }
    measurement = current_best.get("measurement")
    measurement = measurement if isinstance(measurement, Mapping) else {}
    path = measurement.get("raw_result_path")
    if not path:
        result["reason"] = "raw_result_missing"
        return result
    result["raw_result_path"] = str(path)
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result["reason"] = "raw_result_unreadable"
        return result
    if not isinstance(raw, dict):
        result["reason"] = "raw_result_invalid"
        return result
    output = _positive(raw.get("output_throughput"))
    if output is None or output != _positive(current_best.get("tput")) or output != _positive(measurement.get("tput")):
        result["reason"] = "measurement_mismatch"
        return result
    total = _positive(raw.get("total_token_throughput"))
    if total is None:
        inp = _positive(raw.get("input_throughput"))
        total = inp + output if inp is not None else None
    for key, actual in (
        ("total_throughput", total),
        ("e2e_norm_intvty_p90", _positive(raw.get("e2e_norm_intvty_p90"))),
    ):
        if key in current_best and _positive(current_best[key]) != actual:
            result["reason"] = "measurement_mismatch"
            return result
    if raw.get("submission_valid") is not True:
        result["reason"] = "submission_not_valid"
        return result
    evidence = measurement.get("launch_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    benchmark, recipe_reason = _measurement_recipe(evidence)
    result["precision"] = str(benchmark.get("precision") or "").strip().lower()
    gpu_count, conc, scope_reason = (None, None, recipe_reason) if recipe_reason else _recipe_scope(benchmark, evidence)
    result.update(gpu_count=gpu_count, conc=conc)
    if gpu_count is not None and total is not None:
        result["total_tput_per_gpu"] = total / gpu_count
    else:
        result["throughput_reason"] = scope_reason or "total_throughput_missing"
    comparison = raw.get("comparison_metrics")
    comparison = comparison if isinstance(comparison, dict) else {}
    if (
        comparison.get("status") == "ok"
        and comparison.get("metric_basis") == _COMPARISON_BASIS
        and comparison.get("unit") == "tok/s/user"
    ):
        result["e2e_norm_intvty_p90"] = _positive(comparison.get("e2e_norm_intvty_p90"))
    if result["e2e_norm_intvty_p90"] is None:
        result["interactivity_reason"] = str(comparison.get("reason") or "exact_p90_missing")
    if result["total_tput_per_gpu"] is not None or result["e2e_norm_intvty_p90"] is not None:
        result["status"] = "ok"
    else:
        result["reason"] = "comparison_metrics_unavailable"
    return result
