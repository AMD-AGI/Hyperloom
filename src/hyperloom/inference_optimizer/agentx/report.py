# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map Magpie's validated AgentX report into Hyperloom measurement fields."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from typing import Any

import yaml

from hyperloom.common.launch_log_evidence import (
    engine_adjusted_settings_from_log,
    observed_server_config_from_log,
    observed_server_identity_from_log,
)


def agentx_measurement(report: dict[str, Any]) -> dict[str, Any]:
    """Use the public report exclusively; invalid replays cannot be salvaged."""
    metrics = report.get("agentx_metrics") or {}
    throughput = metrics.get("throughput") or {}
    requests = metrics.get("requests") or {}
    latency = metrics.get("latency_seconds") or {}
    recipe = metrics.get("recipe") or {}
    diagnostic = metrics.get("diagnostic_only") is True
    managed = metrics.get("execution_owner") == "magpie" and bool(metrics.get("server_launch"))
    valid = (
        managed
        and all(report.get(key) is True for key in ("success", "benchmark_valid", "publishable"))
        and not diagnostic
    )
    identity = {key: metrics.get(key) for key in ("mode", "scenario_type", "dataset", "recipe")}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = throughput.get("output_tokens_per_second")
    return {
        "benchmark_backend": "magpie_agentx",
        "workload_identity": identity_hash,
        "reported_success": report.get("success"),
        "benchmark_valid": valid,
        "valid_measurement": valid,
        "publishable": report.get("publishable") is True,
        "diagnostic_only": diagnostic,
        "round_succeeded": report.get("success") is True and managed and (valid or diagnostic),
        "framework": report.get("framework"),
        "model": report.get("model"),
        "request_throughput": throughput.get("request_throughput"),
        "output_throughput": output,
        "input_throughput": throughput.get("input_tokens_per_second"),
        "total_token_throughput": throughput.get("total_tokens_per_second"),
        "total_throughput": throughput.get("total_tokens_per_second"),
        "completed_requests": requests.get("successful"),
        "requested_requests": requests.get("total"),
        "duration_seconds": throughput.get("duration_seconds"),
        "request_error_rate": None if requests.get("error_rate") is None else requests["error_rate"] * 100,
        "e2e_norm_intvty_p50": (latency.get("e2e_norm_intvty") or {}).get("p50"),
        "e2e_norm_intvty_p90": (latency.get("e2e_norm_intvty") or {}).get("p90"),
        "agentx_metrics": metrics,
        "agentx_server_launch": metrics.get("server_launch"),
        "tp": recipe.get("tp"),
        "ep": recipe.get("ep"),
        "conc": recipe.get("conc"),
        "ttft_e2el_source": "magpie_report",
        "tpot_source": "magpie_report",
        "nonfatal_warnings": [],
        **_latency_ms(latency),
    }


def _latency_ms(latency: dict[str, Any]) -> dict[str, float | None]:
    """The TTFT, TPOT and E2E latency percentiles of a report, converted to milliseconds."""
    fields: dict[str, float | None] = {}
    for name in ("ttft", "tpot", "e2el"):
        for percentile in ("mean", "p50", "p90", "p99"):
            value = (latency.get(name) or {}).get(percentile)
            fields[f"{name}_{percentile}_ms"] = None if value is None else value * 1000
    return fields


def native_launch_evidence(
    measurement: dict[str, Any], config_path: Path, *, actual_server_log: str | None = None
) -> dict[str, Any]:
    """Adapt the launch Magpie already verified into the optimizer's evidence shape."""
    bench = yaml.safe_load(config_path.read_text())["benchmark"]
    launch = measurement["agentx_metrics"]["server_launch"]
    argv = launch["effective_argv"]
    first_option = next((index for index, token in enumerate(argv) if token.startswith("--")), len(argv))
    flags = shlex.join(argv[first_option:])
    env = {name: value for name, value in launch["effective_env"].items() if value is not None}
    model = str(bench.get("envs", {}).get("MODEL_PATH") or bench["model"])
    digest = "sha256:" + hashlib.sha256(model.encode()).hexdigest()
    recipe = measurement["agentx_metrics"]["recipe"]
    framework = bench["framework"]
    return {
        "schema_version": 1,
        "materialized_config_path": str(config_path),
        "recipe_digest": "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "framework": framework,
        "model_path": model,
        "requested_server_args": flags,
        "requested_server_flags": flags,
        "requested_server_env": env,
        "observed_server_launch_flags": flags,
        "observed_server_env": env,
        "observed_server_config": observed_server_config_from_log(actual_server_log, framework)
        if actual_server_log
        else {},
        "observed_server_identity": observed_server_identity_from_log(actual_server_log, framework)
        if actual_server_log
        else {},
        "engine_adjusted_settings": engine_adjusted_settings_from_log(actual_server_log, framework)
        if actual_server_log
        else {},
        "requested_model_digest": digest,
        "observed_model_binding": {
            "model_digest": digest,
            "tp": str(recipe["tp"]),
            "pp": str(recipe.get("pp", 1)),
            "dp": str(recipe.get("dp", 1)),
        },
        "actual_server_log_path": actual_server_log or "",
        "warm_reuse": {"reused": False},
        "magpie_server_launch": launch,
    }


def read_agentx_report(workspace: Path) -> dict[str, Any] | None:
    """Read the latest public AgentX report for the shared correctness gate."""
    paths = list(workspace.rglob("benchmark_report.json"))
    if not paths:
        return None
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if report.get("scenario") != "agentx":
        return None
    measured = agentx_measurement(report)
    return measured if measured["valid_measurement"] else {}
