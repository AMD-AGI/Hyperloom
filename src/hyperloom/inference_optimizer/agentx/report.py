# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Map Magpie's validated AgentX report into Hyperloom measurement fields."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typing import Any

import yaml

from hyperloom.common.agentx_workload import find_aiperf_artifact
from hyperloom.common.launch_log_evidence import (
    engine_adjusted_settings_from_log,
    observed_server_config_from_log,
    observed_server_identity_from_log,
    operand_digest,
)

#: Where InferenceX's ``install_agentic_deps`` (``benchmarks/benchmark_lib.sh``) builds the round's aiperf,
#: relative to the round's workspace: Magpie points ``AIPERF_RUNTIME_DIR`` at ``<workspace>/.agentx-runtime``.
_ROUND_AIPERF = Path(".agentx-runtime", "venv", "bin", "aiperf")


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
    intvty = latency.get("e2e_norm_intvty") or {}
    result: dict[str, Any] = {
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
        "output_throughput": throughput.get("output_tokens_per_second"),
        "input_throughput": throughput.get("input_tokens_per_second"),
        "total_token_throughput": throughput.get("total_tokens_per_second"),
        "total_throughput": throughput.get("total_tokens_per_second"),
        "completed_requests": requests.get("successful"),
        "requested_requests": requests.get("profiled_total"),
        "duration_seconds": throughput.get("duration_seconds"),
        "request_error_rate": None if requests.get("error_rate") is None else requests["error_rate"] * 100,
        "e2e_norm_intvty_p50": intvty.get("p50"),
        "e2e_norm_intvty_p90": intvty.get("p90"),
        "agentx_metrics": metrics,
        "tp": recipe.get("tp"),
        "ep": recipe.get("ep"),
        "conc": recipe.get("conc"),
        "ttft_e2el_source": "magpie_report",
        "tpot_source": "magpie_report",
        "nonfatal_warnings": [],
    }
    for name in ("ttft", "tpot", "e2el"):
        for percentile in ("mean", "p50", "p90", "p99"):
            value = (latency.get(name) or {}).get(percentile)
            result[f"{name}_{percentile}_ms"] = None if value is None else value * 1000
    return result


def server_launch_environment(launch: Mapping[str, Any]) -> tuple[dict[str, str], set[str]]:
    """The environment a Magpie-verified server launch ran with, and the names its launch cleared.

    ``runtime_environment`` is the server process's own environment, which carries what the launcher injects (an
    InferenceX recipe's simulated acceptance, say); ``effective_env`` is what the recipe set over it, where ``None``
    records a name it cleared.
    """
    env = {str(name): str(value) for name, value in (launch.get("runtime_environment") or {}).items()}
    cleared: set[str] = set()
    for name, value in (launch.get("effective_env") or {}).items():
        if value is None:
            env.pop(str(name), None)
            cleared.add(str(name))
        else:
            env[str(name)] = str(value)
    return env, cleared


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
    digest = operand_digest(model)
    recipe = measurement["agentx_metrics"]["recipe"]
    framework = bench["framework"]
    return {
        "schema_version": 1,
        "materialized_config_path": str(config_path),
        "recipe_digest": "sha256:" + hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "framework": framework,
        "model_path": model,
        "requested_server_args": flags,
        "requested_server_env": env,
        "observed_server_launch_flags": flags,
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


_SHAPE_PERCENTILES = ("avg", "p50", "p75", "p90", "p99")
# Magpie's AgentX report carries no sequence lengths; the aiperf summary it leaves in the round's workspace does.
_AIPERF_SUMMARY = "profile_export_aiperf.json"


def measured_corpus_shape(workspace: Path | None) -> dict[str, Any]:
    """The ISL/OSL percentiles the round replayed, from aiperf's export; ``{}`` when it does not carry both."""
    path = None if workspace is None else find_aiperf_artifact(workspace, _AIPERF_SUMMARY)
    if path is None:
        return {}
    try:
        export = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(export, dict):
        return {}
    shape: dict[str, Any] = {}
    for key, name in (("isl", "input_sequence_length"), ("osl", "output_sequence_length")):
        metric = export.get(name)
        if not isinstance(metric, dict):
            return {}
        shape[key] = {p: int(metric[p]) for p in _SHAPE_PERCENTILES if isinstance(metric.get(p), (int, float))}
    return {**shape, "source": "measured"} if all(shape.values()) else {}


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


@dataclass(frozen=True)
class ReplayedTrace:
    """The corpus a valid AgentX round replayed and the mean request it served."""

    corpus: str
    entries: int
    requests: int
    mean_isl: int
    mean_osl: int


def replayed_trace(workspace: Path) -> ReplayedTrace | None:
    """The trace the latest valid AgentX round under ``workspace`` replayed, or ``None`` without one.

    The report publishes token rates over the measured window rather than token counts, so each
    mean is its rate times the window over the successful requests.
    """
    measured = read_agentx_report(workspace)
    if not measured:
        return None
    dataset = measured["agentx_metrics"].get("dataset") or {}
    requests, window = measured["completed_requests"], measured["duration_seconds"]
    rates = (measured["input_throughput"], measured["output_throughput"])
    if not (dataset.get("loader") and dataset.get("num_dataset_entries") and requests and window and all(rates)):
        return None
    return ReplayedTrace(
        corpus=str(dataset["loader"]),
        entries=int(dataset["num_dataset_entries"]),
        requests=int(requests),
        mean_isl=round(rates[0] * window / requests),
        mean_osl=round(rates[1] * window / requests),
    )


def round_replay_client(workspace: Path) -> Path | None:
    """The aiperf the AgentX round in ``workspace`` replayed with, or ``None`` when it cannot run here.

    Its venv links an interpreter the round installed outside the workspace, so the client runs only
    in a container where a Magpie round has run.
    """
    client = workspace / _ROUND_AIPERF
    return client if os.access(client, os.X_OK) and client.with_name("python").is_file() else None
