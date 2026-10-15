# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind an explicit Magpie workload to one optimization session."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from hyperloom.common.benchmark_config import (
    BENCHMARK_CONFIG_ENV,
    benchmark_config_path,
    configured_benchmark,
    is_magpie_agentx,
)
from hyperloom.common.agentx_workload import is_mlperf_backend
from hyperloom.common.env import env_bool


_SESSION_CONFIG = "benchmark.yaml"


def configure_benchmark(args: argparse.Namespace) -> None:
    """Set launch identity from YAML before topology and model resolution."""
    if env_bool("HYPERLOOM_AGENTX") and not is_mlperf_backend():
        raise ValueError("HYPERLOOM_AGENTX is retired; use --benchmark-config with benchmark.agentx: enable")
    source = getattr(args, "benchmark_config", None)
    resume = getattr(args, "resume_from", None)
    if resume and (Path(resume) / _SESSION_CONFIG).is_file():
        archived = Path(resume) / _SESSION_CONFIG
        if source and yaml.safe_load(Path(source).read_text()) != yaml.safe_load(archived.read_text()):
            raise ValueError("A resumed session must use its recorded benchmark.yaml; start a new session to change it")
        source = archived
    if source:
        os.environ[BENCHMARK_CONFIG_ENV] = str(Path(source).expanduser().resolve(strict=True))
    benchmark = configured_benchmark()
    if not benchmark:
        return
    if is_mlperf_backend():
        raise ValueError("--benchmark-config cannot be combined with the MLPerf backend")
    if not is_magpie_agentx(benchmark):
        raise ValueError("--benchmark-config currently requires benchmark.agentx: enable")
    _validate_agentx_options(args, benchmark)
    envs = benchmark.get("envs", {})
    values = {
        "framework": benchmark["framework"],
        "model": envs.get("MODEL_PATH") or benchmark["model"],
        "precision": benchmark["precision"],
        "tp": int(envs["TP"]),
        "ep": int(envs.get("EP_SIZE", envs.get("EP", 1))),
        "conc": int(envs["CONC"]),
    }
    for name, value in values.items():
        current = getattr(args, name, None)
        if current is not None and str(current) != str(value):
            raise ValueError(f"--{name} conflicts with benchmark YAML; define the workload once")
        setattr(args, name, value)
    args.enable_conc_sweep = False


def _validate_agentx_options(args: argparse.Namespace, benchmark: dict) -> None:
    if benchmark.get("framework") not in {"sglang", "vllm"}:
        raise ValueError("Magpie AgentX optimization supports SGLang or vLLM")
    if int(getattr(args, "nodes", 1)) != 1:
        raise ValueError("Magpie AgentX optimization requires one node")
    if getattr(args, "enable_conc_sweep", None):
        raise ValueError("AgentX concurrency is fixed per session; create a session for each CONC")
    conflicting = [
        name
        for name in ("isl", "osl", "profile_osl", "max_model_len", "reference_script", "quantize", "quantize_scheme")
        if getattr(args, name, None)
    ]
    if conflicting:
        raise ValueError("AgentX workload comes from YAML; unsupported CLI overrides: " + ", ".join(conflicting))
    agentx = benchmark["agentx"]
    if isinstance(agentx, dict) and agentx.get("mode", "canonical") != "canonical":
        raise ValueError("Optimization requires canonical AgentX; fast runs are diagnostic only")
    if benchmark.get("profiler", {}).get("torch_profiler", {}).get("enabled"):
        raise ValueError("Disable torch_profiler in the benchmark YAML; the profile phase enables it separately")
    envs = benchmark.get("envs", {})
    required = ("model", "precision", "docker_image")
    if any(not benchmark.get(key) for key in required):
        raise ValueError("AgentX YAML requires model, precision and docker_image")
    if any(int(envs.get(key, 0)) <= 0 for key in ("TP", "CONC")):
        raise ValueError("AgentX YAML requires explicit positive envs.TP and envs.CONC for resource accounting")
    legacy = [key for key in envs if key.startswith(("AIPERF_", "AGENTX_", "AGENTIC_"))]
    if legacy:
        raise ValueError("Use Magpie agentx configuration instead of legacy client controls: " + ", ".join(legacy))


def bind_benchmark(session_dir: Path) -> None:
    """Snapshot the source before any round can mutate a candidate configuration."""
    source = benchmark_config_path()
    if source is None:
        return
    target = session_dir / _SESSION_CONFIG
    if source.resolve() != target.resolve():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ[BENCHMARK_CONFIG_ENV] = str(target.resolve())


def add_benchmark_argument(parser: argparse.ArgumentParser) -> None:
    """Register the workload's YAML entrypoint."""
    parser.add_argument(
        "--benchmark-config", type=Path, help="Magpie AgentX YAML: model, image, topology and fixed concurrency."
    )
