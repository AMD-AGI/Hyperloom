# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind a native Magpie workload to one optimization session."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import (
    BENCHMARK_CONFIG_ENV,
    benchmark_config_path,
    configured_benchmark,
    is_magpie_agentx,
)
from hyperloom.common.agentx_workload import is_mlperf_backend
from common.env import env_bool
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.gpu_types import gpu_runner_type


_SESSION_CONFIG = "benchmark.yaml"
_CANONICAL = "canonical"


def configure_benchmark(args: argparse.Namespace) -> None:
    """Validate the requested workload before resolving the ordinary CLI defaults."""
    from .bootstrap import parse_operator_extra_env

    pins = parse_operator_extra_env(args)
    if BENCHMARK_CONFIG_ENV in pins:
        raise ValueError("Use --benchmark-config to select the workload YAML")
    runtime = {**os.environ, **pins}
    mlperf = is_mlperf_backend(runtime)
    native = env_bool("HYPERLOOM_AGENTX", env=runtime) and not mlperf
    source = getattr(args, "benchmark_config", None) or benchmark_config_path()
    resume = getattr(args, "resume_from", None)
    if resume:
        archived = Path(resume) / _SESSION_CONFIG
        if archived.is_file():
            explicit = getattr(args, "benchmark_config", None)
            if explicit and yaml.safe_load(Path(explicit).read_text()) != yaml.safe_load(archived.read_text()):
                raise ValueError(
                    "A resumed session must use its recorded benchmark.yaml; start a new session to change it"
                )
            source = archived
        elif source or native:
            raise ValueError("Native AgentX resume requires the session's benchmark.yaml; start a new session")
    if source:
        os.environ[BENCHMARK_CONFIG_ENV] = str(Path(source).expanduser().resolve(strict=True))
        if mlperf:
            raise ValueError("--benchmark-config cannot be combined with the MLPerf backend")
        benchmark = configured_benchmark()
        _validate_agentx_benchmark(benchmark)
        _bind_yaml_arguments(args, benchmark)
    elif not native:
        if getattr(args, "docker_image", None):
            raise ValueError("--docker-image requires native AgentX")
        return
    _validate_agentx_options(args)
    args.enable_conc_sweep = False


def _bind_yaml_arguments(args: argparse.Namespace, benchmark: dict) -> None:
    """Keep CLI flags and operator pins consistent with the archived workload."""
    from .bootstrap import parse_operator_extra_env

    envs = benchmark.get("envs", {})
    values = {
        "framework": benchmark["framework"],
        "model": envs.get("MODEL_PATH") or benchmark["model"],
        "precision": benchmark["precision"],
        "tp": int(envs["TP"]),
        "ep": int(envs.get("EP_SIZE", envs.get("EP", 1))),
        "conc": int(envs["CONC"]),
        "docker_image": benchmark.get("docker_image"),
    }
    pins = parse_operator_extra_env(args)
    for name, value in values.items():
        current = getattr(args, name, None)
        if current is not None and str(current) != str(value):
            raise ValueError(f"--{name.replace('_', '-')} conflicts with benchmark YAML; define the workload once")
        key = "MODEL_PATH" if name == "model" else name.upper()
        if key in pins and str(pins[key]) != str(value):
            raise ValueError(f"--extra-env {key} conflicts with benchmark YAML")
        setattr(args, name, value)


def _validate_agentx_options(args: argparse.Namespace) -> None:
    from .bootstrap import parse_operator_extra_env

    pins = parse_operator_extra_env(args)
    framework = (
        args.framework or pins.get("FRAMEWORK") or os.environ.get("FRAMEWORK") or framework_registry.DEFAULT_FRAMEWORK
    )
    if framework not in {"sglang", "vllm"}:
        raise ValueError("Magpie AgentX optimization supports SGLang or vLLM")
    if int(getattr(args, "nodes", 1)) != 1:
        raise ValueError("Magpie AgentX optimization requires one node")
    if any(getattr(args, key, None) is not None and int(getattr(args, key)) <= 0 for key in ("tp", "ep", "conc")):
        raise ValueError("AgentX TP, EP and CONC must be positive")
    if getattr(args, "enable_conc_sweep", None):
        raise ValueError("AgentX concurrency is fixed per session; create a session for each CONC")
    conflicting = [
        name
        for name in ("isl", "osl", "profile_osl", "max_model_len", "reference_script", "quantize", "quantize_scheme")
        if getattr(args, name, None)
    ]
    if conflicting:
        raise ValueError("AgentX trace replay does not support these CLI overrides: " + ", ".join(conflicting))
    unsupported = [
        key
        for key in pins
        if key.startswith(("AIPERF_", "AGENTX_", "AGENTIC_"))
        or key in {"ISL", "OSL", "RANDOM_RANGE_RATIO", "MAX_MODEL_LEN", "EP_SIZE"}
    ]
    if unsupported:
        raise ValueError("Unsupported AgentX --extra-env controls (use --ep for EP_SIZE): " + ", ".join(unsupported))


def _validate_agentx_benchmark(benchmark: dict) -> None:
    if not is_magpie_agentx(benchmark):
        raise ValueError("--benchmark-config currently requires benchmark.agentx: enable")
    agentx = benchmark["agentx"]
    if isinstance(agentx, dict) and agentx.get("mode", _CANONICAL) != _CANONICAL:
        raise ValueError("Optimization requires canonical AgentX; fast runs are diagnostic only")
    if benchmark.get("profiler", {}).get("torch_profiler", {}).get("enabled"):
        raise ValueError("Disable torch_profiler in the benchmark YAML; the profile phase enables it separately")
    envs = benchmark.get("envs", {})
    required = ("framework", "model", "precision")
    if any(not benchmark.get(key) for key in required):
        raise ValueError("AgentX YAML requires framework, model and precision")
    if any(int(envs.get(key, 0)) <= 0 for key in ("TP", "CONC")):
        raise ValueError("AgentX YAML requires explicit positive envs.TP and envs.CONC for resource accounting")
    legacy = [key for key in envs if key.startswith(("AIPERF_", "AGENTX_", "AGENTIC_"))]
    if legacy:
        raise ValueError("Use Magpie agentx configuration instead of legacy client controls: " + ", ".join(legacy))


def _generated_agentx_benchmark(args: argparse.Namespace) -> dict:
    """Translate resolved Hyperloom workload values into Magpie's public schema."""
    benchmark: dict[str, Any] = {
        "framework": os.environ["FRAMEWORK"],
        "model": str(args.model),
        "precision": args.precision,
        "run_mode": "local",
        "agentx": {"enabled": True, "mode": _CANONICAL, "launch_overrides": {"version": 1}},
        "envs": {"TP": args.tp, "EP_SIZE": args.ep, "CONC": args.conc},
        "profiler": {"torch_profiler": {"enabled": False}},
    }
    model_path = os.environ.get("MODEL_PATH", "")
    if model_path and Path(model_path).is_dir():
        benchmark["envs"]["MODEL_PATH"] = str(Path(model_path).resolve())
    if args.gpu_type:
        # Magpie resolves {framework}_{runner_type}.sh and ships no script for mi308x or
        # mi325x; both run the mi300x one. ``args.gpu_type`` is the real board, so the
        # runner label has to be derived here the way $GPU_TYPE already is.
        benchmark["runner_type"] = gpu_runner_type(args.gpu_type)
    if args.docker_image:
        benchmark["docker_image"] = args.docker_image
    _validate_agentx_benchmark(benchmark)
    return benchmark


def bind_benchmark(session_dir: Path, args: argparse.Namespace) -> None:
    """Archive the workload after CLI defaults resolve, before publishing the session."""
    source = benchmark_config_path()
    target = session_dir / _SESSION_CONFIG
    if source is not None:
        if source.resolve() != target.resolve():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    elif env_bool("HYPERLOOM_AGENTX") and not is_mlperf_backend():
        target.write_text(
            yaml.safe_dump({"benchmark": _generated_agentx_benchmark(args)}, sort_keys=False), encoding="utf-8"
        )
    else:
        return
    os.environ[BENCHMARK_CONFIG_ENV] = str(target.resolve())


def add_benchmark_argument(parser: argparse.ArgumentParser) -> None:
    """Register the native workload's optional YAML and image inputs."""
    parser.add_argument(
        "--benchmark-config", type=Path, help="Optional Magpie AgentX YAML instead of Hyperloom workload flags."
    )
    parser.add_argument(
        "--docker-image", help="AgentX serving image identity (run Hyperloom inside it); required for custom models."
    )
