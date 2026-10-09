# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build a fresh native AgentX source from resolved operator workload inputs."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.agentx_mode import native_agentx_enabled, native_agentx_session


def _source_benchmark() -> dict[str, Any]:
    source = os.environ.get("HYPERLOOM_BENCHMARK_CONFIG", "").strip()
    if not source:
        return {}
    content = Path(source).read_bytes()
    expected = os.environ.get("HYPERLOOM_BENCHMARK_CONFIG_SHA256", "").strip()
    if not expected or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"benchmark config changed after initial validation: {source}")
    parsed = yaml.safe_load(content)
    benchmark = parsed.get("benchmark") if isinstance(parsed, Mapping) else None
    if not isinstance(benchmark, dict):
        raise ValueError(f"benchmark config must contain a benchmark mapping: {source}")
    if "agentx" in benchmark and not native_agentx_enabled(benchmark["agentx"]):
        raise ValueError("HYPERLOOM_AGENTX=1 conflicts with disabled benchmark.agentx")
    if "agentx" not in benchmark:
        benchmark.pop("benchmark_script", None)
    return benchmark


def _canonical_model(args: argparse.Namespace, benchmark: Mapping[str, Any]) -> str:
    model = str(os.environ.get("AGENTX_MODEL_ID") or benchmark.get("model") or getattr(args, "model", "") or "").strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model) or Path(model).expanduser().exists():
        raise ValueError(
            "native AgentX requires a canonical model id for recipe selection; "
            "pass --model <organization/model> or set AGENTX_MODEL_ID when serving a local checkpoint"
        )
    return model


def _workload_envs(args: argparse.Namespace, benchmark: dict[str, Any], model: str) -> dict[str, Any]:
    """Preserve source inputs and project only explicit topology/context overrides."""
    envs = dict(benchmark.get("envs") or {})
    envs.update(CONC=int(args.conc), AGENTX_MODEL_ID=model)
    for attr, key in (("tp", "TP"), ("ep", "EP_SIZE"), ("max_model_len", "MAX_MODEL_LEN")):
        value = getattr(args, attr, None)
        if value is not None:
            envs[key] = int(value)
    model_path = str(os.environ.get("MODEL_PATH") or args.model or "")
    if Path(model_path).expanduser().exists():
        envs["MODEL_PATH"] = str(Path(model_path).expanduser().resolve())
    image = os.environ.get("HYPERLOOM_IMAGE", "").strip()
    if image:
        source_image = str(benchmark.get("docker_image") or "").strip()
        if source_image and source_image != image:
            raise ValueError("HYPERLOOM_IMAGE conflicts with benchmark.docker_image")
        benchmark["docker_image"] = image
    return envs


def prepare_native_agentx_source(args: argparse.Namespace) -> bool:
    """Snapshot fresh managed AgentX inputs without modifying an operator's YAML."""
    if getattr(args, "resume_from", None) or not native_agentx_session():
        return False
    benchmark = _source_benchmark()
    model = _canonical_model(args, benchmark)
    runner = str(getattr(args, "gpu_type", "") or "").strip().lower()
    if not runner:
        raise ValueError("native AgentX requires a detected GPU or explicit --gpu-type to resolve its recipe")
    configured_runner = str(benchmark.get("runner_type") or "").strip().lower()
    if configured_runner and configured_runner != runner:
        raise ValueError(f"benchmark.runner_type {configured_runner!r} conflicts with detected GPU {runner!r}")
    raw_agentx = benchmark.get("agentx")
    agentx = dict(raw_agentx) if isinstance(raw_agentx, Mapping) else {}
    agentx["enabled"] = True
    agentx.setdefault("launch_overrides", {"version": 1})
    benchmark.update(
        agentx=agentx,
        model=model,
        framework=str(args.framework or os.environ.get("FRAMEWORK") or ""),
        precision=str(args.precision),
        runner_type=runner,
    )
    benchmark["envs"] = _workload_envs(args, benchmark, model)
    with tempfile.NamedTemporaryFile(mode="w", prefix="hyperloom-agentx-", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump({"benchmark": benchmark}, stream, sort_keys=False)
        source = Path(stream.name)
    args._generated_agentx_source = str(source)
    os.environ["HYPERLOOM_BENCHMARK_CONFIG"] = str(source)
    os.environ["HYPERLOOM_BENCHMARK_CONFIG_SHA256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    os.environ["AGENTX_MODEL_ID"] = model
    return True
