# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The operator's Magpie configuration, shared by the CLI and execution workers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

BENCHMARK_CONFIG_ENV = "HYPERLOOM_BENCHMARK_CONFIG"


def benchmark_config_path(env: Mapping[str, str] | None = None) -> Path | None:
    runtime = os.environ if env is None else env
    value = runtime.get(BENCHMARK_CONFIG_ENV, "").strip()
    return Path(value) if value else None


@lru_cache(maxsize=16)
def _read_config(path: Path, _modified_ns: int) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("benchmark"), dict):
        raise ValueError(f"{path} must contain a benchmark mapping")
    return dict(data["benchmark"])


def configured_benchmark(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    path = benchmark_config_path(env)
    if path is None:
        return {}
    return _read_config(path, path.stat().st_mtime_ns).copy()


def is_magpie_agentx(benchmark: Mapping[str, Any]) -> bool:
    value = benchmark.get("agentx")
    if isinstance(value, dict):
        value = value.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "enabled", "enable", "yes", "1"}
    return value is True
