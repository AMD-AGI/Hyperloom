# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Identify native AgentX without changing the public legacy AgentX switch."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def config_enables_native_agentx(path: str | Path) -> bool:
    """Read the explicit native AgentX selector from a benchmark YAML."""
    try:
        parsed = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return False
    benchmark = parsed.get("benchmark") if isinstance(parsed, Mapping) else None
    if not isinstance(benchmark, Mapping) or benchmark.get("agentx") is None or benchmark.get("agentx") is False:
        return False
    from hyperloom.inference_optimizer.agentx.native import native_agentx_enabled

    return native_agentx_enabled(benchmark["agentx"])


def native_agentx_session(state: Any = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Resolve the native measurement contract from persisted state or source YAML.

    A supplied state is authoritative over the caller's shell. The native epoch
    keeps a resumed session native even if its accepted config is unavailable;
    resume validation must then report the missing pins instead of using legacy.
    Diagnostic profiling templates cannot replace this session identity.
    """
    if state is not None:
        read = state.get if isinstance(state, Mapping) else lambda name, default=None: getattr(state, name, default)
        mode = str(read("benchmark_mode", "") or "").strip().lower()
        if mode and mode != "agentx":
            return False
        if mode == "agentx" and int(read("agentx_epoch", 0) or 0) >= 2:
            return True
        return any(
            config_enables_native_agentx(path)
            for name in ("baseline_config_path", "benchmark_source_config_path")
            if (path := str(read(name, "") or "").strip())
        )
    source = os.environ if env is None else env
    path = str(source.get("HYPERLOOM_BENCHMARK_CONFIG", "") or "").strip()
    return bool(path) and config_enables_native_agentx(path)
