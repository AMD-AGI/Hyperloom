# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve AgentX routing from saved session identity or fresh launch inputs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def native_agentx_enabled(value: Any) -> bool:
    """Return whether a serialized Magpie ``benchmark.agentx`` enables AgentX."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "enable", "enabled"}
    if isinstance(value, dict):
        raw = value.get("enabled", True)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "enable", "enabled"}
        return bool(raw)
    return False


def config_enables_native_agentx(path: str | Path) -> bool:
    """Read the explicit native AgentX selector from a benchmark YAML."""
    try:
        parsed = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return False
    benchmark = parsed.get("benchmark") if isinstance(parsed, Mapping) else None
    return isinstance(benchmark, Mapping) and native_agentx_enabled(benchmark.get("agentx"))


def _session_state(state: Any, env: Mapping[str, str]) -> Any:
    if state is not None:
        return state
    session = str(env.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "") or "").strip()
    if not session:
        return None
    path = Path(session).expanduser() / "state.json"
    if not path.is_file():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"saved session state must be a mapping: {path}")
    return parsed


def _read(state: Any, name: str, default: Any = None) -> Any:
    return state.get(name, default) if isinstance(state, Mapping) else getattr(state, name, default)


def native_agentx_session(state: Any = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Resolve the native measurement contract from persisted state or source YAML.

    A supplied state is authoritative over the caller's shell. The native epoch
    keeps a resumed session native even if its accepted config is unavailable;
    resume validation must then report the missing pins instead of using legacy.
    Diagnostic profiling templates cannot replace this session identity.
    """
    source = os.environ if env is None else env
    state = _session_state(state, source)
    if state is not None:
        mode = str(_read(state, "benchmark_mode", "") or "").strip().lower()
        if mode and mode != "agentx":
            return False
        epoch = int(_read(state, "agentx_epoch", 0) or 0)
        if mode == "agentx" and epoch:
            return epoch >= 2
        backend = str(_read(state, "agentx_backend", "") or "").strip().lower()
        if backend:
            return backend == "native"
        return any(
            config_enables_native_agentx(path)
            for name in ("baseline_config_path", "benchmark_source_config_path")
            if (path := str(_read(state, name, "") or "").strip())
        )
    path = str(source.get("HYPERLOOM_BENCHMARK_CONFIG", "") or "").strip()
    enabled = str(source.get("HYPERLOOM_AGENTX", "") or "").strip().lower()
    return enabled in {"1", "true", "yes", "on", "enable", "enabled"} or (
        bool(path) and config_enables_native_agentx(path)
    )


def native_agentx_optimization_session(state: Any = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Keep saved measurement-only sessions on their original epoch-2 contract."""
    source = os.environ if env is None else env
    state = _session_state(state, source)
    if not native_agentx_session(state, env=source):
        return False
    return state is None or int(_read(state, "agentx_epoch", 0) or 0) >= 3


def managed_native_agentx_session(state: Any = None, *, env: Mapping[str, str] | None = None) -> bool:
    """Use Magpie-managed serving for fresh sessions and the saved epoch-4 contract."""
    source = os.environ if env is None else env
    state = _session_state(state, source)
    if not native_agentx_session(state, env=source):
        return False
    return state is None or int(_read(state, "agentx_epoch", 0) or 0) >= 4
