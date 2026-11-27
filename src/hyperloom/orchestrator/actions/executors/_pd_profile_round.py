# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD node-list profiling rounds: the round start on the trace filesystem's clock, graph-capture profiling on the
engines, and the client-side profiler left to the engines."""

from __future__ import annotations

import logging
from pathlib import Path
import yaml

from hyperloom.common.benchmark_config import read_benchmark_point

from ._multi_node_env import is_pd_session
from .workload._workload_envs import _SGLANG_PROFILE_CUDA_GRAPH_FLAG, without_dead_cuda_graph_profile

log = logging.getLogger(__name__)

_ENCODING = "utf-8"

_GRAPH_CAPTURE_PROFILE_ENV_KEYS = (
    "SGLANG_GRAPH_BATCH_CAPTURE",
    "DEBUG_CLR_GRAPH_PACKET_CAPTURE",
    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING",
    "DEBUG_HIP_GRAPH_CLASSIC_PATH",
)


def trace_fs_round_start(round_trace_root: str, fallback_unix: float) -> float:
    """Return the round start on the trace filesystem's clock, read from a marker file's mtime.

    Trace mtimes are stamped by the filesystem that stores them, so this round's
    traces are compared against a time from that same clock; ``fallback_unix``
    is returned when the marker cannot be written.
    """
    marker = Path(round_trace_root) / ".round_start"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{fallback_unix:.3f}\n", encoding=_ENCODING)
        return marker.stat().st_mtime
    except OSError as exc:
        log.warning(
            "profile_executor: cannot stamp round start under %s (%s); using the local clock", round_trace_root, exc
        )
        return fallback_unix


def graph_capture_round(extra_server_args: str, config_path: Path) -> tuple[str, dict[str, str]]:
    """Return the round's server args and extra env with graph-capture profiling on, as the profile YAML at
    ``config_path`` sets it, in a PD node-list session; the args unchanged and no env elsewhere, or when the YAML
    cannot be read."""
    if not is_pd_session():
        return extra_server_args, {}
    try:
        envs = read_benchmark_point(config_path).envs
    except (OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("profile_executor: cannot read %s for graph-capture profiling settings: %s", config_path, exc)
        return extra_server_args, {}
    env = {key: str(envs[key]) for key in _GRAPH_CAPTURE_PROFILE_ENV_KEYS if key in envs}
    tokens = extra_server_args.split()
    if _SGLANG_PROFILE_CUDA_GRAPH_FLAG in str(envs.get("EXTRA_SGLANG_ARGS") or "").split():
        if _SGLANG_PROFILE_CUDA_GRAPH_FLAG not in tokens:
            tokens.append(_SGLANG_PROFILE_CUDA_GRAPH_FLAG)
        extra_server_args = " ".join(without_dead_cuda_graph_profile(tokens))
    return extra_server_args, env


def leave_capture_to_engines(config_path: Path) -> None:
    """Turn the client-side profiler off in a PD node-list point, whose engines the executor starts and stops
    profiling on itself: the round loads a server it does not hold."""
    if not is_pd_session():
        return
    cfg = yaml.safe_load(config_path.read_text(encoding=_ENCODING)) or {}
    bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bench, dict):
        return
    torch_profiler = (bench.get("profiler") or {}).get("torch_profiler")
    if isinstance(torch_profiler, dict):
        torch_profiler["enabled"] = False
    envs = bench.get("envs")
    if isinstance(envs, dict):
        envs.pop("PROFILE", None)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding=_ENCODING)
