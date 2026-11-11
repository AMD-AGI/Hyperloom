# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared single-node ``server_lifecycle`` helpers."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from hyperloom.common import rocm_smi
from hyperloom.common.benchmark_config import BenchmarkPointError, read_benchmark_point
from hyperloom.common.ports import pick_server_port
from hyperloom.inference_optimizer import framework_registry
from inference_suite import stop
from inference_suite.contracts.layout import SERVER_PID


log = logging.getLogger(__name__)

#: How often a hand-off re-reads VRAM while it waits for the GPUs to drain.
_GPU_FREE_POLL_SECONDS = 5.0


# Default HTTP port for the persistent server when ``benchmark.envs.PORT`` is unset; pinned into the per-round YAML.
REUSE_PORT_DEFAULT = 8888

# Server-boot budget for the persistent server phase.
SERVER_READY_TIMEOUT_SEC = 7200


def resolve_lifecycle_params(materialized_config_path: Path) -> dict[str, Any]:
    """Inspect the materialized YAML for server_lifecycle eligibility."""
    info: dict[str, Any] = {
        "eligible": False,
        "framework": "",
        "port": REUSE_PORT_DEFAULT,
        "reason": "",
    }
    try:
        point = read_benchmark_point(Path(materialized_config_path))
    except (OSError, yaml.YAMLError, BenchmarkPointError) as exc:
        info["reason"] = f"could not read materialized config: {exc}"
        return info
    info["framework"] = point.framework.lower()
    try:
        info["port"] = int(point.envs.get("PORT", REUSE_PORT_DEFAULT))
    except (TypeError, ValueError):
        info["port"] = REUSE_PORT_DEFAULT

    workload_kind = point.workload_kind.strip().lower()
    if workload_kind == framework_registry.SCRIPTABLE or framework_registry.is_scriptable(info["framework"]):
        info["reason"] = "scriptable framework (server-less; no server_lifecycle)"
        return info

    from ._multi_node_env import is_multi_node

    if is_multi_node():
        info["reason"] = "multi-node (server_lifecycle is local-only)"
        return info

    if point.torch_profiler:
        info["reason"] = "torch_profiler enabled (incompatible with reuse)"
        return info

    # A per-session free port rather than the fixed default: a server a prior attempt left up, or a co-tenant job,
    # may hold the default.
    info["port"] = pick_server_port()

    info["eligible"] = True
    return info


def inject_lifecycle(
    bench: dict[str, Any],
    *,
    cleanup: bool,
    pid_dir: Path | str,
    port: int,
) -> None:
    """Mutate ``bench`` in place to enable the server_lifecycle protocol."""
    ready_timeout = int(
        os.environ.get(
            "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC",
            SERVER_READY_TIMEOUT_SEC,
        )
    )
    bench["server_lifecycle"] = {
        "enabled": True,
        "cleanup": bool(cleanup),
        "force_reuse": False,
        "pid_dir": str(pid_dir),
        "server_ready_timeout_s": ready_timeout,
    }
    # Pin PORT so every round of the lifecycle reaches the same server.
    envs = bench.setdefault("envs", {})
    envs["PORT"] = int(port)


def inject_lifecycle_settings(bench: dict[str, Any], settings: dict[str, Any]) -> None:
    """Enable the server_lifecycle protocol on ``bench`` from a grid's ``server_lifecycle`` settings."""
    inject_lifecycle(
        bench,
        cleanup=bool(settings.get("cleanup", True)),
        pid_dir=settings["pid_dir"],
        port=int(settings["port"]),
    )


def reap_orphaned_servers(session_dir: Path | str) -> list[int]:
    """Stop every server a pidfile under this session's ``runs/`` still records; return the pids signalled.

    Each ``server.pid`` directory goes through :func:`inference_suite.stop`: the recorded pid is signalled only while
    its start time still matches, and once that leader is gone its workers are found by the pidfile's run mark, so a
    co-located session's server and a recycled pid are never touched. A pidfile whose server outlives SIGKILL is kept.
    """
    runs = Path(session_dir) / "runs"
    reaped: list[int] = []
    for pid_dir in _suite_pid_dirs(runs):
        outcome = stop(pid_dir)
        reaped.extend(outcome.reaped)
        if outcome.survivors:
            log.error("orphan-reaper: pids %s from %s outlived SIGKILL", list(outcome.survivors), pid_dir)
    return reaped


def _suite_pid_dirs(root: Path) -> list[Path]:
    """Every pid_dir under ``root`` holding an inference_suite ``server.pid``."""
    return [pidfile.parent for pidfile in sorted(root.rglob(SERVER_PID))] if root.is_dir() else []


class GpuHandoff(NamedTuple):
    """Whether the GPUs a hand-off names are free, and what freeing them took."""

    #: This session's leftover servers that were stopped first.
    reaped: list[int]
    #: ``(card, used_mib, total_mib)`` for every named card still holding VRAM at the deadline.
    busy: list[tuple[int, float, float]]
    #: The directories whose ``server.pid`` outlived the reap, when ``busy`` is non-empty.
    holders: list[str]
    #: Why the cards could not be checked at all; empty when they were.
    unverified: str


def free_gpus_for_handoff(
    session_dir: Path | str,
    cards: Sequence[int] | None,
    *,
    timeout_sec: float,
    poll_sec: float = _GPU_FREE_POLL_SECONDS,
) -> GpuHandoff:
    """Stop this session's leftover servers, then wait up to ``timeout_sec`` for ``cards`` to hold no VRAM.

    For a caller holding every serving lane: no round of this session can be serving then, so a server its pidfiles
    still name was left behind by a teardown that never reached it. The reap runs whatever VRAM reads, since a
    leftover still loading weights holds none yet.

    Args:
        session_dir: The session whose ``runs/`` pidfiles name the leftovers.
        cards: Physical card indices as rocm-smi numbers them; ``None`` when the caller could not resolve them, which
            still reaps but verifies nothing.
        timeout_sec: How long the cards are given to drain.
        poll_sec: How often VRAM is re-read meanwhile.

    Returns:
        What was reaped, and either which cards were still busy at the deadline and who holds them, or why the cards
        could not be checked.
    """
    reaped = reap_orphaned_servers(session_dir)
    if cards is None:
        return GpuHandoff(reaped, [], [], "the cards could not be resolved to physical GPU indices")
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        usage = rocm_smi.gpu_vram_usage()
        if usage is None:
            return GpuHandoff(reaped, [], [], "rocm-smi could not report VRAM")
        if any(not 0 <= card < len(usage) for card in cards):
            return GpuHandoff(
                reaped, [], [], f"rocm-smi reports {len(usage)} GPU(s), which does not cover cards {list(cards)}"
            )
        busy = [
            (card, gpu.used_mib, gpu.total_mib)
            for card, gpu in enumerate(usage)
            if card in cards and rocm_smi.vram_busy(gpu)
        ]
        if not busy:
            return GpuHandoff(reaped, [], [], "")
        if time.monotonic() >= deadline:
            return GpuHandoff(
                reaped, busy, [str(pid_dir) for pid_dir in _suite_pid_dirs(Path(session_dir) / "runs")], ""
            )
        time.sleep(poll_sec)
