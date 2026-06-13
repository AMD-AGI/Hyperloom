# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared single-node ``server_lifecycle`` helpers.

Magpie's ``server_lifecycle`` reuse protocol lets two benchmark rounds share
one persistent server: round 1 boots it (``cleanup=false``) and round 2
re-attaches as a client-only run (``cleanup=true``). Used by the baseline
cold-start double-run guard and by the explore warm-rebench gate.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import yaml

from ._subprocess_kill import _process_group_alive, _signal_group


log = logging.getLogger(__name__)


# Magpie built-in benchmark scripts that honour ``MAGPIE_RUN_PHASE`` and
# support the server_lifecycle reuse protocol. Static mirror of Magpie's
# ``benchmarker.MAGPIE_BUILTIN_SCRIPTS`` (duplicated to avoid an import-time
# Magpie dependency); keep in sync. ``atom_*`` per AMD-AGI/Magpie#34.
MAGPIE_BUILTIN_SCRIPTS = frozenset(
    {
        "vllm_mi300x.sh",
        "vllm_mi355x.sh",
        "sglang_mi300x.sh",
        "sglang_mi355x.sh",
        "atom_mi300x.sh",
        "atom_mi355x.sh",
    }
)

# Default HTTP port for the persistent server when ``benchmark.envs.PORT`` is
# unset; pinned into the per-round YAML so Magpie's reuse keying and our
# teardown agree.
REUSE_PORT_DEFAULT = 8888

# Server-boot budget for the persistent server phase.
# Override via ``INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC``.
SERVER_READY_TIMEOUT_SEC = 2700


def resolve_lifecycle_params(materialized_config_path: Path) -> dict[str, Any]:
    """Inspect the materialized YAML for server_lifecycle eligibility.

    Returns a dict with ``eligible`` (bool), ``framework`` (str),
    ``port`` (int) and ``reason`` (str, populated when ineligible).
    Reuse is single-node only, requires a Magpie built-in script, and is
    incompatible with torch_profiler.
    """
    info: dict[str, Any] = {
        "eligible": False,
        "framework": "",
        "port": REUSE_PORT_DEFAULT,
        "reason": "",
    }
    try:
        with Path(materialized_config_path).open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        info["reason"] = f"could not read materialized config: {exc}"
        return info
    bench = cfg.get("benchmark") or {}
    info["framework"] = str(bench.get("framework") or "").lower()
    envs = bench.get("envs") or {}
    try:
        info["port"] = int(envs.get("PORT", REUSE_PORT_DEFAULT))
    except (TypeError, ValueError):
        info["port"] = REUSE_PORT_DEFAULT

    from ._multi_node_env import is_multi_node
    if is_multi_node():
        info["reason"] = "multi-node (server_lifecycle is local-only)"
        return info

    script_name = Path(str(bench.get("benchmark_script") or "")).name
    if script_name not in MAGPIE_BUILTIN_SCRIPTS:
        info["reason"] = (
            f"benchmark_script={script_name!r} is not a Magpie built-in "
            f"({sorted(MAGPIE_BUILTIN_SCRIPTS)})"
        )
        return info

    profiler_on = bool(
        (bench.get("profiler") or {})
        .get("torch_profiler", {})
        .get("enabled")
    )
    if profiler_on:
        info["reason"] = "torch_profiler enabled (incompatible with reuse)"
        return info

    info["eligible"] = True
    return info


def inject_lifecycle(
    bench: dict[str, Any], *, cleanup: bool, pid_dir: Path | str, port: int,
) -> None:
    """Mutate ``bench`` in place to enable the server_lifecycle protocol.

    Both rounds share ``pid_dir`` + ``port`` so round 2 re-attaches; only
    ``cleanup`` differs (round 1 persists, round 2 tears down).
    """
    ready_timeout = int(os.environ.get(
        "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC",
        SERVER_READY_TIMEOUT_SEC,
    ))
    bench["server_lifecycle"] = {
        "enabled": True,
        "cleanup": bool(cleanup),
        "force_reuse": False,
        "pid_dir": str(pid_dir),
        "server_ready_timeout_s": ready_timeout,
    }
    # Pin PORT so Magpie's reuse keying and our teardown agree.
    envs = bench.setdefault("envs", {})
    envs["PORT"] = int(port)


def teardown_lifecycle_server(
    *, pid_dir: Path | str, framework: str, port: int,
) -> None:
    """Best-effort teardown of a persistent server left by a lifecycle round.

    Idempotent and never raises (safe in ``finally``); a no-op on the happy
    path, real work only on abnormal paths.
    """
    base = Path(pid_dir)
    tag = f"{framework}_{port}"
    pid_file = base / f"{tag}.pid"
    meta_file = base / f"{tag}.json"
    server_pid: int | None = None
    server_pgid: int | None = None
    try:
        if pid_file.exists():
            parts = pid_file.read_text(encoding="utf-8").split()
            if parts:
                server_pid = int(parts[0])
            if len(parts) > 1:
                server_pgid = int(parts[1])
    except (OSError, ValueError):
        # Best-effort: proceed with whatever was parsed; never raise.
        pass

    if server_pid is not None and os.name == "posix":
        # Server is setsid'd, so pgid == pid unless the pid file gave one.
        pgid = server_pgid if server_pgid is not None else server_pid
        _signal_group(pgid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not _process_group_alive(pgid):
                break
            time.sleep(0.1)
        if _process_group_alive(pgid):
            _signal_group(pgid, signal.SIGKILL)
        log.info(
            "server_lifecycle teardown — reaped persistent server "
            "pgid=%d (%s:%d)", pgid, framework, port,
        )
    for p in (pid_file, meta_file):
        try:
            p.unlink()
        except OSError:
            # Already gone or unremovable; teardown must not raise.
            pass
