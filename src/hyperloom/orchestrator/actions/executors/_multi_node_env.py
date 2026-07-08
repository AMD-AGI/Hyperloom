# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Helper that bridges the multi-node CLI state into Magpie subprocesses.

Lives in the executors package (called by ``baseline.py`` / ``_grid_runner.py``
before they launch Magpie) so the dependency edge stays one-way: executors
import this; ``multi_node/`` knows nothing about them.

Reads ``$INFERENCE_OPTIMIZER_NODES`` + ``$MULTI_NODE_STATE_FILE`` (session-scoped
under ``<session_dir>/runtime/`` when pinned, else legacy ``/tmp/``). Single
node (< 2): returns ``{}`` (single-pod path preserved). Multi-node (>= 2) with
a ``service_url``: returns ``MAGPIE_RUN_PHASE=client`` +
``BENCHMARK_BASE_URL=<service_url>`` so Magpie skips its own server launch and
points ``benchmark_serving`` at the head pod. Missing state file: WARN + ``{}``.
:func:`export_ray_address_to_os` also copies ``ray_address`` into
``RAY_ADDRESS`` for kernel-agent ``ray.init``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.multi_node.state_paths import (
    legacy_state_file,
    resolve_state_file,
    state_file_safe_to_read,
)

log = logging.getLogger(__name__)


def _state_path() -> Path:
    """Resolve where the multi_node CLI dropped its state file.

    Returns:
        Path: The state-file path from :func:`resolve_state_file`.
    """
    return resolve_state_file()


def _read_state() -> dict[str, Any]:
    """Best-effort read of the state file. Returns {} on any failure.

    Returns:
        dict[str, Any]: The parsed state dict, or ``{}`` if the file is
            missing, unreadable, or not a JSON object.
    """
    p = _state_path()
    if not p.is_file():
        legacy = legacy_state_file()
        if p != legacy and legacy.is_file() and state_file_safe_to_read(legacy):
            log.warning("multi_node state file %s missing; reading legacy %s", p, legacy)
            p = legacy
        else:
            return {}
    if not state_file_safe_to_read(p):
        log.warning("multi_node state file %s failed ownership/permission check", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("multi_node state file %s unreadable: %s", p, exc)
        return {}
    return data if isinstance(data, dict) else {}


def is_multi_node() -> bool:
    """True iff the optimizer is operating on a >=2-node RayJob cluster.

    State file wins over env so ``--resume`` works (manifest.json doesn't
    persist ``nodes``): session-scoped ``multi_node_state.json`` ``nodes`` >= 2
    wins; else fall back to ``$INFERENCE_OPTIMIZER_NODES``.

    Returns:
        True when operating on a >=2-node RayJob cluster, else False.
    """
    state = _read_state()
    try:
        state_n = int(state.get("nodes") or 0)
    except (TypeError, ValueError):
        state_n = 0
    if state_n >= 2:
        return True
    try:
        env_n = int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "1") or 1)
    except ValueError:
        return False
    return env_n >= 2


def ray_gcs_address_from_state() -> str:
    """Ray GCS address for ``ray.init`` (head pod IP + default GCS port).

    Returns:
        str: The explicit ``ray_address`` from state, ``<head_pod_ip>:6379``
            when only the head IP is known, or ``""`` if neither is present.
    """
    state = _read_state()
    addr = str(state.get("ray_address") or "").strip()
    if addr:
        return addr
    head = str(state.get("head_pod_ip") or "").strip()
    if head:
        return f"{head}:6379"
    return ""


def dynamo_ssh_env_from_state() -> dict[str, str]:
    """Env that routes kernel-agent GEAK GPU work to a Dynamo pod over SSH.

    Returns ``{KERNEL_AGENT_GPU_PLACEMENT=ssh, MN_SSH_HOST/PORT/KEY}`` ONLY when
    the multi_node backend is Dynamo and a GPU pod IP + ssh key are known.
    Returns ``{}`` for the RayJob backend and single-node so the Ray placement
    path (``ray_gcs_address_from_state`` / ``RAY_ADDRESS``) is left untouched —
    this is the isolation seam that keeps the SSH path Dynamo-only.

    Returns:
        A ``{KERNEL_AGENT_GPU_PLACEMENT, MN_SSH_HOST/PORT/KEY}`` mapping for the
        Dynamo backend when a GPU pod IP and ssh key are known, else ``{}``.
    """
    state = _read_state()
    if state.get("backend") != "dynamo":
        return {}
    if (state.get("pd_mode") or "").lower() == "disaggregated":
        gpu_ips = list(state.get("prefill_pod_ips") or []) + list(state.get("decode_pod_ips") or [])
    else:
        gpu_ips = list(state.get("worker_pod_ips") or [])
    key = str(state.get("ssh_key_path") or "").strip()
    if not gpu_ips or not key:
        return {}
    return {
        "KERNEL_AGENT_GPU_PLACEMENT": "ssh",
        "MN_SSH_HOST": str(gpu_ips[0]),
        "MN_SSH_PORT": str(state.get("ssh_port") or 2222),
        "MN_SSH_KEY": key,
    }


def rayjob_id_from_state() -> str:
    """Return the SaFE-allocated RayJob workload id, or ``""`` if absent.

    Reads the ``$MULTI_NODE_STATE_FILE`` checkpoint. Used to scope per-RayJob
    shared artefacts when ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` was not exported
    in-process.

    Returns:
        The SaFE-allocated RayJob workload id, or ``""`` if absent.
    """
    return str(_read_state().get("rayjob_id") or "").strip()


def export_ray_address_to_os() -> None:
    """Set ``RAY_ADDRESS`` from multi_node state when running multi-node optimize."""
    if not is_multi_node():
        return
    addr = ray_gcs_address_from_state()
    if addr:
        os.environ["RAY_ADDRESS"] = addr


def magpie_remote_env() -> dict[str, str]:
    """Return env vars to inject into a Magpie ``benchmark`` subprocess.

    Single-node: ``{}`` (Magpie's ``--run-mode local`` untouched). Multi-node
    with a ``service_url``: ``{"MAGPIE_RUN_PHASE": "client",
    "BENCHMARK_BASE_URL": "<service_url>"}`` so Magpie skips its local server
    launch and points ``benchmark_serving`` at the head pod. Multi-node without
    a state file: ``{}`` + WARN (the local-launch failure surfaces clearly).

    Returns:
        Env vars to inject into the Magpie subprocess, or ``{}`` for the
        single-node path or when no service URL is available.
    """
    if not is_multi_node():
        return {}

    state = _read_state()
    service_url = str(state.get("service_url") or "").strip()
    # Prefer head_pod_ip:port over ClusterIP (sandbox may not reach ClusterIP)
    head_ip = str(state.get("head_pod_ip") or "").strip()
    if head_ip and ".svc.cluster.local" in service_url:
        import re

        port = re.search(r":(\d+)$", service_url)
        port = port.group(1) if port else "8888"
        service_url = f"http://{head_ip}:{port}"
    if not service_url:
        log.warning(
            "INFERENCE_OPTIMIZER_NODES=%s but %s has no service_url; "
            "Magpie will try to launch a local server and likely fail. "
            "Run `python3 -m hyperloom.inference_optimizer.multi_node create-rayjob` "
            "before `inference_optimizer optimize` in multi-node mode.",
            os.environ.get("INFERENCE_OPTIMIZER_NODES"),
            _state_path(),
        )
        return {}

    return {
        "MAGPIE_RUN_PHASE": "client",
        "BENCHMARK_BASE_URL": service_url,
    }


def log_mn_banner(
    component: str,
    target_log: logging.Logger,
    **extra: Any,
) -> None:
    """Print a one-line ``[MN ...]`` banner when multi-node, no-op single-node.

    Lets an operator tailing the log tell single-pod from multi-node RayJob
    rounds. No-op (short-circuits via ``is_multi_node()``) when ``nodes < 2``.
    Multi-node prints ``[MN component=<name> nodes=N head=<ip>
    service_url=<url> key=value ...]``; ``**extra`` keys are appended for
    round-specific context (e.g. ``trace_dir=`` / ``variant=``).

    Args:
        component: Name of the component emitting the banner.
        target_log: Logger the banner line is written to.
        **extra: Round-specific key/value context appended to the banner;
            keys with empty or None values are skipped.
    """
    if not is_multi_node():
        return
    state = _read_state()
    try:
        nodes = int(state.get("nodes") or 0)
    except (TypeError, ValueError):
        nodes = 0
    if nodes < 2:
        try:
            nodes = int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "2") or 2)
        except ValueError:
            nodes = 2
    head = str(state.get("head_pod_ip") or "").strip()
    service_url = str(state.get("service_url") or "").strip()
    pairs = [
        f"component={component}",
        f"nodes={nodes}",
    ]
    if head:
        pairs.append(f"head={head}")
    if service_url:
        pairs.append(f"service_url={service_url}")
    for k, v in extra.items():
        if v is None or v == "":
            continue
        pairs.append(f"{k}={v}")
    target_log.info("[MN %s]", " ".join(pairs))


__all__ = [
    "export_ray_address_to_os",
    "is_multi_node",
    "log_mn_banner",
    "magpie_remote_env",
    "ray_gcs_address_from_state",
    "rayjob_id_from_state",
]
