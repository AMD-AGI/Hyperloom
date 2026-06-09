# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Helper that bridges the multi-node CLI state into Magpie subprocesses.

Lives in the executors package (called by ``baseline.py`` / ``_grid_runner.py``
before they launch Magpie) so the dependency edge stays one-way: executors
import this; ``multi_node/`` knows nothing about them.

Reads ``$INFERENCE_OPTIMIZER_NODES`` + ``/tmp/multi_node_state.json``. Single
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

log = logging.getLogger(__name__)

# Mirror of the multi_node/cli.py constant; duplicated (not imported) to avoid
# pulling httpx into the single-node import path.
_DEFAULT_STATE_PATH = "/tmp/multi_node_state.json"


def _state_path() -> Path:
    """Resolve where the multi_node CLI dropped its state file."""
    return Path(os.environ.get("MULTI_NODE_STATE_FILE", _DEFAULT_STATE_PATH))


def _read_state() -> dict[str, Any]:
    """Best-effort read of the state file. Returns {} on any failure."""
    p = _state_path()
    if not p.is_file():
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
    persist ``nodes``): ``/tmp/multi_node_state.json`` ``nodes`` >= 2 wins;
    else fall back to ``$INFERENCE_OPTIMIZER_NODES``.
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
    """Ray GCS address for ``ray.init`` (head pod IP + default GCS port)."""
    state = _read_state()
    addr = str(state.get("ray_address") or "").strip()
    if addr:
        return addr
    head = str(state.get("head_pod_ip") or "").strip()
    if head:
        return f"{head}:6379"
    return ""


def rayjob_id_from_state() -> str:
    """Return the SaFE-allocated RayJob workload id, or ``""`` if absent.

    Reads the ``$MULTI_NODE_STATE_FILE`` checkpoint. Used to scope per-RayJob
    shared artefacts when ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` was not exported
    in-process.
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
            "Run `python3 -m inference_optimizer.multi_node create-rayjob` "
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
