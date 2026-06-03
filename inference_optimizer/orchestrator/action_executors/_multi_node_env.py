"""Helper that bridges the multi-node CLI state into Magpie subprocesses.

Why this lives here (orchestrator/action_executors/), not in
``inference_optimizer/multi_node/``:

* ``multi_node/`` is the agent-facing CLI (subcommands the LLM types).
* This module is the executor-side glue called by ``baseline.py`` and
  ``_grid_runner.py`` right before they ``subprocess.run(magpie ...)``.
  Putting it next to those callers keeps the dependency edge one-way
  (executors ``import _multi_node_env``; ``multi_node/`` knows nothing
  about the executors).

What the bridge does:

1. Read ``$INFERENCE_OPTIMIZER_NODES`` (set by ``cli._run_optimize``).
2. If ``< 2``: return ``{}`` — single-pod path is preserved bit-for-bit.
3. If ``>= 2``: read ``/tmp/multi_node_state.json`` (written by
   ``inference_optimizer.multi_node create-rayjob`` or by
   ``inference_optimizer optimize --nodes N`` provisioning). If present and
   ``service_url`` is set, return the two env vars Magpie's
   ``scripts/benchmark/{sglang,vllm}_mi*x.sh`` honour::

       MAGPIE_RUN_PHASE   = "client"
       BENCHMARK_BASE_URL = state["service_url"]

   This makes Magpie skip its own server launch and just point
   ``benchmark_serving --base-url ...`` at the head pod's ClusterIP.

4. The same state file may carry ``ray_address`` (``<head_ip>:6379``) so
   kernel-agent can ``ray.init`` into the RayJob cluster from the CPU
   sandbox. :func:`export_ray_address_to_os` copies it into
   ``os.environ["RAY_ADDRESS"]`` for the optimizer process and any
   subprocess that inherits its environment.
5. If ``>= 2`` but state file is missing: log a single WARN and return
   ``{}``. The executor will then try to launch a local server (which
   will fail in the no-GPU sandbox) and surface a clear failure — which
   is preferable to silently benchmarking an unrelated localhost URL.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Mirror the constant baked into multi_node/cli.py. Keeping it duplicated
# (rather than importing from multi_node) avoids pulling httpx into the
# single-node import path; multi_node/_internal/safe_client imports httpx
# at module load time.
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

    Resolution order (state file wins over env so ``--resume`` works):

    1. ``/tmp/multi_node_state.json`` ``nodes`` field (>= 2 wins
       immediately). This file is written by
       ``inference_optimizer.multi_node create-rayjob`` and persists
       across sandbox shells / orchestrator restarts. Without this
       probe, a ``--resume``-launched optimizer would fall through to
       ``$INFERENCE_OPTIMIZER_NODES`` -- which argparse defaults to 1
       because manifest.json (schema v1) does not persist ``nodes`` --
       making ``magpie_remote_env`` return ``{}`` and silently shipping
       every Magpie subprocess into PHASE=all + local sglang launch
       (ADDENDUM-14 in the multi_node SKILL).
    2. ``$INFERENCE_OPTIMIZER_NODES`` env (set by ``cli._run_optimize``
       when ``--nodes`` is passed on the CLI). Honoured when the state
       file is absent or has no usable ``nodes``.
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


def dynamo_ssh_env_from_state() -> dict[str, str]:
    """Env that routes kernel-agent GEAK GPU work to a Dynamo pod over SSH.

    Returns ``{KERNEL_AGENT_GPU_PLACEMENT=ssh, MN_SSH_HOST/PORT/KEY}`` ONLY when
    the multi_node backend is Dynamo and a GPU pod IP + ssh key are known.
    Returns ``{}`` for the RayJob backend and single-node so the Ray placement
    path (``ray_gcs_address_from_state`` / ``RAY_ADDRESS``) is left untouched —
    this is the isolation seam that keeps the SSH path Dynamo-only.
    """
    state = _read_state()
    if state.get("backend") != "dynamo":
        return {}
    if (state.get("pd_mode") or "").lower() == "disaggregated":
        gpu_ips = (
            list(state.get("prefill_pod_ips") or [])
            + list(state.get("decode_pod_ips") or [])
        )
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

    Reads from the same ``$MULTI_NODE_STATE_FILE`` checkpoint that
    ``inference_optimizer.multi_node create-rayjob`` writes after SaFE
    returns. Used by call sites that need to scope per-RayJob shared
    artefacts (e.g. profile-trace dir) when
    ``$HYPERLOOM_MN_PROFILE_TRACE_DIR`` was not exported in-process —
    typically when the optimizer was launched out of band from
    ``cli._provision_multi_node_rayjob_stack``.
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

    Single-node (``--nodes`` 1, default): returns ``{}`` so the caller's
    ``env`` dict is untouched and Magpie's ``--run-mode local`` path
    behaves exactly as before this module existed.

    Multi-node (``--nodes >= 2``): if ``/tmp/multi_node_state.json`` has
    a non-empty ``service_url``, returns::

        {"MAGPIE_RUN_PHASE": "client",
         "BENCHMARK_BASE_URL": "<service_url>"}

    The Magpie shell scripts (``Magpie/scripts/benchmark/{sglang,vllm}_mi*x.sh``)
    detect ``BENCHMARK_BASE_URL`` and force PHASE=client, skipping the
    local server launch and pointing ``benchmark_serving --base-url`` at
    the multi_node RayJob's head pod ClusterIP.

    Multi-node WITHOUT state file: returns ``{}`` and emits a WARN log.
    The downstream Magpie call will then try to start a local server,
    fail in the no-GPU sandbox, and surface a clear ``magpie_nonzero``
    error to the agent — which is preferable to silently benchmarking
    against an unrelated local URL.
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

    Why this helper exists: executors (baseline/profile/grid_runner) and
    server lifecycle code all log a generic "launching X" line that does
    NOT tell the operator whether the run is single-pod or multi-node
    RayJob. Without this signal an operator tailing the optimizer log
    cannot tell which code path the round is taking — especially
    important when triaging restart loops where multi-node restart
    failures look indistinguishable from single-pod magpie failures.

    Single-node path is preserved bit-for-bit: the helper short-circuits
    via ``is_multi_node()`` before touching the logger, so callers get
    zero added output when ``nodes < 2``.

    Multi-node path: prints
    ``[MN component=<name> nodes=N head=<ip> service_url=<url> key=value ...]``
    The ``head_pod_ip`` and ``service_url`` come from
    ``/tmp/multi_node_state.json`` (best-effort; both default to empty
    string when the state file is missing or partial). ``**extra``
    keys are appended in insertion order so callers can surface
    round-specific context (e.g. ``trace_dir=...`` for profile rounds,
    ``variant=...`` for grid rows) without each call site having to
    format the banner itself.
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
