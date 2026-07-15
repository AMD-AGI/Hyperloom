# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Helper that bridges the multi-node CLI state into Magpie subprocesses.

Lives in the executors package (the dependency edge stays one-way: executors
import this; ``multi_node/`` knows nothing about them).

Reads ``$INFERENCE_OPTIMIZER_NODES`` + ``$MULTI_NODE_STATE_FILE``. Single node
(< 2): returns ``{}``. Multi-node (>= 2) with a ``service_url``: returns
``MAGPIE_RUN_PHASE=client`` + ``BENCHMARK_BASE_URL=<service_url>`` so Magpie
skips its own server launch and points ``benchmark_serving`` at the head pod.
:func:`export_ray_address_to_os` also copies ``ray_address`` into
``RAY_ADDRESS`` for kernel-agent ``ray.init``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.multi_node._internal.external_state import (
    external_service_url,
    load_multi_node_state,
)
from hyperloom.inference_optimizer.multi_node.state_paths import resolve_state_file

log = logging.getLogger(__name__)


def _state_path() -> Path:
    """Resolve where the multi_node CLI dropped its state file.

    Returns:
        Path: The state-file path from :func:`resolve_state_file`.
    """
    return resolve_state_file()


def _read_state() -> dict[str, Any]:
    """Best-effort read of multi-node state (file, else external env synthesis).

    Returns:
        dict[str, Any]: The effective state dict for this process.
    """
    return load_multi_node_state()


def mn_bench_warmup_enabled() -> bool:
    """Whether multi-node runs a discarded client warmup pass before measuring.

    Multi-node has no local server_lifecycle to reuse, so the single-node
    warmup-before-measure path is ineligible. Instead callers run one extra
    (discarded) Magpie client pass against the already-restarted, persistent
    remote server to warm JIT / steady-state (esp. important now that PD legs
    skip the server-side warmup). Default ON; disable via
    ``INFERENCE_OPTIMIZER_MN_BENCH_WARMUP=0``.

    Returns:
        bool: True when the multi-node client warmup pass is enabled.
    """
    raw = os.environ.get("INFERENCE_OPTIMIZER_MN_BENCH_WARMUP", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def is_multi_node() -> bool:
    """True iff the optimizer is operating on a >=2-node RayJob cluster.

    State file wins over env so ``--resume`` works: state ``nodes`` >= 2 wins,
    else fall back to ``$INFERENCE_OPTIMIZER_NODES``.

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


def resolve_kb_topology() -> dict[str, int]:
    """Resolve ``(nodes, gpus_per_node)`` for the KB hardware topology suffix.

    Mirrors :func:`is_multi_node`'s source priority so the recipe KB key stays
    stable across ``--resume``: the ``multi_node_state.json`` values win
    (persisted), then the ``INFERENCE_OPTIMIZER_NODES`` /
    ``INFERENCE_OPTIMIZER_GPUS_PER_NODE`` env fallbacks. The CLI exports both
    before the T0 anchor, so a fresh run (where the state file is not written
    until provision, which runs after T0) resolves the same world_size at T0
    and at CLOSE — read and write keys never diverge.

    Returns:
        A ``{"nodes", "gpus_per_node", "pd_mode", "pd_prefill_nodes",
        "pd_decode_nodes"}`` mapping ready to splat into
        :func:`kb_hardware_slug`. Single-node returns ``nodes=1`` so the KB key
        is left unchanged (the PD fields are then ignored downstream).
    """
    state = _read_state()
    try:
        nodes = int(state.get("nodes") or 0)
    except (TypeError, ValueError):
        nodes = 0
    if nodes < 2:
        try:
            nodes = int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "1") or 1)
        except ValueError:
            nodes = 1
    try:
        gpn = int(state.get("gpus_per_node") or 0)
    except (TypeError, ValueError):
        gpn = 0
    if gpn <= 0:
        try:
            gpn = int(os.environ.get("INFERENCE_OPTIMIZER_GPUS_PER_NODE", "8") or 8)
        except ValueError:
            gpn = 8

    # PD topology: env (PD_MODE / PD_PREFILL_NODES / PD_DECODE_NODES) is exported
    # before T0 and stable across the run, so it wins; state fields (persisted at
    # create / restart) are the resume fallback. Values come from the same CLI
    # flags, so env vs state never disagree — only availability differs by phase.
    pd_mode = (os.environ.get("PD_MODE", "") or "").strip().lower()
    if not pd_mode:
        pd_mode = str(state.get("pd_mode") or state.get("last_restart_pd_mode") or "aggregated").strip().lower()

    def _pd_nodes(env_key: str, *state_keys: str) -> int:
        raw = (os.environ.get(env_key, "") or "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        for sk in state_keys:
            try:
                v = int(state.get(sk) or 0)
            except (TypeError, ValueError):
                v = 0
            if v:
                return v
        return 0

    pn = _pd_nodes("PD_PREFILL_NODES", "pd_prefill_nodes", "last_restart_pd_prefill_nodes")
    dn = _pd_nodes("PD_DECODE_NODES", "pd_decode_nodes", "last_restart_pd_decode_nodes")

    return {
        "nodes": max(1, nodes),
        "gpus_per_node": max(1, gpn),
        "pd_mode": pd_mode or "aggregated",
        "pd_prefill_nodes": pn,
        "pd_decode_nodes": dn,
    }


def pd_topology_from_state() -> dict[str, Any]:
    """PD-disaggregation topology from multi-node state (empty unless disaggregated).

    Reads the ``pd_mode`` / ``last_restart_pd_*`` / prefill+decode pod fields the
    CLI persisted into ``multi_node_state.json``. Returns ``{}`` when not
    multi-node or when ``pd_mode != 'disaggregated'`` so every caller no-ops on
    the single-node / colocated paths.

    Returns:
        dict[str, Any]: ``{mode, prefill_nodes, decode_nodes, prefill_tp,
        decode_tp, prefill_ep, decode_ep, transfer_backend, prefill_pod_ips,
        decode_pod_ips}`` when disaggregated, else ``{}``.
    """
    if not is_multi_node():
        return {}
    st = _read_state()
    mode = str(st.get("pd_mode") or st.get("last_restart_pd_mode") or "").strip().lower()
    if mode != "disaggregated":
        return {}

    def _iv(*keys: str) -> int:
        """First state value (by key) coercible to int, else 0.

        Args:
            *keys (str): Candidate state keys, tried in order.

        Returns:
            int: The parsed value, or 0 when none parse.
        """
        for k in keys:
            v = st.get(k)
            try:
                if v is not None:
                    return int(v)
            except (TypeError, ValueError):
                continue
        return 0

    def _ips(key: str) -> list[str]:
        """String list from a state field, or ``[]`` when absent/not a list.

        Args:
            key (str): The state key holding a list of pod IPs.

        Returns:
            list[str]: The pod IPs as strings.
        """
        v = st.get(key)
        return [str(x) for x in v] if isinstance(v, list) else []

    return {
        "mode": "disaggregated",
        "prefill_nodes": _iv("last_restart_pd_prefill_nodes"),
        "decode_nodes": _iv("last_restart_pd_decode_nodes"),
        "prefill_tp": _iv("last_restart_pd_prefill_tp"),
        "decode_tp": _iv("last_restart_pd_decode_tp"),
        "prefill_ep": _iv("last_restart_pd_prefill_ep"),
        "decode_ep": _iv("last_restart_pd_decode_ep"),
        "transfer_backend": str(
            st.get("last_restart_pd_transfer_backend")
            or st.get("pd_transfer_backend")
            or ""
        ).strip(),
        "prefill_pod_ips": _ips("prefill_pod_ips"),
        "decode_pod_ips": _ips("decode_pod_ips"),
    }


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


def infera_ssh_env_from_state() -> dict[str, str]:
    """Env that routes kernel-agent GEAK GPU work to a Infera pod over SSH.

    Returns ``{KERNEL_AGENT_GPU_PLACEMENT=ssh, MN_SSH_HOST/PORT/KEY}`` ONLY when
    the multi_node backend is Infera and a GPU pod IP + ssh key are known.
    Returns ``{}`` for the RayJob backend and single-node so the Ray placement
    path (``ray_gcs_address_from_state`` / ``RAY_ADDRESS``) is left untouched —
    this is the isolation seam that keeps the SSH path Infera-only.

    Returns:
        A ``{KERNEL_AGENT_GPU_PLACEMENT, MN_SSH_HOST/PORT/KEY}`` mapping for the
        Infera backend when a GPU pod IP and ssh key are known, else ``{}``.
    """
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    state = _read_state()
    if state.get("backend") != "infera":
        return {}
    targets = infera_support.gpu_ssh_targets_from_state(state)
    key = str(state.get("ssh_key_path") or "").strip()
    if not targets or not key:
        return {}
    first = targets[0]
    return {
        "KERNEL_AGENT_GPU_PLACEMENT": "ssh",
        "MN_SSH_HOST": str(first.get("podIP") or ""),
        "MN_SSH_PORT": str(first.get("sshPort") or state.get("ssh_port") or 2233),
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
    # External mode: point benchmarks at the env-provided endpoint when multi-node.
    ext = external_service_url()
    if ext and is_multi_node():
        return {"MAGPIE_RUN_PHASE": "client", "BENCHMARK_BASE_URL": ext}
    if not is_multi_node():
        return {}

    state = _read_state()
    service_url = str(state.get("service_url") or "").strip()
    # Prefer head_pod_ip:port over ClusterIP (sandbox may not reach ClusterIP).
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
            "before `python -m hyperloom.inference_optimizer.cli optimize` in multi-node mode.",
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

    Prints ``[MN component=<name> nodes=N head=<ip> service_url=<url> key=value
    ...]``; ``**extra`` keys are appended for round-specific context.

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
    "resolve_kb_topology",
]
