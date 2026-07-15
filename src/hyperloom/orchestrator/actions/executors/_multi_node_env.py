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

import json
import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.multi_node.state_paths import legacy_state_file, resolve_state_file, state_file_safe_to_read

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


def safe_available() -> bool:
    """True when SaFE is reachable (both SAFE_API_URL + SAFE_API_KEY set).

    SaFE presence is signaled by these env vars (see safe_client). When SaFE
    is available the optimizer uses the normal SaFE create flow; the
    ``HYPERLOOM_MN_EXT_*`` external bypass only engages when SaFE is absent.

    Returns:
        bool: Whether SaFE API credentials are both present.
    """
    return bool(os.environ.get("SAFE_API_URL", "").strip()) and bool(
        os.environ.get("SAFE_API_KEY", "").strip()
    )


def external_service_url() -> str:
    """Explicit external benchmark endpoint that bypasses SaFE provisioning.

    When ``$HYPERLOOM_MN_EXT_SERVICE_URL`` is a http(s) URL, the optimizer skips
    the SaFE rayjob/infera create flow entirely and drives benchmarks (and,
    when SSH is supplied, server restarts + GPU sampling) against an
    already-provisioned cluster described purely by env vars. Empty otherwise.

    Returns:
        str: The external service URL, or ``""`` when not in external mode.
    """
    if safe_available():
        return ""
    u = os.environ.get("HYPERLOOM_MN_EXT_SERVICE_URL", "").strip()
    return u if u.startswith(("http://", "https://")) else ""


def build_external_state_from_env() -> dict[str, Any]:
    """Synthesize the multi-node state.json purely from ``HYPERLOOM_MN_EXT_*``.

    Lets the optimizer run with SaFE assumed absent: every field SaFE would
    normally write (service_url, prefill/decode/worker pods, ssh key/port) is
    read from env instead, so all downstream code (restart, worker-ready gate,
    #3 GPU sampling, magpie_remote_env, pd_topology) works unchanged.

    Env vars:
        HYPERLOOM_MN_EXT_SERVICE_URL   frontend/benchmark URL (required)
        HYPERLOOM_MN_EXT_PREFILL_IPS   comma-separated prefill pod IPs
        HYPERLOOM_MN_EXT_DECODE_IPS    comma-separated decode pod IPs
        HYPERLOOM_MN_EXT_WORKER_IPS    comma-separated worker pod IPs (aggregated)
        HYPERLOOM_MN_EXT_SSH_KEY       SSH private key path (enables restart/sampling)
        HYPERLOOM_MN_EXT_SSH_PORT      SSH base port (default 2233; decode role-offset)
        HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS  known_hosts path (optional)
    Reuses INFERENCE_OPTIMIZER_NODES / INFERENCE_OPTIMIZER_GPUS_PER_NODE / PD_MODE.

    Returns:
        dict[str, Any]: A synthetic state dict, or ``{}`` when not in external
        mode (no ``HYPERLOOM_MN_EXT_SERVICE_URL``).
    """
    url = external_service_url()
    if not url:
        return {}

    def _ips(name: str) -> list[str]:
        return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]

    def _int_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, "") or default)
        except ValueError:
            return default

    ssh_key = os.environ.get("HYPERLOOM_MN_EXT_SSH_KEY", "").strip()
    ssh_port = _int_env("HYPERLOOM_MN_EXT_SSH_PORT", 2233)
    known_hosts = os.environ.get("HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS", "").strip()
    nodes = _int_env("INFERENCE_OPTIMIZER_NODES", 1)
    gpn = _int_env("INFERENCE_OPTIMIZER_GPUS_PER_NODE", 8)
    pd_mode = (os.environ.get("PD_MODE", "") or "aggregated").strip().lower()
    if pd_mode in ("colocated", "mixed"):
        pd_mode = "aggregated"

    # Per-role SSH port mirrors the infera backend (decode offset by role stride)
    # so restart-server's SSH fan-out targets the right port per pod.
    try:
        from hyperloom.inference_optimizer.multi_node._internal.infera_support import ssh_role_port_offset
    except Exception:  # noqa: BLE001
        def ssh_role_port_offset(role: str) -> int:  # fallback: decode +10
            return 10 if (role or "").lower() == "decode" else 0

    def _pods(ips: list[str], role: str) -> list[dict[str, Any]]:
        base = ssh_port + ssh_role_port_offset(role)
        return [
            {"podIP": ip, "podId": f"external-{role}-{i}", "role": role, "sshPort": base}
            for i, ip in enumerate(ips)
        ]

    prefill, decode, worker = (
        _ips("HYPERLOOM_MN_EXT_PREFILL_IPS"),
        _ips("HYPERLOOM_MN_EXT_DECODE_IPS"),
        _ips("HYPERLOOM_MN_EXT_WORKER_IPS"),
    )
    backend = (os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "") or "infera").strip().lower()
    # rayjob control plane: head IP is the single required field; the Ray GCS
    # address (head:6379) and Ray Dashboard (head:8265) both derive from it.
    head_ip = os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip()
    ray_address = f"{head_ip}:6379" if head_ip else ""
    ray_dash_token = os.environ.get("HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN", "").strip()
    # rayjob_id / workspace are SaFE constructs (no SaFE workload in external
    # mode). No dedicated env var; carry the standard SAFE_WORKSPACE (k8s
    # namespace) through when present, purely as a k8s-scope passthrough.
    workspace = os.environ.get("SAFE_WORKSPACE", "").strip()
    state: dict[str, Any] = {
        "backend": backend if backend in ("infera", "rayjob") else "infera",
        "external": True,
        "service_url": url,
        "nodes": nodes,
        "gpus_per_node": gpn,
        "pd_mode": pd_mode,
        "ssh_port": ssh_port,
        "prefill_pod_ips": prefill,
        "prefill_pods": _pods(prefill, "prefill"),
        "decode_pod_ips": decode,
        "decode_pods": _pods(decode, "decode"),
        "worker_pod_ips": worker,
        "worker_pods": _pods(worker, "worker"),
    }
    if ssh_key:
        state["ssh_key_path"] = ssh_key
    if known_hosts:
        state["ssh_known_hosts"] = known_hosts
    if head_ip:
        state["head_pod_ip"] = head_ip
        state["ray_address"] = ray_address
    if ray_dash_token:
        state["ray_dashboard_token"] = ray_dash_token
    if workspace:
        state["workspace"] = workspace
    return state


def external_has_ssh_control() -> bool:
    """True when external mode can SSH-manage servers (restart + GPU sampling).

    Requires ``HYPERLOOM_MN_EXT_SSH_KEY`` plus at least one prefill/decode/worker
    pod IP. When False (benchmark-only external), restart is a no-op and the
    optimizer just benchmarks the already-running server.

    Returns:
        bool: Whether external SSH server control is available.
    """
    if not external_service_url():
        return False
    if not os.environ.get("HYPERLOOM_MN_EXT_SSH_KEY", "").strip():
        return False
    return any(
        os.environ.get(k, "").strip()
        for k in ("HYPERLOOM_MN_EXT_PREFILL_IPS", "HYPERLOOM_MN_EXT_DECODE_IPS", "HYPERLOOM_MN_EXT_WORKER_IPS")
    )


def external_has_server_control() -> bool:
    """True when external mode can restart the server (backend-aware).

    infera: needs SSH (see :func:`external_has_ssh_control`). rayjob: needs
    a Ray address (via ``HYPERLOOM_MN_EXT_HEAD_IP``). When False, restart is
    a no-op and the optimizer benchmarks the already-running server as-is.

    Returns:
        bool: Whether external server restart control is available.
    """
    if not external_service_url():
        return False
    if external_has_ssh_control():
        return True
    return bool(os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip())


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
    # External mode: point benchmarks straight at the env-provided endpoint
    # (SaFE bypassed); works regardless of state / node count.
    ext = external_service_url()
    if ext:
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
