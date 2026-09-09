"""Cluster hand-off: synthesize multi-node state from HYPERLOOM_MN_EXT_* env vars."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..state_paths import resolve_state_file, state_file_safe_to_read
from .ssh_client import DEFAULT_SSH_PORT

log = logging.getLogger(__name__)

# Prefix shared by every session-bookkeeping key the CLI checkpoints (``last_restart_*``, ``last_kill_*``,
# ``last_server_*``).
_SESSION_KEY_PREFIX = "last_"

# Launcher-derived, cluster-specific keys that lack the ``last_`` prefix but must still survive a same-hand-off
# reload.
_HANDOFF_CARRIED_KEYS = (
    "pd_prefill_url",
    "pd_decode_url",
)

# The fields that identify *which* cluster a state describes.
_HANDOFF_IDENTITY_KEYS = (
    "service_url",
    "head_pod_ip",
    "prefill_pod_ips",
    "decode_pod_ips",
    "worker_pod_ips",
)

# Port assumed when a ClusterIP service URL carries none.
_DEFAULT_FRONTEND_PORT = "8888"

_MN_BACKENDS = ("infera", "rayjob")
# Mirrors the CLI's --mn-backend default (cli/multi_node.py::_resolve_mn_backend) so a hand-off and a fresh `optimize`
# never disagree about the same cluster.
_DEFAULT_MN_BACKEND = "rayjob"


def _handoff_backend(*, ssh_key: str, has_pod_ips: bool, head_ip: str) -> str:
    """Resolve which control plane a handed-over cluster speaks."""
    explicit = os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "").strip().lower()
    if explicit in _MN_BACKENDS:
        return explicit
    if ssh_key and has_pod_ips:
        return "infera"
    if head_ip:
        return "rayjob"
    return _DEFAULT_MN_BACKEND


def _handoff_pod_count(pd_mode: str, *, prefill: list[str], decode: list[str], worker: list[str]) -> int:
    """How many GPU pods this hand-off actually puts to work."""
    used = prefill + decode if pd_mode == "disaggregated" else worker
    return len(set(used))


def _handoff_nodes(pod_ip_count: int) -> int:
    """Resolve the run's node count from the only source allowed to state it."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_NODES", "").strip()
    if raw:
        try:
            stated = int(raw)
        except ValueError:
            log.warning(
                "external mode: INFERENCE_OPTIMIZER_NODES=%r is not an integer; treating the run "
                "as single-pod. Pass --nodes to state the cluster's size.",
                raw,
            )
        else:
            if pod_ip_count and stated != pod_ip_count:
                log.warning(
                    "external mode: INFERENCE_OPTIMIZER_NODES=%d disagrees with the %d GPU pod IPs "
                    "handed over; honouring the stated value",
                    stated,
                    pod_ip_count,
                )
            return max(1, stated)
    elif pod_ip_count > 1:
        log.warning(
            "external mode: the hand-off carries %d GPU pod IPs but INFERENCE_OPTIMIZER_NODES is "
            "unset, so this run takes the single-pod path and will start one detached server "
            "instead of one cluster. Pass --nodes %d.",
            pod_ip_count,
            pod_ip_count,
        )
    return 1


def external_service_url() -> str:
    """Return the handed-over cluster's benchmark URL, or empty when there is none."""
    u = os.environ.get("HYPERLOOM_MN_EXT_SERVICE_URL", "").strip()
    return u if u.startswith(("http://", "https://")) else ""


def build_external_state_from_env() -> dict[str, Any]:
    """Synthesize multi-node state purely from ``HYPERLOOM_MN_EXT_*`` env vars."""
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
    ssh_port = _int_env("HYPERLOOM_MN_EXT_SSH_PORT", DEFAULT_SSH_PORT)
    known_hosts = os.environ.get("HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS", "").strip()
    gpn = _int_env("INFERENCE_OPTIMIZER_GPUS_PER_NODE", 8)
    # From the --pd-mode flag, mirrored into the env by `optimize`.
    pd_mode = (os.environ.get("PD_MODE", "") or "aggregated").strip().lower()

    try:
        from .infera_support import ssh_role_port_offset
    except Exception:  # noqa: BLE001

        def ssh_role_port_offset(role: str) -> int:  # type: ignore[misc]
            return 10 if (role or "").lower() == "decode" else 0

    def _pods(ips: list[str], role: str) -> list[dict[str, Any]]:
        """Build SSH targets for one role, mirroring the pods' own port math."""
        base = ssh_port + ssh_role_port_offset(role)
        return [
            {
                "podIP": ip,
                "podId": f"external-{role}-{i}",
                "role": role,
                "lwsIndex": i,
                "sshPort": base + i,
            }
            for i, ip in enumerate(ips)
        ]

    prefill, decode, worker = (
        _ips("HYPERLOOM_MN_EXT_PREFILL_IPS"),
        _ips("HYPERLOOM_MN_EXT_DECODE_IPS"),
        _ips("HYPERLOOM_MN_EXT_WORKER_IPS"),
    )
    head_ip = os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip()
    backend = _handoff_backend(
        ssh_key=ssh_key,
        has_pod_ips=bool(prefill or decode or worker),
        head_ip=head_ip,
    )
    nodes = _handoff_nodes(_handoff_pod_count(pd_mode, prefill=prefill, decode=decode, worker=worker))
    ray_address = f"{head_ip}:6379" if head_ip else ""
    ray_dash_token = os.environ.get("HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN", "").strip()

    pn = _int_env("PD_PREFILL_NODES", 0) or len(prefill)
    dn = _int_env("PD_DECODE_NODES", 0) or len(decode)

    state: dict[str, Any] = {
        "backend": backend,
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
    if pd_mode == "disaggregated" and (pn > 0 or dn > 0):
        state["pd_prefill_nodes"] = pn
        state["pd_decode_nodes"] = dn
        state["last_restart_pd_prefill_nodes"] = pn
        state["last_restart_pd_decode_nodes"] = dn
        state["last_restart_pd_mode"] = "disaggregated"
    return state


def external_has_ssh_control() -> bool:
    """True when external env supplies SSH key plus at least one GPU pod IP."""
    if not external_service_url():
        return False
    if not os.environ.get("HYPERLOOM_MN_EXT_SSH_KEY", "").strip():
        return False
    return any(
        os.environ.get(k, "").strip()
        for k in (
            "HYPERLOOM_MN_EXT_PREFILL_IPS",
            "HYPERLOOM_MN_EXT_DECODE_IPS",
            "HYPERLOOM_MN_EXT_WORKER_IPS",
        )
    )


def external_has_server_control() -> bool:
    """True when external mode can restart servers (SSH for infera, head IP for rayjob)."""
    if not external_service_url():
        return False
    if external_has_ssh_control():
        return True
    return bool(os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip())


def _read_state_file(path: Path) -> dict[str, Any]:
    """Load a state dict from disk when the file is present and safe to read."""
    if not path.is_file():
        return {}
    if not state_file_safe_to_read(path):
        log.warning("multi_node state file %s failed ownership/permission check", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("multi_node state file %s unreadable: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def reachable_service_url(state: dict[str, Any]) -> str:
    """Return the cluster's frontend URL as this sandbox can actually reach it."""
    service_url = str(state.get("service_url") or "").strip()
    head_ip = str(state.get("head_pod_ip") or "").strip()
    if head_ip and ".svc.cluster.local" in service_url:
        matched = re.search(r":(\d+)$", service_url)
        return f"http://{head_ip}:{matched.group(1) if matched else _DEFAULT_FRONTEND_PORT}"
    return service_url


def _describes_same_handoff(disk: dict[str, Any], ext_state: dict[str, Any]) -> bool:
    """Whether an on-disk state was written against the cluster now handed over."""
    return all(disk.get(key) == ext_state.get(key) for key in _HANDOFF_IDENTITY_KEYS)


def load_multi_node_state() -> dict[str, Any]:
    """Load multi-node state; a cluster hand-off wins over on-disk state."""
    if external_service_url():
        ext_state = build_external_state_from_env()
        if ext_state:
            try:
                path = resolve_state_file()
            except RuntimeError:
                return ext_state
            disk = _read_state_file(path)
            if disk and not disk.get("external"):
                log.warning(
                    "external mode: HYPERLOOM_MN_EXT_* env overrides stale "
                    "on-disk state at %s (non-external backend=%r)",
                    path,
                    disk.get("backend"),
                )
            elif disk and not _describes_same_handoff(disk, ext_state):
                log.warning(
                    "external mode: on-disk state at %s describes a different cluster; dropping its %s* bookkeeping",
                    path,
                    _SESSION_KEY_PREFIX,
                )
            elif disk:
                for key, value in disk.items():
                    carried = key.startswith(_SESSION_KEY_PREFIX) or key in _HANDOFF_CARRIED_KEYS
                    if carried and key not in ext_state:
                        ext_state[key] = value
            return ext_state

    try:
        path = resolve_state_file()
    except RuntimeError:
        return {}
    return _read_state_file(path)
