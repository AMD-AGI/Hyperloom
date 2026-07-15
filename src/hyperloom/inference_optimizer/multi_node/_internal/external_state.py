"""SaFE-less external multi-node mode: synthesize state from HYPERLOOM_MN_EXT_* env vars."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..state_paths import legacy_state_file, resolve_state_file, state_file_safe_to_read

log = logging.getLogger(__name__)


def safe_available() -> bool:
    """True when SaFE API credentials are both present in the environment.

    Returns:
        bool: Whether the normal SaFE create flow should be used.
    """
    return bool(os.environ.get("SAFE_API_URL", "").strip()) and bool(
        os.environ.get("SAFE_API_KEY", "").strip()
    )


def external_service_url() -> str:
    """Return the external benchmark URL when SaFE is absent, else empty.

    Returns:
        str: ``HYPERLOOM_MN_EXT_SERVICE_URL`` when SaFE is unavailable and the
        value is an http(s) URL; ``""`` otherwise.
    """
    if safe_available():
        return ""
    u = os.environ.get("HYPERLOOM_MN_EXT_SERVICE_URL", "").strip()
    return u if u.startswith(("http://", "https://")) else ""


def build_external_state_from_env() -> dict[str, Any]:
    """Synthesize multi-node state purely from ``HYPERLOOM_MN_EXT_*`` env vars.

    Returns:
        dict[str, Any]: Synthetic state mirroring SaFE create-infera/rayjob
        output, or ``{}`` when not in external mode.
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

    try:
        from .infera_support import ssh_role_port_offset
    except Exception:  # noqa: BLE001
        def ssh_role_port_offset(role: str) -> int:  # type: ignore[misc]
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
    head_ip = os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip()
    ray_address = f"{head_ip}:6379" if head_ip else ""
    ray_dash_token = os.environ.get("HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN", "").strip()
    workspace = os.environ.get("SAFE_WORKSPACE", "").strip()

    pn = _int_env("PD_PREFILL_NODES", 0) or len(prefill)
    dn = _int_env("PD_DECODE_NODES", 0) or len(decode)

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
    if pd_mode == "disaggregated" and (pn > 0 or dn > 0):
        state["pd_prefill_nodes"] = pn
        state["pd_decode_nodes"] = dn
        state["last_restart_pd_prefill_nodes"] = pn
        state["last_restart_pd_decode_nodes"] = dn
        state["last_restart_pd_mode"] = "disaggregated"
    return state


def external_has_ssh_control() -> bool:
    """True when external env supplies SSH key plus at least one GPU pod IP.

    Returns:
        bool: Whether infera-style SSH server control is available.
    """
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
    """True when external mode can restart servers (SSH for infera, head IP for rayjob).

    Returns:
        bool: Whether per-round server restart is possible in external mode.
    """
    if not external_service_url():
        return False
    if external_has_ssh_control():
        return True
    return bool(os.environ.get("HYPERLOOM_MN_EXT_HEAD_IP", "").strip())


def _read_state_file(path: Path) -> dict[str, Any]:
    """Load a state dict from disk when the file is present and safe to read.

    Args:
        path: Resolved multi-node state file path.

    Returns:
        dict[str, Any]: Parsed state, or ``{}`` on any failure.
    """
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


def load_multi_node_state() -> dict[str, Any]:
    """Load multi-node state; external env wins over on-disk when SaFE is absent.

    Normal (SaFE) flow: session ``multi_node_state.json`` wins, with legacy
    ``/tmp/multi_node_state.json`` fallback, then ``{}``.

    External flow (``HYPERLOOM_MN_EXT_SERVICE_URL`` set, SaFE creds absent):
    ``HYPERLOOM_MN_EXT_*`` env synthesis wins over any on-disk state so a
    leftover SaFE state file cannot shadow updated pod IPs / service URL when
    the operator switches to external mode mid-session (e.g. manual
    ``restart-server`` without re-running ``optimize``).

    Returns:
        dict[str, Any]: The effective multi-node state for this process.
    """
    if external_service_url():
        ext_state = build_external_state_from_env()
        if ext_state:
            try:
                path = resolve_state_file()
            except RuntimeError:
                return ext_state
            disk = _read_state_file(path)
            if not disk:
                legacy = legacy_state_file()
                if path != legacy:
                    disk = _read_state_file(legacy)
            if disk and not disk.get("external"):
                log.warning(
                    "external mode: HYPERLOOM_MN_EXT_* env overrides stale "
                    "on-disk state at %s (non-external backend=%r)",
                    path,
                    disk.get("backend"),
                )
            return ext_state

    try:
        path = resolve_state_file()
    except RuntimeError:
        return {}
    data = _read_state_file(path)
    if not data:
        legacy = legacy_state_file()
        if path != legacy:
            data = _read_state_file(legacy)
            if data:
                log.warning("multi_node state file %s missing; using legacy %s", path, legacy)
    return data
