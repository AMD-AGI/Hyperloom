"""Cluster hand-off: synthesize multi-node state from HYPERLOOM_MN_EXT_* env vars."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..state_paths import resolve_state_file, state_file_safe_to_read
from .ssh_client import DEFAULT_SSH_PORT

log = logging.getLogger(__name__)

# Prefix shared by every session-bookkeeping key the CLI checkpoints
# (``last_restart_*``, ``last_kill_*``, ``last_server_*``). Matched by prefix
# rather than an explicit allowlist so a newly checkpointed field is carried
# across a hand-off reload without having to be registered here.
_SESSION_KEY_PREFIX = "last_"

_MN_BACKENDS = ("infera", "rayjob")
# Mirrors the CLI's --mn-backend default (cli/multi_node.py::_resolve_mn_backend)
# so a hand-off and a fresh `optimize` never disagree about the same cluster.
_DEFAULT_MN_BACKEND = "rayjob"


def _handoff_backend(*, ssh_key: str, has_pod_ips: bool, head_ip: str) -> str:
    """Resolve which control plane a handed-over cluster speaks.

    ``state["backend"]`` is a routing switch: every ``hyperloom-mn`` subcommand
    sends ``"infera"`` down the SSH fan-out and everything else to the Ray
    Dashboard. An explicit ``$INFERENCE_OPTIMIZER_MN_BACKEND`` therefore wins,
    but the platform can export the ``HYPERLOOM_MN_EXT_*`` block without that
    companion var, so the shape of the hand-off decides next: infera hands over
    an SSH key plus pod IPs, rayjob hands over the Ray head IP. SSH outranks the
    head IP, matching :func:`external_has_server_control`. A hand-off with
    neither is benchmark-only (restarts are skipped), so the routing is moot and
    the CLI default applies.

    Args:
        ssh_key: ``HYPERLOOM_MN_EXT_SSH_KEY`` value.
        has_pod_ips: Whether any prefill/decode/worker IP list is non-empty.
        head_ip: ``HYPERLOOM_MN_EXT_HEAD_IP`` value.

    Returns:
        str: ``"infera"`` or ``"rayjob"``.
    """
    explicit = os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "").strip().lower()
    if explicit in _MN_BACKENDS:
        return explicit
    if ssh_key and has_pod_ips:
        return "infera"
    if head_ip:
        return "rayjob"
    return _DEFAULT_MN_BACKEND


def external_service_url() -> str:
    """Return the handed-over cluster's benchmark URL, or empty when there is none.

    The hand-off is the ONLY signal for external mode. It is deliberately not
    gated on ``SAFE_API_*``: those credentials authenticate the LLM gateway and
    are present in essentially every platform sandbox, so gating on them made a
    handed-over cluster invisible exactly where the integration runs.

    Returns:
        str: ``HYPERLOOM_MN_EXT_SERVICE_URL`` when it is an http(s) URL; ``""``
        otherwise.
    """
    u = os.environ.get("HYPERLOOM_MN_EXT_SERVICE_URL", "").strip()
    return u if u.startswith(("http://", "https://")) else ""


def build_external_state_from_env() -> dict[str, Any]:
    """Synthesize multi-node state purely from ``HYPERLOOM_MN_EXT_*`` env vars.

    Returns:
        dict[str, Any]: Synthetic state in the same shape the infera / rayjob
        commands read, or ``{}`` when no cluster was handed over.
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
    ssh_port = _int_env("HYPERLOOM_MN_EXT_SSH_PORT", DEFAULT_SSH_PORT)
    known_hosts = os.environ.get("HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS", "").strip()
    nodes = _int_env("INFERENCE_OPTIMIZER_NODES", 1)
    gpn = _int_env("INFERENCE_OPTIMIZER_GPUS_PER_NODE", 8)
    pd_mode = (os.environ.get("PD_MODE", "") or "aggregated").strip().lower()

    try:
        from .infera_support import ssh_role_port_offset
    except Exception:  # noqa: BLE001

        def ssh_role_port_offset(role: str) -> int:  # type: ignore[misc]
            return 10 if (role or "").lower() == "decode" else 0

    def _pods(ips: list[str], role: str) -> list[dict[str, Any]]:
        """Build SSH targets for one role, mirroring the pods' own port math.

        A pod's sshd listens on ``role_base + LWS_WORKER_INDEX`` (see
        ``infera_support.idle_worker_entrypoint`` / ``ssh_port_for_pod``), so the
        pods of a multi-node role occupy consecutive ports. The platform lists a
        role's IPs in LWS ordinal order (leader first), hence the list index is
        the ordinal. Dropping it would leave every non-leader unreachable as soon
        as a role spans nodes.
        """
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
    """Load multi-node state; a cluster hand-off wins over on-disk state.

    Without a hand-off: the resolved ``$MULTI_NODE_STATE_FILE`` / session
    ``runtime/multi_node_state.json`` is the only source, else ``{}``.

    With one (``HYPERLOOM_MN_EXT_SERVICE_URL`` set): ``HYPERLOOM_MN_EXT_*`` env
    synthesis wins over any on-disk state, so a leftover state file cannot
    shadow the updated pod IPs / service URL of the cluster actually handed
    over (e.g. a manual ``restart-server`` without re-running ``optimize``).

    Env ownership stops at those connection/topology fields. The ``last_*``
    bookkeeping that ``restart-server`` checkpoints (submission ids, served
    config, pid/log dirs) is carried over from an on-disk state that describes
    the same hand-off: env synthesis never produces it, so dropping it would
    leave it written every round and read never, costing the resume fast path a
    full cold start on each retry.

    What the two backends do with a carried-over entry differs. Infera re-checks
    liveness for real, SSHing every GPU pod and signalling the recorded PID.
    RayJob only reads the launch job's status, and that job is the fan-out
    driver: it reaches ``SUCCEEDED`` once the ranks are spawned and stays there
    while the detached servers live or die, so it is evidence that a launch
    happened, never that a server is up. Kill therefore drops the launch id
    itself (see ``cli._record_kill_and_invalidate_launch``) rather than relying
    on the probe to notice.

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
            if disk and not disk.get("external"):
                log.warning(
                    "external mode: HYPERLOOM_MN_EXT_* env overrides stale "
                    "on-disk state at %s (non-external backend=%r)",
                    path,
                    disk.get("backend"),
                )
            elif disk:
                for key, value in disk.items():
                    if key.startswith(_SESSION_KEY_PREFIX) and key not in ext_state:
                        ext_state[key] = value
            return ext_state

    try:
        path = resolve_state_file()
    except RuntimeError:
        return {}
    return _read_state_file(path)
