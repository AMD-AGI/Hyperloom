"""Pure helpers for the Infera multi-node backend."""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .server_args_safety import prepare_shell_safe_extra_args
from .ssh_client import DEFAULT_SSH_PORT

# Frontend HTTP port (SaFE common.InferaFrontendPort).
INFERA_FRONTEND_PORT = 8000

# SSH control plane: hostNetwork pods on the same node share one IP, so each GPU role binds a distinct MN_SSH_PORT
# (decode offset by ROLE_STRIDE).
INFERA_SSH_PORT_ROLE_STRIDE = 10


def ssh_role_port_offset(role: str) -> int:
    """Return the SSH port offset for a GPU service role."""
    if (role or "").lower() == "decode":
        return INFERA_SSH_PORT_ROLE_STRIDE
    return 0


def pod_targets_from_lists(
    pods: list[dict[str, Any]] | None,
    ips: list[str] | None,
    *,
    default_port: int,
    default_role: str = "worker",
) -> list[dict[str, Any]]:
    """Build SSH targets from rich pod dicts or legacy IP-only state."""
    if pods:
        return [dict(p) for p in pods if isinstance(p, dict) and p.get("podIP")]
    out: list[dict[str, Any]] = []
    for ip in ips or []:
        ip = str(ip or "").strip()
        if not ip:
            continue
        out.append(
            {
                "podId": "",
                "podIP": ip,
                "role": default_role,
                "lwsIndex": None,
                "sshPort": int(default_port),
            }
        )
    return out


def gpu_ssh_targets_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve every GPU pod SSH target from multi_node state."""
    base = int(state.get("ssh_port") or DEFAULT_SSH_PORT)
    if (state.get("pd_mode") or "").lower() == "disaggregated":
        return pod_targets_from_lists(
            state.get("prefill_pods"),
            state.get("prefill_pod_ips"),
            default_port=base,
            default_role="prefill",
        ) + pod_targets_from_lists(
            state.get("decode_pods"),
            state.get("decode_pod_ips"),
            default_port=base + INFERA_SSH_PORT_ROLE_STRIDE,
            default_role="decode",
        )
    return pod_targets_from_lists(
        state.get("worker_pods"),
        state.get("worker_pod_ips"),
        default_port=base,
        default_role="worker",
    )


# sglang PD bootstrap rendezvous port (SaFE common.InferaBootstrapPort).
INFERA_BOOTSTRAP_PORT = 30001


def disagg_flags(mode: str, kv_transfer_backend: str, *, bootstrap_port: int = INFERA_BOOTSTRAP_PORT) -> str:
    """sglang PD disaggregation flags for a prefill/decode group."""
    m = (mode or "").strip().lower()
    if m not in ("prefill", "decode"):
        return ""
    parts = [f"--disaggregation-mode {m}"]
    kv = (kv_transfer_backend or "").strip()
    if kv:
        parts.append(f"--disaggregation-transfer-backend {kv}")
    parts.append(f"--disaggregation-bootstrap-port {int(bootstrap_port)}")
    return " ".join(parts)


_INFERA_PID_FILE = str(Path(tempfile.gettempdir()) / "mn_infera_server.pid")
_INFERA_LOG_FILE = str(Path(tempfile.gettempdir()) / "mn_infera_server.log")
_INFERA_DIST_INIT_PORT = 5000


@dataclass(frozen=True)
class NodeLaunch:
    """One server group launch_infera_node.py starts on each pod; ``disagg_mode`` names its PD role, if any."""

    framework: str
    model: str
    tp: int
    nnodes: int
    ep: int = 1
    extra_args: str = ""
    disagg_mode: str = ""
    kv_transfer_backend: str = ""


def build_node_kill_args(framework: str) -> str:
    """Build the launch_infera_node.py argv string that only kills the server behind the session PID file."""
    parts = ["--framework", framework, "--kill-only", "--pid-file", _INFERA_PID_FILE]
    return " ".join(shlex.quote(x) for x in parts)


def build_node_launch_args(launch: NodeLaunch) -> str:
    """Build the argv string for launch_infera_node.py (shipped over SSH)."""
    parts = [
        "--framework",
        launch.framework,
        "--model",
        launch.model,
        "--tp",
        str(launch.tp),
        "--nnodes",
        str(launch.nnodes),
        "--dist-init-port",
        str(_INFERA_DIST_INIT_PORT),
        "--pid-file",
        _INFERA_PID_FILE,
        "--log-file",
        _INFERA_LOG_FILE,
        "--health-port",
        str(INFERA_FRONTEND_PORT),
        "--health-wait-sec",
        "0",
    ]
    ep = launch.ep
    if ep and int(ep) > 1:
        parts.extend(["--ep", str(ep)])
    quoted = " ".join(shlex.quote(x) for x in parts)
    # Fold the PD disaggregation flags into extra_args.
    merged_extra = (launch.extra_args or "").strip()
    df = disagg_flags(launch.disagg_mode, launch.kv_transfer_backend)
    if df:
        merged_extra = (merged_extra + " " + df).strip()
    if merged_extra:
        safe_extra = prepare_shell_safe_extra_args(
            merged_extra,
            context="build_node_launch_args",
        )
        quoted += " --extra-args " + shlex.quote(safe_extra)
    return quoted
