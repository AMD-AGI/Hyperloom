"""Pure helpers for the Infera multi-node backend."""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
from typing import Any

from .server_args_safety import prepare_shell_safe_extra_args
from .ssh_client import DEFAULT_SSH_PORT

# Frontend HTTP port (SaFE common.InferaFrontendPort).
INFERA_FRONTEND_PORT = 8000

# SSH control plane: hostNetwork pods on the same node share one IP, so each GPU role binds a distinct MN_SSH_PORT
# (decode offset by ROLE_STRIDE).
INFERA_SSH_PORT_ROLE_STRIDE = 10
_INFERA_IDLE_SCRIPT = "/usr/local/bin/mn-idle.sh"


def ssh_role_port_offset(role: str) -> int:
    """Return the SSH port offset for a GPU service role."""
    if (role or "").lower() == "decode":
        return INFERA_SSH_PORT_ROLE_STRIDE
    return 0


def ssh_port_for_pod(
    role: str,
    lws_index: int | None,
    *,
    ssh_port_base: int = DEFAULT_SSH_PORT,
) -> int:
    """Compute the sshd port a pod listens on."""
    idx = lws_index if isinstance(lws_index, int) else 0
    return int(ssh_port_base) + ssh_role_port_offset(role) + idx


def idle_worker_entrypoint(*, role: str, ssh_port_base: int = DEFAULT_SSH_PORT) -> str:
    """Build the idle worker entryPoint with a role-scoped ``MN_SSH_PORT``."""
    role_base = int(ssh_port_base) + ssh_role_port_offset(role)
    return f"export MN_SSH_PORT=$(( {role_base} + ${{LWS_WORKER_INDEX:-0}} )); exec {_INFERA_IDLE_SCRIPT}"


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


def build_node_launch_args(
    *,
    framework: str,
    model: str,
    tp: int,
    nnodes: int,
    ep: int = 1,
    dist_init_port: int = 5000,
    pid_file: str = str(Path(tempfile.gettempdir()) / "mn_infera_server.pid"),
    log_file: str = str(Path(tempfile.gettempdir()) / "mn_infera_server.log"),
    extra_args: str = "",
    health_port: int = INFERA_FRONTEND_PORT,
    health_wait_sec: int = 0,
    kill_only: bool = False,
    disagg_mode: str = "",
    kv_transfer_backend: str = "",
) -> str:
    """Build the argv string for launch_infera_node.py (shipped over SSH)."""
    parts = ["--framework", framework]
    if kill_only:
        parts.append("--kill-only")
        # kill-only still needs framework and the pid-file path.
        parts.extend(["--pid-file", pid_file])
        return " ".join(shlex.quote(x) for x in parts)
    parts.extend(
        [
            "--model",
            model,
            "--tp",
            str(tp),
            "--nnodes",
            str(nnodes),
            "--dist-init-port",
            str(dist_init_port),
            "--pid-file",
            pid_file,
            "--log-file",
            log_file,
            "--health-port",
            str(health_port),
            "--health-wait-sec",
            str(health_wait_sec),
        ]
    )
    if ep and int(ep) > 1:
        parts.extend(["--ep", str(ep)])
    quoted = " ".join(shlex.quote(x) for x in parts)
    # Fold the PD disaggregation flags into extra_args.
    merged_extra = (extra_args or "").strip()
    df = disagg_flags(disagg_mode, kv_transfer_backend)
    if df:
        merged_extra = (merged_extra + " " + df).strip()
    if merged_extra:
        safe_extra = prepare_shell_safe_extra_args(
            merged_extra,
            context="build_node_launch_args",
        )
        quoted += " --extra-args " + shlex.quote(safe_extra)
    return quoted
