"""Pure helpers for the Infera multi-node backend.

No I/O / no env reads: the CLI feeds in the SaFE GetWorkloadResponse / service
info and these functions extract worker pod IPs, the frontend service URL, and
the pod-side launcher argv. Kept pure so the SSH fan-out logic stays
unit-testable without a live cluster.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path
from typing import Any

# Frontend HTTP port (SaFE common.InferaFrontendPort). Benchmarks target this
# OpenAI-compatible endpoint, never sglang rank-0 :8888.
INFERA_FRONTEND_PORT = 8000

# SSH control plane: hostNetwork pods on the same node share one IP, so each
# GPU role binds a distinct MN_SSH_PORT (decode offset by ROLE_STRIDE). Within
# a role, LWS ordinals add to the role base via LWS_WORKER_INDEX pod-side.
INFERA_SSH_PORT_ROLE_STRIDE = 10
_INFERA_IDLE_SCRIPT = "/usr/local/bin/mn-idle.sh"

# Substrings that mark a pod as the Infera worker (LWS) role vs the frontend.
_WORKER_PODID_HINTS = ("worker", "-lws-", "lws-")
_FRONTEND_PODID_HINTS = ("frontend",)


def ssh_role_port_offset(role: str) -> int:
    """Return the SSH port offset for a GPU service role.

    Args:
        role: Service role (``worker`` / ``prefill`` / ``decode``).

    Returns:
        int: ``0`` for worker/prefill; ``INFERA_SSH_PORT_ROLE_STRIDE`` for decode.
    """
    if (role or "").lower() == "decode":
        return INFERA_SSH_PORT_ROLE_STRIDE
    return 0


def ssh_port_for_pod(
    role: str,
    lws_index: int | None,
    *,
    ssh_port_base: int = 2222,
) -> int:
    """Compute the sshd port a pod listens on.

    Args:
        role: Classified service role.
        lws_index: LWS worker ordinal (leader = 0), or ``None``.
        ssh_port_base: Base port (``MN_SSH_PORT`` default / CLI ``--ssh-port``).

    Returns:
        int: ``ssh_port_base + role_offset + lws_index``.
    """
    idx = lws_index if isinstance(lws_index, int) else 0
    return int(ssh_port_base) + ssh_role_port_offset(role) + idx


def idle_worker_entrypoint(*, role: str, ssh_port_base: int = 2222) -> str:
    """Build the idle worker entryPoint with a role-scoped ``MN_SSH_PORT``.

    The port is ``role_base + LWS_WORKER_INDEX`` so multi-node LWS groups on
    different nodes can reuse the same role base while co-located roles (e.g.
    prefill + decode on one node under hostNetwork) bind distinct ports.

    Args:
        role: GPU service role (``worker`` / ``prefill`` / ``decode``).
        ssh_port_base: Base SSH port from the create-infera CLI.

    Returns:
        str: Shell command executed as the pod entryPoint (before base64).
    """
    role_base = int(ssh_port_base) + ssh_role_port_offset(role)
    return (
        f"export MN_SSH_PORT=$(( {role_base} + ${{LWS_WORKER_INDEX:-0}} )); "
        f"exec {_INFERA_IDLE_SCRIPT}"
    )


def _service_roles_for(pd_mode: str) -> list[str]:
    """Positional serviceRoles list for the deployment topology (matches
    ``build_infera_workload_body``): PD -> [frontend, prefill, decode];
    aggregated -> [frontend, worker].

    Args:
        pd_mode: Deployment topology mode (``"disaggregated"`` or
            ``"aggregated"``).

    Returns:
        The positional service roles for the topology.
    """
    if (pd_mode or "").lower() == "disaggregated":
        return ["frontend", "prefill", "decode"]
    return ["frontend", "worker"]


def _parse_role_index(pod_id: str) -> int | None:
    """Parse the slot index from a IDEP pod name ``<wid>-role<N>-<hash>``.

    Args:
        pod_id: The IDEP pod name.

    Returns:
        The parsed role slot index, or ``None`` when the pattern is absent.
    """
    import re

    m = re.search(r"-role(\d+)-", pod_id)
    return int(m.group(1)) if m else None


def _classify_pod_role(
    pod_id: str,
    resource_id: Any,
    service_roles: list[str],
) -> str | None:
    """Classify a IDEP pod into frontend / prefill / decode / worker.

    Priority:
      1. Explicit role substrings in podId (prefillworker / decodeworker /
         frontend) — present when SaFE renames the pods.
      2. Slot index -> ``service_roles[index]``. The index comes from
         ``resourceId`` (SaFE sets it per IDEP pod) or, as a fallback, the
         ``-role<N>-`` suffix in the pod name (SaFE keeps role0/role1/role2
         deployment names). This is the robust path for the observed
         ``<wid>-role<N>-<hash>`` naming.

    Args:
        pod_id: The IDEP pod name.
        resource_id: SaFE-provided resource id (fallback slot index when an
            integer).
        service_roles: Positional service roles to map a slot index onto.

    Returns:
        The classified role (``frontend`` / ``prefill`` / ``decode`` /
        ``worker``), or ``None`` for an unclassifiable pod.
    """
    pl = pod_id.lower()
    if "prefill" in pl:
        return "prefill"
    if "decode" in pl:
        return "decode"
    if "frontend" in pl:
        return "frontend"
    # The IDEP pod NAME reliably encodes the slot (``<wid>-role<N>-<hash>``);
    # prefer it over resourceId, which SaFE leaves 0 for IDEP pods (no
    # resource.id annotation) and would otherwise map every pod to role 0.
    idx = _parse_role_index(pod_id)
    if idx is None and isinstance(resource_id, int):
        idx = resource_id
    if isinstance(idx, int) and 0 <= idx < len(service_roles):
        return service_roles[idx]
    if any(h in pl for h in _WORKER_PODID_HINTS):
        return "worker"
    return None


def discover_role_pods(
    workload: dict[str, Any],
    *,
    pd_mode: str = "aggregated",
    ssh_port_base: int = 2222,
) -> dict[str, list[dict[str, Any]]]:
    """Group a SaFE GetWorkloadResponse's pods by role.

    ``pd_mode`` selects the positional serviceRoles used to map a pod's slot
    index (resourceId / ``-role<N>-``) to its role. Returns
    ``{"frontend": [...], "prefill": [...], "decode": [...], "worker": [...]}``;
    each entry is ``{"podId", "podIP", "role", "lwsIndex", "sshPort"}`` for pods
    with a non-empty ``podIP``, sorted by LWS ordinal (leader = 0) for
    deterministic rank order.

    Args:
        workload: A SaFE GetWorkloadResponse mapping with a ``pods`` list.
        pd_mode: Deployment topology selecting the positional service roles.
        ssh_port_base: Base SSH port matching the deployed ``--ssh-port``.

    Returns:
        A mapping of role to its list of pod SSH target dicts, sorted by LWS
        ordinal then pod id.
    """
    service_roles = _service_roles_for(pd_mode)
    groups: dict[str, list[dict[str, Any]]] = {
        "frontend": [],
        "prefill": [],
        "decode": [],
        "worker": [],
    }
    for p in workload.get("pods") or []:
        if not isinstance(p, dict):
            continue
        pod_ip = str(p.get("podIP") or "").strip()
        if not pod_ip:
            continue
        # Skip terminal / dead pods. A IDEP role pod that crashed during early
        # scheduling lingers in GetWorkload.pods with a stale podIP but no sshd
        # (phase=Failed/Succeeded). Including it makes restart-server SSH-fan-out
        # to a dead replica -> "Connection refused" rc=1 -> baseline_failed.
        # The live replacement replica (same role index) is the one we want.
        pod_phase = str(p.get("phase") or "").strip().lower()
        if pod_phase in ("failed", "succeeded", "terminating"):
            continue
        pod_id = str(p.get("podId") or "")
        role = _classify_pod_role(pod_id, p.get("resourceId"), service_roles)
        if role is None:
            continue
        lws_idx = _parse_lws_ordinal(pod_id)
        groups[role].append(
            {
                "podId": pod_id,
                "podIP": pod_ip,
                "role": role,
                "lwsIndex": lws_idx,
                "sshPort": ssh_port_for_pod(role, lws_idx, ssh_port_base=ssh_port_base),
            }
        )
    for role in groups:
        groups[role].sort(
            key=lambda d: (
                d["lwsIndex"] if isinstance(d["lwsIndex"], int) else 1 << 30,
                d["podId"],
            )
        )
    return groups


def discover_worker_pods(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the aggregated worker pods — convenience wrapper over
    :func:`discover_role_pods` for the non-PD path. Frontend pods are excluded.

    Args:
        workload: A SaFE GetWorkloadResponse mapping with a ``pods`` list.

    Returns:
        The aggregated worker pod entries.
    """
    return discover_role_pods(workload, pd_mode="aggregated")["worker"]


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
    base = int(state.get("ssh_port") or 2222)
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


def _parse_lws_ordinal(pod_id: str) -> int | None:
    """Parse the trailing ``-<n>`` ordinal from an LWS pod name, else None.

    LWS pods are named ``<group>-<ordinal>`` (leader = 0). KubeRay-style random
    suffixes (``-x6fkf``) are non-numeric and return None.

    Args:
        pod_id: The LWS pod name.

    Returns:
        The trailing ordinal as an int, or ``None`` when it is non-numeric.
    """
    tail = pod_id.rsplit("-", 1)[-1] if "-" in pod_id else ""
    return int(tail) if tail.isdigit() else None


def frontend_service_url(
    workload_id: str,
    workspace: str,
    service_info: dict[str, Any] | None = None,
    *,
    port: int = INFERA_FRONTEND_PORT,
) -> str:
    """Resolve the Infera frontend base URL for benchmarks.

    Prefers the live SaFE service info (clusterIp / dns) when present; falls
    back to the conventional ``http://<wid>.<workspace>.svc.cluster.local:<port>``.

    Args:
        workload_id: The Infera workload id.
        workspace: The Kubernetes namespace / workspace name.
        service_info: Optional live SaFE service info (internalDomain / dns /
            clusterIp / port).
        port: Default frontend HTTP port used when none is in ``service_info``.

    Returns:
        The resolved frontend base URL.
    """
    if service_info:
        # Prefer the ready-made internalDomain.
        internal = str(service_info.get("internalDomain") or "").strip()
        if internal:
            internal = internal.split("://", 1)[-1].rstrip("/")
            return f"http://{internal}"
        # ``port`` may be a nested object {protocol, port, targetPort} or a bare int.
        raw_port = service_info.get("port")
        if isinstance(raw_port, dict):
            svc_port = raw_port.get("port") or raw_port.get("targetPort") or port
        else:
            svc_port = raw_port or port
        dns = str(service_info.get("dns") or service_info.get("dnsName") or "").strip()
        cluster_ip = str(service_info.get("clusterIp") or "").strip()
        host = dns or cluster_ip
        if host:
            host = host.split("://", 1)[-1].rstrip("/")
            return f"http://{host}:{svc_port}"
    return f"http://{workload_id}.{workspace}.svc.cluster.local:{port}"


# sglang PD bootstrap rendezvous port (SaFE common.InferaBootstrapPort).
INFERA_BOOTSTRAP_PORT = 30001


def disagg_flags(mode: str, kv_transfer_backend: str, *, bootstrap_port: int = INFERA_BOOTSTRAP_PORT) -> str:
    """sglang PD disaggregation flags for a prefill/decode group.

    Mirrors the SaFE dispatcher's ``sglangDisaggFlags`` so the SSH-launched
    server matches the native deploy path. ``infera.sglang`` parses these via
    argparse (it does NOT read SGLANG_DISAGGREGATION_* env), so they must be on
    the command line.

    Args:
        mode: Disaggregation mode (``"prefill"`` or ``"decode"``); any other
            value yields an empty string.
        kv_transfer_backend: Optional KV transfer backend name.
        bootstrap_port: sglang PD bootstrap rendezvous port.

    Returns:
        The space-joined sglang PD disaggregation flags, or ``""`` when
        ``mode`` is neither prefill nor decode.
    """
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
    """Build the argv string for launch_infera_node.py (shipped over SSH).

    The same string is sent to every pod in a group; each pod self-determines
    its node-rank from ``$LWS_WORKER_INDEX`` pod-side. ``disagg_mode``
    (prefill/decode) folds the sglang PD flags into the launched command.

    Args:
        framework: Framework name (``"sglang"`` or ``"vllm"``).
        model: Model path passed to the launcher.
        tp: Tensor-parallel size.
        nnodes: Number of nodes in the group.
        ep: Expert-parallel size (only emitted when > 1).
        dist_init_port: torch.distributed rendezvous port.
        pid_file: Pod-side server pid file path.
        log_file: Pod-side server log file path.
        extra_args: Extra args string folded into ``--extra-args``.
        health_port: Leader local readiness probe port.
        health_wait_sec: Seconds to wait for local ``/health`` (0 = skip).
        kill_only: When ``True``, build a kill-only argv that frees the GPU.
        disagg_mode: PD disaggregation mode folded into ``extra_args``.
        kv_transfer_backend: KV transfer backend for PD disaggregation.

    Returns:
        The shell-quoted argv string for ``launch_infera_node.py``.
    """
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
        quoted += " --extra-args " + shlex.quote(merged_extra)
    return quoted
