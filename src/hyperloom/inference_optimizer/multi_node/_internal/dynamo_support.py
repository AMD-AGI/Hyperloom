"""Pure helpers for the Dynamo multi-node backend.

No I/O / no env reads (mirrors ``workload_spec.py``): the CLI feeds in the
SaFE GetWorkloadResponse / service info and these functions extract worker pod
IPs, the frontend service URL, and the pod-side launcher argv. Kept pure so the
SSH fan-out logic stays unit-testable without a live cluster.
"""

from __future__ import annotations

import shlex
from typing import Any

# Frontend HTTP port (SaFE common.DynamoFrontendPort). Benchmarks target this
# OpenAI-compatible endpoint, never sglang rank-0 :8888.
DYNAMO_FRONTEND_PORT = 8000

# Substrings that mark a pod as the Dynamo worker (LWS) role vs the frontend.
_WORKER_PODID_HINTS = ("worker", "-lws-", "lws-")


def _service_roles_for(pd_mode: str) -> list[str]:
    """Positional serviceRoles list for the deployment topology (matches
    ``build_dynamo_workload_body``): PD -> [frontend, prefill, decode];
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
    """Parse the slot index from a DGD pod name ``<wid>-role<N>-<hash>``.

    Args:
        pod_id: The DGD pod name.

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
    """Classify a DGD pod into frontend / prefill / decode / worker.

    Priority:
      1. Explicit role substrings in podId (prefillworker / decodeworker /
         frontend) — present when SaFE renames the pods.
      2. Slot index -> ``service_roles[index]``. The index comes from
         ``resourceId`` (SaFE sets it per DGD pod) or, as a fallback, the
         ``-role<N>-`` suffix in the pod name (SaFE keeps role0/role1/role2
         deployment names). This is the robust path for the observed
         ``<wid>-role<N>-<hash>`` naming.

    Args:
        pod_id: The DGD pod name.
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
    # The DGD pod NAME reliably encodes the slot (``<wid>-role<N>-<hash>``);
    # prefer it over resourceId, which SaFE leaves 0 for DGD pods (no
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
) -> dict[str, list[dict[str, Any]]]:
    """Group a SaFE GetWorkloadResponse's pods by role.

    ``pd_mode`` selects the positional serviceRoles used to map a pod's slot
    index (resourceId / ``-role<N>-``) to its role. Returns
    ``{"frontend": [...], "prefill": [...], "decode": [...], "worker": [...]}``;
    each entry is ``{"podId", "podIP", "lwsIndex"}`` for pods with a non-empty
    ``podIP``, sorted by LWS ordinal (leader = 0) for deterministic rank order.

    Args:
        workload: A SaFE GetWorkloadResponse mapping with a ``pods`` list.
        pd_mode: Deployment topology selecting the positional service roles.

    Returns:
        A mapping of role to its list of ``{"podId", "podIP", "lwsIndex"}``
        entries, sorted by LWS ordinal then pod id.
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
        # Skip terminal / dead pods. A DGD role pod that crashed during early
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
        groups[role].append(
            {
                "podId": pod_id,
                "podIP": pod_ip,
                "lwsIndex": _parse_lws_ordinal(pod_id),
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
    port: int = DYNAMO_FRONTEND_PORT,
) -> str:
    """Resolve the Dynamo frontend base URL for benchmarks.

    Prefers the live SaFE service info (clusterIp / dns) when present; falls
    back to the conventional ``http://<wid>.<workspace>.svc.cluster.local:<port>``.

    Args:
        workload_id: The Dynamo workload id.
        workspace: The Kubernetes namespace / workspace name.
        service_info: Optional live SaFE service info (internalDomain / dns /
            clusterIp / port).
        port: Default frontend HTTP port used when none is in ``service_info``.

    Returns:
        The resolved frontend base URL.
    """
    if service_info:
        # Prefer the ready-made internalDomain ("<wid>.<ns>.svc.cluster.local:8000").
        internal = str(service_info.get("internalDomain") or "").strip()
        if internal:
            internal = internal.split("://", 1)[-1].rstrip("/")
            return f"http://{internal}"
        # SaFE returns ``port`` as a nested object {protocol, port, targetPort};
        # extract the integer (older shapes may return a bare int).
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


# sglang PD bootstrap rendezvous port (SaFE common.DynamoBootstrapPort).
DYNAMO_BOOTSTRAP_PORT = 30001


def disagg_flags(mode: str, kv_transfer_backend: str, *, bootstrap_port: int = DYNAMO_BOOTSTRAP_PORT) -> str:
    """sglang PD disaggregation flags for a prefill/decode group.

    Mirrors the SaFE dispatcher's ``sglangDisaggFlags`` so the SSH-launched
    server matches the native deploy path. ``dynamo.sglang`` parses these via
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
    pid_file: str = "/tmp/mn_dynamo_server.pid",
    log_file: str = "/tmp/mn_dynamo_server.log",
    extra_args: str = "",
    health_port: int = DYNAMO_FRONTEND_PORT,
    health_wait_sec: int = 0,
    kill_only: bool = False,
    disagg_mode: str = "",
    kv_transfer_backend: str = "",
) -> str:
    """Build the argv string for launch_dynamo_node.py (shipped over SSH).

    The SAME string is sent to every pod IN A GROUP — each pod self-determines
    its node-rank from ``$LWS_WORKER_INDEX`` pod-side, so the controller does
    not encode the rank here. ``disagg_mode`` (prefill/decode) folds the sglang
    PD flags into the launched command for that group.

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
        The shell-quoted argv string for ``launch_dynamo_node.py``.
    """
    parts = ["--framework", framework]
    if kill_only:
        parts.append("--kill-only")
        # kill-only still needs framework (vllm tears down its ray node) and
        # the pid-file path so it kills the right server.
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
    # Fold the PD disaggregation flags into extra_args (the pod-side script
    # re-splits --extra-args with shlex and appends them to the sglang cmd).
    merged_extra = (extra_args or "").strip()
    df = disagg_flags(disagg_mode, kv_transfer_backend)
    if df:
        merged_extra = (merged_extra + " " + df).strip()
    if merged_extra:
        quoted += " --extra-args " + shlex.quote(merged_extra)
    return quoted
