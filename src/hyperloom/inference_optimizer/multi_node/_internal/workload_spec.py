# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure builders for the SaFE CreateWorkloadRequest body of a session-scoped multi-node-inference RayJob.

Pure (no I/O / env); the CLI feeds inputs in. Non-overridable decisions:

* ``groupVersionKind = RayJob/v1``; ``priority = 1``,
  ``isSupervised/isTolerateAll = false``, ``useWorkspaceStorage = true``
  (isTolerateAll=false keeps it off tainted nodes that corrupt benchmarks).
* ``maxRetry = 0`` (a SaFE retry would spawn a duplicate cluster; we
  restart the server in-cluster instead).
* ``service`` on port 8888/ClusterIP with ``extraSelectors`` pinning the
  head role, since only the head pod serves.
* ``entryPoints`` default ``["", ""]`` (optional per-role install payloads;
  must exit if used). The submitter long-run driver is only
  ``env.RAY_JOB_ENTRYPOINT`` (base64 ``tail -f /dev/null``).
* ``env`` merges user ``extra_env`` (reserved keys stripped); inject debug
  knobs via ``--extra-env``. ``RAY_JOB_ENTRYPOINT`` can't be overridden.
* ``labels`` strip Brain-managed prefixes; ``primus-claw/session-id`` is
  injected from ``session_id`` for Brain correlation.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from . import infera_support

# Hard-coded per design (Brain TS impl + SKILL.md contract).
_INFERENCE_SERVER_PORT = 8888
# Ray head-role selector label. Neutral default for portability; internal
# deployments can restore their cluster's label via the env override.
_HEAD_ROLE_LABEL = os.environ.get(
    "HYPERLOOM_RAY_HEAD_ROLE_LABEL", "hyperloom.io/ray-role"
)
_HEAD_ROLE_VALUE = "head"
# Submitter-only signal-interruptable driver for RayJob.spec.entrypoint.
_SUBMITTER_BLOCK_ENTRYPOINT = "tail -f /dev/null"

# Stripped from user ``extra_env`` (overwritten below, so accepting them would mislead).
_STRIP_FROM_USER_ENV = frozenset({"RAY_JOB_ENTRYPOINT"})

# Brain-managed label prefixes; caller labels with these are stripped.
_RESERVED_LABEL_PREFIXES = (
    "primus-safe.",
    "primus-claw/",
)


def _b64(s: str) -> str:
    """Base64-encode a string as ASCII.

    Args:
        s (str): The text to encode (UTF-8).

    Returns:
        str: The base64-encoded value as an ASCII string.
    """
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _sanitize_extra_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    """Drop reserved and legacy keys from user-supplied env.

    Args:
        extra_env (dict[str, str] | None): User-supplied environment
            variables, or ``None``.

    Returns:
        dict[str, str]: A copy of ``extra_env`` with reserved keys removed,
        or an empty dict when input is falsy.
    """
    if not extra_env:
        return {}
    return {k: v for k, v in extra_env.items() if k not in _STRIP_FROM_USER_ENV}


def _sanitize_extra_labels(extra_labels: dict[str, str] | None) -> dict[str, str]:
    """Drop labels whose key starts with any reserved prefix.

    Args:
        extra_labels (dict[str, str] | None): User-supplied labels, or
            ``None``.

    Returns:
        dict[str, str]: A copy of ``extra_labels`` excluding keys that start
        with any Brain-managed reserved prefix; empty when input is falsy.
    """
    if not extra_labels:
        return {}
    out: dict[str, str] = {}
    for k, v in extra_labels.items():
        if any(k.startswith(p) for p in _RESERVED_LABEL_PREFIXES):
            continue
        out[k] = v
    return out


def build_rayjob_workload_body(
    *,
    workspace: str,
    display_name: str,
    image: str,
    nodes: int,
    gpus_per_node: int,
    cpus_per_node: int,
    mem_gi_per_node: int,
    ephemeral_gi_per_node: int,
    description: str | None = None,
    owner_id: str | None = None,
    session_id: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a SaFE CreateWorkloadRequest body for a multi-node RayJob (resource quantities as K8s-notation strings; json.dumps-safe).

    Args:
        workspace: SaFE workspace id the workload belongs to.
        display_name: Human-readable name for the workload.
        image: Container image used for both head and worker roles.
        nodes: Total node count; head is 1 and workers are ``nodes - 1``.
        gpus_per_node: GPUs requested per node.
        cpus_per_node: CPUs requested per node.
        mem_gi_per_node: Memory in GiB requested per node.
        ephemeral_gi_per_node: Ephemeral storage in GiB requested per node.
        description: Optional workload description.
        owner_id: Optional owner id to attach to the workload.
        session_id: Optional session id injected as ``primus-claw/session-id``
            for Brain correlation.
        extra_env: Optional user environment variables (reserved keys stripped).
        extra_labels: Optional user labels (Brain-managed prefixes stripped).

    Returns:
        dict[str, Any]: A json.dumps-safe CreateWorkloadRequest body.

    Raises:
        ValueError: If ``nodes`` or ``gpus_per_node`` is below 1, or if
            ``workspace``, ``display_name``, or ``image`` is empty.
    """
    if nodes < 1:
        raise ValueError(f"nodes must be >= 1, got {nodes}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be >= 1, got {gpus_per_node}")
    if not workspace:
        raise ValueError("workspace is required")
    if not display_name:
        raise ValueError("display_name is required")
    if not image:
        raise ValueError("image is required")

    # head replica is 1; worker is N-1, clamped to >=1 (SaFE requires replica >=1).
    worker_replica = max(1, nodes - 1)

    head_resource = {
        "replica": 1,
        "cpu": str(cpus_per_node),
        "memory": f"{mem_gi_per_node}Gi",
        "gpu": str(gpus_per_node),
        "ephemeralStorage": f"{ephemeral_gi_per_node}Gi",
    }
    worker_resource = {
        "replica": worker_replica,
        "cpu": str(cpus_per_node),
        "memory": f"{mem_gi_per_node}Gi",
        "gpu": str(gpus_per_node),
        "ephemeralStorage": f"{ephemeral_gi_per_node}Gi",
    }

    # entryPoints: optional per-role install payloads (base64); empty = none.
    entry_points: list[str] = ["", ""]

    # env: user extra_env (reserved keys stripped).
    env: dict[str, str] = _sanitize_extra_env(extra_env)
    env["RAY_JOB_ENTRYPOINT"] = _b64(_SUBMITTER_BLOCK_ENTRYPOINT)

    # labels: sanitized caller labels + injected ``primus-claw/session-id`` for
    # Brain correlation.
    labels = _sanitize_extra_labels(extra_labels)
    if session_id:
        labels["primus-claw/session-id"] = session_id

    body: dict[str, Any] = {
        "displayName": display_name,
        "workspaceId": workspace,
        "groupVersionKind": {
            "kind": "RayJob",
            "version": "v1",
        },
        "priority": 1,
        "isSupervised": False,
        "isTolerateAll": False,
        "useWorkspaceStorage": True,
        "privileged": False,
        "forceHostNetwork": False,
        "preheat": False,
        "maxRetry": 0,
        "dependencies": [],
        "resources": [head_resource, worker_resource],
        "images": [image, image],
        "entryPoints": entry_points,
        "env": env,
        "labels": labels,
        "service": {
            "protocol": "TCP",
            "port": _INFERENCE_SERVER_PORT,
            "targetPort": _INFERENCE_SERVER_PORT,
            "serviceType": "ClusterIP",
            "extraSelectors": {
                _HEAD_ROLE_LABEL: _HEAD_ROLE_VALUE,
            },
        },
    }
    if description:
        body["description"] = description
    if owner_id:
        body["ownerId"] = owner_id
    return body


# ---------------------------------------------------------------------------
# Infera (InferaDeployment) idle-pod body
#
# The Infera backend reuses the RayJob "long-lived pod + external server
# restart" pattern, but with a SaFE InferaDeployment (LeaderWorkerSet-backed
# multi-node worker) instead of a RayJob, and SSH instead of the Ray Dashboard
# control plane. The worker pods are deployed IDLE (entryPoint =
# ``/usr/local/bin/mn-idle.sh``, which starts sshd then blocks) so the
# optimizer can SSH in and (re)launch infera.engine.sglang/vllm with per-round
# flags without redeploying the workload (preserving the aiter JIT cache across
# restarts).
#
# See multi_node/SKILL.md (Infera section).

# Frontend HTTP port — infera.server listens here; benchmarks target this
# (NOT sglang rank-0 :8888) so the OpenAI-compatible router fronts every
# worker registration. Matches the SaFE Infera fixtures.
_INFERA_FRONTEND_PORT = 8000

# Idle worker entryPoint: role-scoped MN_SSH_PORT (see infera_support.idle_worker_entrypoint).

# Frontend launch command (role 0). round-robin router is the simplest mode;
# --enable-profiling lets the controller drive torch profiling via the
# frontend /v1/admin/profile/* fan-out API.
_INFERA_FRONTEND_ENTRYPOINT_TMPL = (
    "python3 -m infera.server --host 0.0.0.0 --port {port} "
    "--router-policy round-robin --router-tokenizer-path {model} --enable-profiling"
)

# Valid enum values mirrored from the SaFE Infera webhook.
_INFERA_BACKENDS = frozenset({"sglang", "vllm"})
_INFERA_KV_BACKENDS = frozenset({"nixl", "mori", "mooncake"})


def build_infera_workload_body(
    *,
    workspace: str,
    display_name: str,
    image: str,
    nodes: int,
    gpus_per_node: int,
    cpus_per_node: int,
    mem_gi_per_node: int,
    ephemeral_gi_per_node: int,
    ssh_authorized_key: str,
    model: str,
    backend_framework: str = "sglang",
    kv_transfer_backend: str = "mori",
    ssh_port: int = 2222,
    shared_mem_gi: int = 200,
    rdma_resource: str = "1",
    frontend_cpu: int = 4,
    frontend_mem_gi: int = 16,
    frontend_port: int = _INFERA_FRONTEND_PORT,
    pd_mode: str = "aggregated",
    pd_prefill_nodes: int = 0,
    pd_decode_nodes: int = 0,
    pd_prefill_tp: int = 0,
    pd_decode_tp: int = 0,
    description: str | None = None,
    owner_id: str | None = None,
    session_id: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a SaFE CreateWorkloadRequest body for an IDLE multi-node Infera
    deployment (frontend + LeaderWorkerSet worker).

    Topology: ``resources = [frontend, worker]`` with
    ``inferaOptions.serviceRoles = ["frontend", "worker"]``. When ``nodes >= 2``
    the worker is listed in ``multinodeRoles`` and ``worker.replica = nodes`` so
    the dispatcher sets ``multinode.numberOfNodes = nodes`` and the operator
    materialises one LeaderWorkerSet group of ``nodes`` pods (a single
    tensor-parallel model spanning nodes). ``worker.replica`` IS the node count
    per the new API — NOT a Deployment replica count.

    The worker entryPoint is the idle ``mn-idle.sh`` (sshd + block), so no
    inference server starts at deploy time; the optimizer SSHes in to launch
    sglang/vllm per round. ``ssh_authorized_key`` is injected as
    ``MN_SSH_AUTHORIZED_KEY`` so ``mn-sshd-init.sh`` can authorise the
    controller's key at container start.

    ``entryPoints`` are base64-encoded per the SaFE contract (the apiserver
    stores them verbatim; the dispatcher decodes/append/re-encodes the
    launcher payload).

    Args:
        workspace: SaFE workspace id the workload belongs to.
        display_name: Human-readable name for the workload.
        image: Container image used for the frontend and worker roles.
        nodes: Total node count for the aggregated worker role.
        gpus_per_node: GPUs requested per GPU pod.
        cpus_per_node: CPUs requested per GPU pod.
        mem_gi_per_node: Memory in GiB requested per GPU pod.
        ephemeral_gi_per_node: Ephemeral storage in GiB requested per GPU pod.
        ssh_authorized_key: Public key injected as ``MN_SSH_AUTHORIZED_KEY``
            for the idle-pod control plane.
        model: Model path or HF id for the frontend ``--router-tokenizer-path``.
        backend_framework: Infera backend (``sglang`` / ``vllm``).
        kv_transfer_backend: KV transfer backend (``nixl`` / ``mori`` /
            ``mooncake``).
        ssh_port: SSH port exported as ``MN_SSH_PORT``.
        shared_mem_gi: Shared memory in GiB requested per GPU pod.
        rdma_resource: RDMA resource quantity for multinode / PD roles.
        frontend_cpu: CPUs requested for the frontend pod.
        frontend_mem_gi: Memory in GiB for the frontend pod.
        frontend_port: Frontend HTTP port (benchmark target).
        pd_mode: ``aggregated`` or ``disaggregated`` prefill/decode topology.
        pd_prefill_nodes: Prefill group node count (disaggregated).
        pd_decode_nodes: Decode group node count (disaggregated).
        pd_prefill_tp: Prefill group tensor-parallel size (disaggregated).
        pd_decode_tp: Decode group tensor-parallel size (disaggregated).
        description: Optional workload description.
        owner_id: Optional owner id to attach to the workload.
        session_id: Optional session id injected as ``primus-claw/session-id``.
        extra_env: Optional user environment variables (reserved keys stripped).
        extra_labels: Optional user labels (Brain-managed prefixes stripped).

    Returns:
        A json.dumps-safe CreateWorkloadRequest body for the Infera deployment.

    Raises:
        ValueError: If ``nodes`` / ``gpus_per_node`` is below 1; if
            ``workspace`` / ``display_name`` / ``image`` / ``model`` /
            ``ssh_authorized_key`` is empty; or if ``backend_framework`` /
            ``kv_transfer_backend`` is not a supported enum value.
    """
    if nodes < 1:
        raise ValueError(f"nodes must be >= 1, got {nodes}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be >= 1, got {gpus_per_node}")
    if not workspace:
        raise ValueError("workspace is required")
    if not display_name:
        raise ValueError("display_name is required")
    if not image:
        raise ValueError("image is required")
    if not model or not model.strip():
        raise ValueError("model is required (frontend --router-tokenizer-path)")
    if not ssh_authorized_key or not ssh_authorized_key.strip():
        raise ValueError(
            "ssh_authorized_key is required for the Infera idle-pod control "
            "plane (injected as MN_SSH_AUTHORIZED_KEY for mn-sshd-init.sh)"
        )
    bf = (backend_framework or "sglang").lower()
    if bf not in _INFERA_BACKENDS:
        raise ValueError(f"backend_framework must be one of {sorted(_INFERA_BACKENDS)}, got {bf!r}")
    kvb = (kv_transfer_backend or "mori").lower()
    if kvb not in _INFERA_KV_BACKENDS:
        raise ValueError(f"kv_transfer_backend must be one of {sorted(_INFERA_KV_BACKENDS)}, got {kvb!r}")

    # Frontend (role 0): CPU-only OpenAI-compatible router/HTTP server.
    frontend_resource = {
        "replica": 1,
        "cpu": str(frontend_cpu),
        "memory": f"{frontend_mem_gi}Gi",
    }

    def _gpu_resource(replica: int, *, multinode: bool) -> dict[str, Any]:
        """Build the resource dict for one GPU pod slot (worker / prefill / decode).

        RDMA is added only when the role spans nodes.

        Args:
            replica: The replica count for this pod role.
            multinode: Whether the role spans nodes (adds ``rdmaResource``).

        Returns:
            The resource dict for one GPU pod slot.
        """
        res: dict[str, Any] = {
            "replica": replica,
            "cpu": str(cpus_per_node),
            "memory": f"{mem_gi_per_node}Gi",
            "gpu": str(gpus_per_node),
            "ephemeralStorage": f"{ephemeral_gi_per_node}Gi",
            "sharedMemory": f"{shared_mem_gi}Gi",
        }
        if multinode:
            res["rdmaResource"] = rdma_resource
        return res

    frontend_ep = _b64(
        _INFERA_FRONTEND_ENTRYPOINT_TMPL.format(port=frontend_port, model=model.strip())
    )
    worker_idle_ep = _b64(infera_support.idle_worker_entrypoint(role="worker", ssh_port_base=ssh_port))
    prefill_idle_ep = _b64(infera_support.idle_worker_entrypoint(role="prefill", ssh_port_base=ssh_port))
    decode_idle_ep = _b64(infera_support.idle_worker_entrypoint(role="decode", ssh_port_base=ssh_port))

    is_pd = (pd_mode or "aggregated").lower() == "disaggregated"
    if is_pd:
        # PD disaggregation: roles = [frontend, prefill, decode]. A role spans
        # nodes (LeaderWorkerSet) when its TP exceeds one pod's GPUs; otherwise
        # its replica is an independent single-node instance count. Both pod
        # roles deploy IDLE (mn-idle.sh) — restart-server SSH-launches
        # infera.engine.sglang with --disaggregation-mode prefill/decode per group.
        pn = max(1, int(pd_prefill_nodes or 0))
        dn = max(1, int(pd_decode_nodes or 0))
        ptp = int(pd_prefill_tp or 0)
        dtp = int(pd_decode_tp or 0)
        # A role spans nodes (LeaderWorkerSet) only when its TP exceeds one pod's
        # GPUs; otherwise its replica is an independent single-node instance count.
        prefill_mn = ptp > gpus_per_node
        decode_mn = dtp > gpus_per_node
        prefill_res = _gpu_resource(pn, multinode=prefill_mn)
        decode_res = _gpu_resource(dn, multinode=decode_mn)
        # Both GPU roles need an RDMA device to stream the KV cache across pods,
        # even when single-node; without it the KV transfer plane silently no-ops.
        pd_rdma = rdma_resource if (rdma_resource and rdma_resource != "1") else "1k"
        prefill_res["rdmaResource"] = pd_rdma
        decode_res["rdmaResource"] = pd_rdma
        resources = [frontend_resource, prefill_res, decode_res]
        images = [image, image, image]
        entry_points = [frontend_ep, prefill_idle_ep, decode_idle_ep]
        service_roles = ["frontend", "prefill", "decode"]
        multinode_roles = (["prefill"] if prefill_mn else []) + (["decode"] if decode_mn else [])
    else:
        # Aggregated: [frontend, worker]; worker.replica == node count.
        resources = [frontend_resource, _gpu_resource(nodes, multinode=nodes > 1)]
        images = [image, image]
        entry_points = [frontend_ep, worker_idle_ep]
        service_roles = ["frontend", "worker"]
        multinode_roles = ["worker"] if nodes > 1 else []

    # env: SSH control-plane knobs + user extra_env (reserved keys stripped).
    env: dict[str, str] = _sanitize_extra_env(extra_env)
    env["MN_SSH_AUTHORIZED_KEY"] = ssh_authorized_key.strip()
    env["MN_SSH_PORT"] = str(ssh_port)

    labels = _sanitize_extra_labels(extra_labels)
    if session_id:
        labels["primus-claw/session-id"] = session_id

    infera_options: dict[str, Any] = {
        "backendFramework": bf,
        "kvTransferBackend": kvb,
        "serviceRoles": service_roles,
    }
    if multinode_roles:
        infera_options["multinodeRoles"] = multinode_roles

    body: dict[str, Any] = {
        "displayName": display_name,
        "workspaceId": workspace,
        "groupVersionKind": {
            "kind": "InferaDeployment",
            "version": "v1",
        },
        "priority": 1,
        "isSupervised": False,
        "isTolerateAll": False,
        "useWorkspaceStorage": True,
        "privileged": False,
        "forceHostNetwork": False,
        "preheat": False,
        "maxRetry": 0,
        "dependencies": [],
        "resources": resources,
        "images": images,
        "entryPoints": entry_points,
        "env": env,
        "labels": labels,
        "inferaOptions": infera_options,
        "service": {
            "protocol": "TCP",
            "port": frontend_port,
            "targetPort": frontend_port,
            "serviceType": "ClusterIP",
        },
    }
    if description:
        body["description"] = description
    if owner_id:
        body["ownerId"] = owner_id
    return body
