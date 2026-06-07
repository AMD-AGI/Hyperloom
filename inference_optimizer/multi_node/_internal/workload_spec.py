# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure functions that build the SaFE CreateWorkloadRequest body for a
session-scoped RayJob that hosts a multi-node inference server.

Pure: no I/O, no env reads. The CLI (`inference_optimizer.multi_node.cli`)
is responsible for collecting inputs (env + argparse) and feeding them in.
This makes the spec unit-testable and lets future callers (e.g. another
tool) reuse the exact same JSON contract.

Decisions baked in here that the caller cannot override:

* ``groupVersionKind = {"kind": "RayJob", "version": "v1"}`` — matches
  the production sample request format SaFE expects.
* ``priority = 1``, ``isSupervised = false``, ``isTolerateAll = false``,
  ``useWorkspaceStorage = true`` — production defaults. priority=1 keeps
  the multi-node optimization workload above background batch jobs in
  the SaFE scheduler; isTolerateAll=false stops it landing on tainted
  / drained nodes (a multi-node TP run on a flaky node corrupts the
  whole benchmark).
* ``maxRetry = 0`` — SaFE-level retries would spawn a duplicate cluster.
  The orchestrator restarts the inference server inside the existing
  cluster instead.
* ``preheat = false``, ``privileged = false``, ``forceHostNetwork = false``,
  ``dependencies = []`` — production defaults.
* ``service.port = 8888 / targetPort = 8888 / serviceType = ClusterIP``
  with ``extraSelectors = {primus-safe.amd.com/ray-role: head}`` — the
  inference server only listens on the Ray head pod, so the K8s Service
  must select only that pod (otherwise traffic round-robins to workers
  that don't listen on 8888 and clients see connection refused).
* ``entryPoints`` (two strings aligned with head/worker ``resources[]``) are
  **optional** per-role **install** payloads: SaFE passes each non-empty value
  as the argument to ``/shared-data/launcher.sh`` before KubeRay appends
  ``ray start`` (see ``dispatcher_help.go`` ``buildEntryPoint`` /
  ``buildCommands``). If you do not need extra install scripts, use **empty**
  strings for both roles (this builder defaults to ``["", ""]``). Non-empty
  payloads must **exit** so ``launcher.sh`` can finish waiting and ``ray
  start`` runs; never use a non-terminating script (e.g. ``tail -f``) here.
  The RayJob **submitter** long-run driver is **only** ``env.RAY_JOB_ENTRYPOINT``
  (below), not ``entryPoints``.
* ``env`` — SaFE workload ``Env map[string]string`` (JSON ``"env": {...}``).
  User ``extra_env`` is merged first (reserved keys stripped); no defaults
  are injected, so debug knobs (``NCCL_DEBUG``, ``NCCL_DEBUG_FILE``,
  ``TORCH_DISTRIBUTED_DEBUG``, etc.) must be passed explicitly via
  ``--extra-env`` when triaging. ``RAY_JOB_ENTRYPOINT`` is then set to
  base64(``tail -f /dev/null``) for the **KubeRay submitter**
  (``updateRayJob`` / ``spec.entrypoint``) and cannot be overridden by callers.
* ``labels`` — caller-supplied labels are accepted with two carve-outs.
  Brain-managed prefixes (``primus-safe.``, ``primus-claw/``) are stripped
  from user input so callers cannot collide with platform bookkeeping.
  ``primus-claw/session-id`` is injected by the builder when the caller
  passes a non-empty ``session_id`` so Brain can correlate this RayJob
  with its parent sandbox session.
"""

from __future__ import annotations

import base64
from typing import Any

# Hard-coded per design (matches Brain's previous TS implementation and
# the inference-optimization SKILL.md contract).
_INFERENCE_SERVER_PORT = 8888
_HEAD_ROLE_LABEL = "primus-safe.amd.com/ray-role"
_HEAD_ROLE_VALUE = "head"
# Submitter-only: long-running driver for RayJob.spec.entrypoint (SaFE
# updateRayJob). Signal-interruptable (vs sleep infinity).
_SUBMITTER_BLOCK_ENTRYPOINT = "tail -f /dev/null"

# Keys stripped from user ``extra_env`` before merge. The builder
# overwrites these below, so accepting them from callers is misleading
# (the user value would be silently ignored). Strip explicitly so the
# contract is obvious.
_STRIP_FROM_USER_ENV = frozenset({"RAY_JOB_ENTRYPOINT"})

# Brain-managed label keys. Caller-supplied labels with these prefixes are
# silently stripped to avoid colliding with SaFE / brain bookkeeping.
_RESERVED_LABEL_PREFIXES = (
    "primus-safe.",
    "primus-claw/",
)


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _sanitize_extra_env(extra_env: dict[str, str] | None) -> dict[str, str]:
    """Drop reserved and legacy keys from user-supplied env."""
    if not extra_env:
        return {}
    return {k: v for k, v in extra_env.items() if k not in _STRIP_FROM_USER_ENV}


def _sanitize_extra_labels(extra_labels: dict[str, str] | None) -> dict[str, str]:
    """Drop labels whose key starts with any reserved prefix."""
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
    """Build a SaFE CreateWorkloadRequest body for a multi-node RayJob.

    All resource quantities are emitted as STRING values per SaFE
    convention. Quantities follow Kubernetes resource notation
    (e.g. ``"32"`` for CPUs, ``"128Gi"`` for memory).

    The returned dict is safe to pass to ``json.dumps`` — every value is
    a primitive, list of primitives, or nested dict of the same.
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

    # head replica is always 1; worker replica is N-1 (clamped to >= 1
    # because SaFE requires every entry in resources[] to have replica >= 1
    # even in the degenerate single-node case where the worker_group is
    # effectively unused).
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
    # See module docstring and SaFE dispatcher_help.buildEntryPoint.
    entry_points: list[str] = ["", ""]

    # env: SaFE CreateWorkloadRequest.Env (map[string]string) is injected into
    # RayJob pods after dispatcher merges with chart defaults. User keys from
    # extra_env win for the same name; reserved keys are stripped in sanitize.
    # Pass NCCL_DEBUG / NCCL_DEBUG_FILE / TORCH_DISTRIBUTED_DEBUG via --extra-env
    # only when triaging — they are not injected here.
    env: dict[str, str] = _sanitize_extra_env(extra_env)
    env["RAY_JOB_ENTRYPOINT"] = _b64(_SUBMITTER_BLOCK_ENTRYPOINT)

    # labels: caller-supplied first (sanitized), then the Brain
    # correlation key. ``primus-claw/session-id`` is injected (when
    # provided) so Brain can correlate this RayJob with its parent
    # sandbox session for GC / dashboard linking; the value comes from
    # the sandbox env via the CLI layer. SaFE-namespace labels are NOT
    # written here -- SaFE strips ``primus-safe.amd.com/*`` from caller
    # input on its way to the K8s RayJob CRD, so adding them would be a
    # no-op (verified May 2026 against a live ``hl-glm5-...`` workload).
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
# Dynamo (DynamoDeployment) idle-pod body
#
# The Dynamo backend reuses the RayJob "long-lived pod + external server
# restart" pattern, but with a SaFE DynamoDeployment (LeaderWorkerSet-backed
# multi-node worker) instead of a RayJob, and SSH instead of the Ray Dashboard
# control plane. The worker pods are deployed IDLE (entryPoint =
# ``/usr/local/bin/mn-idle.sh``, which starts sshd then blocks) so the
# optimizer can SSH in and (re)launch sglang/vllm with per-round flags without
# redeploying the workload (preserving the aiter JIT cache across restarts).
#
# See multi_node/SKILL.md (Dynamo section) and docs/apis/dynamo-options-design.md.

# Frontend HTTP port — dynamo.frontend listens here; benchmarks target this
# (NOT sglang rank-0 :8888) so the OpenAI-compatible router fronts every
# worker registration. Matches the SaFE Dynamo fixtures.
_DYNAMO_FRONTEND_PORT = 8000

# Idle worker entrypoint baked into the sshd image layer (docker/dynamo/
# mn-idle.sh). Starts the SSH control plane then `tail -f /dev/null`; it
# ignores positional args so the dispatcher's appended sglang multi-node
# flags (--nnodes/--node-rank/--dist-init-addr) are inert in idle mode.
_DYNAMO_IDLE_WORKER_ENTRYPOINT = "/usr/local/bin/mn-idle.sh"

# Frontend launch command (role 0). round-robin router is the simplest mode;
# the optimizer benchmarks through this OpenAI-compatible endpoint.
_DYNAMO_FRONTEND_ENTRYPOINT_TMPL = (
    "python3 -m dynamo.frontend --http-port {port} --router-mode round-robin"
)

# Valid enum values mirrored from the webhook (validateDynamoDeployment).
_DYNAMO_BACKENDS = frozenset({"sglang", "vllm", "trtllm"})
_DYNAMO_KV_BACKENDS = frozenset({"nixl", "mori", "mooncake"})


def build_dynamo_workload_body(
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
    backend_framework: str = "sglang",
    kv_transfer_backend: str = "nixl",
    ssh_port: int = 2222,
    shared_mem_gi: int = 200,
    rdma_resource: str = "1",
    frontend_cpu: int = 4,
    frontend_mem_gi: int = 16,
    frontend_port: int = _DYNAMO_FRONTEND_PORT,
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
    """Build a SaFE CreateWorkloadRequest body for an IDLE multi-node Dynamo
    deployment (frontend + LeaderWorkerSet worker).

    Topology: ``resources = [frontend, worker]`` with
    ``dynamoOptions.serviceRoles = ["frontend", "worker"]``. When ``nodes >= 2``
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
    if not ssh_authorized_key or not ssh_authorized_key.strip():
        raise ValueError(
            "ssh_authorized_key is required for the Dynamo idle-pod control "
            "plane (injected as MN_SSH_AUTHORIZED_KEY for mn-sshd-init.sh)"
        )
    bf = (backend_framework or "sglang").lower()
    if bf not in _DYNAMO_BACKENDS:
        raise ValueError(
            f"backend_framework must be one of {sorted(_DYNAMO_BACKENDS)}, got {bf!r}"
        )
    kvb = (kv_transfer_backend or "nixl").lower()
    if kvb not in _DYNAMO_KV_BACKENDS:
        raise ValueError(
            f"kv_transfer_backend must be one of {sorted(_DYNAMO_KV_BACKENDS)}, got {kvb!r}"
        )

    # Frontend (role 0): CPU-only OpenAI-compatible router/HTTP server.
    frontend_resource = {
        "replica": 1,
        "cpu": str(frontend_cpu),
        "memory": f"{frontend_mem_gi}Gi",
    }

    def _gpu_resource(replica: int, *, multinode: bool) -> dict[str, Any]:
        """One GPU pod slot (worker / prefill / decode). RDMA only when the
        role spans nodes (a single-node role has no cross-node NCCL/KV)."""
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

    frontend_ep = _b64(_DYNAMO_FRONTEND_ENTRYPOINT_TMPL.format(port=frontend_port))
    idle_ep = _b64(_DYNAMO_IDLE_WORKER_ENTRYPOINT)

    is_pd = (pd_mode or "aggregated").lower() == "disaggregated"
    if is_pd:
        # PD disaggregation: roles = [frontend, prefill, decode]. A role spans
        # nodes (LeaderWorkerSet) when its TP exceeds one pod's GPUs; otherwise
        # its replica is an independent single-node instance count. Both pod
        # roles deploy IDLE (mn-idle.sh) — restart-server SSH-launches
        # dynamo.sglang with --disaggregation-mode prefill/decode per group.
        pn = max(1, int(pd_prefill_nodes or 0))
        dn = max(1, int(pd_decode_nodes or 0))
        ptp = int(pd_prefill_tp or 0)
        dtp = int(pd_decode_tp or 0)
        # A role spans nodes (LeaderWorkerSet) ONLY when its TP exceeds one
        # pod's GPUs. Otherwise its replica is an independent single-node
        # instance count (matches the canonical PD body: replica=2, TP=8, no
        # multinodeRoles -> 2 standalone prefill + 2 standalone decode).
        prefill_mn = ptp > gpus_per_node
        decode_mn = dtp > gpus_per_node
        resources = [
            frontend_resource,
            _gpu_resource(pn, multinode=prefill_mn),
            _gpu_resource(dn, multinode=decode_mn),
        ]
        images = [image, image, image]
        entry_points = [frontend_ep, idle_ep, idle_ep]
        service_roles = ["frontend", "prefill", "decode"]
        multinode_roles = (
            (["prefill"] if prefill_mn else [])
            + (["decode"] if decode_mn else [])
        )
    else:
        # Aggregated: [frontend, worker]. worker.replica == node count; the
        # dispatcher sets multinode.numberOfNodes and forces replicas=1.
        resources = [frontend_resource, _gpu_resource(nodes, multinode=nodes > 1)]
        images = [image, image]
        entry_points = [frontend_ep, idle_ep]
        service_roles = ["frontend", "worker"]
        multinode_roles = ["worker"] if nodes > 1 else []

    # env: SSH control-plane knobs + user extra_env (reserved keys stripped).
    env: dict[str, str] = _sanitize_extra_env(extra_env)
    env["MN_SSH_AUTHORIZED_KEY"] = ssh_authorized_key.strip()
    env["MN_SSH_PORT"] = str(ssh_port)

    labels = _sanitize_extra_labels(extra_labels)
    if session_id:
        labels["primus-claw/session-id"] = session_id

    dynamo_options: dict[str, Any] = {
        "backendFramework": bf,
        "kvTransferBackend": kvb,
        "serviceRoles": service_roles,
    }
    if multinode_roles:
        dynamo_options["multinodeRoles"] = multinode_roles

    body: dict[str, Any] = {
        "displayName": display_name,
        "workspaceId": workspace,
        "groupVersionKind": {
            "kind": "DynamoDeployment",
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
        "dynamoOptions": dynamo_options,
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
