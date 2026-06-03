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
    """Build a SaFE CreateWorkloadRequest body for a multi-node RayJob.

    All resource quantities are emitted as STRING values per SaFE
    convention. Quantities follow Kubernetes resource notation
    (e.g. ``"32"`` for CPUs, ``"128Gi"`` for memory).

    The returned dict is safe to pass to ``json.dumps`` — every value is
    a primitive, list of primitives, or nested dict of the same.

    Args:
        workspace (str): SaFE workspace id the workload belongs to.
        display_name (str): Human-readable workload name.
        image (str): Container image used for both head and worker roles.
        nodes (int): Total number of Ray nodes (head + workers); must be >= 1.
        gpus_per_node (int): GPUs requested per node; must be >= 1.
        cpus_per_node (int): CPUs requested per node.
        mem_gi_per_node (int): Memory per node in GiB.
        ephemeral_gi_per_node (int): Ephemeral storage per node in GiB.
        description (str | None): Optional workload description.
        owner_id (str | None): Optional owner identifier.
        session_id (str | None): Optional parent sandbox session id; when set,
            a ``primus-claw/session-id`` label is injected for correlation.
        extra_env (dict[str, str] | None): Extra environment variables merged
            into the workload env (reserved keys are stripped).
        extra_labels (dict[str, str] | None): Extra labels merged into the
            workload labels (reserved-prefix keys are stripped).

    Returns:
        dict[str, Any]: A JSON-serializable SaFE CreateWorkloadRequest body.

    Raises:
        ValueError: If ``nodes`` < 1, ``gpus_per_node`` < 1, or any of
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
