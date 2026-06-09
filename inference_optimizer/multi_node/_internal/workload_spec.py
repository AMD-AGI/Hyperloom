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
  ``env.RAY_JOB_ENTRYPOINT`` (base64 ``tail -f /dev/null``), not entryPoints.
* ``env`` merges user ``extra_env`` (reserved keys stripped); inject debug
  knobs via ``--extra-env``. ``RAY_JOB_ENTRYPOINT`` can't be overridden.
* ``labels`` strip Brain-managed prefixes; ``primus-claw/session-id`` is
  injected from ``session_id`` for Brain correlation.
"""

from __future__ import annotations

import base64
from typing import Any

# Hard-coded per design (Brain TS impl + SKILL.md contract).
_INFERENCE_SERVER_PORT = 8888
_HEAD_ROLE_LABEL = "primus-safe.amd.com/ray-role"
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
    """Build a SaFE CreateWorkloadRequest body for a multi-node RayJob (resource quantities as K8s-notation strings; json.dumps-safe)."""
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

    # head replica is 1; worker is N-1, clamped to >=1 (SaFE requires every resources[] replica >=1).
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

    # env: user extra_env (reserved keys stripped); no debug knobs injected here.
    env: dict[str, str] = _sanitize_extra_env(extra_env)
    env["RAY_JOB_ENTRYPOINT"] = _b64(_SUBMITTER_BLOCK_ENTRYPOINT)

    # labels: sanitized caller labels + injected ``primus-claw/session-id`` for
    # Brain correlation. (SaFE strips ``primus-safe.amd.com/*``, verified May 2026.)
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
