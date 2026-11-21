# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cluster the platform hands over as a node list (PD-disaggregated or aggregated), and the multi-node state it
describes."""

from __future__ import annotations

import os
from typing import Any

from inference_suite.multinode import NODE_LIST_ENV, NodeList, NodeListError, Role, read_node_list

from ...session.paths import workspace_root
from .ssh_client import SSH_USER
from .topology_file import TopologyHandoffError

ATOMESH = "atomesh"
SGLANGROUTER = "sglangrouter"
INFERA = "infera"
PD_BACKENDS = (ATOMESH, SGLANGROUTER)
#: The backends whose GPU hosts Hyperloom reaches over SSH.
SSH_BACKENDS = (INFERA, *PD_BACKENDS)
#: The multi-node state key that marks a node-list cluster.
NODE_LIST_KEY = "node_list"
#: Every host of a node-list cluster's launch with the command it serves.
PODS_KEY = "pods"
#: The pod keys a node-list session writes from its launch, which a reload of the same hand-off keeps.
LAUNCH_POD_KEYS = (PODS_KEY, "prefill_pods", "decode_pods", "worker_pods", "router_pods")
PD_LAUNCH_KEY = "pd_launch"
PD_DIGEST_KEY = "pd_launch_digest"
TRACE_SHARED_KEY = "profile_trace_shared"
TRACE_SHARE_ERROR_KEY = "profile_trace_share_error"

_FRONTEND_BACKEND = {"atomesh": ATOMESH, "sglang-router": SGLANGROUTER, "infera": INFERA}
_FRAMEWORK_BACKEND = {"atom": ATOMESH, "sglang": SGLANGROUTER}


def handed_node_list() -> NodeList | None:
    """The node list ``$HYPERLOOM_MN_NODE_LIST`` hands over, or ``None`` when it is unset.

    Raises:
        TopologyHandoffError: the node list is malformed, or names an SSH user other than root.
    """
    if not os.environ.get(NODE_LIST_ENV, "").strip():
        return None
    try:
        nodes = read_node_list(os.environ)
    except NodeListError as exc:
        raise TopologyHandoffError(f"{NODE_LIST_ENV}: {exc}") from exc
    if nodes.ssh is not None and nodes.ssh.user != SSH_USER:
        raise TopologyHandoffError(
            f"{NODE_LIST_ENV}: ssh.user is {nodes.ssh.user!r}, but Hyperloom sessions connect as {SSH_USER}"
        )
    return nodes


def node_list_backend(nodes: NodeList) -> str:
    """The backend a node list's frontend, else its framework, names."""
    if nodes.frontend is not None:
        return _FRONTEND_BACKEND[nodes.frontend.value]
    return _FRAMEWORK_BACKEND[nodes.framework.value]


def is_pd_state(state: dict[str, Any]) -> bool:
    """Whether a multi-node state describes a node-list cluster the suite launches: an atomesh, sglangrouter, or
    Infera one handed over as a node list."""
    return str(state.get("backend") or "") in SSH_BACKENDS and state.get(NODE_LIST_KEY) is True


def ssh_pod(nodes: NodeList, host: str, role: str, index: int) -> dict[str, Any]:
    """The SSH target of the ``index``-th host of ``role``."""
    return {
        "podIP": host,
        "podId": f"external-{role}-{index}",
        "role": role,
        "lwsIndex": index,
        "sshPort": nodes.ssh.port_of(host) if nodes.ssh else 0,
    }


def pd_state(nodes: NodeList, backend: str) -> dict[str, Any]:
    """The multi-node state of a node-list cluster served by ``backend``."""
    prefill, decode, worker = nodes.prefill, nodes.decode, nodes.worker
    state: dict[str, Any] = {
        "backend": backend,
        "external": True,
        NODE_LIST_KEY: True,
        "service_url": nodes.service_url,
        "nodes": len(set(prefill + decode + worker)),
        "gpus_per_node": nodes.gpus_per_node,
        "pd_mode": nodes.mode.value,
        "ssh_port": nodes.ssh.port if nodes.ssh else 0,
        "router_pod_ip": nodes.router,
        **{f"{role.value}_pod_ips": list(nodes.hosts(role)) for role in Role},
        **{
            f"{role.value}_pods": [ssh_pod(nodes, host, role.value, i) for i, host in enumerate(nodes.hosts(role))]
            for role in Role
        },
        "pd_prefill_nodes": len(prefill),
        "pd_decode_nodes": len(decode),
        "user_data_path": str(workspace_root()),
    }
    if nodes.prefill_url:
        state["pd_prefill_url"] = nodes.prefill_url
    if nodes.decode_url:
        state["pd_decode_url"] = nodes.decode_url
    if nodes.ssh is not None:
        state["ssh_key_path"] = nodes.ssh.key_file
        if nodes.ssh.known_hosts_file:
            state["ssh_known_hosts"] = nodes.ssh.known_hosts_file
    return state


__all__ = [
    "INFERA",
    "LAUNCH_POD_KEYS",
    "NODE_LIST_KEY",
    "PODS_KEY",
    "PD_BACKENDS",
    "PD_DIGEST_KEY",
    "PD_LAUNCH_KEY",
    "SSH_BACKENDS",
    "TRACE_SHARED_KEY",
    "TRACE_SHARE_ERROR_KEY",
    "handed_node_list",
    "is_pd_state",
    "node_list_backend",
    "pd_state",
    "ssh_pod",
]
