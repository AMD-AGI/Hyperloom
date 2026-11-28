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
#: The multi-node state keys recording whether the engine hosts share the profile trace root, and why not.
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


def is_pd_backend(backend: str, *, node_list: bool) -> bool:
    """Whether ``backend`` serves a node-list cluster the suite launches: atomesh or sglangrouter, or Infera when the
    cluster is handed over as a node list."""
    return backend in PD_BACKENDS or (backend == INFERA and node_list)


def is_pd_state(state: dict[str, Any]) -> bool:
    """Whether a multi-node state describes a node-list cluster the suite launches."""
    return is_pd_backend(str(state.get("backend") or ""), node_list=state.get(NODE_LIST_KEY) is True)


def is_ssh_backed(state: dict[str, Any]) -> bool:
    """Whether a cluster's GPU hosts are reached over SSH: an Infera idle-pod cluster, or a node-list PD one."""
    return str(state.get("backend") or "") == INFERA or is_pd_state(state)


def pod(nodes: NodeList, role: str, index: int, host: str) -> dict[str, Any]:
    """The SSH target of ``host``, the ``index``-th host of ``role``."""
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
    ssh_port = nodes.ssh.port if nodes.ssh else 0
    state: dict[str, Any] = {
        "backend": backend,
        "external": True,
        NODE_LIST_KEY: True,
        "service_url": nodes.service_url,
        "nodes": len(set(prefill + decode + worker)),
        "gpus_per_node": nodes.gpus_per_node,
        "pd_mode": nodes.mode.value,
        "ssh_port": ssh_port,
        "router_pod_ip": nodes.router,
        **{f"{role.value}_pod_ips": list(nodes.hosts(role)) for role in Role},
        **{
            f"{role.value}_pods": [pod(nodes, role.value, index, host) for index, host in enumerate(nodes.hosts(role))]
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
    "NODE_LIST_KEY",
    "PODS_KEY",
    "PD_BACKENDS",
    "SSH_BACKENDS",
    "handed_node_list",
    "is_pd_backend",
    "is_pd_state",
    "is_ssh_backed",
    "node_list_backend",
    "pd_state",
    "pod",
]
