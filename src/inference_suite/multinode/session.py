# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-node launch of a session on a node list's cluster, InferenceX read from GitHub."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from inference_suite.contracts.base_launch import BaseLaunch, LaunchQuery
from inference_suite.multinode.inferencex import CHECKOUT_DIRECTORIES
from inference_suite.multinode.launch import Cluster
from inference_suite.multinode.nodes import NodeList, NodeListError, RoleTopology
from inference_suite.multinode.recipe import ResolveInputs, resolve
from inference_suite.multinode.record import PdLaunch, Role, read_record
from inference_suite.multinode.ssh import SshClient
from inference_suite.recipe.inferencex import INFERENCEX_REPO
from inference_suite.recipe.resolve import Miss, resolve_base
from inference_suite.recipe.sources import Checkout, checkout_commit, checkout_main, fetch_sources


def worker_base(session: LaunchQuery, nodes: NodeList, inferencex_commit: str = "") -> BaseLaunch | Miss:
    """The single-node base launch the suite resolves for ``session`` on the GPUs of every worker host, at the TP
    and EP the node list fixes for the workers over the session's."""
    fixed = nodes.topology.get(Role.WORKER, RoleTopology())
    query = dataclasses.replace(session, tp=fixed.tp or session.tp, ep=fixed.ep or session.ep)
    snapshot = fetch_sources(query, inferencex_commit=inferencex_commit)
    return resolve_base(query, snapshot, gpus=len(nodes.worker) * nodes.gpus_per_node)


def resolve_session(
    session: LaunchQuery, nodes: NodeList, replay: PdLaunch | None = None, inferencex_commit: str = ""
) -> PdLaunch:
    """The launch the sources give ``session`` on ``nodes``: ``replay`` when given, else the node list's operator
    args, else for a disaggregated cluster InferenceX at ``inferencex_commit`` (the head of ``main`` when empty), else
    the PD default table, and for an aggregated one the single-node base launch of its workers.

    Raises:
        NodeListError: the session runs another framework than the node list.
        PdResolveError: no source serves the session, or the replayed launch does not fit it.
    """
    with tempfile.TemporaryDirectory(prefix="pd-inferencex-") as scratch:
        into = Path(scratch) / "InferenceX"

        def fetch() -> Checkout:
            if inferencex_commit:
                return checkout_commit(INFERENCEX_REPO, inferencex_commit, CHECKOUT_DIRECTORIES, into)
            return checkout_main(INFERENCEX_REPO, CHECKOUT_DIRECTORIES, into)

        inputs = ResolveInputs(
            operator=nodes.operator,
            replay=replay,
            inferencex=fetch,
            pinned=bool(inferencex_commit),
            base=lambda: worker_base(session, nodes, inferencex_commit),
        )
        return resolve(nodes.query(session), inputs)


def cluster_of(nodes: NodeList, record: Path) -> Cluster:
    """The cluster of ``nodes``, host keys pinned in the node list's known_hosts file or beside ``record``.

    Raises:
        NodeListError: the node list names no SSH access.
    """
    if nodes.ssh is None:
        raise NodeListError("the node list names no SSH access")
    known = Path(nodes.ssh.known_hosts_file) if nodes.ssh.known_hosts_file else record.parent / "known_hosts"
    return Cluster(nodes=nodes, client=SshClient(nodes.ssh, known))


def placed_launch(nodes: NodeList, record: Path) -> PdLaunch:
    """The launch at ``record`` on the node list's hosts.

    Raises:
        NodeListError: the launch is of another mode than the node list's cluster.
    """
    launch = read_record(record)
    if launch.mode is not nodes.mode:
        raise NodeListError(f"the launch at {record} is {launch.mode.value}, the node list {nodes.mode.value}")
    return launch.with_hosts(nodes.role_hosts(), nodes.router)


__all__ = ["cluster_of", "placed_launch", "resolve_session", "worker_base"]
