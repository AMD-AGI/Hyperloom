# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The PD launch of a session on a node list's cluster, InferenceX read from GitHub."""

from __future__ import annotations

import tempfile
from pathlib import Path

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.multinode.inferencex import CHECKOUT_DIRECTORIES
from inference_suite.multinode.launch import Cluster
from inference_suite.multinode.nodes import NodeList, NodeListError
from inference_suite.multinode.recipe import ResolveInputs, resolve
from inference_suite.multinode.record import PdLaunch, Role, read_record
from inference_suite.multinode.ssh import SshClient
from inference_suite.recipe.inferencex import INFERENCEX_REPO
from inference_suite.recipe.sources import Checkout, checkout_commit, checkout_main


def resolve_session(
    session: LaunchQuery, nodes: NodeList, replay: PdLaunch | None = None, inferencex_commit: str = ""
) -> PdLaunch:
    """The launch the sources give ``session`` on ``nodes``: ``replay`` when given, else the node list's operator
    args, else InferenceX at ``inferencex_commit`` (the head of ``main`` when empty), else the PD default table.

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

        inputs = ResolveInputs(operator=nodes.operator, replay=replay, inferencex=fetch, pinned=bool(inferencex_commit))
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
    """The launch at ``record`` on the node list's hosts."""
    launch = read_record(record)
    return launch.with_hosts(nodes.hosts(Role.PREFILL), nodes.hosts(Role.DECODE), nodes.router)


__all__ = ["cluster_of", "placed_launch", "resolve_session"]
