# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-node launch of a session on a node list's cluster, InferenceX read from GitHub."""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path

from inference_suite.contracts.base_launch import (
    BaseLaunch,
    ClientLaunch,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    SourceKind,
    Topology,
)
from inference_suite.frameworks import Framework
from inference_suite.multinode.inferencex import CHECKOUT_DIRECTORIES
from inference_suite.multinode.launch import Cluster
from inference_suite.multinode.nodes import NodeList, NodeListError, RoleTopology
from inference_suite.multinode.recipe import ResolveInputs, resolve
from inference_suite.multinode.record import PdLaunch, PdMode, PdSourceKind, Role, read_record
from inference_suite.multinode.ssh import SshClient
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT, INFERENCEX_TOOL
from inference_suite.recipe.image_version import NO_RECIPE_IMAGE
from inference_suite.recipe.inferencex import INFERENCEX_REPO
from inference_suite.recipe.model import ATOM_ENTRYPOINT, SGLANG_ENTRYPOINT, VLLM_ENTRYPOINT
from inference_suite.recipe.resolve import Miss, resolve_base
from inference_suite.recipe.sources import Checkout, checkout_commit, checkout_main, fetch_sources

_ENTRYPOINTS = {Framework.SGLANG: SGLANG_ENTRYPOINT, Framework.VLLM: VLLM_ENTRYPOINT, Framework.ATOM: ATOM_ENTRYPOINT}
#: The PD sources an operator wrote; every other one is read from a table or recipe tree.
_OPERATOR_SOURCES = frozenset({PdSourceKind.OPERATOR, PdSourceKind.REPLAY})


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


def client_base(launch: PdLaunch, session: LaunchQuery) -> BaseLaunch:
    """The base a round's client measures ``launch`` with: its framework, model and serving role's topology, and the
    pinned InferenceX client. It carries no server layer, since the cluster serves ``launch`` itself."""
    role = launch.role(Role.WORKER if launch.mode is PdMode.AGGREGATED else Role.DECODE)
    kind = SourceKind.USER_FILE if launch.source.kind in _OPERATOR_SOURCES else SourceKind.DEFAULT_TABLE
    return BaseLaunch(
        query=dataclasses.replace(session, tp=None, ep=None),
        source=LaunchSource(kind=kind, strategy=f"{launch.source.kind.value} {launch.digest()}"),
        point=LaunchPoint(conc=session.concurrencies[0], exact=True),
        image=NO_RECIPE_IMAGE,
        framework=launch.framework.value,
        entrypoint=_ENTRYPOINTS[launch.framework],
        model=LaunchModel(session.model_id, launch.model),
        topology=Topology(tp=role.tp, ep=role.ep),
        args=(),
        env={},
        client=ClientLaunch(tool=INFERENCEX_TOOL, commit=INFERENCEX_CLIENT_COMMIT),
        resolved_unix=int(time.time()),
    )


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


__all__ = ["client_base", "cluster_of", "placed_launch", "resolve_session", "worker_base"]
