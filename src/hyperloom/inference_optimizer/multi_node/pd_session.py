# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sessions on a cluster handed over as a node list, PD-disaggregated or aggregated: its state, its frozen launch, its
rounds."""

from __future__ import annotations

import logging
import os
import shlex
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.multinode import (
    Cluster,
    NODE_LIST_ENV,
    RAN_RECORD_NAME,
    RECORD_NAME,
    LaunchError,
    LaunchOptions,
    NodeList,
    NodeListError,
    PdLaunch,
    PdMode,
    PdRecordError,
    PdResolveError,
    Role,
    SshError,
    cluster_commands,
    cluster_of,
    exportable_env_key,
    placed_launch,
    read_record,
    replace_flags,
    resolve_session,
    restart,
    split_launcher_flags,
    stop,
    write_record,
)

from ._internal.pd_handoff import (
    INFERA,
    PD_BACKENDS,
    PODS_KEY,
    SSH_BACKENDS,
    handed_node_list,
    is_pd_state,
    node_list_backend,
    pod,
)

log = logging.getLogger(__name__)

PD_LAUNCH_KEY = "pd_launch"
PD_DIGEST_KEY = "pd_launch_digest"
PREFILL_EXTRA_ENV = "PD_PREFILL_EXTRA_ARGS"
DECODE_EXTRA_ENV = "PD_DECODE_EXTRA_ARGS"
SERVER_LOG_DIR_ENV = "HYPERLOOM_MN_SERVER_LOG_DIR"
PROFILER_DIR_ENV = "SGLANG_TORCH_PROFILER_DIR"

_SUITE_ERRORS = (LaunchError, NodeListError, PdRecordError, PdResolveError, SshError, OSError, ValueError)


class PdSessionError(RuntimeError):
    """A PD session cannot resolve, verify, start or stop its launch."""


@contextmanager
def _as_session_error(what: str) -> Iterator[None]:
    """Re-raise a suite error of the block as a :class:`PdSessionError` prefixed with ``what``."""
    try:
        yield
    except _SUITE_ERRORS as exc:
        raise PdSessionError(f"{what}: {exc}") from exc


def freeze_launch(session_dir: Path, session: LaunchQuery, nodes: NodeList, inferencex_commit: str = "") -> PdLaunch:
    """Resolve the session's PD launch on ``nodes`` and freeze it as ``session_dir/pd_launch.json``.

    Raises:
        PdSessionError: no source serves the session, or the node list does not fit it.
    """
    with _as_session_error("the PD launch could not be resolved"):
        launch = resolve_session(session, nodes, None, inferencex_commit)
        write_record(launch, session_dir / RECORD_NAME)
    return launch


def verified_launch(record: Path, digest: str) -> PdLaunch:
    """The frozen launch at ``record``, refused when it is not the one whose digest the session recorded.

    Raises:
        PdSessionError: the record is missing, unreadable, or carries another digest.
    """
    try:
        launch = read_record(record)
    except (PdRecordError, OSError) as exc:
        raise PdSessionError(f"the frozen PD launch {record} cannot be read: {exc}") from exc
    if digest and launch.digest() != digest:
        raise PdSessionError(f"the frozen PD launch {record} is {launch.digest()}, but the session froze {digest}")
    return launch


def _round_overrides(shared: str, prefill: str, decode: str, *, mode: PdMode) -> dict[str, list[str]]:
    """Per-role overrides of one round: ``shared`` for every engine, each PD role's own args replacing its flags; an
    aggregated launch's workers take ``shared`` alone."""
    common = shlex.split(shared)
    if mode is PdMode.AGGREGATED:
        return {Role.WORKER.value: common} if common else {}
    given = {
        Role.PREFILL.value: replace_flags(common, shlex.split(prefill)),
        Role.DECODE.value: replace_flags(common, shlex.split(decode)),
    }
    return {role: list(args) for role, args in given.items() if args}


def launch_options(forward_env: Mapping[str, str], ready_timeout_s: int) -> LaunchOptions:
    """The options a round runs with: ``forward_env`` exported to every engine (shell-safe keys the runner does not
    reserve), its server-log directory shared by the engines, and its torch profiler directory narrowed to
    ``<dir>/<role>`` for each engine role (the router gets none)."""
    env = {key: value for key, value in forward_env.items() if exportable_env_key(key)}
    log_dir = env.pop(SERVER_LOG_DIR_ENV, "")
    profiler_dir = env.pop(PROFILER_DIR_ENV, "").rstrip("/")
    roles = (Role.PREFILL.value, Role.DECODE.value)
    role_env = {role: {PROFILER_DIR_ENV: f"{profiler_dir}/{role}"} for role in roles} if profiler_dir else {}
    return LaunchOptions(extra_env=env, role_env=role_env, log_dir=log_dir, ready_timeout_s=ready_timeout_s)


class _SessionCluster(NamedTuple):
    """The frozen record, the launch placed on the node list's hosts, and the cluster to run it on."""

    record: Path
    launch: PdLaunch
    cluster: Cluster


def _session_cluster(state: dict[str, Any]) -> _SessionCluster:
    """The session's frozen launch on the handed-over cluster.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch is not the one the session recorded.
    """
    nodes = handed_node_list()
    if nodes is None:
        raise PdSessionError(f"a node-list session needs {NODE_LIST_ENV}, which is unset")
    record = Path(str(state.get(PD_LAUNCH_KEY) or ""))
    verified_launch(record, str(state.get(PD_DIGEST_KEY) or ""))
    return _SessionCluster(record, placed_launch(nodes, record), cluster_of(nodes, record))


def restart_round(state: dict[str, Any], shared_args: str, options: LaunchOptions) -> PdLaunch:
    """Restart the session's cluster with one round's args over the frozen launch; returns the launch as run.

    ``shared_args`` apply to every engine; ``$PD_PREFILL_EXTRA_ARGS`` and ``$PD_DECODE_EXTRA_ARGS`` to their own PD
    role.

    Raises:
        PdSessionError: the launch cannot be verified, a host refused, or the cluster did not come up.
    """
    with _as_session_error("the PD cluster did not come up"):
        session = _session_cluster(state)
        overrides = _round_overrides(
            shared_args,
            os.environ.get(PREFILL_EXTRA_ENV, ""),
            os.environ.get(DECODE_EXTRA_ENV, ""),
            mode=session.launch.mode,
        )
        ran = restart(session.launch, _without_launcher_flags(session.launch, overrides), session.cluster, options)
        write_record(ran, session.record.with_name(RAN_RECORD_NAME))
    return ran


def _without_launcher_flags(launch: PdLaunch, overrides: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """``overrides`` less the flags the launcher owns, which the frozen launch keeps; logs what each role dropped."""
    kept: dict[str, list[str]] = {}
    for role, args in overrides.items():
        role_args, dropped = split_launcher_flags(launch.framework, args)
        if dropped:
            log.warning("pd round: %s keeps the frozen launch's %s; the round's values are dropped", role, dropped)
        if role_args:
            kept[role] = list(role_args)
    return kept


def stop_cluster(state: dict[str, Any]) -> None:
    """Stop the session cluster's router and engines.

    Raises:
        PdSessionError: the launch cannot be verified, or a host refused.
    """
    with _as_session_error("the PD cluster could not be stopped"):
        session = _session_cluster(state)
        stop(session.launch, session.cluster)


def stamp_launch_pods(state: dict[str, Any], launch: PdLaunch | None = None) -> None:
    """Record every host of ``launch`` (the session's frozen launch by default) in ``state``'s ``pods`` as a pod with
    the command it serves, the shape readers of the multi-node state such as GEAK take a cluster in.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch cannot be verified or placed.
    """
    with _as_session_error("the PD launch cannot be placed"):
        session = _session_cluster(state)
        nodes = session.cluster.nodes
        commands = cluster_commands(launch or session.launch, nodes)
    indices: dict[str, int] = {}
    pods: list[dict[str, Any]] = []
    for command in commands:
        index = indices[command.role] = indices.get(command.role, -1) + 1
        pods.append({**pod(nodes, command.role, index, command.host), "launch_command": command.command})
    state[PODS_KEY] = pods


__all__ = [
    "INFERA",
    "PD_BACKENDS",
    "PD_DIGEST_KEY",
    "PD_LAUNCH_KEY",
    "SSH_BACKENDS",
    "PdSessionError",
    "freeze_launch",
    "handed_node_list",
    "is_pd_state",
    "launch_options",
    "node_list_backend",
    "restart_round",
    "stamp_launch_pods",
    "stop_cluster",
    "verified_launch",
]
