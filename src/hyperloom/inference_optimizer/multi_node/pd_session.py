# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD-disaggregated sessions on a cluster handed over as a node list: its state, its frozen launch, its rounds."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.multinode import (
    Cluster,
    NODE_LIST_ENV,
    RECORD_NAME,
    LaunchError,
    LaunchOptions,
    NodeList,
    NodeListError,
    PdLaunch,
    PdRecordError,
    PdResolveError,
    SshError,
    cluster_of,
    engines_gone,
    placed_launch,
    read_record,
    replace_flags,
    resolve_session,
    restart,
    stop,
    write_record,
)

from ._internal.pd_handoff import PD_BACKENDS, handed_node_list, is_pd_state, node_list_backend, pd_state

PD_LAUNCH_KEY = "pd_launch"
PD_DIGEST_KEY = "pd_launch_digest"
#: The launch as the last round ran it, beside the frozen ``pd_launch.json``.
RAN_RECORD_NAME = "pd_launch.ran.json"
PREFILL_EXTRA_ENV = "PD_PREFILL_EXTRA_ARGS"
DECODE_EXTRA_ENV = "PD_DECODE_EXTRA_ARGS"
SERVER_LOG_DIR_ENV = "HYPERLOOM_MN_SERVER_LOG_DIR"

_PREFILL = "prefill"
_DECODE = "decode"
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_ENV_PREFIX = "HL_"
_SUITE_ERRORS = (LaunchError, NodeListError, PdRecordError, PdResolveError, SshError, OSError)


class PdSessionError(RuntimeError):
    """A PD session cannot resolve, verify, start or stop its launch."""


def freeze_launch(session_dir: Path, session: LaunchQuery, nodes: NodeList, inferencex_commit: str = "") -> PdLaunch:
    """Resolve the session's PD launch on ``nodes`` and freeze it as ``session_dir/pd_launch.json``.

    Raises:
        PdSessionError: no source serves the session, or the node list does not fit it.
    """
    try:
        launch = resolve_session(session, nodes, None, inferencex_commit)
        write_record(launch, session_dir / RECORD_NAME)
    except (*_SUITE_ERRORS, ValueError) as exc:
        raise PdSessionError(f"the PD launch could not be resolved: {exc}") from exc
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


def round_overrides(shared: str, prefill: str = "", decode: str = "") -> dict[str, list[str]]:
    """Per-role overrides of one round: ``shared`` for both engines, each role's own args replacing its flags."""
    common = shlex.split(shared)
    given = {_PREFILL: replace_flags(common, shlex.split(prefill)), _DECODE: replace_flags(common, shlex.split(decode))}
    return {role: list(args) for role, args in given.items() if args}


def launch_options(forward_env: Mapping[str, str], ready_timeout_s: int) -> LaunchOptions:
    """The options a round runs with: ``forward_env`` exported to every engine (shell-safe keys the runner does not
    reserve), its server-log directory shared by the engines."""
    env = {
        key: value
        for key, value in forward_env.items()
        if _ENV_KEY.fullmatch(key) and not key.startswith(_RESERVED_ENV_PREFIX)
    }
    log_dir = env.pop(SERVER_LOG_DIR_ENV, "")
    return LaunchOptions(extra_env=env, log_dir=log_dir, ready_timeout_s=ready_timeout_s)


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
        raise PdSessionError(f"a PD session needs {NODE_LIST_ENV}, which is unset")
    record = Path(str(state.get(PD_LAUNCH_KEY) or ""))
    verified_launch(record, str(state.get(PD_DIGEST_KEY) or ""))
    return _SessionCluster(record, placed_launch(nodes, record), cluster_of(nodes, record))


def restart_round(state: dict[str, Any], shared_args: str, options: LaunchOptions) -> PdLaunch:
    """Restart the session's cluster with one round's args over the frozen launch; returns the launch as run.

    ``shared_args`` apply to both engines; ``$PD_PREFILL_EXTRA_ARGS`` and ``$PD_DECODE_EXTRA_ARGS`` to their own.

    Raises:
        PdSessionError: the launch cannot be verified, a host refused, or the cluster did not come up.
    """
    overrides = round_overrides(
        shared_args, os.environ.get(PREFILL_EXTRA_ENV, ""), os.environ.get(DECODE_EXTRA_ENV, "")
    )
    try:
        session = _session_cluster(state)
        ran = restart(session.launch, overrides, session.cluster, options)
        write_record(ran, session.record.with_name(RAN_RECORD_NAME))
    except _SUITE_ERRORS as exc:
        raise PdSessionError(f"the PD cluster did not come up: {exc}") from exc
    return ran


def stop_cluster(state: dict[str, Any]) -> None:
    """Stop the session cluster's router and engines.

    Raises:
        PdSessionError: the launch cannot be verified, or a host refused.
    """
    try:
        session = _session_cluster(state)
        stop(session.launch, session.cluster, LaunchOptions())
    except _SUITE_ERRORS as exc:
        raise PdSessionError(f"the PD cluster could not be stopped: {exc}") from exc


def engines_lost(state: dict[str, Any]) -> str:
    """Why the session's engines are no longer served (``""`` when every one still runs, or none can be checked)."""
    try:
        session = _session_cluster(state)
        gone = engines_gone(session.launch, session.cluster)
    except (*_SUITE_ERRORS, PdSessionError):
        return ""
    return ", ".join(f"{command.role} engine on {command.host} is gone" for command in gone)


__all__ = [
    "DECODE_EXTRA_ENV",
    "PD_BACKENDS",
    "PD_DIGEST_KEY",
    "PD_LAUNCH_KEY",
    "PREFILL_EXTRA_ENV",
    "RAN_RECORD_NAME",
    "PdSessionError",
    "engines_lost",
    "freeze_launch",
    "handed_node_list",
    "is_pd_state",
    "launch_options",
    "node_list_backend",
    "pd_state",
    "restart_round",
    "round_overrides",
    "stop_cluster",
    "verified_launch",
]
