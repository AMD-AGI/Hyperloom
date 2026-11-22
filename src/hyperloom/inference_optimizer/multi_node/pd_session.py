# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sessions on a cluster handed over as a node list, PD-disaggregated or aggregated: its state, its frozen launch, its
rounds."""

from __future__ import annotations

import logging
import os
import shlex
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from inference_suite.contracts.base_launch import LaunchQuery
from hyperloom.common.env_safety import valid_env_key
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
    engine_port,
    engine_urls,
    engines_gone,
    placed_launch,
    read_record,
    replace_flags,
    resolve_session,
    restart,
    split_launcher_flags,
    stop,
    worker_hosts,
    write_record,
)

from ._internal.pd_handoff import (
    INFERA,
    PD_BACKENDS,
    PD_DIGEST_KEY,
    PD_LAUNCH_KEY,
    PODS_KEY,
    SSH_BACKENDS,
    TRACE_SHARE_ERROR_KEY,
    TRACE_SHARED_KEY,
    handed_node_list,
    is_pd_state,
    node_list_backend,
    ssh_pod,
)

log = logging.getLogger(__name__)

ROUTER_ROLE = "router"
#: ``managed_by`` of the router pod: the frontend in front of the platform's service URL.
PLATFORM_MANAGED = "platform"
PREFILL_EXTRA_ENV = "PD_PREFILL_EXTRA_ARGS"
DECODE_EXTRA_ENV = "PD_DECODE_EXTRA_ARGS"
SERVER_LOG_DIR_ENV = "HYPERLOOM_MN_SERVER_LOG_DIR"
PROFILER_DIR_ENV = "SGLANG_TORCH_PROFILER_DIR"
TRACE_SHARE_PROBE_TIMEOUT_S = 30

_PREFILL = "prefill"
_DECODE = "decode"
_WORKER = "worker"
_RESERVED_ENV_PREFIX = "HL_"
_SUITE_ERRORS = (LaunchError, NodeListError, PdRecordError, PdResolveError, SshError, OSError, ValueError)


class PdSessionError(RuntimeError):
    """A PD session cannot resolve, verify, start or stop its launch."""


@contextmanager
def _session_errors(what: str) -> Iterator[None]:
    """Re-raise a suite error as a :class:`PdSessionError` prefixed with ``what``."""
    try:
        yield
    except _SUITE_ERRORS as exc:
        raise PdSessionError(f"{what}: {exc}") from exc


def freeze_launch(session_dir: Path, session: LaunchQuery, nodes: NodeList, inferencex_commit: str = "") -> PdLaunch:
    """Resolve the session's PD launch on ``nodes`` and freeze it as ``session_dir/pd_launch.json``.

    Raises:
        PdSessionError: no source serves the session, or the node list does not fit it.
    """
    with _session_errors("the PD launch could not be resolved"):
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


def round_overrides(
    shared: str, prefill: str = "", decode: str = "", *, mode: PdMode = PdMode.DISAGGREGATED
) -> dict[str, list[str]]:
    """Per-role overrides of one round: ``shared`` for every engine, each PD role's own args replacing its flags; an
    aggregated launch's workers take ``shared`` alone."""
    common = shlex.split(shared)
    if mode is PdMode.AGGREGATED:
        return {_WORKER: common} if common else {}
    given = {_PREFILL: replace_flags(common, shlex.split(prefill)), _DECODE: replace_flags(common, shlex.split(decode))}
    return {role: list(args) for role, args in given.items() if args}


def launch_options(forward_env: Mapping[str, str], ready_timeout_s: int) -> LaunchOptions:
    """The options a round runs with: ``forward_env`` exported to every engine (shell-safe keys the runner does not
    reserve), its server-log directory shared by the engines, and its torch profiler directory narrowed to
    ``<dir>/<role>`` for each engine role (the router gets none)."""
    env = {
        key: value
        for key, value in forward_env.items()
        if valid_env_key(key) and not key.startswith(_RESERVED_ENV_PREFIX)
    }
    log_dir = env.pop(SERVER_LOG_DIR_ENV, "")
    profiler_dir = env.pop(PROFILER_DIR_ENV, "").rstrip("/")
    role_env = (
        {role: {PROFILER_DIR_ENV: f"{profiler_dir}/{role}"} for role in (_PREFILL, _DECODE)} if profiler_dir else {}
    )
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
    with _session_errors("the PD cluster did not come up"):
        session = _session_cluster(state)
        overrides = round_overrides(
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
    with _session_errors("the PD cluster could not be stopped"):
        session = _session_cluster(state)
        stop(session.launch, session.cluster, LaunchOptions())


def session_engine_urls(state: dict[str, Any]) -> dict[str, list[str]]:
    """The HTTP base URL of every prefill and decode engine worker, as the frozen launch is placed on the node list.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch cannot be verified or placed.
    """
    with _session_errors("the PD engines cannot be placed"):
        session = _session_cluster(state)
        return engine_urls(session.launch, session.cluster.nodes)


def session_worker_hosts(state: dict[str, Any]) -> dict[str, list[tuple[str, ...]]]:
    """The hosts of every engine worker per role, each its first host first, in the order of
    :func:`session_engine_urls`.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch cannot be verified or placed.
    """
    with _session_errors("the PD engines cannot be placed"):
        launch = _session_cluster(state).launch
        return {role.value: worker_hosts(launch.role(role)) for role in launch.served_roles}


def stamp_launch_pods(state: dict[str, Any], launch: PdLaunch | None = None) -> None:
    """Record every host of ``launch`` (the session's frozen launch by default) in ``state`` as a pod with the command
    it serves: ``pods`` and each role's ``<role>_pods``, the shape readers of the multi-node state such as GEAK take a
    cluster in.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch cannot be verified or placed.
    """
    with _session_errors("the PD launch cannot be placed"):
        session = _session_cluster(state)
        nodes = session.cluster.nodes
        commands = cluster_commands(launch or session.launch, nodes)
    indices: dict[str, int] = {}
    pods: list[dict[str, Any]] = []
    for command in commands:
        index = indices[command.role] = indices.get(command.role, -1) + 1
        pod = {**ssh_pod(nodes, command.host, command.role, index), "launch_command": command.command}
        if command.role == ROUTER_ROLE:
            pod["managed_by"] = PLATFORM_MANAGED
        else:
            pod["enginePort"] = engine_port(nodes, Role(command.role))
        pods.append(pod)
    state[PODS_KEY] = pods
    for role in (*(role.value for role in Role), ROUTER_ROLE):
        state[f"{role}_pods"] = [pod for pod in pods if pod["role"] == role]


def _host_reads_marker(cluster: Cluster, host: str, marker: Path, token: str) -> str:
    """``""`` when ``host`` reads ``token`` from ``marker`` over SSH, else why it does not."""
    result = cluster.client.run(host, f"cat {shlex.quote(str(marker))}", TRACE_SHARE_PROBE_TIMEOUT_S)
    if result.returncode != 0:
        detail = result.stderr.strip()[-200:]
        return f"engine host {host}: cannot read {marker} (rc={result.returncode}) {detail}".strip()
    if result.stdout.strip() != token:
        return f"engine host {host}: {marker} is a different file there"
    return ""


def probe_trace_share(state: dict[str, Any], trace_root: Path) -> str:
    """``""`` when every engine host reads a marker written here under ``trace_root``, else why not."""
    token = uuid.uuid4().hex
    marker = trace_root / f".share_probe-{token}"
    try:
        trace_root.mkdir(parents=True, exist_ok=True)
        marker.write_text(token, encoding="utf-8")
    except OSError as exc:
        return f"cannot write {marker}: {exc}"
    try:
        session = _session_cluster(state)
        launch = session.launch
        for host in dict.fromkeys(host for role in launch.served_roles for host in launch.role(role).hosts):
            reason = _host_reads_marker(session.cluster, host, marker, token)
            if reason:
                return reason
    except (*_SUITE_ERRORS, PdSessionError) as exc:
        return f"the engine hosts cannot be probed: {exc}"
    finally:
        marker.unlink(missing_ok=True)
    return ""


def record_trace_share(state: dict[str, Any], trace_root: Path) -> str:
    """Record on ``state`` whether the engine hosts share ``trace_root`` (:func:`probe_trace_share`); returns why
    not (``""`` when shared)."""
    reason = probe_trace_share(state, trace_root)
    state[TRACE_SHARED_KEY] = not reason
    if reason:
        state[TRACE_SHARE_ERROR_KEY] = reason
    else:
        state.pop(TRACE_SHARE_ERROR_KEY, None)
    return reason


def engines_lost(state: dict[str, Any]) -> str:
    """Why the session's engines are no longer served (``""`` when every one still runs, or none can be checked)."""
    try:
        session = _session_cluster(state)
        gone = engines_gone(session.launch, session.cluster)
    except (*_SUITE_ERRORS, PdSessionError):
        return ""
    return ", ".join(f"{command.role} engine on {command.host} is gone" for command in gone)


__all__ = [
    "INFERA",
    "PD_BACKENDS",
    "PD_DIGEST_KEY",
    "PD_LAUNCH_KEY",
    "PREFILL_EXTRA_ENV",
    "PROFILER_DIR_ENV",
    "RAN_RECORD_NAME",
    "SSH_BACKENDS",
    "TRACE_SHARED_KEY",
    "TRACE_SHARE_ERROR_KEY",
    "PdSessionError",
    "engines_lost",
    "freeze_launch",
    "handed_node_list",
    "is_pd_state",
    "launch_options",
    "node_list_backend",
    "record_trace_share",
    "restart_round",
    "round_overrides",
    "session_engine_urls",
    "session_worker_hosts",
    "stamp_launch_pods",
    "stop_cluster",
    "verified_launch",
]
