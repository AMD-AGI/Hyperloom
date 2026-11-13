# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sessions on a cluster handed over as a node list, PD-disaggregated or aggregated: its state, its frozen launch, its
rounds."""

from __future__ import annotations

import os
import re
import shlex
import uuid
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
    PdMode,
    PdRecordError,
    PdResolveError,
    SshError,
    cluster_of,
    engine_urls,
    engines_gone,
    placed_launch,
    read_record,
    replace_flags,
    resolve_session,
    restart,
    stop,
    write_record,
)

from ._internal.pd_handoff import INFERA, PD_BACKENDS, handed_node_list, is_pd_state, node_list_backend, pd_state

PD_LAUNCH_KEY = "pd_launch"
PD_DIGEST_KEY = "pd_launch_digest"
#: The launch as the last round ran it, beside the frozen ``pd_launch.json``.
RAN_RECORD_NAME = "pd_launch.ran.json"
PREFILL_EXTRA_ENV = "PD_PREFILL_EXTRA_ARGS"
DECODE_EXTRA_ENV = "PD_DECODE_EXTRA_ARGS"
SERVER_LOG_DIR_ENV = "HYPERLOOM_MN_SERVER_LOG_DIR"
PROFILER_DIR_ENV = "SGLANG_TORCH_PROFILER_DIR"
TRACE_SHARED_KEY = "profile_trace_shared"
TRACE_SHARE_ERROR_KEY = "profile_trace_share_error"
TRACE_SHARE_PROBE_TIMEOUT_S = 30

_PREFILL = "prefill"
_DECODE = "decode"
_WORKER = "worker"
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
        if _ENV_KEY.fullmatch(key) and not key.startswith(_RESERVED_ENV_PREFIX)
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
    try:
        session = _session_cluster(state)
        overrides = round_overrides(
            shared_args,
            os.environ.get(PREFILL_EXTRA_ENV, ""),
            os.environ.get(DECODE_EXTRA_ENV, ""),
            mode=session.launch.mode,
        )
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


def session_engine_urls(state: dict[str, Any]) -> dict[str, list[str]]:
    """The HTTP base URL of every prefill and decode engine worker, as the frozen launch is placed on the node list.

    Raises:
        PdSessionError: no node list is handed over, or the frozen launch cannot be verified or placed.
    """
    try:
        session = _session_cluster(state)
        return engine_urls(session.launch, session.cluster.nodes)
    except (*_SUITE_ERRORS, ValueError) as exc:
        raise PdSessionError(f"the PD engines cannot be placed: {exc}") from exc


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
    "DECODE_EXTRA_ENV",
    "INFERA",
    "PD_BACKENDS",
    "PD_DIGEST_KEY",
    "PD_LAUNCH_KEY",
    "PREFILL_EXTRA_ENV",
    "PROFILER_DIR_ENV",
    "RAN_RECORD_NAME",
    "TRACE_SHARED_KEY",
    "TRACE_SHARE_ERROR_KEY",
    "PdSessionError",
    "engines_lost",
    "freeze_launch",
    "handed_node_list",
    "is_pd_state",
    "launch_options",
    "node_list_backend",
    "pd_state",
    "probe_trace_share",
    "record_trace_share",
    "restart_round",
    "round_overrides",
    "session_engine_urls",
    "stop_cluster",
    "verified_launch",
]
