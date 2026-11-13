# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node helpers for the ``optimize`` CLI: cluster adoption and backend resolution."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from inference_suite.contracts.base_launch import BaseLaunch, LaunchQuery

from ..multi_node import TopologyHandoffError, topology_file_overlay


def _topology_file_overlay() -> dict[str, str] | None:
    """The TOPOLOGY_HANDOFF_FILE view of the hand-off, or exit 2 naming the file when it cannot be used."""
    try:
        return topology_file_overlay()
    except TopologyHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _hand_off_service_url() -> str:
    """The handed-over cluster's URL; a named but unusable topology file exits 2 before any state is built."""
    from ..multi_node._internal.external_state import external_service_url

    try:
        return external_service_url()
    except TopologyHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _resolve_gpus_per_node(args: argparse.Namespace) -> int:
    """GPUs per node for this run: --gpus-per-node > $TOPOLOGY_HANDOFF_FILE gpus_per_node > 8.

    The file is consulted only for a multi-node run, the only kind that reads a hand-off.
    """
    flag = getattr(args, "gpus_per_node", None)
    if flag is not None:
        return int(flag)
    from_file = ((_topology_file_overlay() if int(args.nodes) >= 2 else None) or {}).get(
        "INFERENCE_OPTIMIZER_GPUS_PER_NODE", ""
    )
    return int(from_file) if from_file else 8


def _hand_off_node_list() -> Any:
    """The node list handed over, or ``None``; a malformed one exits 2."""
    from ..multi_node.pd_session import handed_node_list

    try:
        return handed_node_list()
    except TopologyHandoffError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _default_mn_backend() -> str:
    """The backend a node list's frontend names, else rayjob."""
    from ..multi_node.pd_session import node_list_backend

    nodes = _hand_off_node_list()
    return node_list_backend(nodes) if nodes is not None else "rayjob"


def _resolve_mn_backend(args: argparse.Namespace) -> str:
    """Multi-node backend selector: --mn-backend > $TOPOLOGY_HANDOFF_FILE backend > $INFERENCE_OPTIMIZER_MN_BACKEND >
    the node list's frontend > rayjob."""
    from ..multi_node.pd_session import PD_BACKENDS

    from_file = (_topology_file_overlay() or {}).get("INFERENCE_OPTIMIZER_MN_BACKEND", "")
    backend = (
        (getattr(args, "mn_backend", None) or "").strip()
        or from_file
        or os.environ.get("INFERENCE_OPTIMIZER_MN_BACKEND", "").strip()
        or os.environ.get("MN_BACKEND", "").strip()
        or _default_mn_backend()
    ).lower()
    choices = ("rayjob", "infera", *PD_BACKENDS)
    if backend not in choices:
        print(
            f"ERROR: --mn-backend must be one of {', '.join(choices)}, got {backend!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return backend


def is_pd_session(args: argparse.Namespace) -> bool:
    """Whether a ``--nodes >= 2`` run serves a node-list cluster the suite launches: an atomesh or sglangrouter one,
    or an Infera one handed over as a node list."""
    from ..multi_node.pd_session import INFERA, PD_BACKENDS

    if int(getattr(args, "nodes", 1) or 1) < 2:
        return False
    backend = _resolve_mn_backend(args)
    return backend in PD_BACKENDS or (backend == INFERA and _hand_off_node_list() is not None)


def freeze_pd_session_launch(
    session_dir: Path, args: argparse.Namespace, session: LaunchQuery, inferencex_commit: str = ""
) -> BaseLaunch:
    """Resolve a fresh node-list session's launch on the handed-over cluster and freeze it in ``session_dir``, with the
    client base its rounds measure it with, at whose topology the session then runs; exits 2 when no node list is
    handed over, the session's ``--pd-mode`` is not the node list's mode, or no source serves the session."""
    from inference_suite.multinode import client_base

    from ..multi_node.pd_session import PdSessionError, freeze_launch
    from ..session.base_launch import write_base_launch

    nodes = _hand_off_node_list()
    if nodes is None:
        _exit_refused(f"--mn-backend {_resolve_mn_backend(args)} needs the HYPERLOOM_MN_NODE_LIST hand-off")
    pd_mode = (os.environ.get("PD_MODE", "") or "aggregated").strip().lower()
    if pd_mode != nodes.mode.value:
        _exit_refused(
            f"the handed-over cluster is {nodes.mode.value}, the session {pd_mode}; pass --pd-mode {nodes.mode.value}"
        )
    try:
        launch = freeze_launch(session_dir, session, nodes, inferencex_commit)
    except PdSessionError as exc:
        _exit_refused(str(exc))
    print(f"multi-node: froze the {launch.mode.value} launch {launch.digest()} from {launch.source}")
    base = client_base(launch, session)
    write_base_launch(session_dir, base)
    args.tp, args.ep = base.topology.tp, base.topology.ep
    os.environ["TP"], os.environ["EP"] = str(base.topology.tp), str(base.topology.ep)
    return base


def _exit_refused(problem: str) -> None:
    """Print ``problem`` and exit 2."""
    print(f"ERROR: {problem}", file=sys.stderr)
    sys.exit(2)


def _bind_pd_launch(ext_state: dict[str, Any], session_dir: Path | None) -> None:
    """Record the session's frozen PD launch in ``ext_state``; exits 2 when it is missing or not the one frozen."""
    from inference_suite.multinode import RECORD_NAME

    from ..multi_node._internal.external_state import load_multi_node_state
    from ..multi_node.pd_session import PD_DIGEST_KEY, PD_LAUNCH_KEY, PdSessionError, verified_launch

    if session_dir is None:
        _exit_refused("a PD session binds its frozen launch to a session directory, and none is open")
    frozen = str(load_multi_node_state().get(PD_DIGEST_KEY) or "")
    try:
        launch = verified_launch(session_dir / RECORD_NAME, frozen)
    except PdSessionError as exc:
        _exit_refused(str(exc))
    ext_state[PD_LAUNCH_KEY] = str(session_dir / RECORD_NAME)
    ext_state[PD_DIGEST_KEY] = launch.digest()


def _record_trace_share(ext_state: dict[str, Any]) -> None:
    """Record whether the engine hosts share the multi-node profile trace root; warn when they do not."""
    from ..multi_node.pd_session import record_trace_share
    from ..session.paths import mn_profile_trace_root

    root = os.path.expandvars(os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()) or str(
        mn_profile_trace_root()
    )
    reason = record_trace_share(ext_state, Path(root))
    if reason:
        print(
            f"WARNING: profile trace dir {root} is not shared with the engine hosts ({reason}); "
            "profiling will be skipped for this session.",
            file=sys.stderr,
        )


def _prepare_multi_node_state(args: argparse.Namespace, session_dir: Path | None = None) -> None:
    """Adopt the platform-provisioned cluster for a ``--nodes >= 2`` run."""
    nodes = max(1, int(args.nodes))
    if nodes < 2:
        return

    from ..multi_node._internal.external_state import (
        build_external_state_from_env,
        external_has_server_control,
        external_has_ssh_control,
    )
    from ..multi_node.cli import _save_state
    from hyperloom.orchestrator.actions.executors._multi_node_env import export_ray_address_to_os

    if not _hand_off_service_url():
        print(
            "ERROR: --nodes >= 2 requires a cluster provisioned by the platform. "
            "Set HYPERLOOM_MN_EXT_SERVICE_URL (plus HYPERLOOM_MN_EXT_SSH_KEY and "
            "one of _PREFILL_IPS / _DECODE_IPS / _WORKER_IPS for infera, or "
            "HYPERLOOM_MN_EXT_HEAD_IP for rayjob); see multi_node/SKILL.md "
            '"Cluster hand-off".',
            file=sys.stderr,
        )
        sys.exit(2)

    ext_state = build_external_state_from_env()
    ext_state["backend"] = _resolve_mn_backend(args)
    _reject_unusable_backend(ext_state)
    if is_pd_session(args):
        _bind_pd_launch(ext_state, session_dir)
        _record_trace_share(ext_state)
    try:
        _save_state(ext_state)
    except OSError as exc:
        print(f"ERROR: cannot write multi-node state: {exc}", file=sys.stderr)
        sys.exit(2)
    os.environ["BENCHMARK_BASE_URL"] = ext_state["service_url"]
    export_ray_address_to_os()
    # Report server *control*, not SSH specifically: rayjob restarts through the Ray dashboard with no SSH at all, so
    # keying this on SSH labelled a fully controllable rayjob cluster "benchmark-only" and gave the genuinely
    # uncontrollable one the same words -- exactly backwards on the one line an operator reads to find out whether the
    # run can tune anything.
    has_control = external_has_server_control()
    print(
        "multi-node: adopted the platform-provisioned cluster. "
        f"url={ext_state['service_url']} prefill={ext_state['prefill_pod_ips']} "
        f"decode={ext_state['decode_pod_ips']} worker={ext_state['worker_pod_ips']} "
        f"server_control={'yes' if has_control else 'no (benchmark-only)'} "
        f"ssh_control={'yes' if external_has_ssh_control() else 'no'}"
    )
    if not has_control:
        # Loud, not fatal: benchmark-only is a documented mode (multi_node/SKILL.md), but without a restart every
        # candidate re-measures the one unchanged server, so the run still reports gains that no config produced.
        print(
            "WARNING: no server control -- per-round restarts will be skipped, so every "
            "candidate config measures the SAME unchanged server and the reported gains "
            "are meaningless. Set HYPERLOOM_MN_EXT_HEAD_IP (rayjob) or "
            "HYPERLOOM_MN_EXT_SSH_KEY plus one of _PREFILL_IPS / _DECODE_IPS / _WORKER_IPS "
            "(infera) to tune. Continuing in benchmark-only mode.",
            file=sys.stderr,
        )
    _prepare_adopted_cluster(
        args,
        str(ext_state["backend"]),
        head_ip=str(ext_state.get("head_pod_ip") or ""),
    )


def _reject_unusable_backend(ext_state: dict[str, Any]) -> None:
    """Exit 2 when the selected backend cannot control this hand-off; flag a benchmark-only rayjob."""
    from ..multi_node._internal.external_state import external_has_ssh_control
    from ..multi_node.pd_session import PD_BACKENDS

    if ext_state["backend"] in PD_BACKENDS and not ext_state.get("router_pod_ip"):
        _exit_refused(f"--mn-backend {ext_state['backend']} needs the HYPERLOOM_MN_NODE_LIST hand-off of a PD cluster")
    if ext_state["backend"] == "infera" and not external_has_ssh_control():
        print(
            "ERROR: the infera backend requires SSH control. Set "
            "HYPERLOOM_MN_EXT_SSH_KEY plus HYPERLOOM_MN_EXT_PREFILL_IPS/"
            "DECODE_IPS (or WORKER_IPS). rayjob uses Ray, not SSH.",
            file=sys.stderr,
        )
        sys.exit(2)
    # The mirror image. --mn-backend defaults to rayjob, so an infera-shaped hand-off whose operator forgot the flag
    # arrives here as rayjob and would report server_control=yes -- SSH control is real, it is simply not the control
    # this backend uses.
    if ext_state["backend"] == "rayjob" and not ext_state.get("head_pod_ip"):
        if external_has_ssh_control():
            print(
                "ERROR: --mn-backend rayjob (the default) needs "
                "HYPERLOOM_MN_EXT_HEAD_IP, but this hand-off carries SSH "
                "credentials and GPU pod IPs instead -- it is an infera "
                "cluster. Pass --mn-backend infera.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            "multi-node: rayjob with no HYPERLOOM_MN_EXT_HEAD_IP -- this cluster cannot be "
            "restarted, so the run is benchmark-only (see the warning below).",
            file=sys.stderr,
        )


def _prepare_adopted_cluster(args: argparse.Namespace, backend: str, *, head_ip: str) -> None:
    """Make an adopted cluster usable by this session."""
    from ..multi_node.cli import cmd_bootstrap, install_geak_on_pods_best_effort, resolve_poll_timeout_s

    if backend == "rayjob":
        if not head_ip:
            print(
                "multi-node: no HYPERLOOM_MN_EXT_HEAD_IP -- skipping the head-pod bootstrap "
                "(benchmark-only hand-off has no head pod to prepare).",
                file=sys.stderr,
            )
        else:
            ns_boot = argparse.Namespace(
                script=None,
                force=False,
                print_logs=False,
                poll_interval=6,
                # Adoption runs in-process, so it is not bound by the foreground/MCP 120s ceiling; a slow head pod
                # should not abort the run.
                poll_timeout=resolve_poll_timeout_s(600),
            )
            rc_boot = cmd_bootstrap(ns_boot)
            if rc_boot != 0:
                sys.exit(rc_boot)

    if not getattr(args, "no_kernel", False):
        install_geak_on_pods_best_effort()
