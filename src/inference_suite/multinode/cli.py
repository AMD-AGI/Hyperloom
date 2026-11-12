# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m inference_suite multinode``: ``resolve`` a PD launch, ``start``, ``restart`` or ``stop`` it.

Every command reads the cluster from the node list in ``HYPERLOOM_MN_NODE_LIST``. ``resolve --query <query.json>
[--replay <pd_launch.json>] [--inferencex-commit <sha>] [--out <path>]`` writes the launch the sources give and prints
its digest. ``start``, ``restart [--prefill-args ARGS] [--decode-args ARGS] [--router-args ARGS]`` and ``stop`` act
on the launch at ``--record`` on the node list's hosts; ``start`` and ``restart`` write the launch as it ran back to
the record and print its digest. The record defaults to ``pd_launch.json`` in
``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``. Exit 1 is a cluster that did not come up, exit 2 a refused input.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.multinode.commands import CommandError
from inference_suite.multinode.inferencex import CHECKOUT_DIRECTORIES
from inference_suite.multinode.launch import Cluster, LaunchError, LaunchOptions, restart, start, stop
from inference_suite.multinode.nodes import NodeList, NodeListError, read_node_list
from inference_suite.multinode.recipe import PdResolveError, ResolveInputs, resolve
from inference_suite.multinode.record import RECORD_NAME, PdLaunch, PdRecordError, Role, read_record, write_record
from inference_suite.multinode.ssh import SshClient, SshError
from inference_suite.recipe.inferencex import INFERENCEX_REPO
from inference_suite.recipe.sources import SourceFetchError, checkout_commit, checkout_main

SESSION_DIR_ENV = "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"
EXIT_OK = 0
EXIT_NOT_UP = 1
EXIT_REFUSED = 2
_REFUSED = (NodeListError, PdResolveError, PdRecordError, CommandError, SourceFetchError, OSError, ValueError)


def default_record(environ: Mapping[str, str]) -> Path:
    """``pd_launch.json`` in the session directory, else in the working directory."""
    return Path(environ.get(SESSION_DIR_ENV, "") or ".") / RECORD_NAME


def _refused(detail: object) -> int:
    print(detail, file=sys.stderr)
    return EXIT_REFUSED


def _resolve(args: argparse.Namespace) -> int:
    try:
        nodes = read_node_list(os.environ)
        query = nodes.query(LaunchQuery.from_document(json.loads(args.query.read_text(encoding="utf-8"))))
        replay = read_record(args.replay) if args.replay else None
        with tempfile.TemporaryDirectory(prefix="pd-inferencex-") as scratch:
            into = Path(scratch) / "InferenceX"
            commit = args.inferencex_commit

            def fetch() -> Any:
                if commit:
                    return checkout_commit(INFERENCEX_REPO, commit, CHECKOUT_DIRECTORIES, into)
                return checkout_main(INFERENCEX_REPO, CHECKOUT_DIRECTORIES, into)

            inputs = ResolveInputs(operator=nodes.operator, replay=replay, inferencex=fetch, pinned=bool(commit))
            launch = resolve(query, inputs)
        write_record(launch, args.out)
    except _REFUSED as error:
        return _refused(error)
    print(launch.digest())
    return EXIT_OK


def _cluster(nodes: NodeList, record: Path) -> Cluster:
    """The cluster of ``nodes``, host keys pinned in the node list's known_hosts file or beside ``record``."""
    if nodes.ssh is None:
        raise NodeListError("the node list names no SSH access")
    known = Path(nodes.ssh.known_hosts_file) if nodes.ssh.known_hosts_file else record.parent / "known_hosts"
    return Cluster(nodes=nodes, client=SshClient(nodes.ssh, known))


def _launch(nodes: NodeList, record: Path) -> PdLaunch:
    """The launch at ``record`` on the node list's hosts."""
    launch = read_record(record)
    return launch.with_hosts(nodes.hosts(Role.PREFILL), nodes.hosts(Role.DECODE), nodes.router)


def _options(args: argparse.Namespace) -> LaunchOptions:
    env = dict(item.split("=", 1) for item in args.env)
    return LaunchOptions(extra_env=env, log_dir=args.log_dir, ready_timeout_s=args.ready_timeout)


def _overrides(args: argparse.Namespace) -> dict[str, list[str]]:
    given = {Role.PREFILL.value: args.prefill_args, Role.DECODE.value: args.decode_args, "router": args.router_args}
    return {role: shlex.split(text) for role, text in given.items() if text}


def _run(args: argparse.Namespace) -> int:
    """Start, restart or stop the launch at ``--record``."""
    try:
        nodes = read_node_list(os.environ)
        cluster = _cluster(nodes, args.record)
        launch = _launch(nodes, args.record)
        if args.action == "stop":
            stop(launch, cluster, _options(args))
            return EXIT_OK
        if args.action == "restart":
            ran = restart(launch, _overrides(args), cluster, _options(args))
        else:
            ran = start(launch, cluster, _options(args))
        write_record(ran, args.record)
    except _REFUSED as error:
        return _refused(error)
    except (LaunchError, SshError) as error:
        print(error, file=sys.stderr)
        return EXIT_NOT_UP
    print(ran.digest())
    return EXIT_OK


def _cluster_parser(commands: Any, action: str, help_text: str, record: Path) -> argparse.ArgumentParser:
    parser = commands.add_parser(action, help=help_text)
    parser.add_argument("--record", type=Path, default=record, help="the pd_launch.json to act on")
    parser.add_argument("--log-dir", default="", help="an absolute directory every engine also logs to")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="exported to every server")
    parser.add_argument("--ready-timeout", type=int, default=1800, help="seconds the cluster may take to come up")
    parser.set_defaults(handler=_run, action=action)
    return parser


def add_commands(commands: Any) -> None:
    """Register ``multinode`` and its commands on the top-level subparsers ``commands``."""
    record = default_record(os.environ)
    multinode = commands.add_parser("multinode", help="resolve and run a multi-node PD launch")
    actions = multinode.add_subparsers(dest="multinode_command", required=True)
    resolve_parser = actions.add_parser("resolve", help="resolve the PD launch of the node list's cluster")
    resolve_parser.add_argument("--query", type=Path, required=True, help="the session's LaunchQuery document")
    resolve_parser.add_argument("--replay", type=Path, help="an earlier pd_launch.json to replay")
    resolve_parser.add_argument("--inferencex-commit", default="", metavar="SHA", help="read InferenceX at SHA")
    resolve_parser.add_argument("--out", type=Path, default=record, help="where to write pd_launch.json")
    resolve_parser.set_defaults(handler=_resolve)
    _cluster_parser(actions, "start", "start the launch on the node list's hosts", record)
    restart_parser = _cluster_parser(actions, "restart", "start the launch with per-role flag overrides", record)
    for role in ("prefill", "decode", "router"):
        restart_parser.add_argument(f"--{role}-args", default="", help=f"flags that replace the {role}'s same flags")
    _cluster_parser(actions, "stop", "stop the router and every engine", record)


__all__ = ["EXIT_NOT_UP", "EXIT_OK", "EXIT_REFUSED", "SESSION_DIR_ENV", "add_commands", "default_record"]
