# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command each host of a PD cluster runs, from the frozen launch and the node list.

Every host of a role runs one engine process; a worker that spans several hosts runs one process per host, its first
host serving HTTP. SGLang joins the hosts of a worker with ``--dist-init-addr`` / ``--nnodes`` / ``--node-rank``;
ATOM, whose TP stays within a node, spans them with multi-node data parallelism (``--data-parallel-*``). The launcher
sets the model, served name, context, bind address, ports, TP, disaggregation role and KV transfer; everything else
comes from the launch. The router (``sglang_router`` or ``atomesh``) runs on the router host in front of every worker.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from inference_suite.frameworks import Framework
from inference_suite.multinode.nodes import LaunchMode, NodeList
from inference_suite.multinode.record import FrontendType, PdLaunch, Role, RoleLaunch

ROUTER = "router"
ROUTER_PORT = 8000
_SGLANG_ENGINE_PORT = 8000
#: Engine HTTP ports when the node list names no engine URL.
DEFAULT_ENGINE_PORTS = {
    Framework.SGLANG: {Role.PREFILL: _SGLANG_ENGINE_PORT, Role.DECODE: _SGLANG_ENGINE_PORT},
    Framework.ATOM: {Role.PREFILL: 8010, Role.DECODE: 8020},
}
PREFILL_BOOTSTRAP_PORT = 8998
DECODE_BOOTSTRAP_PORT = 9098
DIST_INIT_PORT = 20000
ATOM_HANDSHAKE_PORT = 6301
ATOM_DP_MASTER_PORT = 29500
ATOM_DP_BASE_PORT = 29501
#: Shell that appends ``--disaggregation-ib-device`` when the host names its RDMA devices in ``HL_IB_DEVICE``.
IB_DEVICE_TAIL = '${HL_IB_DEVICE:+--disaggregation-ib-device "$HL_IB_DEVICE"}'
_PYTHON = "python3"
_BIND_ALL = ("--host", "0.0.0.0")
_ATOM_KV_ROLE = {Role.PREFILL: "kv_producer", Role.DECODE: "kv_consumer"}
_BOOTSTRAP = {Role.PREFILL: PREFILL_BOOTSTRAP_PORT, Role.DECODE: DECODE_BOOTSTRAP_PORT}


class CommandError(ValueError):
    """The launch cannot be laid out on the node list's hosts."""


@dataclass(frozen=True)
class HostCommand:
    """What one host runs: its role, host, the worker and the host's rank in it, and the command line."""

    role: str
    host: str
    worker: int
    rank: int
    command: str


def _join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def engine_port(nodes: NodeList, role: Role) -> int:
    """The port a role's engines serve on: that of the node list's role URL, else the framework's default."""
    url = nodes.prefill_url if role is Role.PREFILL else nodes.decode_url
    port = urlsplit(url).port if url else None
    return port or DEFAULT_ENGINE_PORTS[nodes.framework][role]


def worker_hosts(role: RoleLaunch) -> list[tuple[str, ...]]:
    """The role's hosts split into its workers, each its first host first.

    Raises:
        CommandError: the hosts do not split evenly into the role's workers.
    """
    if len(role.hosts) % role.workers:
        raise CommandError(f"{len(role.hosts)} hosts do not split into {role.workers} workers")
    span = len(role.hosts) // role.workers
    return [role.hosts[index : index + span] for index in range(0, len(role.hosts), span)]


def _context_args(framework: Framework, context_length: int | None) -> list[str]:
    if context_length is None:
        return []
    flag = "--context-length" if framework is Framework.SGLANG else "--max-model-len"
    return [flag, str(context_length)]


def _sglang_engine(launch: PdLaunch, nodes: NodeList, role: Role, hosts: tuple[str, ...], rank: int) -> str:
    """``sglang.launch_server`` for host ``rank`` of a worker on ``hosts``."""
    launched = launch.role(role)
    argv = [_PYTHON, "-m", "sglang.launch_server", "--model-path", launch.model]
    argv += ["--served-model-name", launch.served_model_name, *_BIND_ALL]
    argv += ["--port", str(engine_port(nodes, role)), "--tp-size", str(launched.tp)]
    argv += ["--disaggregation-mode", role.value, "--disaggregation-transfer-backend", launch.transfer_backend]
    argv += ["--disaggregation-bootstrap-port", str(_BOOTSTRAP[role])]
    if len(hosts) > 1:
        argv += ["--dist-init-addr", f"{hosts[0]}:{DIST_INIT_PORT}", "--nnodes", str(len(hosts))]
        argv += ["--node-rank", str(rank)]
    argv += _context_args(Framework.SGLANG, launch.context_length)
    return f"{_join([*argv, *launched.args])} {IB_DEVICE_TAIL}"


def _atom_engine(launch: PdLaunch, nodes: NodeList, role: Role, hosts: tuple[str, ...], rank: int) -> str:
    """``atom.entrypoints.openai_server`` for host ``rank`` of a worker on ``hosts``.

    Raises:
        CommandError: a worker spans hosts with a TP above one host's GPUs.
    """
    launched = launch.role(role)
    kv = {
        "kv_role": _ATOM_KV_ROLE[role],
        "kv_connector": launch.transfer_backend,
        "proxy_ip": hosts[rank],
        "handshake_port": ATOM_HANDSHAKE_PORT,
    }
    argv = [_PYTHON, "-m", "atom.entrypoints.openai_server", "--model", launch.model]
    argv += [*_BIND_ALL, "--server-port", str(engine_port(nodes, role)), "-tp", str(launched.tp)]
    argv += ["--kv-transfer-config", json.dumps(kv, separators=(",", ":"))]
    if len(hosts) > 1:
        if launched.tp > nodes.gpus_per_node:
            raise CommandError(f"ATOM TP {launched.tp} cannot exceed the {nodes.gpus_per_node} GPUs of one host")
        local = nodes.gpus_per_node // launched.tp
        argv += ["--data-parallel-size", str(local * len(hosts)), "--data-parallel-size-local", str(local)]
        argv += ["--data-parallel-rank", str(rank * local), "--data-parallel-master-ip", hosts[0]]
        argv += ["--data-parallel-master-port", str(ATOM_DP_MASTER_PORT)]
        argv += ["--data-parallel-base-port", str(ATOM_DP_BASE_PORT)]
    argv += _context_args(Framework.ATOM, launch.context_length)
    return _join([*argv, *launched.args])


def engine_commands(launch: PdLaunch, nodes: NodeList, role: Role) -> list[HostCommand]:
    """One command per host of ``role``, worker by worker."""
    compose = _sglang_engine if launch.framework is Framework.SGLANG else _atom_engine
    return [
        HostCommand(role.value, host, worker, rank, compose(launch, nodes, role, hosts, rank))
        for worker, hosts in enumerate(worker_hosts(launch.role(role)))
        for rank, host in enumerate(hosts)
    ]


def _worker_urls(launch: PdLaunch, nodes: NodeList, role: Role) -> list[str]:
    """The URLs the router reaches a role at: the role's service URL in place, else each worker's first host."""
    service = nodes.prefill_url if role is Role.PREFILL else nodes.decode_url
    if nodes.launch_mode is LaunchMode.IN_PLACE and service:
        return [service]
    port = engine_port(nodes, role)
    return [f"http://{hosts[0]}:{port}" for hosts in worker_hosts(launch.role(role))]


def _bootstrapped(urls: list[str], flag: str, bootstrap: bool) -> list[str]:
    argv: list[str] = []
    for url in urls:
        argv += [flag, url, str(PREFILL_BOOTSTRAP_PORT)] if bootstrap else [flag, url]
    return argv


def router_command(launch: PdLaunch, nodes: NodeList) -> HostCommand:
    """The router in front of every prefill and decode worker, with the frontend args of the launch."""
    prefill = _worker_urls(launch, nodes, Role.PREFILL)
    decode = _worker_urls(launch, nodes, Role.DECODE)
    sglang = launch.framework is Framework.SGLANG
    if launch.frontend.type is FrontendType.SGLANG_ROUTER:
        argv = [_PYTHON, "-m", "sglang_router.launch_router", "--pd-disaggregation"]
    else:
        argv = ["atomesh", "launch", "--pd-disaggregation", "--backend", launch.framework.value]
    argv += _bootstrapped(prefill, "--prefill", sglang) + _bootstrapped(decode, "--decode", False)
    argv += [*_BIND_ALL, "--port", str(ROUTER_PORT), *launch.frontend.args]
    return HostCommand(ROUTER, nodes.router, 0, 0, _join(argv))


def cluster_commands(launch: PdLaunch, nodes: NodeList) -> list[HostCommand]:
    """Every host's command: prefill, then decode, then the router."""
    return [*engine_commands(launch, nodes, Role.PREFILL), *engine_commands(launch, nodes, Role.DECODE)] + [
        router_command(launch, nodes)
    ]


def process_needle(role: str, framework: Framework, frontend: FrontendType) -> str:
    """``|``-separated cmdline substrings, any of which marks the server process of ``role`` on a host."""
    if role == ROUTER:
        return "sglang_router.launch_router|sglang::router" if frontend is FrontendType.SGLANG_ROUTER else "atomesh"
    return "sglang.launch_server" if framework is Framework.SGLANG else "atom.entrypoints.openai_server"


__all__ = [
    "ROUTER",
    "ROUTER_PORT",
    "CommandError",
    "HostCommand",
    "cluster_commands",
    "engine_commands",
    "engine_port",
    "process_needle",
    "router_command",
    "worker_hosts",
]
