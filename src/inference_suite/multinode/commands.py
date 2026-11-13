# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The command each host of a multi-node cluster runs, from the frozen launch and the node list.

Every host of a role runs one engine process; a worker that spans several hosts runs one process per host, its first
host serving HTTP. SGLang (and Infera's ``infera.engine.sglang``) joins the hosts of a worker with ``--dist-init-addr``
/ ``--nnodes`` / ``--node-rank``; ATOM, whose TP stays within a node, spans them with multi-node data parallelism
(``--data-parallel-*``). The launcher sets the model, served name, context, bind address, ports, TP, and for a
disaggregated launch the role and KV transfer; everything else comes from the launch. The router (``sglang_router``
or ``atomesh``) runs on the router host in front of every worker, PD-disaggregated or over plain worker URLs. Infera
engines register with the platform's frontend through Kubernetes discovery, so an Infera launch runs no router.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from inference_suite.argv import canonical_flag_name, is_flag
from inference_suite.frameworks import Framework
from inference_suite.multinode.nodes import LaunchMode, NodeList
from inference_suite.multinode.record import PD_ROLES, FrontendType, PdLaunch, PdMode, Role, RoleLaunch

ROUTER = "router"
ROUTER_PORT = 8000
_SGLANG_ENGINE_PORT = 8000
#: Engine HTTP ports when the node list names no engine URL.
DEFAULT_ENGINE_PORTS = {
    Framework.SGLANG: {Role.PREFILL: _SGLANG_ENGINE_PORT, Role.DECODE: _SGLANG_ENGINE_PORT},
    Framework.ATOM: {Role.PREFILL: 8010, Role.DECODE: 8020},
}
#: The HTTP port of an aggregated worker, every framework alike.
WORKER_PORT = _SGLANG_ENGINE_PORT
#: The HTTP port, multi-host rendezvous port and PD bootstrap port of an Infera engine.
INFERA_ENGINE_PORT = 30000
INFERA_DIST_INIT_PORT = 5000
INFERA_BOOTSTRAP_PORT = 30001
INFERA_ENGINE = "infera.engine.sglang"
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
_DIST_INIT_ADDR = "--dist-init-addr"
_PORT = "--port"
_NNODES = "--nnodes"
_NODE_RANK = "--node-rank"
_TRUST_REMOTE_CODE = "--trust-remote-code"
_DP_ENABLE = frozenset({"enable-dp-attention", "enable-dp-lm-head"})


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


def _role_url(nodes: NodeList, role: Role) -> str:
    """The service URL the node list names for a PD role; empty for a worker."""
    return {Role.PREFILL: nodes.prefill_url, Role.DECODE: nodes.decode_url}.get(role, "")


def engine_port(nodes: NodeList, role: Role) -> int:
    """The port a role's engines serve on: Infera's engine port, else that of the node list's role URL, else the
    framework's default."""
    if nodes.frontend is FrontendType.INFERA:
        return INFERA_ENGINE_PORT
    if role is Role.WORKER:
        return WORKER_PORT
    url = _role_url(nodes, role)
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


def engine_urls(launch: PdLaunch, nodes: NodeList) -> dict[str, list[str]]:
    """The HTTP base URL of every engine worker per role: its first host on the role's engine port."""
    return {
        role.value: [f"http://{hosts[0]}:{engine_port(nodes, role)}" for hosts in worker_hosts(launch.role(role))]
        for role in launch.served_roles
    }


def _context_args(framework: Framework, context_length: int | None) -> list[str]:
    if context_length is None:
        return []
    flag = "--context-length" if framework is Framework.SGLANG else "--max-model-len"
    return [flag, str(context_length)]


def _disaggregation_args(launch: PdLaunch, role: Role, bootstrap_port: int) -> list[str]:
    """SGLang's disaggregation flags of a PD role; none for an aggregated worker."""
    if role not in PD_ROLES:
        return []
    argv = ["--disaggregation-mode", role.value, "--disaggregation-transfer-backend", launch.transfer_backend]
    return [*argv, "--disaggregation-bootstrap-port", str(bootstrap_port)]


def _rendezvous_args(hosts: tuple[str, ...], rank: int, port: int) -> list[str]:
    """The flags joining host ``rank`` to a worker on ``hosts``; none for a worker on one host."""
    if len(hosts) == 1:
        return []
    return [_DIST_INIT_ADDR, f"{hosts[0]}:{port}", _NNODES, str(len(hosts)), _NODE_RANK, str(rank)]


def _with_ib_device(command: str, role: Role) -> str:
    """``command`` with the RDMA device tail of a PD role."""
    return f"{command} {IB_DEVICE_TAIL}" if role in PD_ROLES else command


def _sglang_engine(launch: PdLaunch, nodes: NodeList, role: Role, hosts: tuple[str, ...], rank: int) -> str:
    """``sglang.launch_server`` for host ``rank`` of a worker on ``hosts``."""
    launched = launch.role(role)
    argv = [_PYTHON, "-m", "sglang.launch_server", "--model-path", launch.model]
    argv += ["--served-model-name", launch.served_model_name, *_BIND_ALL]
    argv += [_PORT, str(engine_port(nodes, role)), "--tp-size", str(launched.tp)]
    argv += _disaggregation_args(launch, role, _BOOTSTRAP.get(role, 0))
    argv += _rendezvous_args(hosts, rank, DIST_INIT_PORT)
    argv += _context_args(Framework.SGLANG, launch.context_length)
    return _with_ib_device(_join([*argv, *launched.args]), role)


def _canonical_flags(args: Sequence[str]) -> set[str]:
    """The canonical names of the flags ``args`` sets."""
    return {canonical_flag_name(arg.partition("=")[0]) for arg in args if is_flag(arg)}


def _infera_engine(launch: PdLaunch, nodes: NodeList, role: Role, hosts: tuple[str, ...], rank: int) -> str:
    """``infera.engine.sglang`` for host ``rank`` of a worker on ``hosts``, advertised to the platform's frontend
    at the host's address; a PD leg skips the server warmup."""
    launched = launch.role(role)
    given = _canonical_flags(launched.args)
    argv = [_PYTHON, "-m", INFERA_ENGINE, "--model-path", launch.model, "--tp-size", str(launched.tp)]
    argv += [] if "trust-remote-code" in given else [_TRUST_REMOTE_CODE]
    argv += [*_BIND_ALL, _PORT, str(engine_port(nodes, role)), "--discovery-backend", "kubernetes"]
    argv += ["--advertise-host", hosts[rank], "--request-transport", "nats"]
    argv += _rendezvous_args(hosts, rank, INFERA_DIST_INIT_PORT)
    if launched.ep > 1 and "ep-size" not in given:
        argv += ["--ep-size", str(launched.ep)]
    argv += _disaggregation_args(launch, role, INFERA_BOOTSTRAP_PORT)
    argv += _context_args(Framework.SGLANG, launch.context_length)
    tail = []
    if given & _DP_ENABLE and "dp-size" not in given and launched.tp > 1:
        tail += ["--dp-size", str(launched.tp)]
    if role in PD_ROLES and "skip-server-warmup" not in given:
        tail.append("--skip-server-warmup")
    return _with_ib_device(_join([*argv, *launched.args, *tail]), role)


def _atom_engine(launch: PdLaunch, nodes: NodeList, role: Role, hosts: tuple[str, ...], rank: int) -> str:
    """``atom.entrypoints.openai_server`` for host ``rank`` of a worker on ``hosts``.

    Raises:
        CommandError: a worker spans hosts with a TP above one host's GPUs.
    """
    launched = launch.role(role)
    argv = [_PYTHON, "-m", "atom.entrypoints.openai_server", "--model", launch.model]
    argv += [*_BIND_ALL, "--server-port", str(engine_port(nodes, role)), "-tp", str(launched.tp)]
    if role in PD_ROLES:
        kv = {
            "kv_role": _ATOM_KV_ROLE[role],
            "kv_connector": launch.transfer_backend,
            "proxy_ip": hosts[rank],
            "handshake_port": ATOM_HANDSHAKE_PORT,
        }
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
    if launch.frontend.type is FrontendType.INFERA:
        compose = _infera_engine
    return [
        HostCommand(role.value, host, worker, rank, compose(launch, nodes, role, hosts, rank))
        for worker, hosts in enumerate(worker_hosts(launch.role(role)))
        for rank, host in enumerate(hosts)
    ]


def _worker_urls(launch: PdLaunch, nodes: NodeList, role: Role) -> list[str]:
    """The URLs the router reaches a role at: the role's service URL in place, else each worker's first host."""
    service = _role_url(nodes, role)
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
    """The router in front of every worker, with the frontend args of the launch: PD-disaggregated over the prefill
    and decode workers, else over the aggregated workers' URLs.

    Raises:
        CommandError: the launch's frontend is the platform's, which no host here runs.
    """
    if launch.frontend.type is FrontendType.INFERA:
        raise CommandError("the Infera frontend is the platform's; the launch runs no router")
    disaggregated = launch.mode is PdMode.DISAGGREGATED
    if launch.frontend.type is FrontendType.SGLANG_ROUTER:
        argv = [_PYTHON, "-m", "sglang_router.launch_router"]
    else:
        argv = ["atomesh", "launch"]
    argv += ["--pd-disaggregation"] if disaggregated else []
    argv += ["--backend", launch.framework.value] if launch.frontend.type is FrontendType.ATOMESH else []
    if disaggregated:
        prefill = _worker_urls(launch, nodes, Role.PREFILL)
        decode = _worker_urls(launch, nodes, Role.DECODE)
        sglang = launch.framework is Framework.SGLANG
        argv += _bootstrapped(prefill, "--prefill", sglang) + _bootstrapped(decode, "--decode", False)
    else:
        argv += ["--worker-urls", *_worker_urls(launch, nodes, Role.WORKER)]
    argv += [*_BIND_ALL, _PORT, str(ROUTER_PORT), *launch.frontend.args]
    return HostCommand(ROUTER, nodes.router, 0, 0, _join(argv))


def runs_router(launch: PdLaunch) -> bool:
    """Whether the launch runs a router of its own: every frontend but the platform's Infera frontend."""
    return launch.frontend.type is not FrontendType.INFERA


def cluster_commands(launch: PdLaunch, nodes: NodeList) -> list[HostCommand]:
    """Every host's command: each role's engines in launch order, then the router when the launch runs one."""
    engines = [command for role in launch.served_roles for command in engine_commands(launch, nodes, role)]
    return [*engines, router_command(launch, nodes)] if runs_router(launch) else engines


def process_needle(role: str, framework: Framework, frontend: FrontendType) -> str:
    """``|``-separated cmdline substrings, any of which marks the server process of ``role`` on a host."""
    if role == ROUTER:
        return "sglang_router.launch_router|sglang::router" if frontend is FrontendType.SGLANG_ROUTER else "atomesh"
    if frontend is FrontendType.INFERA:
        return INFERA_ENGINE
    return "sglang.launch_server" if framework is Framework.SGLANG else "atom.entrypoints.openai_server"


__all__ = [
    "INFERA_BOOTSTRAP_PORT",
    "INFERA_DIST_INIT_PORT",
    "INFERA_ENGINE",
    "INFERA_ENGINE_PORT",
    "ROUTER",
    "ROUTER_PORT",
    "WORKER_PORT",
    "CommandError",
    "HostCommand",
    "cluster_commands",
    "engine_commands",
    "engine_port",
    "engine_urls",
    "process_needle",
    "router_command",
    "runs_router",
    "worker_hosts",
]
