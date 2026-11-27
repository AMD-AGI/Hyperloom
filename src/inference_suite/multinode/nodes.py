# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The node list: the one input of a multi-node session, a JSON document in ``HYPERLOOM_MN_NODE_LIST``.

It names the framework and model, the cluster's service and engine URLs, the serving mode, the router host and the
hosts of each role of the mode (prefill and decode when disaggregated, worker when aggregated), the GPUs per node, SSH
access (the key by path only, a host entry's own ``ssh_port`` over the shared one), the KV transfer backend, the
frontend it runs, the topology and args the operator fixed, and how the launcher starts processes: ``direct`` over
SSH, or ``in_place`` into the process a platform launcher waits on. The router host is optional for the ``infera``
frontend, which the platform runs. Every key is checked; an unknown one is refused.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from hyperloom_common.contract_fields import ContractFields

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.frameworks import Framework
from inference_suite.multinode.query import SERVED_FRONTENDS, PdQuery, RoleAsk
from inference_suite.multinode.record import MODE_ROLES, FrontendType, PdMode, Role
from inference_suite.multinode.recipe import OperatorArgs

NODE_LIST_ENV = "HYPERLOOM_MN_NODE_LIST"
_FRAMEWORKS = {Framework.SGLANG.value: Framework.SGLANG, Framework.ATOM.value: Framework.ATOM}
_ROUTER = "router"
ROUTER_PORT = 8000
_SGLANG_ENGINE_PORT = 8000
#: Engine HTTP ports when the node list names no engine URL.
_DEFAULT_ENGINE_PORTS = {
    Framework.SGLANG: {Role.PREFILL: _SGLANG_ENGINE_PORT, Role.DECODE: _SGLANG_ENGINE_PORT},
    Framework.ATOM: {Role.PREFILL: 8010, Role.DECODE: 8020},
}
#: The HTTP port of an aggregated worker, every framework alike.
_WORKER_PORT = _SGLANG_ENGINE_PORT
#: The HTTP port of an Infera engine.
_INFERA_ENGINE_PORT = 30000
_SSH_PORT = "ssh_port"
_REQUIRED = ("framework", "model", "service_url", "gpus_per_node")
_KEYS = tuple(
    sorted(
        frozenset(_REQUIRED)
        | {_ROUTER, "mode", *(role.value for role in Role)}
        | {"prefill_url", "decode_url", "ssh", "transfer_backend", "topology", "launch_mode"}
        | {"config", "frontend"}
    )
)
_SSH_KEYS = ("key_file", "known_hosts_file", "port", "user")
_TOPOLOGY_KEYS = ("dp_attention", "ep", "tp")
_ROUTER_ARGS = "router_args"
_HOST_KEYS = ("host", _SSH_PORT)
_HTTP = ("http://", "https://")
_MAX_PORT = 65535


class NodeListError(ValueError):
    """The node list is absent or breaks its contract."""


_FIELDS = ContractFields(NodeListError)


class LaunchMode(str, Enum):
    #: Each process starts over SSH as a detached child of the session.
    DIRECT = "direct"
    #: Each process is exec'd into the pid a platform launcher waits on, so the pod is not recreated.
    IN_PLACE = "in_place"


@dataclass(frozen=True)
class SshAccess:
    """How the launcher reaches every host: port, user, the private key's path, an optional known_hosts file and
    the hosts whose sshd listens on a port of its own."""

    port: int
    user: str
    key_file: str
    known_hosts_file: str = ""
    host_ports: Mapping[str, int] = field(default_factory=dict)

    def port_of(self, host: str) -> int:
        """The port ``host``'s sshd listens on."""
        return self.host_ports.get(host, self.port)


@dataclass(frozen=True)
class RoleTopology:
    """The topology the operator fixed for one role; ``None`` leaves it to the launch source."""

    tp: int | None = None
    ep: int | None = None
    dp_attention: bool | None = None


@dataclass(frozen=True)
class NodeList:
    """A parsed node list."""

    framework: Framework
    model: str
    service_url: str
    router: str
    prefill: tuple[str, ...]
    decode: tuple[str, ...]
    gpus_per_node: int
    mode: PdMode = PdMode.DISAGGREGATED
    worker: tuple[str, ...] = ()
    prefill_url: str = ""
    decode_url: str = ""
    ssh: SshAccess | None = None
    transfer_backend: str = ""
    topology: Mapping[Role, RoleTopology] = field(default_factory=dict)
    launch_mode: LaunchMode = LaunchMode.DIRECT
    operator: OperatorArgs = field(default_factory=OperatorArgs)
    frontend: FrontendType | None = None

    @property
    def roles(self) -> tuple[Role, ...]:
        """The roles the node list's mode runs."""
        return MODE_ROLES[self.mode]

    def hosts(self, role: Role) -> tuple[str, ...]:
        return {Role.PREFILL: self.prefill, Role.DECODE: self.decode, Role.WORKER: self.worker}[role]

    def role_hosts(self) -> dict[Role, tuple[str, ...]]:
        """The hosts of each role of the mode."""
        return {role: self.hosts(role) for role in self.roles}

    def query(self, session: LaunchQuery) -> PdQuery:
        """The query of this cluster for ``session``, whose framework must be the node list's.

        Raises:
            NodeListError: the session runs another framework.
        """
        if session.framework != self.framework.value:
            raise NodeListError(f"the session runs {session.framework}, the node list {self.framework.value}")
        asks = {}
        for role in self.roles:
            fixed = self.topology.get(role, RoleTopology())
            asks[role] = RoleAsk(self.hosts(role), fixed.tp, fixed.ep, fixed.dp_attention)
        return PdQuery(
            model_id=session.model_id,
            model_path=self.model,
            framework=self.framework,
            board=session.board,
            gpus_per_node=self.gpus_per_node,
            workload=session.workload,
            concurrencies=session.concurrencies,
            roles=asks,
            router_host=self.router,
            isl=session.isl,
            osl=session.osl,
            transfer_backend=self.transfer_backend,
            frontend=self.frontend,
            mode=self.mode,
        )


def _integer(value: Any, what: str, maximum: int | None = None) -> int:
    number = _FIELDS.integer(value, what, minimum=1)
    if maximum and number > maximum:
        raise NodeListError(f"{what} must be an integer from 1 to {maximum}, got {value!r}")
    return number


def _url(value: Any, what: str) -> str:
    text = _FIELDS.text(value, what)
    if not text.startswith(_HTTP) or not urlsplit(text).hostname:
        raise NodeListError(f"{what} must be an http(s) URL, got {text!r}")
    return text


def _host(value: Any, what: str) -> str:
    return _FIELDS.text(_FIELDS.known(value, _HOST_KEYS, what).get("host"), f"{what}'s host")


def _hosts(value: Any, role: Role) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise NodeListError(f"{role.value} must be a non-empty list of hosts, got {value!r}")
    hosts = tuple(_host(item, f"{role.value}[{index}]") for index, item in enumerate(value))
    _FIELDS.unique(hosts, f"{role.value} host")
    return hosts


def _host_ports(fields: Mapping[str, Any]) -> dict[str, int]:
    """The SSH port of every host entry that names its own."""
    ports: dict[str, int] = {}
    for key in (_ROUTER, *(role.value for role in Role)):
        entries = fields.get(key)
        for entry in [entries] if isinstance(entries, Mapping) else entries or []:
            if isinstance(entry, Mapping) and entry.get(_SSH_PORT) is not None:
                ports[str(entry.get("host"))] = _integer(entry[_SSH_PORT], f"{key}'s {_SSH_PORT}", _MAX_PORT)
    return ports


def _ssh(value: Any) -> SshAccess:
    fields = _FIELDS.known(value, _SSH_KEYS, "ssh")
    key_file = _FIELDS.text(fields.get("key_file"), "ssh.key_file")
    if not key_file.startswith("/"):
        raise NodeListError("ssh.key_file must be an absolute path")
    known_hosts = fields.get("known_hosts_file", "")
    if known_hosts and not _FIELDS.text(known_hosts, "ssh.known_hosts_file").startswith("/"):
        raise NodeListError("ssh.known_hosts_file must be an absolute path")
    return SshAccess(
        port=_integer(fields.get("port", 22), "ssh.port", _MAX_PORT),
        user=_FIELDS.text(fields.get("user", "root"), "ssh.user"),
        key_file=key_file,
        known_hosts_file=str(known_hosts or ""),
    )


def _role_topology(value: Any, role: Role) -> RoleTopology:
    fields = _FIELDS.known(value, _TOPOLOGY_KEYS, f"topology.{role.value}")
    return RoleTopology(
        tp=_FIELDS.optional_integer(fields.get("tp"), f"topology.{role.value}.tp", minimum=1),
        ep=_FIELDS.optional_integer(fields.get("ep"), f"topology.{role.value}.ep", minimum=1),
        dp_attention=_FIELDS.optional_boolean(fields.get("dp_attention"), f"topology.{role.value}.dp_attention"),
    )


def _args(value: Any, what: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if not isinstance(value, str):
        raise NodeListError(f"{what} must be a string, got {type(value).__name__}")
    try:
        return tuple(shlex.split(value))
    except ValueError as error:
        raise NodeListError(f"{what} is not shell-tokenizable: {error}") from error


def _operator(value: Any, mode: PdMode) -> OperatorArgs:
    """The operator's args of ``config``, which names the args of the mode's roles and the router's only."""
    keys = {f"{role.value}_args": role for role in MODE_ROLES[mode]}
    fields = _FIELDS.known(value, tuple(sorted({*keys, _ROUTER_ARGS})), "config")
    given = {role: _args(fields.get(key), f"config.{key}") for key, role in keys.items()}
    return OperatorArgs(
        prefill=given.get(Role.PREFILL, ()),
        decode=given.get(Role.DECODE, ()),
        worker=given.get(Role.WORKER, ()),
        router=_args(fields.get(_ROUTER_ARGS), f"config.{_ROUTER_ARGS}"),
    )


def _framework(value: Any) -> Framework:
    if value not in _FRAMEWORKS:
        raise NodeListError(f"framework must be one of {sorted(_FRAMEWORKS)}, got {value!r}")
    return _FRAMEWORKS[value]


def _optional_url(fields: Mapping[str, Any], key: str) -> str:
    return _url(fields[key], key) if fields.get(key) else ""


def _frontend(fields: Mapping[str, Any], framework: Framework) -> FrontendType | None:
    """The frontend the node list names, refused when ``framework`` does not serve through it."""
    if not fields.get("frontend"):
        return None
    frontend: FrontendType = _FIELDS.enum(FrontendType, fields["frontend"], "frontend")
    if frontend not in SERVED_FRONTENDS[framework]:
        raise NodeListError(f"framework {framework.value} does not serve through the {frontend.value} frontend")
    return frontend


def _role_hosts(fields: Mapping[str, Any], mode: PdMode) -> dict[Role, tuple[str, ...]]:
    """The hosts of each role of ``mode``, refused when another role names hosts or two roles share one."""
    for role in Role:
        if role not in MODE_ROLES[mode] and fields.get(role.value) is not None:
            raise NodeListError(f"a {mode.value} node list names no {role.value} hosts")
    hosts = {role: _hosts(fields[role.value], role) for role in MODE_ROLES[mode]}
    every = [host for role_hosts in hosts.values() for host in role_hosts]
    shared = sorted({host for host in every if every.count(host) > 1})
    if shared:
        raise NodeListError(f"{' and '.join(role.value for role in hosts)} share hosts {shared}")
    return hosts


def role_url(nodes: NodeList, role: Role) -> str:
    """The service URL the node list names for a PD role; empty for a worker."""
    return {Role.PREFILL: nodes.prefill_url, Role.DECODE: nodes.decode_url}.get(role, "")


def engine_port(nodes: NodeList, role: Role) -> int:
    """The port a role's engines serve on: Infera's engine port, else that of the node list's role URL, else the
    framework's default."""
    if nodes.frontend is FrontendType.INFERA:
        return _INFERA_ENGINE_PORT
    if role is Role.WORKER:
        return _WORKER_PORT
    url = role_url(nodes, role)
    port = urlsplit(url).port if url else None
    return port or _DEFAULT_ENGINE_PORTS[nodes.framework][role]


def parse_node_list(document: Any) -> NodeList:
    """The node list ``document`` states.

    Raises:
        NodeListError: a key is missing, unknown or of the wrong kind, a role the mode does not run names hosts, or
            the roles share a host.
    """
    fields = _FIELDS.known(document, _KEYS, "the node list")
    mode = _FIELDS.enum(PdMode, fields.get("mode", PdMode.DISAGGREGATED.value), "mode")
    frontend = _frontend(fields, _framework(fields.get("framework"))) if "framework" in fields else None
    required = (*_REQUIRED, *(() if frontend is FrontendType.INFERA else (_ROUTER,)))
    missing = [key for key in (*required, *(role.value for role in MODE_ROLES[mode])) if key not in fields]
    if missing:
        raise NodeListError(f"the node list lacks {missing}")
    framework = _framework(fields["framework"])
    hosts = _role_hosts(fields, mode)
    router = _host(fields[_ROUTER], _ROUTER) if fields.get(_ROUTER) is not None else ""
    topology = _FIELDS.known(fields.get("topology") or {}, tuple(role.value for role in MODE_ROLES[mode]), "topology")
    ssh = _ssh(fields["ssh"]) if fields.get("ssh") is not None else None
    ports = _host_ports(fields)
    nodes = NodeList(
        framework=framework,
        model=_FIELDS.text(fields["model"], "model"),
        service_url=_url(fields["service_url"], "service_url"),
        router=router,
        prefill=hosts.get(Role.PREFILL, ()),
        decode=hosts.get(Role.DECODE, ()),
        gpus_per_node=_integer(fields["gpus_per_node"], "gpus_per_node"),
        mode=mode,
        worker=hosts.get(Role.WORKER, ()),
        prefill_url=_optional_url(fields, "prefill_url"),
        decode_url=_optional_url(fields, "decode_url"),
        ssh=replace(ssh, host_ports=ports) if ssh is not None else None,
        transfer_backend=_FIELDS.text(fields["transfer_backend"], "transfer_backend")
        if fields.get("transfer_backend")
        else "",
        topology={Role(key): _role_topology(value, Role(key)) for key, value in topology.items()},
        launch_mode=_FIELDS.enum(LaunchMode, fields.get("launch_mode", LaunchMode.DIRECT.value), "launch_mode"),
        operator=_operator(fields["config"], mode) if fields.get("config") is not None else OperatorArgs(),
        frontend=frontend,
    )
    return _refuse_router_port_clash(nodes)


def _refuse_router_port_clash(nodes: NodeList) -> NodeList:
    """``nodes`` unchanged when its router host serves no engine on the router's port.

    Raises:
        NodeListError: a role's host is the router host and its engines serve on the router's port.
    """
    if not nodes.router or nodes.frontend is FrontendType.INFERA:
        return nodes
    for role in nodes.roles:
        if nodes.router in nodes.hosts(role) and engine_port(nodes, role) == ROUTER_PORT:
            raise NodeListError(f"router host {nodes.router} also serves a {role.value} engine on port {ROUTER_PORT}")
    return nodes


def read_node_list(environ: Mapping[str, str]) -> NodeList:
    """The node list ``environ`` carries in ``HYPERLOOM_MN_NODE_LIST``.

    Raises:
        NodeListError: the variable is unset, not JSON, or its document breaks the contract.
    """
    raw = environ.get(NODE_LIST_ENV, "").strip()
    if not raw:
        raise NodeListError(f"{NODE_LIST_ENV} is unset")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise NodeListError(f"{NODE_LIST_ENV} is not JSON: {error}") from error
    return parse_node_list(document)


__all__ = [
    "NODE_LIST_ENV",
    "ROUTER_PORT",
    "LaunchMode",
    "NodeList",
    "NodeListError",
    "RoleTopology",
    "SshAccess",
    "engine_port",
    "parse_node_list",
    "read_node_list",
    "role_url",
]
