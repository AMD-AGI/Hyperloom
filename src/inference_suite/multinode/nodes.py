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
from collections.abc import Callable, Mapping
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
_FRAMEWORKS = {framework.value: framework for framework in SERVED_FRONTENDS}
_ROUTER = "router"
ROUTER_PORT = 8000
_ENGINE_PORT = 8000
#: Engine HTTP ports when the node list names no engine URL.
DEFAULT_ENGINE_PORTS = {
    Framework.SGLANG: {Role.PREFILL: _ENGINE_PORT, Role.DECODE: _ENGINE_PORT, Role.WORKER: _ENGINE_PORT},
    Framework.ATOM: {Role.PREFILL: 8010, Role.DECODE: 8020, Role.WORKER: _ENGINE_PORT},
}
#: The HTTP port of an Infera engine.
INFERA_ENGINE_PORT = 30000
_SSH_PORT = "ssh_port"
_REQUIRED = ("framework", "model", "service_url", "gpus_per_node")
_KEYS = (
    *_REQUIRED,
    *(_ROUTER, "mode", *(role.value for role in Role)),
    *("prefill_url", "decode_url", "ssh", "transfer_backend", "topology", "launch_mode", "config", "frontend"),
)
_SSH_KEYS = ("port", "user", "key_file", "known_hosts_file")
_TOPOLOGY_KEYS = ("tp", "ep", "dp_attention")
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
        """The hosts of ``role``."""
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
        fixed = {role: self.topology.get(role, RoleTopology()) for role in self.roles}
        asks = {role: RoleAsk(self.hosts(role), own.tp, own.ep, own.dp_attention) for role, own in fixed.items()}
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


def _text(value: Any, what: str) -> str:
    """``value`` stripped, refused unless it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise NodeListError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _port(value: Any, what: str) -> int:
    """``value`` as a TCP port."""
    port = _FIELDS.integer(value, what, minimum=1)
    if port > _MAX_PORT:
        raise NodeListError(f"{what} must be at most {_MAX_PORT}, got {port}")
    return port


def _url(value: Any, what: str) -> str:
    """``value`` as an http(s) URL naming a host."""
    text = _text(value, what)
    if not text.startswith(_HTTP) or not urlsplit(text).hostname:
        raise NodeListError(f"{what} must be an http(s) URL, got {text!r}")
    return text


def _host(value: Any, what: str) -> str:
    """The host a host entry names."""
    return _text(_FIELDS.known(value, _HOST_KEYS, what).get("host"), f"{what}'s host")


def _hosts(value: Any, role: Role) -> tuple[str, ...]:
    """The hosts ``role`` names, each once."""
    if not isinstance(value, list) or not value:
        raise NodeListError(f"{role.value} must be a non-empty list of hosts, got {value!r}")
    hosts = tuple(_host(item, f"{role.value}[{index}]") for index, item in enumerate(value))
    _FIELDS.unique(hosts, f"{role.value} hosts")
    return hosts


def _host_ports(fields: Mapping[str, Any]) -> dict[str, int]:
    """The SSH port of every host entry that names its own."""
    ports: dict[str, int] = {}
    for key in (_ROUTER, *(role.value for role in Role)):
        entries = fields.get(key)
        for entry in [entries] if isinstance(entries, Mapping) else entries or []:
            if isinstance(entry, Mapping) and entry.get(_SSH_PORT) is not None:
                ports[str(entry.get("host"))] = _port(entry[_SSH_PORT], f"{key}'s {_SSH_PORT}")
    return ports


def _ssh(value: Any) -> SshAccess:
    """The SSH access ``value`` names; key paths must be absolute."""
    fields = _FIELDS.known(value, _SSH_KEYS, "ssh")
    key_file = _text(fields.get("key_file"), "ssh.key_file")
    if not key_file.startswith("/"):
        raise NodeListError("ssh.key_file must be an absolute path")
    known_hosts = fields.get("known_hosts_file", "")
    if known_hosts and not _text(known_hosts, "ssh.known_hosts_file").startswith("/"):
        raise NodeListError("ssh.known_hosts_file must be an absolute path")
    return SshAccess(
        port=_port(fields.get("port", 22), "ssh.port"),
        user=_text(fields.get("user", "root"), "ssh.user"),
        key_file=key_file,
        known_hosts_file=str(known_hosts or ""),
    )


def _role_topology(value: Any, role: Role) -> RoleTopology:
    """The topology the operator fixed for ``role``."""
    where = f"topology.{role.value}"
    fields = _FIELDS.known(value, _TOPOLOGY_KEYS, where)
    return RoleTopology(
        tp=_FIELDS.optional_integer(fields.get("tp"), f"{where}.tp", minimum=1),
        ep=_FIELDS.optional_integer(fields.get("ep"), f"{where}.ep", minimum=1),
        dp_attention=_FIELDS.optional_boolean(fields.get("dp_attention"), f"{where}.dp_attention"),
    )


def _args(value: Any, what: str) -> tuple[str, ...]:
    """``value`` split into shell tokens; none when it is absent or empty."""
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
    fields = _FIELDS.known(value, (*keys, _ROUTER_ARGS), "config")
    given = {role.value: _args(fields.get(key), f"config.{key}") for key, role in keys.items()}
    return OperatorArgs(**given, router=_args(fields.get(_ROUTER_ARGS), f"config.{_ROUTER_ARGS}"))


def _framework(value: Any) -> Framework:
    """The multi-node framework ``value`` names."""
    if value not in _FRAMEWORKS:
        raise NodeListError(f"framework must be one of {sorted(_FRAMEWORKS)}, got {value!r}")
    return _FRAMEWORKS[value]


def _optional(fields: Mapping[str, Any], key: str, parse: Callable[[Any, str], str]) -> str:
    """``fields[key]`` checked by ``parse``; empty when the key is absent or empty."""
    return parse(fields[key], key) if fields.get(key) else ""


def _frontend(fields: Mapping[str, Any], framework: Framework) -> FrontendType | None:
    """The frontend the node list names, refused when ``framework`` does not serve through it."""
    if not fields.get("frontend"):
        return None
    frontend = _FIELDS.enum(FrontendType, fields["frontend"], "frontend")
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
    _FIELDS.unique(every, f"hosts across {' and '.join(role.value for role in hosts)}")
    return hosts


def role_url(nodes: NodeList, role: Role) -> str:
    """The service URL the node list names for a PD role; empty for a worker."""
    return {Role.PREFILL: nodes.prefill_url, Role.DECODE: nodes.decode_url}.get(role, "")


def engine_port(nodes: NodeList, role: Role) -> int:
    """The port a role's engines serve on: Infera's engine port, else that of the node list's role URL, else the
    framework's default."""
    if nodes.frontend is FrontendType.INFERA:
        return INFERA_ENGINE_PORT
    return urlsplit(role_url(nodes, role)).port or DEFAULT_ENGINE_PORTS[nodes.framework][role]


def parse_node_list(document: Any) -> NodeList:
    """The node list ``document`` states.

    Raises:
        NodeListError: a key is missing, unknown or of the wrong kind, a role the mode does not run names hosts, or
            the roles share a host.
    """
    fields = _FIELDS.known(document, _KEYS, "the node list")
    mode = _FIELDS.enum(PdMode, fields.get("mode", PdMode.DISAGGREGATED.value), "mode")
    framework = _framework(fields["framework"]) if "framework" in fields else None
    frontend = _frontend(fields, framework) if framework is not None else None
    required = (*_REQUIRED, *(() if frontend is FrontendType.INFERA else (_ROUTER,)))
    missing = [key for key in (*required, *(role.value for role in MODE_ROLES[mode])) if key not in fields]
    if missing or framework is None:
        raise NodeListError(f"the node list lacks {missing}")
    hosts = _role_hosts(fields, mode)
    router = _host(fields[_ROUTER], _ROUTER) if fields.get(_ROUTER) is not None else ""
    topology = _FIELDS.known(fields.get("topology") or {}, tuple(role.value for role in MODE_ROLES[mode]), "topology")
    ssh = _ssh(fields["ssh"]) if fields.get("ssh") is not None else None
    ports = _host_ports(fields)
    nodes = NodeList(
        framework=framework,
        model=_text(fields["model"], "model"),
        service_url=_url(fields["service_url"], "service_url"),
        router=router,
        prefill=hosts.get(Role.PREFILL, ()),
        decode=hosts.get(Role.DECODE, ()),
        gpus_per_node=_FIELDS.integer(fields["gpus_per_node"], "gpus_per_node", minimum=1),
        mode=mode,
        worker=hosts.get(Role.WORKER, ()),
        prefill_url=_optional(fields, "prefill_url", _url),
        decode_url=_optional(fields, "decode_url", _url),
        ssh=replace(ssh, host_ports=ports) if ssh is not None else None,
        transfer_backend=_optional(fields, "transfer_backend", _text),
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
