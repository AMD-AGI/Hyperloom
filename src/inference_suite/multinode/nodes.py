# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The node list: the one input of a PD session, a JSON document in ``HYPERLOOM_MN_NODE_LIST``.

It names the framework and model, the cluster's service and engine URLs, the router, prefill and decode hosts, the
GPUs per node, SSH access (the key by path only), the KV transfer backend, the router it runs, the topology and args
the operator fixed, and how the launcher starts processes: ``direct`` over SSH, or ``in_place`` into the process a
platform launcher waits on. Every key is checked; an unknown one is refused.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.frameworks import Framework
from inference_suite.multinode.query import PdQuery, RoleAsk
from inference_suite.multinode.record import FrontendType, Role
from inference_suite.multinode.recipe import OperatorArgs

NODE_LIST_ENV = "HYPERLOOM_MN_NODE_LIST"
_FRAMEWORKS = {Framework.SGLANG.value: Framework.SGLANG, Framework.ATOM.value: Framework.ATOM}
_REQUIRED = ("framework", "model", "service_url", "router", "prefill", "decode", "gpus_per_node")
_KEYS = (
    frozenset(_REQUIRED)
    | {"prefill_url", "decode_url", "ssh", "transfer_backend", "topology", "launch_mode"}
    | {"config", "frontend"}
)
_SSH_KEYS = frozenset({"port", "user", "key_file", "known_hosts_file"})
_TOPOLOGY_KEYS = frozenset({"tp", "ep", "dp_attention"})
_CONFIG_KEYS = frozenset({"prefill_args", "decode_args", "router_args"})
_HOST_KEYS = frozenset({"host"})
_HTTP = ("http://", "https://")
_MAX_PORT = 65535


class NodeListError(ValueError):
    """The node list is absent or breaks its contract."""


class LaunchMode(str, Enum):
    #: Each process starts over SSH as a detached child of the session.
    DIRECT = "direct"
    #: Each process is exec'd into the pid a platform launcher waits on, so the pod is not recreated.
    IN_PLACE = "in_place"


@dataclass(frozen=True)
class SshAccess:
    """How the launcher reaches every host: port, user, the private key's path and an optional known_hosts file."""

    port: int
    user: str
    key_file: str
    known_hosts_file: str = ""


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
    prefill_url: str = ""
    decode_url: str = ""
    ssh: SshAccess | None = None
    transfer_backend: str = ""
    topology: Mapping[Role, RoleTopology] = field(default_factory=dict)
    launch_mode: LaunchMode = LaunchMode.DIRECT
    operator: OperatorArgs = field(default_factory=OperatorArgs)
    frontend: FrontendType | None = None

    def hosts(self, role: Role) -> tuple[str, ...]:
        return self.prefill if role is Role.PREFILL else self.decode

    def query(self, session: LaunchQuery) -> PdQuery:
        """The PD query of this cluster for ``session``, whose framework must be the node list's.

        Raises:
            NodeListError: the session runs another framework.
        """
        if session.framework != self.framework.value:
            raise NodeListError(f"the session runs {session.framework}, the node list {self.framework.value}")
        asks = {}
        for role in Role:
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
            prefill=asks[Role.PREFILL],
            decode=asks[Role.DECODE],
            router_host=self.router,
            isl=session.isl,
            osl=session.osl,
            transfer_backend=self.transfer_backend,
            frontend=self.frontend,
        )


def _mapping(value: Any, what: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NodeListError(f"{what} must be an object, got {type(value).__name__}")
    unknown = sorted(set(value) - keys)
    if unknown:
        raise NodeListError(f"{what} has unknown keys {unknown}")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NodeListError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _integer(value: Any, what: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
        raise NodeListError(f"{what} must be an integer from 1 to {maximum or 'any'}, got {value!r}")
    return value


def _url(value: Any, what: str) -> str:
    text = _text(value, what)
    if not text.startswith(_HTTP) or not urlsplit(text).hostname:
        raise NodeListError(f"{what} must be an http(s) URL, got {text!r}")
    return text


def _host(value: Any, what: str) -> str:
    return _text(_mapping(value, what, _HOST_KEYS).get("host"), f"{what}'s host")


def _hosts(value: Any, role: Role) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise NodeListError(f"{role.value} must be a non-empty list of hosts, got {value!r}")
    hosts = tuple(_host(item, f"{role.value}[{index}]") for index, item in enumerate(value))
    if len(set(hosts)) != len(hosts):
        raise NodeListError(f"{role.value} names a host twice: {list(hosts)}")
    return hosts


def _ssh(value: Any) -> SshAccess:
    fields = _mapping(value, "ssh", _SSH_KEYS)
    key_file = _text(fields.get("key_file"), "ssh.key_file")
    if not key_file.startswith("/"):
        raise NodeListError("ssh.key_file must be an absolute path")
    known_hosts = fields.get("known_hosts_file", "")
    if known_hosts and not _text(known_hosts, "ssh.known_hosts_file").startswith("/"):
        raise NodeListError("ssh.known_hosts_file must be an absolute path")
    return SshAccess(
        port=_integer(fields.get("port", 22), "ssh.port", _MAX_PORT),
        user=_text(fields.get("user", "root"), "ssh.user"),
        key_file=key_file,
        known_hosts_file=str(known_hosts or ""),
    )


def _optional_integer(value: Any, what: str) -> int | None:
    return None if value is None else _integer(value, what)


def _role_topology(value: Any, role: Role) -> RoleTopology:
    fields = _mapping(value, f"topology.{role.value}", _TOPOLOGY_KEYS)
    dp = fields.get("dp_attention")
    if dp is not None and not isinstance(dp, bool):
        raise NodeListError(f"topology.{role.value}.dp_attention must be true, false or null, got {dp!r}")
    return RoleTopology(
        tp=_optional_integer(fields.get("tp"), f"topology.{role.value}.tp"),
        ep=_optional_integer(fields.get("ep"), f"topology.{role.value}.ep"),
        dp_attention=dp,
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


def _operator(value: Any) -> OperatorArgs:
    fields = _mapping(value, "config", _CONFIG_KEYS)
    return OperatorArgs(
        prefill=_args(fields.get("prefill_args"), "config.prefill_args"),
        decode=_args(fields.get("decode_args"), "config.decode_args"),
        router=_args(fields.get("router_args"), "config.router_args"),
    )


def _enum(kind: type[Enum], value: Any, what: str) -> Any:
    try:
        return kind(value)
    except ValueError as error:
        raise NodeListError(f"{what} must be one of {[member.value for member in kind]}, got {value!r}") from error


def _framework(value: Any) -> Framework:
    if value not in _FRAMEWORKS:
        raise NodeListError(f"framework must be one of {sorted(_FRAMEWORKS)}, got {value!r}")
    return _FRAMEWORKS[value]


def _optional_url(fields: Mapping[str, Any], key: str) -> str:
    return _url(fields[key], key) if fields.get(key) else ""


def parse_node_list(document: Any) -> NodeList:
    """The node list ``document`` states.

    Raises:
        NodeListError: a key is missing, unknown or of the wrong kind, or the roles share a host.
    """
    fields = _mapping(document, "the node list", _KEYS)
    missing = [key for key in _REQUIRED if key not in fields]
    if missing:
        raise NodeListError(f"the node list lacks {missing}")
    prefill, decode = _hosts(fields["prefill"], Role.PREFILL), _hosts(fields["decode"], Role.DECODE)
    router = _host(fields["router"], "router")
    shared = sorted(set(prefill) & set(decode))
    if shared:
        raise NodeListError(f"prefill and decode share hosts {shared}")
    topology = _mapping(fields.get("topology") or {}, "topology", frozenset(role.value for role in Role))
    return NodeList(
        framework=_framework(fields["framework"]),
        model=_text(fields["model"], "model"),
        service_url=_url(fields["service_url"], "service_url"),
        router=router,
        prefill=prefill,
        decode=decode,
        gpus_per_node=_integer(fields["gpus_per_node"], "gpus_per_node"),
        prefill_url=_optional_url(fields, "prefill_url"),
        decode_url=_optional_url(fields, "decode_url"),
        ssh=_ssh(fields["ssh"]) if fields.get("ssh") is not None else None,
        transfer_backend=_text(fields["transfer_backend"], "transfer_backend")
        if fields.get("transfer_backend")
        else "",
        topology={Role(key): _role_topology(value, Role(key)) for key, value in topology.items()},
        launch_mode=_enum(LaunchMode, fields.get("launch_mode", LaunchMode.DIRECT.value), "launch_mode"),
        operator=_operator(fields["config"]) if fields.get("config") is not None else OperatorArgs(),
        frontend=_enum(FrontendType, fields["frontend"], "frontend") if fields.get("frontend") else None,
    )


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
    "LaunchMode",
    "NodeList",
    "NodeListError",
    "RoleTopology",
    "SshAccess",
    "parse_node_list",
    "read_node_list",
]
