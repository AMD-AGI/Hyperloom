# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``pd_launch.json``: the PD launch a session resolved once, frozen for every round and every resume.

The record holds what each role runs (hosts, topology, args, env, the runner-owned keys taken out of its source),
the router, the KV transfer backend and where the launch came from. Its digest covers the roles' launch, the
frontend and the transfer backend, not the hosts, so a resume refuses a record whose launch was changed. No
credential or key path is part of it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from inference_suite.frameworks import Framework

SCHEMA = "inference-suite-pd-launch/1"
RECORD_NAME = "pd_launch.json"
_UTF8 = "utf-8"


class PdRecordError(ValueError):
    """A PD launch document breaks its contract, or its digest does not match its launch."""


class PdSourceKind(str, Enum):
    #: Per-role args the operator gave.
    OPERATOR = "operator"
    #: An earlier ``pd_launch.json``, replayed exactly.
    REPLAY = "replay"
    #: An ``amd-master.yaml`` multi-node disaggregated entry of InferenceX main.
    INFERENCEX_MASTER = "inferencex-master"
    #: A recipe of InferenceX main's multi-node recipe tree that no master entry names.
    INFERENCEX_LOOKUP = "inferencex-lookup"
    #: A row of the PD default table.
    PD_TABLE = "pd-table"


class FrontendType(str, Enum):
    SGLANG_ROUTER = "sglang-router"
    ATOMESH = "atomesh"
    INFERA = "infera"


class Role(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class PdSource:
    """Where a PD launch came from: the source, the repository commit and file read, the entry and variant chosen,
    the sha256 of every file read, the candidates passed over and the warnings the choice carries."""

    kind: PdSourceKind
    repo: str = ""
    commit: str = ""
    path: str = ""
    entry: str = ""
    locator: str = ""
    files: Mapping[str, str] = field(default_factory=dict)
    alternatives: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "repo": self.repo,
            "commit": self.commit,
            "path": self.path,
            "entry": self.entry,
            "locator": self.locator,
            "files": dict(sorted(self.files.items())),
            "alternatives": list(self.alternatives),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_document(cls, document: Any) -> PdSource:
        fields = _mapping(document, "a PD source")
        return cls(
            kind=_enum(PdSourceKind, fields.get("kind"), "a PD source's kind"),
            repo=_text(fields.get("repo", ""), "a PD source's repo"),
            commit=_text(fields.get("commit", ""), "a PD source's commit"),
            path=_text(fields.get("path", ""), "a PD source's path"),
            entry=_text(fields.get("entry", ""), "a PD source's entry"),
            locator=_text(fields.get("locator", ""), "a PD source's locator"),
            files=_string_map(fields.get("files", {}), "a PD source's files"),
            alternatives=_strings(fields.get("alternatives", []), "a PD source's alternatives"),
            warnings=_strings(fields.get("warnings", []), "a PD source's warnings"),
        )


@dataclass(frozen=True)
class RoleLaunch:
    """What one role runs: its hosts, its workers' topology, its args and env after the runner-set flags, and the
    runner-owned keys taken out of its source."""

    hosts: tuple[str, ...]
    workers: int
    tp: int
    ep: int = 1
    dp_attention: bool = False
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    stripped: tuple[str, ...] = ()

    def launch_document(self) -> dict[str, Any]:
        """The fields the digest covers: everything but the hosts."""
        return {
            "workers": self.workers,
            "topology": {"tp": self.tp, "ep": self.ep},
            "dp_attention": self.dp_attention,
            "args": list(self.args),
            "env": dict(sorted(self.env.items())),
            "stripped": list(self.stripped),
        }

    def to_document(self) -> dict[str, Any]:
        return {"hosts": list(self.hosts), **self.launch_document()}

    @classmethod
    def from_document(cls, document: Any, where: str) -> RoleLaunch:
        fields = _mapping(document, where)
        topology = _mapping(fields.get("topology"), f"{where}'s topology")
        return cls(
            hosts=_strings(fields.get("hosts"), f"{where}'s hosts"),
            workers=_integer(fields.get("workers"), f"{where}'s workers"),
            tp=_integer(topology.get("tp"), f"{where}'s tp"),
            ep=_integer(topology.get("ep", 1), f"{where}'s ep"),
            dp_attention=_boolean(fields.get("dp_attention", False), f"{where}'s dp_attention"),
            args=_strings(fields.get("args", []), f"{where}'s args"),
            env=_string_map(fields.get("env", {}), f"{where}'s env"),
            stripped=_strings(fields.get("stripped", []), f"{where}'s stripped"),
        )


@dataclass(frozen=True)
class Frontend:
    """The router: its type, the host it runs on (empty when the platform runs it) and its args after the
    runner-set flags."""

    type: FrontendType
    host: str = ""
    args: tuple[str, ...] = ()

    def launch_document(self) -> dict[str, Any]:
        return {"type": self.type.value, "args": list(self.args)}

    def to_document(self) -> dict[str, Any]:
        return {"type": self.type.value, "host": self.host, "args": list(self.args)}

    @classmethod
    def from_document(cls, document: Any) -> Frontend:
        fields = _mapping(document, "the frontend")
        return cls(
            type=_enum(FrontendType, fields.get("type"), "the frontend's type"),
            host=_text(fields.get("host", ""), "the frontend's host"),
            args=_strings(fields.get("args", []), "the frontend's args"),
        )


@dataclass(frozen=True)
class PdLaunch:
    """A resolved PD launch: the framework, the model path and the name it is served as, the context the engines
    serve (``None`` for the engine's own), the KV transfer backend, both roles, the frontend and its source."""

    source: PdSource
    framework: Framework
    model: str
    served_model_name: str
    context_length: int | None
    transfer_backend: str
    prefill: RoleLaunch
    decode: RoleLaunch
    frontend: Frontend

    def role(self, role: Role) -> RoleLaunch:
        return self.prefill if role is Role.PREFILL else self.decode

    def digest(self) -> str:
        """The sha256 of the launch: the model, its served name and context, the roles but their hosts, the frontend
        but its host, the transfer backend."""
        launch = {
            "model": self.model,
            "served_model_name": self.served_model_name,
            "context_length": self.context_length,
            "transfer_backend": self.transfer_backend,
            "roles": {role.value: self.role(role).launch_document() for role in Role},
            "frontend": self.frontend.launch_document(),
        }
        canonical = json.dumps(launch, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode(_UTF8)).hexdigest()}"

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source": self.source.to_document(),
            "framework": self.framework.value,
            "model": self.model,
            "served_model_name": self.served_model_name,
            "context_length": self.context_length,
            "transfer_backend": self.transfer_backend,
            "roles": {role.value: self.role(role).to_document() for role in Role},
            "frontend": self.frontend.to_document(),
            "digest": self.digest(),
        }

    @classmethod
    def from_document(cls, document: Any) -> PdLaunch:
        """The launch ``document`` records, refused when its schema or digest does not hold."""
        fields = _mapping(document, "a PD launch")
        if fields.get("schema") != SCHEMA:
            raise PdRecordError(f"a PD launch has schema {fields.get('schema')!r}, not {SCHEMA}")
        roles = _mapping(fields.get("roles"), "a PD launch's roles")
        launch = cls(
            source=PdSource.from_document(fields.get("source")),
            framework=_enum(Framework, fields.get("framework"), "a PD launch's framework"),
            model=_text(fields.get("model"), "a PD launch's model"),
            served_model_name=_text(fields.get("served_model_name"), "a PD launch's served model name"),
            context_length=_optional_integer(fields.get("context_length"), "a PD launch's context length"),
            transfer_backend=_text(fields.get("transfer_backend", ""), "a PD launch's transfer backend"),
            prefill=RoleLaunch.from_document(roles.get(Role.PREFILL.value), "the prefill role"),
            decode=RoleLaunch.from_document(roles.get(Role.DECODE.value), "the decode role"),
            frontend=Frontend.from_document(fields.get("frontend")),
        )
        if fields.get("digest") != launch.digest():
            raise PdRecordError(f"a PD launch's digest {fields.get('digest')!r} is not that of its launch")
        return launch

    def with_hosts(self, prefill: Sequence[str], decode: Sequence[str], router: str) -> PdLaunch:
        """The same launch on other hosts: the digest does not change."""
        return replace(
            self,
            prefill=replace(self.prefill, hosts=tuple(prefill)),
            decode=replace(self.decode, hosts=tuple(decode)),
            frontend=replace(self.frontend, host=router),
        )


def write_record(launch: PdLaunch, path: Path) -> str:
    """Write ``launch`` to ``path`` and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(launch.to_document(), indent=2) + "\n", encoding=_UTF8)
    return launch.digest()


def read_record(path: Path) -> PdLaunch:
    """The launch frozen at ``path``.

    Raises:
        PdRecordError: the file is not a PD launch, or its launch no longer matches its digest.
        OSError: the file cannot be read.
    """
    try:
        document = json.loads(path.read_text(encoding=_UTF8))
    except json.JSONDecodeError as error:
        raise PdRecordError(f"{path} is not JSON: {error}") from error
    return PdLaunch.from_document(document)


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PdRecordError(f"{what} must be a mapping, got {value!r}")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise PdRecordError(f"{what} must be a string, got {value!r}")
    return value


def _integer(value: Any, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PdRecordError(f"{what} must be a positive integer, got {value!r}")
    return value


def _optional_integer(value: Any, what: str) -> int | None:
    return None if value is None else _integer(value, what)


def _boolean(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise PdRecordError(f"{what} must be true or false, got {value!r}")
    return value


def _strings(value: Any, what: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PdRecordError(f"{what} must be a list of strings, got {value!r}")
    return tuple(value)


def _string_map(value: Any, what: str) -> dict[str, str]:
    mapping = _mapping(value, what)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in mapping.items()):
        raise PdRecordError(f"{what} must map strings to strings, got {value!r}")
    return dict(mapping)


def _enum(kind: type[Enum], value: Any, what: str) -> Any:
    try:
        return kind(value)
    except ValueError as error:
        raise PdRecordError(f"{what} must be one of {[member.value for member in kind]}, got {value!r}") from error


__all__ = [
    "RECORD_NAME",
    "SCHEMA",
    "Frontend",
    "FrontendType",
    "PdLaunch",
    "PdRecordError",
    "PdSource",
    "PdSourceKind",
    "Role",
    "RoleLaunch",
    "read_record",
    "write_record",
]
