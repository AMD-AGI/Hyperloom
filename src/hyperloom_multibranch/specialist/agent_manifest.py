# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch runs its third-party agent by: the manifest its onboarding writes.

Onboarding reads the agent's repository once and answers with one JSON object, which :meth:`AgentManifest.read`
validates before anything runs: every command is a non-empty string, every path is relative and stays inside the
agent's checkout, and a key the schema does not have is refused, so a misspelt field fails the attempt instead of
silently meaning nothing. The manifest says how to install and launch the agent, the config it is launched with, how
a kept change shows (:class:`KeepSignal`), where the agent keeps the server arguments and environment it runs with
(:class:`ConfigSource`) and what it found (:class:`LedgerSource`); an agent with no keep signal is read once, from its
workspace.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind

#: The field that names a file in the agent's checkout, in every part of the manifest that reads one.
_PATH_FIELD = "path"


class ManifestError(ValueError):
    """An onboarding answer that is not a valid manifest; the message names the field."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _optional(value: object, name: str) -> str:
    """An optional text field: absent and ``null`` are both empty."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ManifestError(f"{name} must be a string or null, got {value!r}")
    return value


def _relative(value: object, name: str) -> str:
    path = PurePosixPath(_text(value, name))
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{name} must be a path inside the agent's checkout, got {str(path)!r}")
    return str(path)


def _fields(document: object, name: str, allowed: set[str]) -> Mapping[str, Any]:
    if not isinstance(document, Mapping):
        raise ManifestError(f"{name} must be an object, got {document!r}")
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ManifestError(f"{name} has fields the manifest does not define: {', '.join(unknown)}")
    return document


def _commands(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ManifestError(f"{name} must be a list of shell commands, got {value!r}")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def read_field(state: object, dotted: str) -> object:
    """The value at the ``.``-separated path ``dotted`` of the parsed JSON ``state``, a step that is an integer
    indexing a list; raise ``KeyError``, ``IndexError`` or ``ValueError`` when it is not there."""

    for step in dotted.split("."):
        if isinstance(state, Mapping):
            state = state[step]
        elif isinstance(state, Sequence) and not isinstance(state, str):
            state = state[int(step)]
        else:
            raise KeyError(dotted)
    return state


@dataclass(frozen=True)
class KeepSignal:
    """How a kept change shows: the JSON file at ``path`` whose ``counter`` field counts the keeps, as a number or as
    the list items matching every pair of ``where``; with no ``counter``, every change of the file is a keep."""

    path: str
    counter: str = ""
    where: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def read(cls, document: object) -> KeepSignal:
        fields = _fields(document, "keep", {_PATH_FIELD, "field", "where"})
        where = fields.get("where", {})
        if not isinstance(where, Mapping):
            raise ManifestError(f"keep.where must be an object, got {where!r}")
        return cls(
            _relative(fields.get("path"), "keep.path"), _optional(fields.get("field"), "keep.field"), dict(where)
        )

    def mark(self, raw: bytes) -> int | str:
        """What the keep file's bytes ``raw`` say of the keeps: the count of :meth:`count`, or, with no ``counter``,
        the digest of the file, whose every change is one keep; raise ``ValueError``, ``KeyError`` or ``IndexError``
        when it cannot be read, as a file the agent is writing cannot."""

        if not self.counter:
            return hashlib.sha256(raw).hexdigest()
        return self.count(json.loads(raw))

    def count(self, state: object) -> int:
        """The keeps the parsed keep file ``state`` records."""

        found = read_field(state, self.counter)
        if isinstance(found, Sequence) and not isinstance(found, str):
            return sum(
                1
                for item in found
                if isinstance(item, Mapping) and all(item.get(key) == value for key, value in self.where.items())
            )
        if isinstance(found, int) and not isinstance(found, bool):
            return found
        raise ValueError(f"keep.field {self.counter} is neither a list nor a count: {found!r}")

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "field": self.counter, "where": dict(self.where)}


@dataclass(frozen=True)
class ConfigSource:
    """Where the agent keeps what it launches the server with: the JSON file ``path`` in its workspace, with the
    server arguments as one string under ``server_args`` and the environment as an object under ``envs``."""

    path: str
    server_args: str
    envs: str

    @classmethod
    def read(cls, document: object) -> ConfigSource:
        fields = _fields(document, "config_source", {_PATH_FIELD, "server_args", "envs"})
        return cls(
            _relative(fields.get("path"), "config_source.path"),
            _text(fields.get("server_args"), "config_source.server_args"),
            _text(fields.get("envs"), "config_source.envs"),
        )

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "server_args": self.server_args, "envs": self.envs}


@dataclass(frozen=True)
class LedgerSource:
    """Where the agent writes down what it found: the JSON-lines file ``path`` in its checkout, whose entries hold an
    id under ``id``, a kind under ``kind`` and the finding under ``text``; ``kinds`` maps each of the agent's kinds
    worth a learning to the learning it is, and an entry whose id comes again revises the earlier one."""

    path: str
    id: str
    kind: str
    text: str
    kinds: Mapping[str, LearningKind]

    @classmethod
    def read(cls, document: object) -> LedgerSource:
        fields = _fields(document, "ledger", {_PATH_FIELD, "id", "kind", "text", "kinds"})
        kinds = fields.get("kinds")
        if not isinstance(kinds, Mapping) or not kinds:
            raise ManifestError(f"ledger.kinds must be a non-empty object, got {kinds!r}")
        try:
            learning_kinds = {str(kind): LearningKind(learning) for kind, learning in kinds.items()}
        except ValueError as error:
            names = ", ".join(kind.value for kind in LearningKind)
            raise ManifestError(f"ledger.kinds maps to {names} only: {error}") from error
        return cls(
            _relative(fields.get("path"), "ledger.path"),
            _text(fields.get("id"), "ledger.id"),
            _text(fields.get("kind"), "ledger.kind"),
            _text(fields.get("text"), "ledger.text"),
            learning_kinds,
        )

    def to_document(self) -> dict[str, object]:
        kinds = {kind: learning.value for kind, learning in self.kinds.items()}
        return {"path": self.path, "id": self.id, "kind": self.kind, "text": self.text, "kinds": kinds}


_MANIFEST_FIELDS = {
    "source",
    "commit",
    "install",
    "config_file",
    "config",
    "launch",
    "resume",
    "keep",
    "config_source",
    "summary_field",
    "needs",
    "ledger",
}


@dataclass(frozen=True)
class AgentManifest:
    """How to run one agent: ``install`` once, write ``config`` to ``config_file``, run ``launch`` (``resume`` after a
    crash, ``launch`` again when empty); ``keep`` is ``None`` for an agent collected once, from its workspace;
    ``summary_field`` is where the keep file holds the agent's words for its latest keep; ``needs`` is what
    the agent requires of its sandbox; ``ledger`` is ``None`` for an agent that writes down no findings."""

    source: str
    commit: str
    install: tuple[str, ...]
    config_file: str
    config: Mapping[str, Any]
    launch: str
    resume: str = ""
    keep: KeepSignal | None = None
    config_source: ConfigSource | None = None
    summary_field: str = ""
    needs: tuple[str, ...] = ()
    ledger: LedgerSource | None = None

    @classmethod
    def read(cls, document: object) -> AgentManifest:
        """The manifest ``document`` holds; raise :class:`ManifestError` naming the first field it gets wrong."""

        fields = _fields(document, "the manifest", _MANIFEST_FIELDS)
        config = fields.get("config")
        if not isinstance(config, Mapping):
            raise ManifestError(f"config must be an object, got {config!r}")
        keep, source, ledger = fields.get("keep"), fields.get("config_source"), fields.get("ledger")
        return cls(
            source=_text(fields.get("source"), "source"),
            commit=_text(fields.get("commit"), "commit"),
            install=_commands(fields.get("install", []), "install"),
            config_file=_relative(fields.get("config_file"), "config_file"),
            config=dict(config),
            launch=_text(fields.get("launch"), "launch"),
            resume=_optional(fields.get("resume"), "resume"),
            keep=None if keep is None else KeepSignal.read(keep),
            config_source=None if source is None else ConfigSource.read(source),
            summary_field=_optional(fields.get("summary_field"), "summary_field"),
            needs=_commands(fields.get("needs", []), "needs"),
            ledger=None if ledger is None else LedgerSource.read(ledger),
        )

    def to_document(self) -> dict[str, object]:
        return {
            "source": self.source,
            "commit": self.commit,
            "install": list(self.install),
            "config_file": self.config_file,
            "config": dict(self.config),
            "launch": self.launch,
            "resume": self.resume,
            "keep": None if self.keep is None else self.keep.to_document(),
            "config_source": None if self.config_source is None else self.config_source.to_document(),
            "summary_field": self.summary_field,
            "needs": list(self.needs),
            "ledger": None if self.ledger is None else self.ledger.to_document(),
        }


__all__ = ["AgentManifest", "ConfigSource", "KeepSignal", "LedgerSource", "ManifestError", "read_field"]
