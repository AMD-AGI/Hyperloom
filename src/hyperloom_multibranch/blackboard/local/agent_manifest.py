# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An autonomous agent's manifest as the blackboard keeps it: every credential dropped, the rest as the runtime gave it.

The manifest says how a third-party agent runs: where its source is, its commands, and the config its campaign is
written into, whose values (model, TP, ISL/OSL, ports, server flags) are what tells two agents' runs apart. A name that
says it holds a credential (one of :data:`CREDENTIAL_WORDS`, as a word of the name, so ``HF_TOKEN`` and ``apiKey`` do
and ``max_tokens`` and ``tokenizer`` do not) has its value replaced by :data:`REDACTED`, at any depth of the document;
so does the value of such a name assigned (``NAME=value``) or passed as a flag (``--name value``) in a command, and the
``source`` URL loses any user and password.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

CREDENTIAL_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "credential"})
REDACTED = "<redacted>"
COMMAND_FIELDS = ("install", "launch", "resume")
SOURCE_FIELD = "source"
_WORDS = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_VALUE = r"(?P<value>\"[^\"]*\"|'[^']*'|\S+)"
_ASSIGNED = re.compile(r"(?P<name>-{0,2}[A-Za-z_][\w.-]*)=" + _VALUE)
_FLAG = re.compile(r"(?P<name>--?[A-Za-z][\w-]*)\s+(?=[^\s-])" + _VALUE)


def is_credential(name: str) -> bool:
    """Whether ``name`` says it holds a credential: one of its words is one of :data:`CREDENTIAL_WORDS`."""

    return any(word.lower() in CREDENTIAL_WORDS for word in _WORDS.findall(name))


def _redacted_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): REDACTED if is_credential(str(key)) else _redacted(item) for key, item in value.items()}


def _redacted(value: object) -> object:
    if isinstance(value, Mapping):
        return _redacted_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    return value


def redacted_command(command: str) -> str:
    """``command`` with the value of every credential it assigns or passes as a flag replaced by :data:`REDACTED`."""

    def replaced(match: re.Match[str]) -> str:
        if not is_credential(match["name"].lstrip("-")):
            return match[0]
        return match[0][: match.start("value") - match.start()] + REDACTED

    return _FLAG.sub(replaced, _ASSIGNED.sub(replaced, command))


def without_userinfo(source: str) -> str:
    """``source`` without the user and password of a URL; a source that names none is returned as it is."""

    parts = urlsplit(source)
    if parts.username is None and parts.password is None:
        return source
    host = parts.hostname or ""
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def sanitized_manifest(document: Mapping[str, object]) -> dict[str, object]:
    """``document`` with every credential dropped, as the blackboard keeps an agent's manifest."""

    kept = _redacted_mapping(document)
    for name in COMMAND_FIELDS:
        value = kept.get(name)
        if isinstance(value, str):
            kept[name] = redacted_command(value)
        elif isinstance(value, list):
            kept[name] = [redacted_command(item) if isinstance(item, str) else item for item in value]
    source = kept.get(SOURCE_FIELD)
    if isinstance(source, str):
        kept[SOURCE_FIELD] = without_userinfo(source)
    return kept


__all__ = [
    "COMMAND_FIELDS",
    "CREDENTIAL_WORDS",
    "REDACTED",
    "SOURCE_FIELD",
    "is_credential",
    "redacted_command",
    "sanitized_manifest",
    "without_userinfo",
]
