# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An autonomous agent's manifest as the blackboard keeps it: every credential dropped, the rest as the runtime gave it.

The manifest says how a third-party agent runs: where its source is, its commands, and the config its campaign is
written into, whose values (model, TP, ISL/OSL, ports, server flags) are what tells two agents' runs apart.

Credentials go as they go from every record that leaves a process, through :mod:`hyperloom.common.env_safety`: at any
depth of the document, a name it reads as a credential has its value replaced by :data:`REDACTED`, and every string
has the credentials it recognises replaced (an ``Authorization`` header, a bearer or API token, a secret assigned in a
command). Three shapes it does not read are replaced here as well: a name one of whose words is one of
:data:`CREDENTIAL_WORDS`, as a JSON key is camel-cased (``apiKey``, ``accessToken``) where an env name is not, so
``max_tokens`` and ``tokenizer`` stay; the value of such a name assigned (``NAME=value``) or passed as a flag
(``--name value``) in a command; and the user and password of a URL. The ``source`` URL loses its user and password
altogether.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from hyperloom.common.env_safety import redact_secret_env_values, redact_secret_values

CREDENTIAL_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "credential"})
#: The mask :mod:`~hyperloom.common.env_safety` writes, so a value it masked and one masked here read alike.
REDACTED = "[REDACTED]"
COMMAND_FIELDS = ("install", "launch", "resume")
SOURCE_FIELD = "source"
_WORDS = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
_VALUE = r"(?P<value>\"[^\"]*\"|'[^']*'|\S+)"
_ASSIGNED = re.compile(r"(?P<name>-{0,2}[A-Za-z_][\w.-]*)=" + _VALUE)
_FLAG = re.compile(r"(?P<name>--?[A-Za-z][\w-]*)\s+(?=[^\s-])" + _VALUE)
_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@")


def is_credential(name: str) -> bool:
    """Whether ``name`` says it holds a credential: one of its words is one of :data:`CREDENTIAL_WORDS`."""

    return any(word.lower() in CREDENTIAL_WORDS for word in _WORDS.findall(name))


def _masked(text: str) -> str:
    return _USERINFO.sub(rf"\1{REDACTED}@", redact_secret_values(text))


def _redacted_mapping(value: Mapping[str, object]) -> dict[str, object]:
    screened = redact_secret_env_values({str(key): item for key, item in value.items()})
    return {key: REDACTED if is_credential(key) else _redacted(item) for key, item in screened.items()}


def _redacted(value: object) -> object:
    if isinstance(value, Mapping):
        return _redacted_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    return _masked(value) if isinstance(value, str) else value


def redacted_command(command: str) -> str:
    """``command`` with every credential replaced by :data:`REDACTED`: those
    :func:`~hyperloom.common.env_safety.redact_secret_values` recognises, a URL's user and password, and the value of
    every credential it assigns or passes as a flag."""

    def replaced(match: re.Match[str]) -> str:
        if not is_credential(match["name"].lstrip("-")):
            return match[0]
        return match[0][: match.start("value") - match.start()] + REDACTED

    return _FLAG.sub(replaced, _ASSIGNED.sub(replaced, _masked(command)))


def without_userinfo(source: str) -> str:
    """``source`` without the user and password of a URL; a source that names none, or is no URL Python can split, is
    returned as it is."""

    try:
        parts = urlsplit(source)
        if parts.username is None and parts.password is None:
            return source
        host = parts.hostname or ""
        netloc = host if parts.port is None else f"{host}:{parts.port}"
    except ValueError:
        return source
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def sanitized_manifest(document: Mapping[str, object]) -> dict[str, object]:
    """``document`` with every credential dropped, as the blackboard keeps an agent's manifest."""

    kept = _redacted_mapping(document)
    for name in COMMAND_FIELDS:
        value = document.get(name)
        if isinstance(value, str):
            kept[name] = redacted_command(value)
        elif isinstance(value, list):
            kept[name] = [redacted_command(item) if isinstance(item, str) else _redacted(item) for item in value]
    source = document.get(SOURCE_FIELD)
    if isinstance(source, str):
        kept[SOURCE_FIELD] = _masked(without_userinfo(source))
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
