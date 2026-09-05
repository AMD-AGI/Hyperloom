# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Narrow, single-round exemptions from the blocked environment-variable set.

A grant lifts one name from ``hyperloom.common.env_safety``'s blocked set for
one round, and only for one exact value. Loader search paths are granted as a
prefix so the existing search path survives.

A credential name is never liftable, listed or unlisted: the request comes from
a round's own deliverable, so a grantable secret name is a way for an authored
round to put a key into the benchmark environment and into the materialized
``config.yaml`` on disk.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.common.env_safety import (
    BENCHMARK_SECRET_ENV_NAMES,
    BLOCKED_UNTRUSTED_ENV_NAMES,
    BLOCKED_VARIANT_ENV_NAMES,
    is_secret_shaped_env_name,
    valid_env_key,
)

#: Blocked names that select which copy of a library or module answers. These
#: are grantable, and only as a prefix.
LOADER_SEARCH_PATH_NAMES: frozenset[str] = frozenset({"LD_LIBRARY_PATH", "PYTHONPATH"})

#: Blocked names no grant may ever lift. Two families, for two reasons:
#: every untrusted name that is not a loader search path runs code before the
#: served process reaches its entrypoint; and
#: every credential name is scrubbed on purpose, so re-admitting one would put
#: a key back into a benchmark environment and into the ``config.yaml`` that
#: environment is written to. ``BLOCKED_VARIANT_ENV_NAMES`` is exactly these
#: two families plus the loader paths, so without the second line the only
#: grantable non-path names in the whole set are the credentials.
#:
#: Derived so a name added to ``env_safety`` is refused without a second edit
#: here.
NEVER_GRANTABLE: frozenset[str] = (BLOCKED_UNTRUSTED_ENV_NAMES - LOADER_SEARCH_PATH_NAMES) | BENCHMARK_SECRET_ENV_NAMES

#: Refusal reasons, recorded on the round.
NOT_BLOCKED = "name_is_not_blocked"
NAME_NOT_GRANTABLE = "name_is_never_grantable"
SECRET_SHAPED_NAME = "name_looks_like_a_credential"
INVALID_NAME = "invalid_env_key"
EMPTY_VALUE = "empty_value"
UNSAFE_VALUE = "value_contains_control_characters"

#: Characters that would let a value break out of the assignment it rides in.
_UNSAFE_VALUE_CHARS = ("\x00", "\n", "\r")


class GrantRefused(ValueError):
    """A requested grant cannot be issued; the reason is the message."""


@dataclass(frozen=True)
class EnvGrant:
    """One round's authorisation to set one blocked name to one value.

    Attributes:
        name: The blocked environment variable name, upper-cased.
        value: The exact value authorised; no other value is covered.
        round_key: The round that raised the need and may consume it.
        prefix: Whether the value is prepended rather than replacing. Always
            true for a loader search path.
        reason: Why the round needed it, for the record.
    """

    name: str
    value: str
    round_key: str
    prefix: bool
    reason: str = ""

    def covers(self, name: str, value: str) -> bool:
        """Whether this grant authorises a specific assignment.

        Args:
            name: The name being assigned.
            value: The value being assigned.

        Returns:
            bool: True only when both name and value match.
        """
        return name.strip().upper() == self.name and value == self.value

    def applied_to(self, existing: str) -> str:
        """Return the value to write, given what the configuration already holds.

        Args:
            existing: The value already in the materialized configuration; may
                be empty.

        Returns:
            str: The prefixed value for a prefix grant, skipping an entry the
            path already carries; the granted value for a plain one.
        """
        current = existing.strip()
        if not self.prefix:
            return self.value
        if not current:
            return self.value
        if self.value in current.split(os.pathsep):
            return current
        return f"{self.value}{os.pathsep}{current}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "name": self.name,
            "value": self.value,
            "round_key": self.round_key,
            "prefix": self.prefix,
            "reason": self.reason,
        }


def issue_grant(*, name: str, value: str, round_key: str, reason: str = "") -> EnvGrant:
    """Issue a grant for one name/value pair, or refuse it.

    Args:
        name: The blocked name the round needs to set.
        value: The exact value it needs to set it to.
        round_key: The round raising the need.
        reason: Why the round needs it.

    Returns:
        EnvGrant: The issued grant; a loader search path is always a prefix
        grant.

    Raises:
        GrantRefused: When the name needs no grant, can never be granted, looks
            like a credential, or the value is unusable.
    """
    key = name.strip().upper()
    if not valid_env_key(key):
        raise GrantRefused(f"{INVALID_NAME}: {name!r}")
    if key in NEVER_GRANTABLE:
        raise GrantRefused(f"{NAME_NOT_GRANTABLE}: {key}")
    # The listed credential names are already in ``NEVER_GRANTABLE``; this
    # catches the unlisted ones by shape, the same test that keeps a credential
    # out of a session YAML by name alone.
    if is_secret_shaped_env_name(key):
        raise GrantRefused(f"{SECRET_SHAPED_NAME}: {key}")
    if key not in BLOCKED_VARIANT_ENV_NAMES:
        raise GrantRefused(f"{NOT_BLOCKED}: {key}")
    text = value.strip()
    if not text:
        raise GrantRefused(f"{EMPTY_VALUE}: {key}")
    if any(ch in text for ch in _UNSAFE_VALUE_CHARS):
        raise GrantRefused(f"{UNSAFE_VALUE}: {key}")
    return EnvGrant(
        name=key,
        value=text,
        round_key=round_key,
        prefix=key in LOADER_SEARCH_PATH_NAMES,
        reason=reason,
    )


def parse_requests(raw: Any, *, round_key: str) -> tuple[tuple[EnvGrant, ...], tuple[str, ...]]:
    """Turn a round's declared grant requests into grants, naming every refusal.

    Args:
        raw: The ``env_grant_requests`` value from the round's deliverable: a
            list of ``{"name", "value", "reason"}`` mappings.
        round_key: The round raising the requests.

    Returns:
        tuple: ``(grants, refusals)``. Refusals are returned rather than raised
        so one bad request does not discard a good one.
    """
    grants: list[EnvGrant] = []
    refusals: list[str] = []
    if raw is None:
        return (), ()
    if not isinstance(raw, (list, tuple)):
        return (), (f"env_grant_requests must be a list, got {type(raw).__name__}",)
    for entry in raw:
        if not isinstance(entry, Mapping):
            refusals.append("malformed grant request")
            continue
        name = entry.get("name")
        value = entry.get("value")
        reason = entry.get("reason", "")
        if not isinstance(name, str) or not isinstance(value, str) or not isinstance(reason, str):
            refusals.append("malformed grant request")
            continue
        try:
            grants.append(issue_grant(name=name, value=value, round_key=round_key, reason=reason))
        except GrantRefused as exc:
            refusals.append(str(exc))
    return tuple(grants), tuple(refusals)


__all__ = [
    "LOADER_SEARCH_PATH_NAMES",
    "NEVER_GRANTABLE",
    "EnvGrant",
    "GrantRefused",
    "parse_requests",
]
