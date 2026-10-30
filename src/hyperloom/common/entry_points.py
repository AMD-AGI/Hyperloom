# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Entry-point discovery shared by every pluggable registry.

One entry point is one third-party contribution. Loading it runs foreign code, so a
failure to load is returned as data for the caller to record instead of raised: a
broken plugin must not take down the others, or the host process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Iterator

log = logging.getLogger(__name__)

#: ``EntryPointObject.name`` of the one record that reports a failure to read the package metadata at all.
DISCOVERY_FAILED = "<discovery>"


@dataclass(frozen=True)
class EntryPointObject:
    """One entry point, loaded, or the reason it could not be."""

    name: str
    group: str
    #: What the entry point resolved to, with a zero-argument factory already called. None when ``error`` is set.
    value: object | None = None
    error: str | None = None
    #: ``<distribution>==<version>`` of the package that declared it; empty when the metadata does not say.
    origin: str = ""


def _origin_of(entry: object) -> str:
    """Name the distribution that declared ``entry``, as ``<name>==<version>``."""
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None)
    version = getattr(dist, "version", None)
    return f"{name}=={version}" if name and version else ""


def _load(entry: object, name: str, group: str) -> EntryPointObject:
    """Load one entry point; a callable that is not a factory result is called with no arguments."""
    origin = _origin_of(entry)
    try:
        loaded = entry.load()  # type: ignore[attr-defined]
        value = loaded() if callable(loaded) else loaded
    except Exception as exc:  # isolation boundary: one third-party entry point
        log.warning("failed to load entry point %s of group %s: %s", name, group, exc, exc_info=True)
        return EntryPointObject(name=name, group=group, error=f"{type(exc).__name__}: {exc}", origin=origin)
    return EntryPointObject(name=name, group=group, value=value, origin=origin)


def iter_entry_point_objects(group: str) -> Iterator[EntryPointObject]:
    """Yield every entry point of ``group``, loaded one by one.

    An entry point may resolve to the object itself or to a zero-argument factory for it. Validating what it
    resolved to is the caller's: this function knows nothing of the registry's types. Two distributions declaring
    the same name are both yielded, so the caller's own duplicate check sees and records the conflict.

    When the package metadata cannot be read, one record named :data:`DISCOVERY_FAILED` carries the error and
    nothing else is yielded.
    """
    try:
        entries = list(metadata.entry_points().select(group=group))
    except (OSError, ValueError, TypeError) as exc:
        log.warning("entry-point discovery of group %s failed: %s", group, exc)
        yield EntryPointObject(name=DISCOVERY_FAILED, group=group, error=f"{type(exc).__name__}: {exc}")
        return
    for entry in entries:
        yield _load(entry, str(getattr(entry, "name", "")), group)
