# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a socket is, and what fills one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class Socket:
    """A named place in the product where one of several implementations plugs in."""

    name: str
    protocol: type
    default: str = "local"
    #: Set when the socket is open to third parties: they register under hyperloom.plugs.<name>.
    entry_point_group: str | None = None


@dataclass(frozen=True)
class PlugContext:
    """What an implementation's factory is given."""

    socket: str
    config: Mapping[str, Any]
    session_dir: "Path | None"
    #: Injected by the hosted entry; its type belongs to that entry, which the kernel must not import.
    host_info: object | None
    #: Fetch a socket this implementation declared in ``requires``; it is already started.
    get: Callable[[str], object]


@dataclass(frozen=True)
class Implementation:
    """One way of filling a socket."""

    socket: str
    name: str
    factory: Callable[[PlugContext], object]
    #: Sockets that must be started before this one, and that the factory may ``get``.
    requires: tuple[str, ...] = ()
    #: ``builtin``, or ``<distribution>==<version>`` for a third-party plugin.
    origin: str = "builtin"
