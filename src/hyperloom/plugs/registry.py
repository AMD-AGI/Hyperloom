# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which sockets exist and which implementations can fill them."""

from __future__ import annotations

import dataclasses
import threading
from typing import Mapping

from hyperloom.common.entry_points import DISCOVERY_FAILED, iter_entry_point_objects
from hyperloom.plugs.errors import PlugError
from hyperloom.plugs.socket import Implementation, Socket


class PlugRegistry:
    """Sockets, their built-in implementations and the third-party ones discovered for them."""

    def __init__(self) -> None:
        self._sockets: dict[str, Socket] = {}
        self._impls: dict[tuple[str, str], Implementation] = {}
        self._errors: dict[str, str] = {}
        self._discovered = False
        self._lock = threading.RLock()

    def define(self, socket: Socket) -> None:
        """Add a socket; naming one twice is a programming error."""
        with self._lock:
            if socket.name in self._sockets:
                raise PlugError(f"socket {socket.name!r} is already defined")
            self._sockets[socket.name] = socket

    def socket(self, name: str) -> Socket:
        """Return the defined socket ``name``."""
        try:
            return self._sockets[name]
        except KeyError:
            raise PlugError(f"unknown socket {name!r}; defined: {', '.join(sorted(self._sockets))}") from None

    def socket_names(self) -> frozenset[str]:
        """Every defined socket name."""
        return frozenset(self._sockets)

    def register(self, impl: Implementation) -> None:
        """Add an implementation of a defined socket; the first one registered under a name keeps it."""
        with self._lock:
            self.socket(impl.socket)
            key = (impl.socket, impl.name)
            if key in self._impls:
                raise PlugError(f"implementation {impl.name!r} of socket {impl.socket!r} is already registered")
            self._impls[key] = impl

    def discover(self, *, force: bool = False) -> None:
        """Register the third-party implementations of every open socket, once; a broken plugin is only recorded."""
        with self._lock:
            if self._discovered and not force:
                return
            self._discovered = True
            for socket in self._sockets.values():
                if socket.entry_point_group:
                    self._discover_socket(socket)

    def _discover_socket(self, socket: Socket) -> None:
        for found in iter_entry_point_objects(socket.entry_point_group or ""):
            key = f"{socket.name}/{found.name}"
            if found.error is not None:
                self._errors[DISCOVERY_FAILED if found.name == DISCOVERY_FAILED else key] = found.error
                continue
            try:
                self._register_found(socket, found.name, found.value, found.origin)
            except (PlugError, TypeError, ValueError) as exc:
                self._errors[key] = f"{type(exc).__name__}: {exc}"

    def _register_found(self, socket: Socket, entry_name: str, value: object, origin: str) -> None:
        if not isinstance(value, Implementation):
            raise TypeError(
                "entry point must resolve to Implementation or a zero-argument factory returning Implementation"
            )
        if value.socket != socket.name:
            raise ValueError(f"entry point of socket {socket.name!r} resolved to an implementation of {value.socket!r}")
        if value.name != entry_name:
            raise ValueError(f"entry-point name {entry_name!r} does not match implementation name {value.name!r}")
        # A plugin cannot declare itself built-in: its origin is what the package metadata says it is.
        self.register(dataclasses.replace(value, origin=origin or "unknown"))

    def implementation(self, socket: str, name: str) -> Implementation:
        """Return the implementation ``name`` of ``socket``, discovering plugins first if it is not built in."""
        self.socket(socket)
        found = self._impls.get((socket, name))
        if found is None:
            self.discover()
            found = self._impls.get((socket, name))
        if found is None:
            known = sorted(impl for (sock, impl) in self._impls if sock == socket)
            raise PlugError(
                f"socket {socket!r} has no implementation {name!r}; available: {', '.join(known) or '(none)'}"
                + self._error_suffix(socket)
            )
        return found

    def _error_suffix(self, socket: str) -> str:
        related = {
            key: err for key, err in self._errors.items() if key.startswith(f"{socket}/") or key == DISCOVERY_FAILED
        }
        return f"; plugin errors: {related}" if related else ""

    def plugin_errors(self) -> Mapping[str, str]:
        """Why each plugin that failed to load was skipped, keyed ``<socket>/<entry-point name>``."""
        return dict(self._errors)
