# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The built-in sockets and implementations. The only place an implementation is registered, by import path."""

from __future__ import annotations

import importlib
from typing import Callable

from hyperloom.plugs.events import ProgressEvents
from hyperloom.plugs.registry import PlugRegistry
from hyperloom.plugs.socket import Implementation, PlugContext, Socket


def lazy(target: str) -> Callable[[PlugContext], object]:
    """A factory that imports ``module:function`` when it is first called, not when it is registered."""
    module_name, _, attr = target.partition(":")

    def factory(context: PlugContext) -> object:
        return getattr(importlib.import_module(module_name), attr)(context)

    return factory


#: A socket whose protocol is defined by the change that adds its first implementation uses ``object`` until then.
_SOCKETS = (
    Socket("measurement", object, entry_point_group="hyperloom.plugs.measurement"),
    Socket("sandbox", object, entry_point_group="hyperloom.plugs.sandbox"),
    Socket("agent_runner", object, entry_point_group="hyperloom.plugs.agent_runner"),
    Socket("events", ProgressEvents),
    Socket("storage", object),
    Socket("knowledge", object),
    Socket("gpu_token", object),
    Socket("credentials", object),
)

_IMPLEMENTATIONS = (Implementation("events", "local", lazy("hyperloom.plugs.events_local:create")),)


def builtin_registry() -> PlugRegistry:
    """A registry holding every socket and built-in implementation, with no plugin discovered yet."""
    registry = PlugRegistry()
    for socket in _SOCKETS:
        registry.define(socket)
    for impl in _IMPLEMENTATIONS:
        registry.register(impl)
    return registry
