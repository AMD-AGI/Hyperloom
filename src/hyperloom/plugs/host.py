# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The plug host: starts the manifest's implementations in dependency order and closes them in reverse."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from hyperloom.plugs.errors import PlugError
from hyperloom.plugs.builtin import builtin_registry
from hyperloom.plugs.manifest import ResolvedManifest, expand_manifest, fingerprint, resolve_manifest
from hyperloom.plugs.registry import PlugRegistry
from hyperloom.plugs.socket import Implementation, PlugContext

log = logging.getLogger(__name__)

EVENT_LOCK_MISMATCH = "hyperloom.plugs_lock_mismatch"
_EVENTS_SOCKET = "events"


class PlugHost:
    """The implementations a manifest selects, started once and handed out by socket name."""

    def __init__(
        self,
        manifest: ResolvedManifest,
        registry: PlugRegistry,
        *,
        session_dir: Path | None,
        host_info: object | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._manifest = manifest
        self._registry = registry
        self._session_dir = session_dir
        self._host_info = host_info
        self._configs = expand_manifest(manifest, env)
        self._impls: dict[str, Implementation] = {}
        self._started: dict[str, object] = {}
        self._order: list[str] = []
        self._close_errors: dict[str, str] = {}
        self._lock = threading.RLock()
        self._begun = False

    def plan(self) -> tuple[str, ...]:
        """Check that every implementation exists and the dependencies resolve; return the start order.

        Nothing is constructed, so a configuration error can be refused before any side effect.
        """
        impls = {name: self._registry.implementation(name, s.impl) for name, s in self._manifest.sockets.items()}
        self._impls = impls
        return tuple(_start_order(impls))

    def start(self) -> None:
        """Construct every implementation, dependencies first, and call ``start()`` on those that have one."""
        with self._lock:
            if self._begun:
                raise PlugError("plug host was already started")
            order = self.plan()
            self._begun = True
            for socket in order:
                try:
                    self._start_one(socket)
                except Exception as exc:
                    self.close()
                    raise PlugError(
                        f"socket {socket!r} implementation {self._impls[socket].name!r} failed to start: "
                        f"{type(exc).__name__}: {exc}; plugin errors: {dict(self._registry.plugin_errors())}"
                    ) from exc
            self._report_lock_mismatches()

    def _start_one(self, socket: str) -> None:
        impl = self._impls[socket]
        context = PlugContext(
            socket=socket,
            config=self._configs[socket],
            session_dir=self._session_dir,
            host_info=self._host_info,
            get=lambda name, _impl=impl: self._dependency(_impl, name),
        )
        obj = impl.factory(context)
        starter = getattr(obj, "start", None)
        if callable(starter):
            starter()
        self._started[socket] = obj
        self._order.append(socket)

    def _dependency(self, owner: Implementation, name: str) -> object:
        if name not in owner.requires:
            raise PlugError(f"socket {owner.socket!r} did not declare {name!r} in requires")
        return self.get(name)

    def get(self, socket: str) -> object:
        """The started implementation of ``socket``."""
        try:
            return self._started[socket]
        except KeyError:
            raise PlugError(
                f"socket {socket!r} is not started; started: {', '.join(self._order) or '(none)'}"
            ) from None

    def has(self, socket: str) -> bool:
        """Whether ``socket`` is started."""
        return socket in self._started

    def close(self) -> None:
        """Close what was started, newest first; an I/O failure to close one is recorded and the rest still close."""
        with self._lock:
            while self._order:
                socket = self._order.pop()
                obj = self._started.pop(socket)
                closer = getattr(obj, "close", None)
                if not callable(closer):
                    continue
                try:
                    closer()
                except OSError as exc:
                    self._close_errors[socket] = f"{type(exc).__name__}: {exc}"
                    log.warning("closing plug %s failed: %s", socket, exc, exc_info=True)

    def close_errors(self) -> Mapping[str, str]:
        """Why each implementation that failed to close did."""
        return dict(self._close_errors)

    def _report_lock_mismatches(self) -> None:
        for mismatch in self._manifest.lock_mismatches:
            log.warning(
                "plugs lock mismatch: session file chose %s for %s, locked to %s",
                mismatch.file_impl,
                mismatch.socket,
                mismatch.locked_impl,
            )
            if self.has(_EVENTS_SOCKET):
                self.get(_EVENTS_SOCKET).emit(  # type: ignore[attr-defined]
                    EVENT_LOCK_MISMATCH,
                    {"socket": mismatch.socket, "file_impl": mismatch.file_impl, "locked_impl": mismatch.locked_impl},
                )

    def fingerprint(self) -> str:
        """Fingerprint of the selection, which covers the implementations' origins and expanded configuration."""
        if not self._impls:
            self.plan()
        rows = {
            name: {"impl": s.impl, "origin": self._impls[name].origin, "config": self._configs[name]}
            for name, s in self._manifest.sockets.items()
        }
        return fingerprint(self._manifest.profile, rows)

    def describe(self) -> dict[str, Any]:
        """The final manifest as it is traced: configuration as written, never expanded."""
        if not self._impls:
            self.plan()
        return {
            "profile": self._manifest.profile,
            "fingerprint": self.fingerprint(),
            "sockets": {
                name: {
                    "impl": s.impl,
                    "origin": self._impls[name].origin,
                    "layer": s.layer,
                    "config": dict(s.config),
                }
                for name, s in sorted(self._manifest.sockets.items())
            },
        }


def _start_order(impls: Mapping[str, Implementation]) -> list[str]:
    """Topological order of the sockets by ``requires``; a missing dependency or a cycle is refused."""
    for socket, impl in impls.items():
        missing = [dep for dep in impl.requires if dep not in impls]
        if missing:
            raise PlugError(
                f"socket {socket!r} implementation {impl.name!r} requires {missing}, which the manifest does not select"
            )
    order: list[str] = []
    waiting = {socket: set(impl.requires) for socket, impl in impls.items()}
    while waiting:
        ready = sorted(socket for socket, deps in waiting.items() if deps <= set(order))
        if not ready:
            raise PlugError(f"plug dependencies form a cycle among {sorted(waiting)}")
        for socket in ready:
            order.append(socket)
            del waiting[socket]
    return order


_current: PlugHost | None = None
_default: PlugHost | None = None
_current_lock = threading.RLock()


def active() -> PlugHost:
    """The process's current host. With none activated, one is created on first use from the built-in local profile."""
    global _default
    with _current_lock:
        if _current is not None:
            return _current
        if _default is None:
            registry = builtin_registry()
            host = PlugHost(resolve_manifest(registry), registry, session_dir=None)
            host.start()
            _default = host
        return _default


@contextmanager
def activated(host: PlugHost) -> Iterator[PlugHost]:
    """Make ``host`` the current host for the duration, then restore the previous one."""
    global _current
    with _current_lock:
        previous = _current
        _current = host
    try:
        yield host
    finally:
        with _current_lock:
            _current = previous
