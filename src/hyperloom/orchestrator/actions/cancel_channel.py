# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cooperative cancellation channel between the dispatcher and blocking work."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "CancelScope",
    "cancel_scope_listener",
    "current_cancel_scope",
    "stop_was_asked_for",
    "use_cancel_scope",
]


class CancelScope:
    """One action's cancel channel: the flag, why it was raised, who watches it."""

    def __init__(self) -> None:
        """Create an uncancelled scope with no listeners."""
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._listeners = 0

    def cancel(self, *, reason: str) -> None:
        """Ask the work running in this scope to stop."""
        with self._lock:
            if not self._reason:
                self._reason = str(reason)
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """bool: Whether this scope has been cancelled."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """str: Why the scope was cancelled; empty while it is not."""
        with self._lock:
            return self._reason

    @property
    def has_listeners(self) -> bool:
        """bool: Whether any blocking call is currently watching this scope."""
        with self._lock:
            return self._listeners > 0

    @contextmanager
    def listening(self) -> Iterator["CancelScope"]:
        """Count the caller as a watcher for the duration of the block."""
        with self._lock:
            self._listeners += 1
        try:
            yield self
        finally:
            with self._lock:
                self._listeners = max(0, self._listeners - 1)


_CURRENT_SCOPE: ContextVar[CancelScope | None] = ContextVar(
    "hyperloom_cancel_scope",
    default=None,
)


def current_cancel_scope() -> CancelScope | None:
    """Return the cancel scope of the action running in this context."""
    return _CURRENT_SCOPE.get()


def stop_was_asked_for() -> bool:
    """Whether the action running in this context has already been asked to stop."""
    scope = _CURRENT_SCOPE.get()
    return scope is not None and scope.cancelled


@contextmanager
def use_cancel_scope(scope: CancelScope | None) -> Iterator[CancelScope | None]:
    """Publish ``scope`` for the duration of the block."""
    if scope is None:
        yield None
        return
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


@contextmanager
def cancel_scope_listener() -> Iterator[CancelScope | None]:
    """Watch the published scope, if there is one, for the duration of the block."""
    scope = _CURRENT_SCOPE.get()
    if scope is None:
        yield None
        return
    with scope.listening():
        yield scope
