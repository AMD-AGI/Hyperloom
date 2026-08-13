# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cooperative cancellation channel between the dispatcher and blocking work.

Every benchmark executor spends its time inside ``asyncio.to_thread``, and a
thread that has already started cannot be cancelled. Cancelling the coroutine
returns a clean ``CancelledError`` to the canceller while the subprocess it was
waiting on keeps running -- long enough for the lanes and the GPU lease to be
released, and the database closed, under a benchmark that still owns the card.

So the thread needs something it can check. A :class:`CancelScope` carries a
:class:`threading.Event` on a :class:`contextvars.ContextVar`: the dispatcher
publishes one per action, and ``asyncio.to_thread`` copies the context into the
worker thread, so code many frames down finds it without every executor
signature growing a parameter. Blocking code checks the scope at whatever
interval it already polls at and stops itself.

Cooperative means exactly that: work that never looks at the scope cannot be
stopped through it, which is why a scope also counts its listeners -- the
canceller waits only for work that can hear it.

The channel is in-process: this ContextVar is unset inside a Ray actor, so a
round running there cannot read the scope. What crosses is the request, not the
channel -- :class:`..executors._ray_serving.ServingLease` watches the scope on
the submitter's side and forwards a cancel to the actor, which publishes a scope
of its own around the round so the same reaper stops it. Work put on Ray by
anything that does not do that forwarding is out of reach, and stops only on the
session deadline it was handed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

__all__ = [
    "CancelScope",
    "cancel_scope_listener",
    "current_cancel_scope",
    "use_cancel_scope",
]


class CancelScope:
    """One action's cancel channel: the flag, why it was raised, who watches it.

    Every method is safe to call from any thread: the flag is a
    :class:`threading.Event` and the listener count is taken under a lock, since
    the writer is the event loop and the readers are worker threads.
    """

    def __init__(self) -> None:
        """Create an uncancelled scope with no listeners."""
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._listeners = 0

    def cancel(self, *, reason: str) -> None:
        """Ask the work running in this scope to stop.

        Returns as soon as the flag is raised -- whoever is watching it decides
        how to stop, and how long that takes.

        Args:
            reason (str): Short cause, kept for the log line the stopped work
                writes. The first reason wins, so a later blanket cancel cannot
                overwrite the specific one that got there first.
        """
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
        """Count the caller as a watcher for the duration of the block.

        Yields:
            CancelScope: This scope, so the block can check it.
        """
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
    """Return the cancel scope of the action running in this context.

    Returns:
        CancelScope | None: The published scope, or ``None`` when the caller is
            not running under one -- a Ray worker, a unit test, or any code the
            dispatcher did not start.
    """
    return _CURRENT_SCOPE.get()


@contextmanager
def use_cancel_scope(scope: CancelScope | None) -> Iterator[CancelScope | None]:
    """Publish ``scope`` for the duration of the block.

    Must be entered inside the task that runs the action: a task copies the
    context at creation, so a value set afterwards from outside never reaches
    it.

    Args:
        scope (CancelScope | None): The scope to publish; ``None`` leaves the
            context untouched, for callers with nothing to cancel.

    Yields:
        CancelScope | None: The published scope, unchanged.
    """
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
    """Watch the published scope, if there is one, for the duration of the block.

    Registering is what tells the canceller this work can be asked to stop, so
    the window has to cover everything the cancel is meant to reach -- from the
    moment the child is spawned, not from the first poll.

    Yields:
        CancelScope | None: The scope to check, or ``None`` when none is
            published, in which case the block is a plain no-op.
    """
    scope = _CURRENT_SCOPE.get()
    if scope is None:
        yield None
        return
    with scope.listening():
        yield scope
