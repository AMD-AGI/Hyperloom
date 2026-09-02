# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session that recording entry points write into, bound once at startup.

A fact is recorded where it is born, which is usually many frames below
whoever knows the session directory. The alternative to binding it here is a
``session_dir`` parameter on every recorder entry point, and that pushes the
problem outward rather than solving it: a call site that wants to record a fact
must first acquire the session directory, so recording only spreads as far as
that path already reaches. Binding once means the path appears in one place and
every entry point below it takes no path at all.

The binding lives on a :class:`contextvars.ContextVar` rather than a module
global for two reasons.

Tests get :func:`session_scope`, so a session bound by one case cannot leak
into the next -- a module global would have to be torn down by hand in every
test that touches recording.

And a ContextVar is unset inside a Ray actor, so a subprocess that tries to
record fails on :class:`SessionNotBoundError` instead of silently writing
fragments into a path it inherited from its parent. That failure is the point:
``Recorder.record_upsert_*`` reads, merges and rewrites a fragment while
holding an in-process lock only, so two processes upserting the same fragment
lose one side of the merge. ``test_breakdown_recorder_no_subprocess_writers``
stops a subprocess from importing the recorder at all; this is the runtime
backstop for anything that gets past it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path

__all__ = [
    "SessionNotBoundError",
    "bind_session",
    "bound_session",
    "bound_session_or_none",
    "session_is_bound",
    "session_scope",
    "unbind_session",
]


class SessionNotBoundError(RuntimeError):
    """Raised when a recording entry point runs with no session bound.

    Signals one of two things, and the message says which is likelier: startup
    never called :func:`bind_session`, or the caller is running in a
    subprocess, where the binding does not exist and writing fragments is
    unsafe anyway.
    """


_CURRENT_SESSION: ContextVar[Path | None] = ContextVar(
    "hyperloom_recording_session",
    default=None,
)


def _canonical(session_dir: Path | str) -> Path:
    """Return the canonical form of ``session_dir`` for binding.

    Resolving matters because downstream caches are keyed by the derived spool
    path: two spellings of one session directory must not produce two
    recorders, each holding its own lock, both upserting the same fragments.

    Args:
        session_dir (Path | str): The session root directory, in any spelling.

    Returns:
        Path: The expanded, resolved session root.

    Raises:
        ValueError: If ``session_dir`` is empty.
    """
    if not str(session_dir or "").strip():
        raise ValueError("session_dir must be non-empty")
    return Path(session_dir).expanduser().resolve()


def bind_session(session_dir: Path | str) -> Token[Path | None]:
    """Bind ``session_dir`` as the session every recording entry point writes to.

    Called once per process that owns a session -- the coordinator at startup,
    or a pre-session CLI stage once its session directory exists. Callers that
    need the binding to end at a known point should use :func:`session_scope`
    instead of resetting the token by hand.

    Args:
        session_dir (Path | str): The session root directory to bind.

    Returns:
        Token[Path | None]: The token restoring the previous binding, for
            :func:`unbind_session`.

    Raises:
        ValueError: If ``session_dir`` is empty.
    """
    return _CURRENT_SESSION.set(_canonical(session_dir))


def unbind_session(token: Token[Path | None]) -> None:
    """Restore the binding that ``token`` was taken before.

    Args:
        token (Token[Path | None]): The token returned by :func:`bind_session`.
    """
    _CURRENT_SESSION.reset(token)


@contextmanager
def session_scope(session_dir: Path | str) -> Iterator[Path]:
    """Bind ``session_dir`` for the duration of the block.

    Args:
        session_dir (Path | str): The session root directory to bind.

    Yields:
        Path: The canonical form of the bound session root.

    Raises:
        ValueError: If ``session_dir`` is empty.
    """
    token = bind_session(session_dir)
    try:
        yield _CURRENT_SESSION.get() or _canonical(session_dir)
    finally:
        unbind_session(token)


def bound_session() -> Path:
    """Return the bound session root, or fail loudly.

    Returns:
        Path: The canonical session root bound for this context.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    session = _CURRENT_SESSION.get()
    if session is None:
        raise SessionNotBoundError(
            "no session is bound: call bind_session(session_dir) at startup. "
            "Inside a subprocess this is expected and writing breakdown "
            "fragments there is unsafe -- write a conclusion JSON for the "
            "coordinator to replay instead."
        )
    return session


def bound_session_or_none() -> Path | None:
    """Return the bound session root, or ``None`` when nothing is bound.

    For callers that degrade to doing nothing rather than failing.

    Returns:
        Path | None: The canonical session root, or ``None``.
    """
    return _CURRENT_SESSION.get()


def session_is_bound() -> bool:
    """Report whether a session is bound in this context.

    Returns:
        bool: ``True`` when a session is bound.
    """
    return _CURRENT_SESSION.get() is not None
