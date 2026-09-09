# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session that recording entry points write into, bound once at startup."""

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
    """Raised when a recording entry point runs with no session bound."""


_CURRENT_SESSION: ContextVar[Path | None] = ContextVar(
    "hyperloom_recording_session",
    default=None,
)


def _canonical(session_dir: Path | str) -> Path:
    """Return the canonical form of ``session_dir`` for binding."""
    if not str(session_dir or "").strip():
        raise ValueError("session_dir must be non-empty")
    return Path(session_dir).expanduser().resolve()


def bind_session(session_dir: Path | str) -> Token[Path | None]:
    """Bind ``session_dir`` as the session every recording entry point writes to."""
    return _CURRENT_SESSION.set(_canonical(session_dir))


def unbind_session(token: Token[Path | None]) -> None:
    """Restore the binding that ``token`` was taken before."""
    _CURRENT_SESSION.reset(token)


@contextmanager
def session_scope(session_dir: Path | str) -> Iterator[Path]:
    """Bind ``session_dir`` for the duration of the block."""
    token = bind_session(session_dir)
    try:
        yield _CURRENT_SESSION.get() or _canonical(session_dir)
    finally:
        unbind_session(token)


def bound_session() -> Path:
    """Return the bound session root, or fail loudly."""
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
    """Return the bound session root, or ``None`` when nothing is bound."""
    return _CURRENT_SESSION.get()


def session_is_bound() -> bool:
    """Report whether a session is bound in this context."""
    return _CURRENT_SESSION.get() is not None
