# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Single-optimizer session lock (issue #592).

A long ``inference_optimizer optimize`` run is guarded by a robustness monitor
that re-launches the optimizer via ``--resume`` if it judges the process dead.
During the slow serving cold-start that liveness check can misfire and spawn a
**second** optimizer on the same ``session_dir``; the two then contend for the
shared ``coordinator.db`` leases and both write ``state.json``, corrupting the
session.

This module is the authoritative backstop: whichever optimizer owns a session
holds an exclusive advisory lock (``flock`` on POSIX) on
``<session_dir>/runtime/optimizer.lock`` for its whole lifetime. A second
optimizer fails to acquire the lock and must refuse to run *before* touching
``state.json`` or any lease. The kernel drops the lock automatically when the
holder exits or crashes, so there is no stale-lock recovery to get wrong.

The lock file body is a small JSON document (``pid`` / ``hostname`` /
``started_at`` / ``heartbeat_at``) so the monitor can read the *authoritative*
owner pid instead of trusting a wrapper pidfile.

The lock file is intentionally never unlinked on release: unlinking a flock'd
path races a concurrent acquirer (it would flock a now-unlinked inode while a
newcomer creates and flocks a fresh file). A stale body is harmless — the next
acquirer flocks the same inode and overwrites it.
"""

from __future__ import annotations

import errno
import json
import os
import socket
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import session_paths

try:  # POSIX runtime (Linux): authoritative flock-based exclusion.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev hosts (e.g. Windows).
    fcntl = None  # type: ignore[assignment]


def _now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pid_alive(pid: int | None) -> bool:
    """Best-effort liveness probe for ``pid`` (used only on the fcntl-less path).

    Args:
        pid (int | None): Candidate owner pid; ``None`` / non-positive → dead.

    Returns:
        bool: ``True`` when the pid appears to reference a live process.
    """
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — treat as alive.
        return True
    except OSError:
        return False
    return True


def read_owner(session_dir: Path) -> dict[str, Any] | None:
    """Read the lock file's owner metadata without acquiring the lock.

    Args:
        session_dir (Path): The session root directory.

    Returns:
        dict | None: The parsed owner document, or ``None`` when the lock file
            is absent / empty / malformed.
    """
    path = session_paths.optimizer_lock_path(session_dir)
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class SessionAlreadyRunning(RuntimeError):
    """Raised when another live optimizer already owns the session."""

    def __init__(self, session_dir: Path, owner: dict[str, Any] | None):
        """Capture the contended session and the current owner metadata.

        Args:
            session_dir (Path): The session whose lock could not be acquired.
            owner (dict | None): The existing owner document, if readable.
        """
        self.session_dir = Path(session_dir)
        self.owner = owner or {}
        who = ""
        if owner:
            who = (
                f" (held by pid={owner.get('pid')} "
                f"host={owner.get('hostname')} "
                f"since={owner.get('started_at')})"
            )
        super().__init__(
            f"another optimizer is already running for session "
            f"{self.session_dir}{who}"
        )


class SessionLock:
    """Exclusive, crash-safe, single-optimizer-per-session lock.

    Acquire once at optimizer startup (both fresh ``optimize`` and
    ``--resume``); hold for the whole run. Use as a context manager or call
    :meth:`acquire` / :meth:`release` explicitly.
    """

    def __init__(self, session_dir: Path):
        """Bind the lock to a session directory (does not acquire yet).

        Args:
            session_dir (Path): The session root directory to guard.
        """
        self.session_dir = Path(session_dir)
        self.path = session_paths.optimizer_lock_path(self.session_dir)
        self._fd: int | None = None
        self._started_at: str = ""

    def acquire(self) -> SessionLock:
        """Acquire the session lock or raise :class:`SessionAlreadyRunning`.

        Returns:
            SessionLock: ``self`` once the lock is held.

        Raises:
            SessionAlreadyRunning: When a live optimizer already owns the
                session.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # os.open returns a non-inheritable fd by default (PEP 446), so spawned
        # serving subprocesses never keep the lock alive past the optimizer.
        # 0o600: owner-only (the lock body carries pid/host metadata).
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                owner = self._read_owner_fd(fd)
                os.close(fd)
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    raise SessionAlreadyRunning(self.session_dir, owner) from exc
                raise
        else:  # pragma: no cover - non-POSIX fallback (no real exclusion).
            owner = self._read_owner_fd(fd)
            owner_pid = owner.get("pid") if owner else None
            if owner and owner_pid != os.getpid() and _pid_alive(owner_pid):
                os.close(fd)
                raise SessionAlreadyRunning(self.session_dir, owner)
        self._fd = fd
        self._write_owner(self._now_owner(started_at=_now_iso()))
        return self

    def heartbeat(self) -> None:
        """Refresh ``heartbeat_at`` in the lock body (best-effort, never raises)."""
        if self._fd is None:
            return
        with suppress(OSError):
            self._write_owner(self._now_owner(started_at=self._started_at))

    def release(self) -> None:
        """Release the lock and close the fd. The file body is left in place."""
        if self._fd is None:
            return
        if fcntl is not None:
            with suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(self._fd)
        self._fd = None

    def _now_owner(self, *, started_at: str) -> dict[str, Any]:
        """Build the owner document written into the lock body."""
        now = _now_iso()
        self._started_at = started_at or now
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": self._started_at,
            "heartbeat_at": now,
        }

    def _write_owner(self, owner: dict[str, Any]) -> None:
        """Atomically rewrite the lock body (truncate + write while holding it)."""
        assert self._fd is not None
        blob = json.dumps(owner).encode("utf-8")
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, blob)
        with suppress(OSError):
            os.fsync(self._fd)

    @staticmethod
    def _read_owner_fd(fd: int) -> dict[str, Any] | None:
        """Read + parse the owner JSON from an already-open fd (no lock needed)."""
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 65536).decode("utf-8", "replace").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def __enter__(self) -> SessionLock:
        """Context-manager entry: acquire the lock."""
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit: release the lock."""
        self.release()

    def __del__(self) -> None:
        """Release on GC so a returning caller drops the lock promptly."""
        self.release()


__all__ = ["SessionAlreadyRunning", "SessionLock", "read_owner"]
