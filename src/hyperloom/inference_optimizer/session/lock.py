# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-optimizer session lock."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import socket
from contextlib import suppress
from pathlib import Path
from typing import Any

from hyperloom.common.timeutil import now_iso

from . import session_paths

log = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SEC = 30.0

try:  # POSIX runtime (Linux): authoritative flock-based exclusion.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX dev hosts (e.g. Windows).
    fcntl = None  # type: ignore[assignment]


def _pid_alive(pid: int | None) -> bool:
    """Best-effort liveness probe for ``pid`` (used only on the fcntl-less path)."""
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
    """Read the lock file's owner metadata without acquiring the lock."""
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
        """Capture the contended session and the current owner metadata."""
        self.session_dir = Path(session_dir)
        self.owner = owner or {}
        who = ""
        if owner:
            who = f" (held by pid={owner.get('pid')} host={owner.get('hostname')} since={owner.get('started_at')})"
        super().__init__(f"another optimizer is already running for session {self.session_dir}{who}")


class SessionLockPathError(RuntimeError):
    """Raised when the lock path itself is unsafe to open."""

    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"session lock path {self.path} is unsafe: {reason}")


class SessionLock:
    """Exclusive, crash-safe, single-optimizer-per-session lock."""

    def __init__(self, session_dir: Path):
        """Bind the lock to a session directory (does not acquire yet)."""
        self.session_dir = Path(session_dir)
        self.path = session_paths.optimizer_lock_path(self.session_dir)
        self._fd: int | None = None
        self._started_at: str = ""

    def acquire(self) -> SessionLock:
        """Acquire the session lock or raise :class:`SessionAlreadyRunning`."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Non-inheritable fd (PEP 446) so serving subprocesses don't keep the lock alive past the optimizer. 0o600:
        # owner-only.
        open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, open_flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SessionLockPathError(self.path, "lock file must not be a symlink") from exc
            raise
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
        owner = self._now_owner(started_at=now_iso(timespec="seconds"))
        self._write_owner(owner)
        self._append_pod_history(owner)
        return self

    def heartbeat(self) -> None:
        """Refresh ``heartbeat_at`` in the lock body (best-effort, never raises)."""
        if self._fd is None:
            return
        with suppress(OSError):
            self._write_owner(self._now_owner(started_at=self._started_at))

    async def pulse(self) -> None:
        """Refresh owner liveness while the coordinator event loop runs."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SEC)
            self.heartbeat()

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
        now = now_iso(timespec="seconds")
        self._started_at = started_at or now
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": self._started_at,
            "heartbeat_at": now,
        }

    def _append_pod_history(self, owner: dict[str, Any]) -> None:
        """Append this acquisition to the pod-ownership ledger (never raises)."""
        record = {
            "acquired_at": owner.get("started_at"),
            "hostname": owner.get("hostname"),
            "pid": owner.get("pid"),
        }
        try:
            path = session_paths.pod_history_path(self.session_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            log.debug("session lock: pod history append failed", exc_info=True)

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
