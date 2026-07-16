# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the single-optimizer session lock.

The lock guarantees that a second ``optimize`` / ``--resume`` attaching to the
same ``session_dir`` cannot run, so a misfiring robustness monitor can never
spawn a duplicate optimizer that corrupts the shared leases / ``state.json``.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys

import pytest

from hyperloom.inference_optimizer.session import lock as session_lock
from hyperloom.inference_optimizer.session import session_paths
from hyperloom.inference_optimizer.session.lock import (
    SessionAlreadyRunning,
    SessionLock,
)


def test_lock_path_under_runtime(tmp_path):
    """The lock lives at ``<session_dir>/runtime/optimizer.lock``."""
    assert (
        session_paths.optimizer_lock_path(tmp_path)
        == tmp_path / "runtime" / "optimizer.lock"
    )


def test_acquire_writes_owner_metadata(tmp_path):
    """Acquiring publishes the owner pid so the monitor can read it back."""
    lock = SessionLock(tmp_path)
    lock.acquire()
    try:
        owner = session_lock.read_owner(tmp_path)
        assert owner is not None
        assert owner["pid"] == os.getpid()
        assert owner["started_at"]
        assert owner["heartbeat_at"]
    finally:
        lock.release()


def test_read_owner_missing_is_none(tmp_path):
    """No lock file yet -> ``read_owner`` returns ``None`` (not an error)."""
    assert session_lock.read_owner(tmp_path) is None


def test_release_allows_reacquire(tmp_path):
    """After release the same session can be locked again."""
    first = SessionLock(tmp_path)
    first.acquire()
    first.release()
    second = SessionLock(tmp_path)
    second.acquire()  # must not raise
    second.release()


@pytest.mark.skipif(
    session_lock.fcntl is None,
    reason="flock-based mutual exclusion requires POSIX fcntl",
)
def test_second_holder_rejected(tmp_path):
    """A second live holder of the same session is refused (flock backstop)."""
    held = SessionLock(tmp_path)
    held.acquire()
    try:
        with pytest.raises(SessionAlreadyRunning) as exc:
            SessionLock(tmp_path).acquire()
        # The error surfaces the contended session for the operator.
        assert exc.value.session_dir == tmp_path
    finally:
        held.release()


def test_context_manager_releases(tmp_path):
    """The context manager acquires on enter and releases on exit."""
    with SessionLock(tmp_path):
        owner = session_lock.read_owner(tmp_path)
        assert owner is not None and owner["pid"] == os.getpid()
    # Released: a fresh acquire succeeds immediately.
    again = SessionLock(tmp_path)
    again.acquire()
    again.release()


def test_pid_fallback_rejects_live_owner(tmp_path, monkeypatch):
    """Without fcntl, a lock owned by a *different* live pid is still refused.

    Exercises the non-POSIX fallback path using a separate live child process
    as owner.
    """
    monkeypatch.setattr(session_lock, "fcntl", None)
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_path = session_paths.optimizer_lock_path(tmp_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            '{"pid": %d, "hostname": "x", "started_at": "t", '
            '"heartbeat_at": "t"}' % (live.pid),
            encoding="utf-8",
        )
        with pytest.raises(SessionAlreadyRunning):
            SessionLock(tmp_path).acquire()
    finally:
        live.terminate()
        live.wait(timeout=10)


def test_pid_fallback_takes_over_dead_owner(tmp_path, monkeypatch):
    """Without fcntl, a lock owned by a dead pid is taken over (not refused)."""
    monkeypatch.setattr(session_lock, "fcntl", None)
    lock_path = session_paths.optimizer_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # A pid that is essentially guaranteed to be dead.
    lock_path.write_text(
        '{"pid": 2147480000, "hostname": "x", '
        '"started_at": "t", "heartbeat_at": "t"}',
        encoding="utf-8",
    )
    lock = SessionLock(tmp_path)
    lock.acquire()  # must not raise — stale owner is taken over
    try:
        owner = session_lock.read_owner(tmp_path)
        assert owner is not None and owner["pid"] == os.getpid()
    finally:
        lock.release()


def test_pid_alive_edge_cases(monkeypatch):
    """_pid_alive covers None/non-positive, ProcessLookup, Permission, OSError."""
    from hyperloom.inference_optimizer.session.lock import _pid_alive

    assert _pid_alive(None) is False
    assert _pid_alive(0) is False
    assert _pid_alive(-5) is False
    assert _pid_alive(os.getpid()) is True

    def raise_perm(_pid, _sig):
        raise PermissionError()

    monkeypatch.setattr(session_lock.os, "kill", raise_perm)
    assert _pid_alive(12345) is True  # exists but owned by another user

    def raise_oserror(_pid, _sig):
        raise OSError("weird")

    monkeypatch.setattr(session_lock.os, "kill", raise_oserror)
    assert _pid_alive(12345) is False


def test_read_owner_empty_and_malformed(tmp_path):
    """read_owner returns None for blank / non-JSON / non-dict bodies."""
    lock_path = session_paths.optimizer_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path.write_text("   \n", encoding="utf-8")
    assert session_lock.read_owner(tmp_path) is None

    lock_path.write_text("not json {{{", encoding="utf-8")
    assert session_lock.read_owner(tmp_path) is None

    lock_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert session_lock.read_owner(tmp_path) is None


def test_read_owner_fd_empty_and_malformed(tmp_path):
    """_read_owner_fd handles blank / non-JSON / non-dict bodies and OSError."""
    lock_path = session_paths.optimizer_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path.write_text("", encoding="utf-8")
    fd = os.open(lock_path, os.O_RDWR)
    try:
        assert SessionLock._read_owner_fd(fd) is None
    finally:
        os.close(fd)

    lock_path.write_text("garbage }{", encoding="utf-8")
    fd = os.open(lock_path, os.O_RDWR)
    try:
        assert SessionLock._read_owner_fd(fd) is None
    finally:
        os.close(fd)

    lock_path.write_text('["not", "a", "dict"]', encoding="utf-8")
    fd = os.open(lock_path, os.O_RDWR)
    try:
        assert SessionLock._read_owner_fd(fd) is None
    finally:
        os.close(fd)

    # A closed fd triggers the OSError branch.
    fd2 = os.open(lock_path, os.O_RDWR)
    os.close(fd2)
    assert SessionLock._read_owner_fd(fd2) is None


def test_heartbeat_refreshes_body(tmp_path):
    """heartbeat rewrites the lock body while held; no-op when not held."""
    lock = SessionLock(tmp_path)
    # Not held yet -> no-op, no raise.
    lock.heartbeat()
    lock.acquire()
    try:
        before = session_lock.read_owner(tmp_path)
        lock.heartbeat()
        after = session_lock.read_owner(tmp_path)
        assert before is not None and after is not None
        assert after["pid"] == os.getpid()
    finally:
        lock.release()


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="O_NOFOLLOW is POSIX-only",
)
def test_acquire_refuses_symlinked_lock_path(tmp_path):
    """A symlinked lock path is refused rather than followed and written."""
    lock_path = session_paths.optimizer_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_target"
    outside.write_text("victim", encoding="utf-8")
    os.symlink(outside, lock_path)
    with pytest.raises(OSError):
        SessionLock(tmp_path).acquire()
    # The symlink target must be untouched (not opened/truncated/locked).
    assert outside.read_text(encoding="utf-8") == "victim"


def test_acquire_plain_file_still_works(tmp_path):
    """A regular (non-symlink) lock file acquires normally with O_NOFOLLOW."""
    lock = SessionLock(tmp_path)
    lock.acquire()
    try:
        assert session_lock.read_owner(tmp_path) is not None
    finally:
        lock.release()
    # Re-acquire on the now-existing regular file also works.
    again = SessionLock(tmp_path)
    again.acquire()
    again.release()


@pytest.mark.skipif(
    session_lock.fcntl is None,
    reason="flock error-path requires POSIX fcntl",
)
def test_acquire_reraises_non_contention_oserror(tmp_path, monkeypatch):
    """A flock OSError that is not EAGAIN/EACCES/EWOULDBLOCK propagates."""

    def raise_eio(_fd, _op):
        raise OSError(errno.EIO, "io error")

    monkeypatch.setattr(session_lock.fcntl, "flock", raise_eio)
    with pytest.raises(OSError):
        SessionLock(tmp_path).acquire()
