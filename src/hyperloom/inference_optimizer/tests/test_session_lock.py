# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the single-optimizer session lock."""

from __future__ import annotations

import errno
import json
import os

import pytest

from hyperloom.inference_optimizer.session import lock as session_lock
from hyperloom.inference_optimizer.session import session_paths
from hyperloom.inference_optimizer.session.lock import (
    SessionAlreadyRunning,
    SessionLock,
    SessionLockPathError,
)


def _read_owner(session_dir):
    return json.loads(session_paths.optimizer_lock_path(session_dir).read_text(encoding="utf-8"))


def test_lock_path_under_runtime(tmp_path):
    """The lock lives at ``<session_dir>/runtime/optimizer.lock``."""
    assert session_paths.optimizer_lock_path(tmp_path) == tmp_path / "runtime" / "optimizer.lock"


def test_acquire_writes_owner_metadata(tmp_path):
    """Acquiring publishes the owner pid so the monitor can read it back."""
    lock = SessionLock(tmp_path)
    lock.acquire()
    try:
        owner = _read_owner(tmp_path)
        assert owner is not None
        assert owner["pid"] == os.getpid()
        assert owner["started_at"]
    finally:
        lock.release()


def test_release_allows_reacquire(tmp_path):
    """After release the same session can be locked again."""
    first = SessionLock(tmp_path)
    first.acquire()
    first.release()
    second = SessionLock(tmp_path)
    second.acquire()  # must not raise
    second.release()


def test_pod_history_path_under_runtime(tmp_path):
    """The ownership ledger lives at ``<session_dir>/runtime/pod_history.jsonl``."""
    assert session_paths.pod_history_path(tmp_path) == tmp_path / "runtime" / "pod_history.jsonl"


def test_acquire_appends_pod_history(tmp_path):
    """Each acquisition appends one ledger line, so the owner chain survives."""
    first = SessionLock(tmp_path)
    first.acquire()
    first.release()
    second = SessionLock(tmp_path)
    second.acquire()
    second.release()

    lines = session_paths.pod_history_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, lines
    for line in lines:
        record = json.loads(line)
        assert record["pid"] == os.getpid()
        assert record["hostname"]
        assert record["acquired_at"]


def test_pod_history_failure_does_not_break_acquire(tmp_path):
    """The ledger is observational: a write failure must not fail the acquire."""
    ledger = session_paths.pod_history_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.mkdir()

    lock = SessionLock(tmp_path)
    lock.acquire()  # must not raise
    try:
        assert _read_owner(tmp_path) is not None
    finally:
        lock.release()


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
        owner = _read_owner(tmp_path)
        assert owner is not None and owner["pid"] == os.getpid()
    # Released: a fresh acquire succeeds immediately.
    again = SessionLock(tmp_path)
    again.acquire()
    again.release()


@pytest.mark.parametrize("body", ["", "garbage }{", '["not", "a", "dict"]'], ids=["blank", "not-json", "not-a-dict"])
def test_a_contended_lock_whose_owner_cannot_be_parsed_is_still_refused(tmp_path, body):
    """The refusal names no owner rather than failing on a lock body it cannot parse."""
    held = SessionLock(tmp_path)
    held.acquire()
    try:
        session_paths.optimizer_lock_path(tmp_path).write_text(body, encoding="utf-8")
        with pytest.raises(SessionAlreadyRunning) as exc:
            SessionLock(tmp_path).acquire()
        assert exc.value.owner == {}
        assert "held by" not in str(exc.value)
    finally:
        held.release()


def test_a_contended_lock_whose_body_cannot_be_read_is_still_refused(tmp_path, monkeypatch):
    held = SessionLock(tmp_path)
    held.acquire()
    try:

        def raise_eio(_fd, _size):
            raise OSError(errno.EIO, "io error")

        monkeypatch.setattr(session_lock.os, "read", raise_eio)
        with pytest.raises(SessionAlreadyRunning) as exc:
            SessionLock(tmp_path).acquire()
        assert exc.value.owner == {}
    finally:
        held.release()


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
    with pytest.raises(SessionLockPathError, match="must not be a symlink"):
        SessionLock(tmp_path).acquire()
    # The symlink target must be untouched (not opened/truncated/locked).
    assert outside.read_text(encoding="utf-8") == "victim"


def test_acquire_plain_file_still_works(tmp_path):
    """A regular (non-symlink) lock file acquires normally with O_NOFOLLOW."""
    lock = SessionLock(tmp_path)
    lock.acquire()
    try:
        assert _read_owner(tmp_path) is not None
    finally:
        lock.release()
    # Re-acquire on the now-existing regular file also works.
    again = SessionLock(tmp_path)
    again.acquire()
    again.release()


def test_acquire_reraises_non_contention_oserror(tmp_path, monkeypatch):
    """A flock OSError that is not EAGAIN/EACCES/EWOULDBLOCK propagates."""

    def raise_eio(_fd, _op):
        raise OSError(errno.EIO, "io error")

    monkeypatch.setattr(session_lock.fcntl, "flock", raise_eio)
    with pytest.raises(OSError):
        SessionLock(tmp_path).acquire()
