# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The lock the builders of one commit into a runner cache exclude each other with, and the error a build ends with.

Builders of one commit take an exclusive ``flock`` on ``<cache>/.<sha>.flock`` (:func:`build_lock`); the one that waits
gives up after its timeout. In a cache directory set up to be shared the lock is ``0664``, so every account of the
group can take it after the one that made it; anywhere else it is made under the builder's own umask.
"""

from __future__ import annotations

import fcntl
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hyperloom_common.shared_storage import current_umask, ownership, shared_parent
from hyperloom.measure.runner.cache_layout import CacheError

__all__ = [
    "DEFAULT_LOCK_TIMEOUT_SEC",
    "CacheBuildError",
    "build_lock",
    "exclusive",
    "shared_umask",
]

DEFAULT_LOCK_TIMEOUT_SEC = 1800.0
_LOCK_POLL_SEC = 0.1
#: How a lock is opened: for writing, which ``flock`` over NFS needs, and never through a link.
_LOCK_FLAGS = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
_ALL_FILE_BITS = 0o666


class CacheBuildError(CacheError):
    """The cache cannot be built; the message says why. A campaign that gets it does not start."""


def shared_umask() -> int:
    """The umask of what a builder makes for the other builders of the cache, the cache directory and the locks,
    below storage set up to be shared (:func:`~hyperloom_common.shared_storage.make_shared_dirs`): its own, with every
    group bit let through, so the group may read, enter and write whatever the umask keeps from others. Elsewhere they
    are made under the builder's own umask, private as anything else it makes."""
    return current_umask() & ~stat.S_IRWXG


def _open_lock(path: Path) -> int:
    """``path`` opened for writing, its directory reached and judged as the shared-storage helpers do theirs
    (:func:`~hyperloom_common.shared_storage.shared_parent`): no other account's link on the way is followed. Made by
    this call in a cache directory set up to be shared, it is ``0664`` (:func:`shared_umask`) for every builder of the
    group; anywhere else under the builder's umask, as private as the cache directory.

    Raises:
        CacheBuildError: When it cannot be opened so, naming it: a lock made without group write, a link refused.
    """
    try:
        with shared_parent(path) as (directory, name, shared):
            mode = _ALL_FILE_BITS & ~(shared_umask() if shared else current_umask())
            try:
                descriptor = os.open(name, _LOCK_FLAGS | os.O_CREAT | os.O_EXCL, mode, dir_fd=directory)
            except FileExistsError:
                return os.open(name, _LOCK_FLAGS, dir_fd=directory)
            try:
                os.fchmod(descriptor, mode)  # the umask never decides it
            except BaseException:
                os.close(descriptor)
                raise
            return descriptor
    except OSError as exc:
        raise CacheBuildError(
            f"cannot open the build lock {path} for writing: {exc.strerror}{ownership(path)}; every account that builds "
            "into this cache must be able to"
        ) from None


def build_lock(cache_root: Path, sha: str) -> Path:
    """The lock the builders of commit ``sha`` into ``cache_root`` take. Named apart from the ``.<sha>.lock`` of builders
    whose locks were ``0644``, which another account cannot open for writing."""
    return cache_root / f".{sha}.flock"


@contextmanager
def exclusive(path: Path, timeout_sec: float) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for the block; wait for it for at most ``timeout_sec``. The native build
    service (``hyperloom.measure.runner.native``) coalesces its per-key builds with it too."""
    descriptor = _open_lock(path)
    try:
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CacheBuildError(f"another build still holds {path} after {timeout_sec:.0f} s") from None
                time.sleep(_LOCK_POLL_SEC)
        yield
    finally:
        os.close(descriptor)
