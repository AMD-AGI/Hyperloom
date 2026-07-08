# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared best-effort cross-process file lock (``fcntl.flock``)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger(__name__)


@contextmanager
def best_effort_file_lock(lock_path: str, *, label: str = "file_lock") -> Iterator[None]:
    """Best-effort mutex; falls through without exclusion when fcntl/lock-file is unavailable."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    try:
        fp = open(lock_path, "w")  # noqa: SIM115
    except OSError as e:
        log.warning(
            "%s: cannot open lock file %s (%s); proceeding without exclusion",
            label,
            lock_path,
            e,
        )
        yield
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()
