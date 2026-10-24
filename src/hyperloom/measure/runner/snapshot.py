# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check a runtime snapshot against the manifest hash the sheet names (design 4.6.3, 4.8.3 step 1)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from hyperloom.measure.hosted_spec import dir_manifest_sha256

__all__ = ["SnapshotError", "check_snapshot"]

_CHUNK = 1 << 20


class SnapshotError(Exception):
    """The snapshot cannot be read, holds a symbolic link, or is not the one the sheet names."""


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(root: Path) -> list[tuple[str, int, str]]:
    """``(relative path, size, sha256)`` of every file under ``root``; a link anywhere is refused, as for an attachment."""
    entries = []
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        for name in (*subdirectories, *names):
            if (Path(directory) / name).is_symlink():
                raise SnapshotError(f"the runtime snapshot holds a symbolic link: {Path(directory) / name}")
        for name in names:
            path = Path(directory) / name
            entries.append((path.relative_to(root).as_posix(), path.stat().st_size, _sha256_of(path)))
    return entries


def check_snapshot(snapshot: Path, manifest_sha256: str) -> None:
    """Hash every file of ``snapshot`` and compare the list's hash with ``manifest_sha256``."""
    if not snapshot.is_dir():
        raise SnapshotError(f"the runtime snapshot {snapshot} is not a directory")
    try:
        actual = dir_manifest_sha256(_entries(snapshot))
    except OSError as exc:
        raise SnapshotError(f"the runtime snapshot {snapshot} cannot be read: {exc}") from exc
    if actual != manifest_sha256:
        raise SnapshotError(f"the runtime snapshot {snapshot} does not match its manifest hash {manifest_sha256}")
