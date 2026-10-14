# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""List the files a test left in the output directory, with their sizes and hashes (design 4.8.3 step 6)."""

from __future__ import annotations

import os
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.manifest import RESERVED_NAMES, ManifestFile

__all__ = ["OutputRejected", "inventory"]


_UNSAFE = "unsafe_output"


class OutputRejected(Exception):
    """The output cannot be listed: it holds a link or special file, or it is larger than the sheet allows."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def inventory(files_dir: Path, *, max_total_bytes: int) -> tuple[ManifestFile, ...]:
    """Every regular file below ``files_dir``, hashed through the descriptor that was checked.

    Raises:
        OutputRejected: ``unsafe_output`` for a symlink or special file anywhere below, ``output_too_large`` when the
            files add up to more than ``max_total_bytes``.
    """
    try:
        root_fd = fsafe.open_root(files_dir)
    except (fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
        raise OutputRejected(_UNSAFE, f"the output directory is not usable: {exc}") from exc
    entries: list[ManifestFile] = []
    total = 0
    try:
        for relative in fsafe.walk_regular_files(root_fd):
            if relative in RESERVED_NAMES:
                raise OutputRejected(_UNSAFE, f"{relative} is a name the bundle owns")
            fd = fsafe.open_beneath(root_fd, relative)
            try:
                sha256, size = fsafe.sha256_and_size(fd)
            finally:
                os.close(fd)
            total += size
            if total > max_total_bytes:
                raise OutputRejected("output_too_large", f"the output exceeds {max_total_bytes} bytes")
            entries.append(ManifestFile(relative, size, sha256))
    except fsafe.UnsafeEntry as exc:
        raise OutputRejected(_UNSAFE, str(exc)) from exc
    finally:
        os.close(root_fd)
    return tuple(entries)
