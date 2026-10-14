# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-file writes for files every participant of a session reads.

``hyperloom.common.io`` writes owner-only files by design; the notebook may live on a shared folder that participants
running as other users read, so its files are group and world readable. A reader never sees a partial file.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

SHARED_FILE_MODE = 0o644


def encode_json(value: object) -> bytes:
    """Canonical JSON bytes: sorted keys, ASCII only, trailing newline."""

    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()


def _staged(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(staged, SHARED_FILE_MODE)
    return Path(staged)


def publish_once(path: Path, data: bytes) -> bool:
    """Create ``path`` holding ``data`` unless it exists; ``False`` when another writer created it first.

    The file is linked into place complete, so its existence and its content arrive together. ``link`` is atomic on
    NFSv3 and later, where ``O_EXCL`` alone would expose the file before its bytes are written.
    """

    staged = _staged(path, data)
    try:
        os.link(staged, path)
    except FileExistsError:
        return False
    finally:
        with suppress(FileNotFoundError):
            staged.unlink()
    return True


__all__ = ["SHARED_FILE_MODE", "encode_json", "publish_once"]
