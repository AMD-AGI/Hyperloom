# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-file writes for files every participant of a session reads.

The participants may be different accounts of one group: each leg of a hosted run may be another account, and the
RSI archive is read by every session of a deployment. So a published file is readable by them (``644``, set by
``fchmod`` whatever the umask) and written by none once it is there, and the directories it goes in are the group's
(``2775``, :func:`~hyperloom_common.shared_storage.make_shared_dirs`). Segments (:mod:`.segments`) are created ``664``,
as each is appended to by whichever leg runs.
A reader never sees a partial file, and a host crash leaves no empty one: a file's bytes reach the disk before its name
appears.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from hyperloom_common.shared_storage import make_shared_dirs

#: A published file: read by every account of the group, written by none once it is in place.
PUBLISHED_MODE = 0o644


def encode_json(value: object) -> bytes:
    """Canonical JSON bytes: sorted keys, ASCII only, trailing newline."""

    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()


def _staged(path: Path, data: bytes) -> Path:
    make_shared_dirs(path.parent)
    fd, staged = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        os.fchmod(handle.fileno(), PUBLISHED_MODE)
        handle.flush()
        os.fsync(handle.fileno())
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


__all__ = ["PUBLISHED_MODE", "encode_json", "publish_once"]
