# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-file writes for files every participant of a session reads.

The participants may be different accounts of one group: each leg of a hosted run may be another account, and the
RSI archive is read by every session of a deployment. So below shared storage a published file is readable by them
(``644``, set by ``fchmod`` whatever the umask) and written by none once it is there, and the directories it goes in
are the group's (``2775``, :func:`~hyperloom_common.shared_storage.make_shared_dirs`). Anywhere else every participant
runs as the session's user, so a published file stays owner-only (``600``) and the directories follow the umask. No
link a member of the group could have put on the way is followed
(:func:`~hyperloom_common.shared_storage.publish_shared_bytes`). Segments (:mod:`.segments`) are created ``664`` below
shared storage, as each is appended to by whichever leg runs, and owner-only anywhere else.
A reader never sees a partial file, and a host crash leaves no empty one: a file's bytes reach the disk before its name
appears.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_common.shared_storage import parent_is_shared, publish_shared_bytes

#: A published file below shared storage: read by every account of the group, written by none once it is in place.
PUBLISHED_MODE = 0o644
#: A file of the session anywhere else: its participants all run as the session's user.
OWNER_ONLY_MODE = 0o600


def session_file_mode(path: Path, shared: int) -> int:
    """The mode of a new session file at ``path``: ``shared`` below shared storage (judged before a missing parent is
    made), owner-only anywhere else."""

    return shared if parent_is_shared(Path(path)) else OWNER_ONLY_MODE


def encode_json(value: object) -> bytes:
    """Canonical JSON bytes: sorted keys, ASCII only, trailing newline."""

    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()


def publish_once(path: Path, data: bytes) -> bool:
    """Create ``path`` holding ``data`` unless it exists; ``False`` when another writer created it first.

    The file is linked into place complete, so its existence and its content arrive together. ``link`` is atomic on
    NFSv3 and later, where ``O_EXCL`` alone would expose the file before its bytes are written.
    """

    path = Path(path)
    return publish_shared_bytes(path, data, mode=session_file_mode(path, PUBLISHED_MODE), make_parents=True)


__all__ = ["OWNER_ONLY_MODE", "PUBLISHED_MODE", "encode_json", "publish_once", "session_file_mode"]
