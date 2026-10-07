# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The files of one Experience KB, under ``<root>/<kb_id>/files``, each kept once under its content's sha256.

A record names a file by a ``FileRef``; the file itself lives here. The database decides which files a KB holds, and
a file is only stored once its bytes hash to the name it is stored under.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO

from hyperloom_kb.schema import FileRef
from hyperloom_kb.storage import StorageContractError, _fsync_directory

FILES_DIR = "files"
_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FileMismatch(StorageContractError):
    """Raised when the bytes sent as a file are not the size or content its name says."""


def file_sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StorageContractError("a file is named by the sha256 of its content, 64 lowercase hex characters")
    return value


def file_ref(path: Path, name: str = "") -> FileRef:
    """The ``FileRef`` a record names the local file at ``path`` by; ``name`` defaults to its file name."""

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return FileRef(name=name or path.name, sha256=digest.hexdigest(), bytes=size)


class FileStore:
    def __init__(self, root: Path, kb_id: str) -> None:
        self.directory = root / kb_id / FILES_DIR

    def path(self, sha256: str) -> Path:
        return self.directory / file_sha256(sha256)

    def holds(self, sha256: str, size: int) -> bool:
        """Whether the file ``sha256`` is here with the ``size`` the database holds for it."""

        path = self.path(sha256)
        return path.is_file() and path.stat().st_size == size

    def receive(self, sha256: str, size: int, stream: BinaryIO) -> None:
        """Store the ``size`` bytes read from ``stream`` as the file ``sha256``, once they hash to it."""

        target = self.path(sha256)
        self.directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        remaining = size
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{sha256}.", dir=self.directory, delete=False) as out:
            temporary = Path(out.name)
            try:
                while remaining:
                    chunk = stream.read(min(_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise FileMismatch(f"file {sha256} ended after {size - remaining} of its {size} bytes")
                    digest.update(chunk)
                    out.write(chunk)
                    remaining -= len(chunk)
                if digest.hexdigest() != sha256:
                    raise FileMismatch(f"the bytes sent as file {sha256} hash to {digest.hexdigest()}")
                out.flush()
                os.fsync(out.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, target)
            _fsync_directory(self.directory)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["FILES_DIR", "FileMismatch", "FileStore", "file_ref", "file_sha256"]
