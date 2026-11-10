# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The canonical record files of one Experience KB, under ``<root>/<kb_id>/records``.

The database decides which records exist and what content each must have; a file is only ever read against the
content hash the database holds for it, so a file a failed write left behind is never served.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_kb.schema import Experience
from hyperloom_kb.storage import (
    StorageContractError,
    _atomic_write,
    _experience_id,
    experience_content_hash,
)

RECORDS_DIR = "records"


class RecordFiles:
    def __init__(self, root: Path, kb_id: str) -> None:
        self.directory = root / kb_id / RECORDS_DIR

    def _path(self, experience_id: str) -> Path:
        return self.directory / f"{_experience_id(experience_id)}.json"

    def write(self, experience_id: str, data: bytes) -> None:
        _atomic_write(self._path(experience_id), data)

    def holds(self, experience_id: str, size: int) -> bool:
        """Whether the file of ``experience_id`` is there with the ``size`` the database holds for it."""

        path = self._path(experience_id)
        return path.is_file() and path.stat().st_size == size

    def read(self, experience_id: str, content_hash: str) -> Experience:
        path = self._path(experience_id)
        try:
            experience = Experience.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise StorageContractError(f"cannot read the record of {experience_id}: {exc}") from exc
        if experience_content_hash(experience) != content_hash:
            raise StorageContractError(f"the record file of {experience_id} does not hold its stored content")
        return experience


__all__ = ["RECORDS_DIR", "RecordFiles"]
