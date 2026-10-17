# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The files under a log root, and bounded reads of them."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: The most of one file a reader takes; a longer file is read as its first TEXT_LIMIT bytes.
TEXT_LIMIT = 64_000_000
#: A JSON file larger than this is not read at all: every JSON file a run writes is far smaller.
JSON_LIMIT = 16_000_000


@dataclass(frozen=True)
class FileRecord:
    """One regular file under a log root, and its size when the root was listed."""

    path: Path
    size: int


def _raise(error: OSError) -> None:
    raise error


def inventory(root: Path) -> list[FileRecord]:
    """Every regular file under ``root``, in path order. Symlinked directories are not entered, and a directory that
    cannot be listed raises its ``OSError`` rather than hiding the files in it."""

    records: list[FileRecord] = []
    for directory, _subdirectories, names in os.walk(root, onerror=_raise):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                records.append(FileRecord(path, path.stat().st_size))
    return sorted(records, key=lambda record: record.path)


def is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def read_text(path: Path, limit: int = TEXT_LIMIT) -> str:
    """The first ``limit`` bytes of ``path`` as UTF-8, with undecodable bytes replaced and NUL characters dropped."""

    with path.open("rb") as stream:
        data = stream.read(limit)
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object ``path`` holds, or None when it is absent, unreadable, too large, or not one object."""

    try:
        if path.stat().st_size > JSON_LIMIT:
            return None
        value = json.loads(read_text(path, JSON_LIMIT))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_yaml(path: Path) -> dict[str, Any]:
    """The YAML mapping ``path`` holds; raises ``ValueError`` for any other document."""

    value = yaml.safe_load(read_text(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return value
