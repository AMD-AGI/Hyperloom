# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable record of the backups a non-git apply takes, written before it mutates.

Each record is appended the moment its backup lands and before the file it
describes is touched, so a revert can run from disk alone.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl

log = logging.getLogger(__name__)

#: Filename of the ledger within a backup root.
LEDGER_NAME = "backup_ledger.jsonl"


def ledger_path(backup_root: Path | str) -> Path:
    """Return ``<backup_root>/backup_ledger.jsonl``."""
    return Path(backup_root) / LEDGER_NAME


def append_record(backup_root: Path | str, record: Mapping[str, Any]) -> bool:
    """Append one backup record, before the file it describes is mutated.

    Args:
        backup_root: Directory the apply writes its backups under.
        record: The backup record to persist.

    Returns:
        bool: Whether the record reached disk. A caller that gets ``False``
        must not mutate the file the record describes.
    """
    target = ledger_path(backup_root)
    try:
        append_jsonl(target, dict(record), make_parents=True, fsync=True)
    except OSError as exc:
        log.error("delivery: could not append backup ledger %s (%s)", target, exc)
        return False
    return True


def load_records(backup_root: Path | str) -> list[dict[str, Any]]:
    """Read back every persisted backup record, in the order they were taken.

    Args:
        backup_root: Directory the apply wrote its backups under.

    Returns:
        list[dict[str, Any]]: The records, empty when no backup was ever taken
        under this root. A malformed line is skipped so the surviving records
        still restore the files they name.
    """
    target = ledger_path(backup_root)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            log.warning("delivery: skipping malformed backup ledger line in %s", target)
            continue
        out.append(row)
    return out


def merge_records(
    in_memory: Sequence[Mapping[str, Any]],
    backup_root: Path | str,
) -> list[dict[str, Any]]:
    """Return the records to revert, the persisted ledger first.

    Args:
        in_memory: Records the caller accumulated this process, folded in for
            the case where the ledger could not be written.
        backup_root: Directory the apply wrote its backups under.

    Returns:
        list[dict[str, Any]]: De-duplicated records in the order they were
        taken, so a caller can revert them in reverse.
    """
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in [*load_records(backup_root), *in_memory]:
        key = (str(record["target"]), str(record["backup_path"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(record))
    return merged


__all__ = [
    "append_record",
    "ledger_path",
    "load_records",
    "merge_records",
]
