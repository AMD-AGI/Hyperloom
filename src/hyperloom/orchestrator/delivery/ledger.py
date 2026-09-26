# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable record of the backups a non-git apply takes, written before it mutates."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hyperloom.common.io import append_jsonl

log = logging.getLogger(__name__)

#: Filename of the ledger within a backup root.
LEDGER_NAME = "backup_ledger.jsonl"


#: Recorded in place of a hash when the path cannot be read.
ABSENT = ""


def file_digest(path: Path) -> str:
    """Return the lowercase sha256 of ``path``'s bytes, :data:`ABSENT` if unreadable.

    Args:
        path: File to hash.

    Returns:
        str: Hex digest, or :data:`ABSENT`.
    """
    try:
        with Path(path).open("rb") as fh:
            digest = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ABSENT
    return digest.hexdigest()


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
        if isinstance(row, dict) and row.get("kind") == "prepare_complete":
            continue
        out.append(row)
    return out


def _backup_of(record: Mapping[str, Any]) -> Any:
    """The copy a record names, under whichever field name its writer used."""
    return record.get("backup_path", record.get("backup"))


def _validate_restore_record(record: Mapping[str, Any]) -> None:
    target = record.get("target")
    if not isinstance(target, str) or not target or not Path(target).is_absolute():
        raise ValueError(f"invalid backup target: {target!r}")
    existed = record.get("existed")
    if not isinstance(existed, bool):
        raise ValueError(f"invalid backup existence: {target}")
    backup = _backup_of(record)
    if backup is not None and (not isinstance(backup, str) or not backup or not Path(backup).is_absolute()):
        raise ValueError(f"invalid backup path: {target}")
    action = record.get("revert_action")
    if action not in (None, "restore", "restore_old", "delete"):
        raise ValueError(f"invalid restore action: {target}")
    if (action == "delete" and existed) or (action in ("restore", "restore_old") and not existed):
        raise ValueError(f"contradictory restore action: {target}")
    mode = record.get("mode")
    if mode is not None and (type(mode) is not int or not 0 <= mode <= 0o7777):
        raise ValueError(f"invalid backup mode: {target}")
    digest = record.get("pre_image_sha256")
    if existed and (
        not backup
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(f"incomplete backup preimage: {target}")


def mark_prepared(backup_root: Path | str) -> bool:
    """Commit all preceding backup bytes before the corresponding mutation starts."""
    target = ledger_path(backup_root)
    try:
        data = target.read_bytes() if target.exists() else b""
        for record in load_records(backup_root):
            backup = _backup_of(record)
            if backup:
                # Windows fsync needs write access; POSIX also accepts read-only backup descriptors.
                with Path(backup).open("rb+" if os.name == "nt" else "rb") as stream:
                    os.fsync(stream.fileno())
        return append_record(
            backup_root,
            {
                "kind": "prepare_complete",
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
    except (OSError, ValueError, TypeError) as exc:
        log.error("delivery: could not commit prepared backups %s (%s)", target, exc)
        return False


def load_prepared_records(backup_root: Path | str) -> list[dict[str, Any]]:
    """Read the records the last checkpoint committed, discarding any later tail.

    One root is shared by every mutation batch an attempt takes, so the ledger
    reads as ``records… checkpoint`` repeated: a batch's records are all written
    before it mutates anything, and its checkpoint is written before the first
    mutation of the batch starts. Records past the last checkpoint therefore
    describe mutations that never ran, and restoring from them would overwrite
    live content with a preimage that was never displaced. Dropping that tail is
    what lets the committed prefix -- an earlier patch of the same attempt, fully
    applied -- still be reverted when a later batch dies mid-preparation.

    Args:
        backup_root: Directory the apply wrote its backups under.

    Returns:
        list[dict[str, Any]]: The committed records, in the order they were
        taken; empty when no batch ever committed under this root.

    Raises:
        ValueError: The committed prefix does not match its checkpoint, or holds
            a record that cannot describe a restore.
    """
    try:
        data = ledger_path(backup_root).read_bytes()
    except FileNotFoundError:
        return []
    committed: list[dict[str, Any]] = []
    staged: list[dict[str, Any]] = []
    offset = 0
    for line in data.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except ValueError:
            break
        if not isinstance(row, dict):
            break
        if row.get("kind") == "prepare_complete":
            if row.get("byte_count") != offset or row.get("sha256") != hashlib.sha256(data[:offset]).hexdigest():
                raise ValueError("prepared backup ledger is truncated or changed")
            for record in staged:
                _validate_restore_record(record)
            committed.extend(staged)
            staged.clear()
        else:
            staged.append(row)
        offset += len(line)
    if staged:
        log.warning(
            "delivery: ignoring %d uncommitted backup record(s) in %s",
            len(staged),
            ledger_path(backup_root),
        )
    return committed


def restore_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """Restore each target's first preimage, verifying before reporting completion.

    Intermediate backups are not recovery destinations: after A -> B -> C the
    target must end at A, including when a retry follows a partially completed
    restore. A target already at that preimage needs no surviving backup file.
    """
    first: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        try:
            _validate_restore_record(record)
            target = os.path.abspath(record["target"])
            if any(parent.is_symlink() for parent in Path(target).parents):
                raise ValueError(f"backup target parent is a symlink: {target}")
            first.setdefault(target, record)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    if errors:
        return [], errors
    restored: list[str] = []
    for name, record in reversed(first.items()):
        target = Path(name)
        try:
            if record["existed"]:
                if target.is_symlink():
                    raise ValueError(f"existing backup target became a symlink: {name}")
                backup = _backup_of(record)
                digest = str(record.get("pre_image_sha256") or "")
                if not digest:
                    raise ValueError(f"missing preimage hash: {name}")
                if file_digest(target) != digest:
                    if not backup or file_digest(Path(backup)) != digest:
                        raise ValueError(f"missing or corrupt backup: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
                mode = record.get("mode")
                if mode is not None:
                    target.chmod(int(mode))
                if file_digest(target) != digest:
                    raise OSError(f"preimage verification failed: {name}")
            else:
                target.unlink(missing_ok=True)
                if target.exists():
                    raise OSError(f"created target still present: {name}")
            restored.append(name)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{name}: {exc}")
    return restored, errors


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
    "ABSENT",
    "LEDGER_NAME",
    "append_record",
    "file_digest",
    "ledger_path",
    "load_prepared_records",
    "load_records",
    "mark_prepared",
    "merge_records",
    "restore_records",
]
