"""Backend-neutral storage contracts for canonical Experience data."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from hyperloom_kb.schema import Experience, ExperienceStatus

_EXPERIENCE_ID_RE = re.compile(r"^exp-[0-9a-f]{32}$")
_SCHEMA_REF_RE = re.compile(r"^schema:sha256:([0-9a-f]{64})$")


class StorageContractError(RuntimeError):
    """Raised when a backend cannot honor the durable storage contract."""


class ImmutableExperienceConflict(StorageContractError):
    """Raised when one Experience id is associated with different content."""


class InsertStatus(str, Enum):
    CREATED = "created"
    UNCHANGED = "unchanged"


def canonical_experience_bytes(experience: Experience) -> bytes:
    """Encode one Experience independently of any page or SQL representation."""

    return json.dumps(
        experience.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def experience_content_hash(experience: Experience) -> str:
    """Return the canonical SHA-256 for one exact Experience snapshot."""

    return hashlib.sha256(canonical_experience_bytes(experience)).hexdigest()


@dataclass(frozen=True)
class StoredExperience:
    experience: Experience
    content_hash: str

    def __post_init__(self) -> None:
        expected = experience_content_hash(self.experience)
        if self.content_hash != expected:
            raise StorageContractError("stored Experience content hash does not match")


@dataclass(frozen=True)
class InsertResult:
    record: StoredExperience
    status: InsertStatus


@runtime_checkable
class ExperienceStore(Protocol):
    """Atomic immutable storage for complete canonical Experiences.

    ``insert_complete`` must be linearizable for one Experience id:
    equal content returns ``UNCHANGED`` and different content raises
    ``ImmutableExperienceConflict``. A read after success returns the exact
    stored record.
    """

    def insert_complete(self, experience: Experience) -> InsertResult:
        """Atomically insert or idempotently replay one complete Experience."""

    def get_experience(self, experience_id: str) -> StoredExperience | None:
        """Return one exact record, or ``None`` only when it does not exist."""

    def list_experiences(self, schema_ref: str) -> tuple[StoredExperience, ...]:
        """Return records for one exact schema_ref in stable id order."""


def _experience_id(value: str) -> str:
    if not isinstance(value, str) or not _EXPERIENCE_ID_RE.fullmatch(value):
        raise StorageContractError("experience_id is invalid")
    return value


def _schema_digest(schema_ref: str) -> str:
    if not isinstance(schema_ref, str):
        raise StorageContractError("schema_ref is invalid")
    match = _SCHEMA_REF_RE.fullmatch(schema_ref)
    if match is None:
        raise StorageContractError("schema_ref is invalid")
    return match.group(1)


def _stored(experience: Experience) -> StoredExperience:
    if experience.status is not ExperienceStatus.COMPLETE:
        raise StorageContractError("ExperienceStore accepts complete Experiences only")
    return StoredExperience(experience, experience_content_hash(experience))


class InMemoryExperienceStore:
    """Thread-safe reference implementation of immutable Experience storage."""

    def __init__(self) -> None:
        self._records: dict[str, StoredExperience] = {}
        self._lock = threading.RLock()

    def insert_complete(self, experience: Experience) -> InsertResult:
        incoming = _stored(experience)
        with self._lock:
            existing = self._records.get(experience.id)
            if existing is not None:
                if existing.content_hash != incoming.content_hash:
                    raise ImmutableExperienceConflict("Experience id already exists with different content")
                return InsertResult(existing, InsertStatus.UNCHANGED)
            self._records[experience.id] = incoming
            return InsertResult(incoming, InsertStatus.CREATED)

    def get_experience(self, experience_id: str) -> StoredExperience | None:
        experience_id = _experience_id(experience_id)
        with self._lock:
            return self._records.get(experience_id)

    def list_experiences(self, schema_ref: str) -> tuple[StoredExperience, ...]:
        _schema_digest(schema_ref)
        with self._lock:
            return tuple(
                self._records[experience_id]
                for experience_id in sorted(self._records)
                if self._records[experience_id].experience.schema_ref == schema_ref
            )


def _fsync_directory(path: Path) -> None:
    # Flushing a directory makes a rename into it durable where a file system can flush one. Windows cannot open a
    # directory as a file, and some network and FUSE file systems refuse to flush one; the renamed file is flushed
    # already, so there the write stands without it.
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ExperienceStore",
    "ImmutableExperienceConflict",
    "InMemoryExperienceStore",
    "InsertResult",
    "InsertStatus",
    "StorageContractError",
    "StoredExperience",
    "canonical_experience_bytes",
    "experience_content_hash",
]
