"""Backend-neutral storage contracts for canonical Experience data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from hyperloom_kb.schema import Experience, ExperienceDeclaration, ExperienceStatus

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


def canonical_schema_bytes(declaration: ExperienceDeclaration) -> bytes:
    """Encode one exact Experience Schema independently of a backend."""

    return json.dumps(
        declaration.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


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


@runtime_checkable
class SchemaRegistry(Protocol):
    """Immutable registry keyed by declaration-derived schema_ref."""

    def register_schema(self, declaration: ExperienceDeclaration) -> str:
        """Register exact schema content and return its stable schema_ref."""

    def get_schema(self, schema_ref: str) -> ExperienceDeclaration | None:
        """Return one exact declaration, or ``None`` when it is unknown."""

    def list_schemas(self) -> tuple[ExperienceDeclaration, ...]:
        """Return every registered declaration in stable schema_ref order."""


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


class InMemorySchemaRegistry:
    """Thread-safe reference registry used by the conformance suite."""

    def __init__(self) -> None:
        self._schemas: dict[str, ExperienceDeclaration] = {}
        self._lock = threading.RLock()

    def register_schema(self, declaration: ExperienceDeclaration) -> str:
        if not isinstance(declaration, ExperienceDeclaration):
            raise StorageContractError("declaration is invalid")
        with self._lock:
            existing = self._schemas.get(declaration.schema_ref)
            if existing is not None and existing != declaration:
                raise StorageContractError("schema_ref collision")
            self._schemas[declaration.schema_ref] = declaration
            return declaration.schema_ref

    def get_schema(self, schema_ref: str) -> ExperienceDeclaration | None:
        _schema_digest(schema_ref)
        with self._lock:
            return self._schemas.get(schema_ref)

    def list_schemas(self) -> tuple[ExperienceDeclaration, ...]:
        with self._lock:
            return tuple(self._schemas[schema_ref] for schema_ref in sorted(self._schemas))


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
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


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


class LocalSchemaRegistry:
    """Durable filesystem SchemaRegistry with atomic immutable registration."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.schemas = self.root / "schemas"
        self.locks = self.root / "locks" / "schemas"

    def _path(self, schema_ref: str) -> Path:
        return self.schemas / f"{_schema_digest(schema_ref)}.json"

    def register_schema(self, declaration: ExperienceDeclaration) -> str:
        if not isinstance(declaration, ExperienceDeclaration):
            raise StorageContractError("declaration is invalid")
        path = self._path(declaration.schema_ref)
        data = canonical_schema_bytes(declaration)
        with _exclusive_lock(self.locks / f"{path.stem}.lock"):
            if path.exists():
                try:
                    existing = path.read_bytes()
                except OSError as exc:
                    raise StorageContractError(f"cannot read registered schema: {exc}") from exc
                if existing != data:
                    raise StorageContractError("schema_ref collision")
                return declaration.schema_ref
            _atomic_write(path, data)
        return declaration.schema_ref

    def get_schema(self, schema_ref: str) -> ExperienceDeclaration | None:
        path = self._path(schema_ref)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise StorageContractError(f"cannot read registered schema: {exc}") from exc
        try:
            declaration = ExperienceDeclaration.from_dict(value)
        except ValueError as exc:
            raise StorageContractError(f"registered schema is invalid: {exc}") from exc
        if declaration.schema_ref != schema_ref:
            raise StorageContractError("registered schema content does not match schema_ref")
        return declaration

    def list_schemas(self) -> tuple[ExperienceDeclaration, ...]:
        if not self.schemas.exists():
            return ()
        declarations = []
        for path in self.schemas.glob("*.json"):
            try:
                declarations.append(ExperienceDeclaration.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError) as exc:
                raise StorageContractError(f"registered schema {path.name} is invalid: {exc}") from exc
        return tuple(sorted(declarations, key=lambda declaration: declaration.schema_ref))


class LocalExperienceStore:
    """Durable filesystem ExperienceStore with per-id process locking."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.experiences = self.root / "experiences"
        self.locks = self.root / "locks" / "experiences"

    def _path(self, experience_id: str) -> Path:
        return self.experiences / f"{_experience_id(experience_id)}.json"

    def _read(self, path: Path) -> StoredExperience | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise StorageContractError(f"cannot read stored Experience: {exc}") from exc
        try:
            experience = Experience.from_dict(value)
        except ValueError as exc:
            raise StorageContractError(f"stored Experience is invalid: {exc}") from exc
        return StoredExperience(experience, experience_content_hash(experience))

    def insert_complete(self, experience: Experience) -> InsertResult:
        incoming = _stored(experience)
        path = self._path(experience.id)
        with _exclusive_lock(self.locks / f"{experience.id}.lock"):
            existing = self._read(path)
            if existing is not None:
                if existing.content_hash != incoming.content_hash:
                    raise ImmutableExperienceConflict("Experience id already exists with different content")
                return InsertResult(existing, InsertStatus.UNCHANGED)
            _atomic_write(path, canonical_experience_bytes(experience))
            persisted = self._read(path)
            if persisted != incoming:
                raise StorageContractError("Experience read-after-write verification failed")
            return InsertResult(persisted, InsertStatus.CREATED)

    def get_experience(self, experience_id: str) -> StoredExperience | None:
        return self._read(self._path(experience_id))

    def list_experiences(self, schema_ref: str) -> tuple[StoredExperience, ...]:
        _schema_digest(schema_ref)
        if not self.experiences.exists():
            return ()
        return tuple(
            record
            for path in sorted(self.experiences.glob("exp-*.json"))
            if (record := self._read(path)) is not None and record.experience.schema_ref == schema_ref
        )


__all__ = [
    "ExperienceStore",
    "ImmutableExperienceConflict",
    "InMemoryExperienceStore",
    "InMemorySchemaRegistry",
    "InsertResult",
    "InsertStatus",
    "LocalExperienceStore",
    "LocalSchemaRegistry",
    "SchemaRegistry",
    "StorageContractError",
    "StoredExperience",
    "canonical_experience_bytes",
    "canonical_schema_bytes",
    "experience_content_hash",
]
