# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared cache of compiled code: an append-only, content-addressed store read with verification (design 4.21).

Layout of :class:`DirJitStore`::

    <root>/blobs/<aa>/<sha256>                       one compiled file, named by the hash of its bytes
    <root>/snapshots/<kind>/<key>/<sha256>.json      the files of a kind under a key, named by the hash of this list

A snapshot lists ``relative path -> (sha256, size)``. It only adds: nothing here rewrites or removes a stored blob or
snapshot, a name becomes visible in one rename, and two publishers of the same name store the same bytes (a blob's name is
its hash). A stored blob is never rewritten, except that bytes which verify against the hash of a stored blob that does
not are put in its place: a damaged blob would otherwise stay damaged for every later test, and the name of the blob is
the hash of what it must hold, so the replacement can only repair. Publishers that extend the same snapshot concurrently
make two snapshots, both valid; a reader takes the one with the most files.

Reading proves every byte: the snapshot's own name is the hash of its bytes, it names the kind and key that were asked
for, every path is one the kind admits, and every blob is copied through descriptors that refuse symlinks while its hash
and size are checked against the list. A snapshot that fails any of that is dropped whole, with a note, and the next
generation is tried; the caller then builds what is missing itself, so a bad cache costs time and never a result.

The one file a copy changes after it has been verified is a kind's index of the files its compiler made, which holds
paths: the reader's directory is written into its private copy (:mod:`hyperloom.measure.jit.locations`).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure import fsafe
from hyperloom.measure.archive import fsync_dir
from hyperloom.measure.docs import SHA256_RE, DocumentError, Reader, canonical_json, parse_json_object, relative_path
from hyperloom.measure.jit.locations import MAX_LOCATED_BYTES, NotPortable, localised
from hyperloom.measure.jit_kinds import KINDS, CacheKind, kind_named

__all__ = [
    "BLOBS_DIR",
    "MANIFEST_NAME",
    "MAX_SNAPSHOT_BYTES",
    "DirJitStore",
    "Fetched",
    "JitStoreError",
    "Published",
    "Snapshot",
    "SnapshotInvalid",
    "StoredFile",
    "read_staged_snapshot",
]

SNAPSHOT_SCHEMA = "hyperloom.jit-cache-snapshot/v1"
#: A kind's snapshot (everything it holds, not only what a test added) over this size is not stored.
MAX_SNAPSHOT_BYTES = 4 * 1024**3
MANIFEST_NAME = "manifest.json"
BLOBS_DIR = "blobs"
SNAPSHOTS_DIR = "snapshots"
_MAX_MANIFEST_BYTES = 32 * 1024**2
_GENERATIONS_TRIED = 3
_INCOMING = ".incoming-"
_FILE_MODE = 0o600
_BLOB_MODE = 0o444
_FETCH_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class JitStoreError(Exception):
    """The store cannot be read or written; the cache is skipped, the test is not failed."""


class SnapshotInvalid(JitStoreError):
    """A stored or staged snapshot does not verify: the message names what differs."""


@dataclass(frozen=True)
class StoredFile:
    sha256: str
    size: int


@dataclass(frozen=True)
class Snapshot:
    """The files of one kind under one key, as the cache lists them."""

    kind: str
    key: str
    parent: str | None
    files: Mapping[str, StoredFile]

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.files.values())

    def document(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "kind": self.kind,
            "key": self.key,
            "parent": self.parent,
            "files": {path: {"sha256": f.sha256, "size": f.size} for path, f in sorted(self.files.items())},
        }

    def to_bytes(self) -> bytes:
        return canonical_json(self.document()).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def parse(cls, raw: bytes) -> "Snapshot":
        """Read a snapshot strictly: unknown keys, an unknown kind, a path its kind does not admit are all refusals."""
        try:
            return cls._parse(raw)
        except (DocumentError, KeyError) as exc:
            raise SnapshotInvalid(f"the snapshot is not valid: {exc}") from None

    @classmethod
    def _parse(cls, raw: bytes) -> "Snapshot":
        r = Reader(parse_json_object(raw, what="snapshot"), "snapshot")
        if r.text("schema") != SNAPSHOT_SCHEMA:
            raise DocumentError(f"snapshot.schema: expected {SNAPSHOT_SCHEMA}")
        kind = kind_named(r.text("kind") or "")
        key = r.text("key", pattern=SHA256_RE) or ""
        parent = r.text("parent", pattern=SHA256_RE, optional=True)
        files = _parse_files(r.raw("files"), kind)
        r.done()
        snapshot = cls(kind.name, key, parent, files)
        if snapshot.total_bytes > MAX_SNAPSHOT_BYTES:
            raise DocumentError("snapshot.files: more bytes than a snapshot may hold")
        return snapshot


def _parse_files(value: Any, kind: CacheKind) -> dict[str, StoredFile]:
    if not isinstance(value, Mapping):
        raise DocumentError("snapshot.files: expected an object")
    files = {}
    for path, item in value.items():
        relative_path(path, "snapshot.files")
        if not kind.admits(path):
            raise DocumentError(f"snapshot.files: {path} is not a file the {kind.name} cache admits")
        e = Reader(item, f"snapshot.files.{path}")
        files[path] = StoredFile(e.text("sha256", pattern=SHA256_RE) or "", e.integer("size", minimum=0) or 0)
        e.done()
    return files


@dataclass(frozen=True)
class Fetched:
    """What a read of the cache gave: the snapshot copied (``None`` when there was none that verified), and every reason a
    stored snapshot was dropped."""

    snapshot: Snapshot | None
    files: int = 0
    bytes: int = 0
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class Published:
    """What a publication added: the snapshot's name, the blobs and bytes that were new, and whether it was there already."""

    snapshot_sha256: str
    new_blobs: int
    new_bytes: int
    existed: bool


def read_staged_snapshot(staging: Path) -> tuple[Snapshot, str]:
    """The snapshot a test staged in ``staging/manifest.json`` and the hash of its bytes; the file is read symlink-proof."""
    try:
        root_fd = fsafe.open_root(staging)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise SnapshotInvalid(f"the staged contribution cannot be read: {exc}") from None
    try:
        raw = _read_bounded(root_fd, MANIFEST_NAME)
    finally:
        os.close(root_fd)
    return Snapshot.parse(raw), hashlib.sha256(raw).hexdigest()


def _read_bounded(root_fd: int, rel: str) -> bytes:
    try:
        fd = fsafe.open_beneath(root_fd, rel)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise SnapshotInvalid(f"{rel} cannot be read: {exc}") from None
    try:
        raw = os.read(fd, _MAX_MANIFEST_BYTES + 1)
        while len(raw) <= _MAX_MANIFEST_BYTES:
            more = os.read(fd, _MAX_MANIFEST_BYTES + 1 - len(raw))
            if not more:
                break
            raw += more
    finally:
        os.close(fd)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise SnapshotInvalid(f"{rel} is larger than a snapshot list may be")
    return raw


class DirJitStore:
    """The cache in a local directory tree; the local sandbox backend and tests use it, an object store follows."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    # -- reading ---------------------------------------------------------------------------------------------------

    def fetch(self, kind: CacheKind, key: str, dest: Path) -> Fetched:
        """Copy the best snapshot of ``kind`` under ``key`` into ``dest``, verified; the next generation when one fails."""
        if kind not in KINDS or not SHA256_RE.fullmatch(key):
            raise JitStoreError("the cache key is not usable")
        listed, problems = self._generations(kind, key)
        for snapshot in listed[:_GENERATIONS_TRIED]:
            try:
                size = self._copy_snapshot(snapshot, Path(dest))
            except SnapshotInvalid as exc:
                problems.append(f"{kind.name} snapshot {snapshot.sha256()[:12]}: {exc}")
                continue
            return Fetched(snapshot, len(snapshot.files), size, tuple(problems))
        return Fetched(None, 0, 0, tuple(problems))

    def _generations(self, kind: CacheKind, key: str) -> tuple[list[Snapshot], list[str]]:
        directory = self._root / SNAPSHOTS_DIR / kind.name / key
        try:
            names = sorted(path.name for path in directory.iterdir())
        except FileNotFoundError:
            return [], []
        except OSError as exc:
            return [], [f"{kind.name} snapshots cannot be listed: {type(exc).__name__}"]
        found, problems = [], []
        for name in names:
            try:
                found.append(self._read_snapshot(directory / name, kind, key))
            except SnapshotInvalid as exc:
                problems.append(f"{kind.name} snapshot {name[:12]}: {exc}")
        return sorted(found, key=lambda s: (-len(s.files), s.sha256())), problems

    @staticmethod
    def _read_snapshot(path: Path, kind: CacheKind, key: str) -> Snapshot:
        stem = path.name.removesuffix(".json")
        if path.name == stem or not SHA256_RE.fullmatch(stem):
            raise SnapshotInvalid("the name is not a snapshot hash")
        try:
            root_fd = fsafe.open_root(path.parent)
        except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
            raise SnapshotInvalid(f"cannot be read: {exc}") from None
        try:
            raw = _read_bounded(root_fd, path.name)
        finally:
            os.close(root_fd)
        if hashlib.sha256(raw).hexdigest() != stem:
            raise SnapshotInvalid("the list does not hash to its name")
        snapshot = Snapshot.parse(raw)
        if snapshot.to_bytes() != raw:
            raise SnapshotInvalid("the list is not in its canonical form")
        if (snapshot.kind, snapshot.key) != (kind.name, key):
            raise SnapshotInvalid("the list is for another kind or key than the directory it is in")
        return snapshot

    def _copy_snapshot(self, snapshot: Snapshot, dest: Path) -> int:
        """Copy every file; on a failure the files this call made are removed again, so no half snapshot is left."""
        try:
            root_fd = fsafe.open_root(self._root)
        except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
            raise SnapshotInvalid(f"the store cannot be read: {exc}") from None
        made: list[Path] = []
        total = 0
        try:
            for relative, stored in snapshot.files.items():
                target = dest / relative
                if not target.parent.exists():
                    os.makedirs(target.parent, mode=0o700, exist_ok=True)
                _copy_blob(root_fd, stored, target)
                made.append(target)
                total += stored.size
            _localise(snapshot, Path(dest))
        except BaseException:
            for path in made:
                path.unlink(missing_ok=True)
            raise
        finally:
            os.close(root_fd)
        return total

    def listing(self, kind: CacheKind, key: str, sha256: str) -> Mapping[str, StoredFile] | None:
        """The files of the stored snapshot ``sha256`` (a contribution's parent), or ``None`` when it is not stored or does
        not verify."""
        if kind not in KINDS or not SHA256_RE.fullmatch(key) or not SHA256_RE.fullmatch(sha256):
            return None
        try:
            return self._read_snapshot(self._root / SNAPSHOTS_DIR / kind.name / key / f"{sha256}.json", kind, key).files
        except SnapshotInvalid:
            return None

    def holds(self, sha256: str) -> bool:
        """Whether the blob with this hash is stored already, so a publisher need not bring it."""
        return SHA256_RE.fullmatch(sha256) is not None and (self._root / BLOBS_DIR / sha256[:2] / sha256).is_file()

    # -- writing ---------------------------------------------------------------------------------------------------

    def publish(self, staging: Path, expected_sha256: str) -> Published:
        """Store what a test staged: ``staging/manifest.json`` must be the list whose hash was signed, and every file it
        lists must be stored already or staged under ``staging/blobs/<sha256>``; nothing stored is replaced."""
        snapshot, actual = read_staged_snapshot(staging)
        if actual != expected_sha256:
            raise SnapshotInvalid("the staged list is not the one whose hash the result signed")
        incoming = self._root / f"{_INCOMING}{uuid.uuid4().hex}"
        try:
            incoming.mkdir(parents=True, mode=0o700)
            new_blobs, new_bytes = self._store_blobs(snapshot, Path(staging), incoming)
            existed = self._store_snapshot(snapshot, actual)
        except OSError as exc:
            raise JitStoreError(f"the cache could not be written: {type(exc).__name__}") from None
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
        return Published(actual, new_blobs, new_bytes, existed)

    def _store_blobs(self, snapshot: Snapshot, staging: Path, incoming: Path) -> tuple[int, int]:
        """Store the blobs the list names that are not stored (or are stored damaged and are staged sound)."""
        new_blobs = new_bytes = 0
        staged_fd = fsafe.open_root(staging)
        try:
            for relative, stored in snapshot.files.items():
                final = self._root / BLOBS_DIR / stored.sha256[:2] / stored.sha256
                if self._is_stored(
                    final, stored, relative, fsafe.exists_beneath(staged_fd, f"{BLOBS_DIR}/{stored.sha256}")
                ):
                    continue
                self._store_blob(staged_fd, relative, stored, incoming, final)
                new_blobs, new_bytes = new_blobs + 1, new_bytes + stored.size
        finally:
            os.close(staged_fd)
        return new_blobs, new_bytes

    @staticmethod
    def _is_stored(final: Path, stored: StoredFile, relative: str, staged: bool) -> bool:
        """Whether the blob is there and needs nothing: a sound one is never rewritten. A file the contributor took from the
        cache is judged by its size (it was verified when it was read); one it also staged is read, and a damaged stored
        copy is replaced by the staged bytes, which are checked against the same hash before they are stored."""
        if not final.is_file():
            return False
        if final.stat().st_size != stored.size and not staged:
            raise SnapshotInvalid(f"{relative}: the stored blob is not {stored.size} bytes")
        if not staged:
            return True
        return _file_sha256(final) == stored.sha256

    @staticmethod
    def _store_blob(staged_fd: int, relative: str, stored: StoredFile, incoming: Path, final: Path) -> None:
        try:
            source = fsafe.open_beneath(staged_fd, f"{BLOBS_DIR}/{stored.sha256}")
        except (fsafe.MissingEntry, fsafe.UnsafeEntry):
            raise SnapshotInvalid(f"{relative} is neither stored nor staged") from None
        pending = incoming / stored.sha256
        try:
            _copy_checked(source, pending, stored, relative, mode=_BLOB_MODE)
        finally:
            os.close(source)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(pending, final)
        except OSError:
            if not final.is_file():  # another publisher stored the same hash first: its bytes are these
                raise
        fsync_dir(final.parent)

    def _store_snapshot(self, snapshot: Snapshot, name: str) -> bool:
        directory = self._root / SNAPSHOTS_DIR / snapshot.kind / snapshot.key
        final = directory / f"{name}.json"
        if final.exists():
            return True
        directory.mkdir(parents=True, exist_ok=True)
        pending = self._root / f"{_INCOMING}{uuid.uuid4().hex}.json"
        try:
            with open(pending, "xb") as out:
                out.write(snapshot.to_bytes())
                out.flush()
                os.fsync(out.fileno())
            os.chmod(pending, _BLOB_MODE)
            os.rename(pending, final)
        finally:
            pending.unlink(missing_ok=True)
        fsync_dir(directory)
        return False


def _file_sha256(path: Path) -> str | None:
    try:
        root_fd = fsafe.open_root(path.parent)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry):
        return None
    try:
        fd = fsafe.open_beneath(root_fd, path.name)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry):
        return None
    finally:
        os.close(root_fd)
    try:
        return fsafe.sha256_and_size(fd)[0]
    finally:
        os.close(fd)


def _localise(snapshot: Snapshot, dest: Path) -> None:
    """Write the directory ``dest`` into each copied file that names the directory it was written in. Every file was
    verified against its hash before this runs, so what is rewritten is the stored bytes and nothing else."""
    kind = kind_named(snapshot.kind)
    for relative in snapshot.files:
        if kind.locates(relative):
            _rewrite(relative, dest, kind)


def _rewrite(relative: str, dest: Path, kind: CacheKind) -> None:
    descriptor = os.open(dest / relative, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "r+b") as handle:
        raw = handle.read(MAX_LOCATED_BYTES + 1)
        try:
            local = localised(raw, dest, kind, relative)
        except NotPortable as exc:
            raise SnapshotInvalid(f"{Path(relative).name}: {exc}") from None
        handle.seek(0)
        handle.truncate()
        handle.write(local)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_blob(root_fd: int, stored: StoredFile, target: Path) -> None:
    try:
        source = fsafe.open_beneath(root_fd, f"{BLOBS_DIR}/{stored.sha256[:2]}/{stored.sha256}")
    except fsafe.MissingEntry:
        raise SnapshotInvalid(f"{target.name}: the blob is missing from the store") from None
    except fsafe.UnsafeEntry as exc:
        raise SnapshotInvalid(f"{target.name}: the blob is not a plain file ({exc.kind})") from None
    try:
        _copy_checked(source, target, stored, target.name, mode=_FILE_MODE)
    finally:
        os.close(source)


def _copy_checked(source: int, target: Path, stored: StoredFile, shown: str, *, mode: int) -> None:
    """Copy ``source`` to a new file ``target`` while hashing it; a file that is not the hash and size listed is removed."""
    digest, total = hashlib.sha256(), 0
    descriptor = os.open(target, _FETCH_FLAGS, mode)
    try:
        with os.fdopen(descriptor, "wb") as out:
            for chunk in fsafe.stream(source):
                total += len(chunk)
                if total > stored.size:
                    break
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != stored.sha256:
        target.unlink(missing_ok=True)
        raise SnapshotInvalid(f"{shown}: the bytes are not the {stored.size} bytes whose hash is listed")
