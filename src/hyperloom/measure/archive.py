# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-once archive of verified result bundles.

A bundle is stored under ``<request_id>/<manifest_sha256>/``, laid out exactly as it was staged (``manifest.json``,
``manifest.sig`` and the listed files), so an archived bundle can be verified again with
:func:`hyperloom.measure.bundle.verify_bundle`. The archive only adds: nothing here updates, replaces or deletes a
stored bundle, and a bundle becomes visible under its final name in one step, so a reader never sees half of one.

:class:`DirResultArchive` is the local-directory implementation used by the local sandbox backend and by tests; the S3
implementation follows the same :class:`ResultArchive` protocol.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Protocol

from hyperloom.measure import fsafe
from hyperloom.measure.docs import SHA256_RE
from hyperloom.measure.manifest import RESERVED_NAMES

ARCHIVE_FAILED = "archive_failed"
_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_INCOMING = ".incoming-"
MANIFEST_NAME = RESERVED_NAMES[0]


class ArchiveError(Exception):
    """The bundle could not be archived; ``error_class`` is ``archive_failed``, an infrastructure failure (4.7)."""

    error_class = ARCHIVE_FAILED


class ResultArchive(Protocol):
    def put_bundle(self, request_id: str, manifest_sha256: str, staging_dir: Path) -> str:
        """Store the bundle in ``staging_dir`` and return where it is stored; storing it again is a success."""

    def bundle_path(self, request_id: str, manifest_sha256: str) -> Path:
        """A readable directory holding the stored bundle, laid out as it was staged.

        Raises :class:`ArchiveError` when nothing is stored under the key or the archive cannot be reached.
        """


class DirResultArchive:
    """Archive into a local directory tree."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def bundle_dir(self, request_id: str, manifest_sha256: str) -> Path:
        """Where a bundle is (or would be) stored."""
        _check_key(request_id, manifest_sha256)
        return self._root / request_id / manifest_sha256

    def bundle_path(self, request_id: str, manifest_sha256: str) -> Path:
        path = self.bundle_dir(request_id, manifest_sha256)
        if not path.is_dir():
            raise ArchiveError("no bundle is archived under this request and manifest hash")
        return path

    def put_bundle(self, request_id: str, manifest_sha256: str, staging_dir: Path) -> str:
        final = self.bundle_dir(request_id, manifest_sha256)
        listing = _read_staging(Path(staging_dir), manifest_sha256)
        try:
            return self._put(final, Path(staging_dir), listing)
        except ArchiveError:
            raise
        except OSError as exc:
            raise ArchiveError(f"the archive could not be written: {type(exc).__name__}") from None

    def _put(self, final: Path, staging_dir: Path, listing: dict[str, str]) -> str:
        if final.exists():
            return self._same_or_conflict(final, listing)
        final.parent.mkdir(parents=True, exist_ok=True)
        incoming = self._root / f"{_INCOMING}{uuid.uuid4().hex}"
        incoming.mkdir(parents=True, mode=0o700)
        try:
            _copy_listing(staging_dir, incoming, listing)
            _fsync_tree(incoming)
            try:
                os.rename(incoming, final)
            except OSError:
                # Another writer stored the same key first, or the name is taken by something else.
                if final.exists():
                    return self._same_or_conflict(final, listing)
                raise
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
        fsync_dir(final.parent)
        return os.fspath(final)

    def _same_or_conflict(self, final: Path, listing: dict[str, str]) -> str:
        """A bundle is already stored under this key: identical content is a success, anything else a refusal."""
        if _hash_tree(final) != listing:
            raise ArchiveError("a different bundle is already archived under this request and manifest hash")
        return os.fspath(final)


def _check_key(request_id: str, manifest_sha256: str) -> None:
    if not _SEGMENT_RE.fullmatch(request_id) or request_id in (".", ".."):
        raise ArchiveError("the request id is not usable as an archive key")
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise ArchiveError("the manifest hash is not a sha256")


def _read_staging(staging_dir: Path, manifest_sha256: str) -> dict[str, str]:
    """Relative path to sha256 of every file staged, after proving ``manifest.json`` is the one the key names."""
    try:
        root_fd = fsafe.open_root(staging_dir)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise ArchiveError(f"the staging directory cannot be archived: {exc}") from None
    try:
        listing = {rel: _hash_one(root_fd, rel) for rel in fsafe.walk_regular_files(root_fd)}
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise ArchiveError(f"the staging directory cannot be archived: {exc}") from None
    finally:
        os.close(root_fd)
    if listing.get(MANIFEST_NAME) != manifest_sha256:
        raise ArchiveError("manifest.json is not the manifest the archive key names")
    return listing


def _hash_one(root_fd: int, rel: str) -> str:
    fd = fsafe.open_beneath(root_fd, rel)
    try:
        return fsafe.sha256_and_size(fd)[0]
    finally:
        os.close(fd)


def _hash_tree(directory: Path) -> dict[str, str]:
    try:
        root_fd = fsafe.open_root(directory)
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise ArchiveError(f"the stored bundle cannot be read: {exc}") from None
    try:
        return {rel: _hash_one(root_fd, rel) for rel in fsafe.walk_regular_files(root_fd)}
    except (fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
        raise ArchiveError(f"the stored bundle cannot be read: {exc}") from None
    finally:
        os.close(root_fd)


def _copy_listing(staging_dir: Path, incoming: Path, listing: dict[str, str]) -> None:
    """Copy each file while hashing it again, so a byte that changed after the listing never reaches the archive."""
    root_fd = fsafe.open_root(staging_dir)
    try:
        for rel, expected in listing.items():
            _copy_verified(root_fd, rel, expected, incoming / rel)
    finally:
        os.close(root_fd)


def _copy_verified(root_fd: int, rel: str, expected: str, target: Path) -> None:
    """Copy ``rel`` below ``root_fd`` to a new file ``target``, refusing a copy whose bytes are not ``expected``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    source = fsafe.open_beneath(root_fd, rel)
    digest = hashlib.sha256()
    try:
        with open(target, "xb") as out:
            for chunk in fsafe.stream(source):
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    finally:
        os.close(source)
    if digest.hexdigest() != expected:
        raise ArchiveError(f"{rel} changed while it was being archived")
    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root):
        fsync_dir(Path(directory))
