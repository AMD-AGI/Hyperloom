# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write-once archive of verified result bundles.

A bundle is stored under ``<request_id>/<manifest_sha256>/``, laid out exactly as it was staged (``manifest.json``,
``manifest.sig`` and the listed files), so an archived bundle can be verified again with
:func:`hyperloom.measure.bundle.verify_bundle`. The archive only adds: nothing here updates, replaces or deletes a
stored bundle, and a bundle becomes visible under its final name in one step, so a reader never sees half of one.

:class:`DirResultArchive` is the local-directory implementation used by the local sandbox backend and by tests; the S3
implementation follows the same :class:`ResultArchive` protocol. Below storage set up to be shared
(:func:`~hyperloom_common.shared_storage.parent_is_shared`), where a later leg of a run may be another account of the
group, the directories it makes on the way are the group's (``2775``) and the directories of a stored bundle are
``2750``, which the group reads and only their writer changes, whatever the umask; anywhere else the umask decides and a
bundle stays its writer's.
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

from hyperloom_common.shared_storage import make_shared_dirs, open_directory, parent_is_shared, read_directory
from hyperloom.measure import fsafe
from hyperloom.measure.docs import SHA256_RE
from hyperloom.measure.manifest import RESERVED_NAMES

ARCHIVE_FAILED = "archive_failed"
_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_INCOMING = ".incoming-"
MANIFEST_NAME = RESERVED_NAMES[0]
#: The directories of a stored bundle below shared storage: the group reads them, and no other account can change them.
BUNDLE_DIR_MODE = 0o2750


class ArchiveError(Exception):
    """The bundle could not be archived; ``error_class`` is ``archive_failed``, an infrastructure failure."""

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
        make_shared_dirs(final.parent)
        dir_mode = BUNDLE_DIR_MODE if parent_is_shared(final) else None
        incoming = f"{_INCOMING}{uuid.uuid4().hex}"
        root_fd = open_directory(self._root)
        try:
            os.mkdir(incoming, 0o700, dir_fd=root_fd)
            top = _open_incoming(root_fd, incoming)
            try:
                _fill(top, staging_dir, listing, dir_mode)
                parent_fd = open_directory(final.parent)
                try:
                    os.rename(incoming, final.name, src_dir_fd=root_fd, dst_dir_fd=parent_fd)
                except OSError:
                    # Another writer stored the same key first, or the name is taken by something else.
                    if final.exists():
                        return self._same_or_conflict(final, listing)
                    raise
                else:
                    _fsync_at(parent_fd)
                finally:
                    os.close(parent_fd)
            finally:
                try:
                    _remove_incoming(root_fd, incoming, top)
                finally:
                    os.close(top)
        finally:
            os.close(root_fd)
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


def _open_incoming(root_fd: int, incoming: str) -> int:
    """The new directory ``incoming`` below ``root_fd``, opened without following a link and refused unless it is this
    account's own and closed to everyone else: the archive's root may be a directory the group writes, so its name may
    have been taken over since it was made. Everything stored is written through this descriptor."""
    top = os.open(incoming, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    try:
        found = os.fstat(top)
        if found.st_uid != os.geteuid() or found.st_mode & 0o077:
            raise ArchiveError("the archive's incoming directory was replaced while the bundle was being stored")
    except BaseException:
        os.close(top)
        raise
    return top


def _fill(top: int, staging_dir: Path, listing: dict[str, str], dir_mode: int | None) -> None:
    """Fill the directory ``top`` with the listed files and flush it, its directories given ``dir_mode`` when it is not
    ``None``."""
    created = _copy_listing(staging_dir, top, listing)
    for rel in ("", *created):
        directory = fsafe.open_dir_beneath(top, rel.split("/") if rel else [])
        try:
            if dir_mode is not None:
                os.fchmod(directory, dir_mode)
            os.fsync(directory)
        finally:
            os.close(directory)


def _remove_incoming(root_fd: int, incoming: str, top: int) -> None:
    """Remove what is left of ``incoming`` when it still names the directory open as ``top``; a name another account
    took over is left as it is."""
    try:
        here, opened = os.stat(incoming, dir_fd=root_fd, follow_symlinks=False), os.fstat(top)
    except OSError:
        return  # renamed into place, or gone
    if (here.st_dev, here.st_ino) == (opened.st_dev, opened.st_ino):
        shutil.rmtree(os.path.join(f"/proc/self/fd/{root_fd}", incoming), ignore_errors=True)


def _copy_listing(staging_dir: Path, incoming: int, listing: dict[str, str]) -> list[str]:
    """Copy each file below the directory ``incoming`` while hashing it again, so a byte that changed after the listing
    never reaches the archive; returns the directories made for them."""
    created: list[str] = []
    root_fd = fsafe.open_root(staging_dir)
    try:
        for rel, expected in listing.items():
            *parents, name = rel.split("/")
            parent = fsafe.open_dir_beneath(incoming, parents, created=created)
            try:
                _copy_verified(root_fd, rel, expected, parent, name)
            finally:
                os.close(parent)
    finally:
        os.close(root_fd)
    return created


def _copy_verified(root_fd: int, rel: str, expected: str, parent: int, name: str) -> None:
    """Copy ``rel`` below ``root_fd`` to a new file ``name`` in ``parent``, refusing a copy whose bytes are not
    ``expected``."""
    source = fsafe.open_beneath(root_fd, rel)
    digest = hashlib.sha256()
    try:
        target = fsafe.create_exclusive(parent, name)
        try:
            for chunk in fsafe.stream(source):
                digest.update(chunk)
                fsafe.write_all(target, chunk)
            os.fsync(target)
            os.fchmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        finally:
            os.close(target)
    finally:
        os.close(source)
    if digest.hexdigest() != expected:
        raise ArchiveError(f"{rel} changed while it was being archived")


def _fsync_at(descriptor: int) -> None:
    directory = read_directory(descriptor)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
