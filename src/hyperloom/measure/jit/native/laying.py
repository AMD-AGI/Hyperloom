# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A runner's read of the ready builds a sheet names (``environment.builds``), for the farm to lay.

Every byte is proven before it is laid: the store's record verifies under a trusted key and hashes to the key the sheet
names, was built for the sheet's image, is a ready build of the root's own component, and its canonical bytes hash to the
manifest SHA-256 the sheet names (so the sheet fixes which build of the key is measured). Each output is copied out of
the store onto the farm's local disk with its hash checked while copying (:meth:`~hyperloom.measure.jit.native.store.
BuildStore.fetch`), never linked to the store on shared storage.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.measure import fsafe
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.jit.native.record import READY, RECORD_FILES, BuildRecord
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused
from hyperloom.measure.jit.store import SnapshotInvalid, read_bounded

__all__ = ["NamedBuild", "NativeLayError", "SheetBuilds", "manifest_sha256", "sheet_builds"]

_MANIFEST = RECORD_FILES[READY][0]
_MODULE_SUFFIX = ".so"


class NativeLayError(Exception):
    """A build the sheet names cannot be laid: missing, refused by a check, or not the one named."""


@dataclass(frozen=True)
class NamedBuild:
    """One build as a sheet names it."""

    root_id: str
    key: str
    manifest_sha256: str


def manifest_sha256(record: BuildRecord) -> str:
    """The SHA-256 of a record's manifest as the store holds it (its canonical bytes)."""
    return hashlib.sha256(record.to_bytes()).hexdigest()


class SheetBuilds:
    """The builds one sheet names, read from ``store`` for ``image_digest``."""

    def __init__(self, store: BuildStore, named: Mapping[str, NamedBuild], image_digest: str) -> None:
        self._store, self._named, self._image = store, named, image_digest

    def record(self, root_id: str, import_name: str) -> BuildRecord:
        """The proven ready record the sheet names for ``root_id``, whose outputs all lie in the package ``import_name``.

        Raises:
            NativeLayError: The sheet names none, or the store's entry is missing or is not the named build.
        """
        return self._proven(root_id, lambda outputs: [path for path in outputs if path.parts[:1] != (import_name,)])

    def module_record(self, root_id: str) -> BuildRecord:
        """The proven ready record of the JIT modules the sheet names for the package root ``root_id``: each output is a
        module file of a JIT directory (``<name>.so``).

        Raises:
            NativeLayError: As :meth:`record`.
        """
        return self._proven(root_id, lambda outputs: [path for path in outputs if not _module_file(path)])

    def _proven(self, root_id: str, outside: Callable[[list[PurePosixPath]], list[PurePosixPath]]) -> BuildRecord:
        named = self._named.get(root_id)
        if named is None:
            raise NativeLayError(f"{root_id}: a runtime root whose build the sheet does not name")
        try:
            record = self._store.read(self._inputs_of(named).inputs)
        except BuildRefused as exc:
            raise NativeLayError(f"{root_id}: the stored build {named.key} is refused: {exc}") from None
        problem = self._problem(record, named, outside)
        if problem:
            raise NativeLayError(f"{root_id}: the stored build {named.key} {problem}")
        return record  # type: ignore[return-value]  # _problem refused None

    def fetch(self, record: BuildRecord, site: Path) -> None:
        """Copy the record's outputs below ``site`` onto local disk, each proven against its hash."""
        try:
            self._store.fetch(record, site)
        except BuildRefused as exc:
            raise NativeLayError(f"the outputs of the build {record.key} cannot be laid: {exc}") from None

    def _inputs_of(self, named: NamedBuild) -> BuildRecord:
        """The record the entry holds, parsed only to learn its inputs; :meth:`BuildStore.read` then proves it."""
        directory = self._store.entry_dir_of(named.root_id, named.key)
        if not os.path.lexists(directory):
            raise NativeLayError(f"{named.root_id}: the build {named.key} is not in the store")
        try:
            root_fd = fsafe.open_root(directory)
        except (fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
            raise NativeLayError(f"{named.root_id}: the build {named.key} is not a directory: {exc}") from None
        try:
            return BuildRecord.parse(read_bounded(root_fd, _MANIFEST))
        except (SnapshotInvalid, DocumentError) as exc:
            raise NativeLayError(f"{named.root_id}: the build {named.key} has no ready manifest: {exc}") from None
        finally:
            os.close(root_fd)

    def _problem(
        self,
        record: BuildRecord | None,
        named: NamedBuild,
        outside: Callable[[list[PurePosixPath]], list[PurePosixPath]],
    ) -> str:
        if record is None or record.status != READY:
            return "is not a ready build"
        if record.key != named.key or record.inputs.component != named.root_id:
            return f"is the build {record.key} of {record.inputs.component}"
        if record.inputs.image_digest != self._image:
            return f"was built for {record.inputs.image_digest}, not {self._image}"
        if manifest_sha256(record) != named.manifest_sha256:
            return f"has the manifest {manifest_sha256(record)}, not {named.manifest_sha256}"
        stray = outside([PurePosixPath(output.path) for output in record.outputs])
        if stray:
            return f"has outputs outside its place: {[path.as_posix() for path in stray[:3]]}"
        return ""


def _module_file(path: PurePosixPath) -> bool:
    return len(path.parts) == 1 and path.suffix == _MODULE_SUFFIX


def sheet_builds(named: tuple[NamedBuild, ...], store: BuildStore | None, image_digest: str) -> SheetBuilds | None:
    """The builds a sheet names, read from ``store``; ``None`` for a sheet that names none.

    Raises:
        NativeLayError: The sheet names builds and this process has no build store.
    """
    if not named:
        return None
    if store is None:
        raise NativeLayError(
            "the sheet names native builds and this runner is given no build store (HYPERLOOM_BUILD_STORE and "
            "HYPERLOOM_BUILD_TRUST)"
        )
    return SheetBuilds(store, {build.root_id: build for build in named}, image_digest)
