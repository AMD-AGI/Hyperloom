# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native build store: one signed entry per key, outputs kept as content-addressed blobs.

Layout::

    <root>/blobs/<aa>/<sha256>                                  one output file (the layout of DirJitStore)
    <root>/builds/<component>/<key>/manifest.json, manifest.sig  a build that made its outputs, and its log.txt
    <root>/builds/<component>/<key>/failed.json, failed.sig      a deterministic compile error, and its log.txt
    <root>/.<key>.lock                                          held while one key is looked up or built

Atomicity is that of the runner cache (:mod:`hyperloom.measure.runner.cache_build`): an entry is made in
``.<key>.tmp-<pid>-<rand>`` beside its final place, its record is the last file written and one ``rename`` makes it
visible. The writer (the build service, :mod:`hyperloom.measure.runner.native.service`) holds an exclusive ``flock`` on
:meth:`BuildStore.lock_path` around reading and publishing, so concurrent requests of one key build it once. An entry is
never overwritten: one that fails a check is refused, not rebuilt. A build that failed for machine reasons
(:class:`BuildInfraError`) leaves nothing.

Reading proves every byte: the record's signature under a trusted key, the key recomputed from the record's inputs against
the one asked for, the image, and each blob's hash while it is copied out (:func:`~hyperloom.measure.jit.store.copy_blob`).
"""

from __future__ import annotations

import os
import secrets
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.archive import fsync_dir
from hyperloom.measure.docs import DocumentError
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import EXECUTABLE, LOG_NAME, READY, RECORD_FILES, BuildRecord
from hyperloom.measure.jit.native.trust import BuildRefused, Refusal, Signer, Trust
from hyperloom.measure.jit.store import BLOBS_DIR, SnapshotInvalid, StoredFile, copy_blob, read_bounded, store_blob

__all__ = ["BUILDS_DIR", "BuildInfraError", "BuildStore"]

BUILDS_DIR = "builds"
#: The log kept beside a record is the compiler's tail, capped so a noisy build cannot fill the store.
MAX_LOG_BYTES = 256 * 1024
#: A record larger than this is not one the service wrote; it is skipped when listing ready builds.
_MAX_RECORD_BYTES = 1 << 20
_ENTRY_FILE_MODE = 0o644
_FETCHED_EXECUTABLE, _FETCHED_PLAIN = 0o700, 0o600


class BuildInfraError(Exception):
    """A build failed for machine reasons (disk, toolchain, timeout, a killed compiler, a lock never released): retried,
    never stored as the candidate's failure."""


class BuildStore:
    """A directory store of native builds, read with :class:`Trust`; the build service writes it with :meth:`publish`."""

    def __init__(self, root: Path, trust: Trust) -> None:
        self._root = Path(root)
        self._trust = trust

    def entry_dir(self, inputs: BuildInputs) -> Path:
        return self.entry_dir_of(inputs.component, inputs.key)

    def entry_dir_of(self, component: str, key: str) -> Path:
        """The entry of ``key``, for a reader that knows only the component and the key (a sheet's build)."""
        return self._root / BUILDS_DIR / component / key

    def holds(self, sha256: str) -> bool:
        return (self._root / BLOBS_DIR / sha256[:2] / sha256).is_file()

    # -- reading ---------------------------------------------------------------------------------------------------

    def read(self, inputs: BuildInputs) -> BuildRecord | None:
        """The verified record of ``inputs``' key; ``None`` when there is no entry.

        Raises:
            BuildRefused: When an entry is there and fails a check; :attr:`BuildRefused.reason` says which.
        """
        directory = self.entry_dir(inputs)
        if not os.path.lexists(directory):
            return None
        try:
            root_fd = fsafe.open_root(directory)
        except fsafe.UnsafeEntry:
            raise BuildRefused(Refusal.RECORD_INVALID, "the entry is not a directory") from None
        try:
            for status, (record_name, signature_name) in RECORD_FILES.items():
                if fsafe.exists_beneath(root_fd, record_name):
                    record = self._verified(root_fd, record_name, signature_name, inputs)
                    if record.status != status:
                        raise BuildRefused(Refusal.RECORD_INVALID, f"{record_name} holds a {record.status} record")
                    return record
        finally:
            os.close(root_fd)
        raise BuildRefused(Refusal.RECORD_INVALID, "the entry holds no record")

    def _verified(self, root_fd: int, record_name: str, signature_name: str, inputs: BuildInputs) -> BuildRecord:
        try:
            raw, signature = read_bounded(root_fd, record_name), read_bounded(root_fd, signature_name)
        except SnapshotInvalid as exc:
            raise BuildRefused(Refusal.RECORD_INVALID, str(exc)) from None
        self._trust.check(raw, signature)
        try:
            record = BuildRecord.parse(raw)
        except DocumentError as exc:
            raise BuildRefused(Refusal.RECORD_INVALID, str(exc)) from None
        if record.inputs.image_digest != inputs.image_digest:
            raise BuildRefused(Refusal.IMAGE_MISMATCH, f"built for {record.inputs.image_digest}")
        if record.key != inputs.key or record.inputs.key != inputs.key:
            raise BuildRefused(Refusal.KEY_MISMATCH, "the record's inputs do not hash to the key asked for")
        return record

    def fetch(self, record: BuildRecord, dest: Path) -> None:
        """Copy each output of a ready record to ``dest/<path>``, proving its hash and size while copying."""
        root_fd = fsafe.open_root(self._root)
        try:
            for output in record.outputs:
                if not self.holds(output.sha256):
                    raise BuildRefused(Refusal.BLOB_MISSING, f"{output.path}: the blob is not in the store")
                target = Path(dest) / output.path
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    copy_blob(root_fd, StoredFile(output.sha256, output.size), target)
                except SnapshotInvalid as exc:
                    raise BuildRefused(Refusal.BLOB_MISMATCH, str(exc)) from None
                os.chmod(target, _FETCHED_EXECUTABLE if output.mode == EXECUTABLE else _FETCHED_PLAIN)
        finally:
            os.close(root_fd)

    def ready_records(self, component: str, image_digest: str) -> Iterator[BuildRecord]:
        """The verified ready records of ``component`` built for ``image_digest``: what a build of it took before. An
        entry that cannot be read or proven is passed over; it tells nothing of how long a build takes."""
        component_dir = self._root / BUILDS_DIR / component
        if not component_dir.is_dir():
            return
        for entry in component_dir.iterdir():
            record = self._ready(entry / RECORD_FILES[READY][0])
            if record is not None and record.inputs.image_digest == image_digest:
                yield record

    def _ready(self, manifest: Path) -> BuildRecord | None:
        try:
            if not manifest.is_file() or manifest.stat().st_size > _MAX_RECORD_BYTES:
                return None
            return self.read(BuildRecord.parse(manifest.read_bytes()).inputs)
        except (OSError, DocumentError, BuildRefused):
            return None

    # -- writing ---------------------------------------------------------------------------------------------------

    def lock_path(self, inputs: BuildInputs) -> Path:
        """The file whose exclusive ``flock`` a writer of ``inputs``' key holds around :meth:`read` and :meth:`publish`."""
        return self._root / f".{inputs.key}.lock"

    def clear_leftovers(self, inputs: BuildInputs) -> None:
        """Remove the unfinished entries of a writer that died; only under the key's lock."""
        for leftover in self.entry_dir(inputs).parent.glob(f".{inputs.key}.tmp-*"):
            shutil.rmtree(leftover, ignore_errors=True)

    def publish(self, record: BuildRecord, staging: Path, log: str, signer: Signer) -> None:
        """Store ``record``'s output blobs from ``staging/blobs/<sha256>`` and its signed entry; only under the key's lock,
        after :meth:`read` found no entry.

        Raises:
            BuildInfraError: When an output could not be stored; no entry is made.
        """
        try:
            self.entry_dir(record.inputs).parent.mkdir(parents=True, exist_ok=True)
            self._store_blobs(record, staging)
            self._write_entry(record, log, signer)
        except OSError as exc:
            raise BuildInfraError(f"the store could not be written: {type(exc).__name__}") from None

    def _store_blobs(self, record: BuildRecord, staging: Path) -> None:
        """Store the outputs' blobs; one already stored is kept (its name is the hash of what it must hold)."""
        incoming = self._root / f".incoming-{uuid.uuid4().hex}"
        incoming.mkdir(parents=True, mode=0o700)
        staged_fd = fsafe.open_root(staging)
        try:
            for output in record.outputs:
                final = self._root / BLOBS_DIR / output.sha256[:2] / output.sha256
                if not final.is_file():
                    stored = StoredFile(output.sha256, output.size)
                    store_blob(staged_fd, output.path, stored, incoming, final)
        except SnapshotInvalid as exc:
            raise BuildInfraError(f"an output could not be stored: {exc}") from None
        finally:
            os.close(staged_fd)
            shutil.rmtree(incoming, ignore_errors=True)

    def _write_entry(self, record: BuildRecord, log: str, signer: Signer) -> None:
        """Write the entry beside its place, the record last, and rename it into place."""
        final = self.entry_dir(record.inputs)
        pending = final.parent / f".{record.key}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
        pending.mkdir()
        try:
            record_name, signature_name = RECORD_FILES[record.status]
            raw = record.to_bytes()
            _write(pending / LOG_NAME, log.encode("utf-8", errors="replace")[-MAX_LOG_BYTES:])
            _write(pending / signature_name, signer.sign(raw))
            _write(pending / record_name, raw)
            fsync_dir(pending)
            os.rename(pending, final)
        except BaseException:
            shutil.rmtree(pending, ignore_errors=True)
            raise
        fsync_dir(final.parent)


def _write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, _ENTRY_FILE_MODE)
    with os.fdopen(descriptor, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
