# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify a result bundle read from a sandbox and place it in the output directory.

A bundle is ``manifest.json``, its signature ``manifest.sig`` and the files the manifest lists. :func:`verify_bundle`
proves that every byte came from the runner of *this* sandbox for *this* measurement; :func:`materialize` then moves
the files into the output directory, checking each byte a second time as it is copied (the staging directory is not
trusted between the two calls) and landing either all files or none.

Every refusal is a :class:`ResultRejected` whose ``reason`` names the check that failed, so a test and an operator can
tell a tampered byte from a wrong sandbox.
"""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from hyperloom.measure import fsafe
from hyperloom.measure.docs import DocumentError, parse_json_object
from hyperloom.measure.manifest import RESERVED_NAMES, RESULT_SCHEMA, ManifestFile, ResultManifest, manifest_sha256
from hyperloom.measure.signing import SigningError, public_key_fingerprint, verify

MANIFEST_NAME, SIGNATURE_NAME = RESERVED_NAMES
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SIGNATURE_BYTES = 4096
_SEAL = object()


class RejectReason(str, Enum):
    """Which check of 4.7 a bundle failed."""

    STAGING_MISSING = "staging_missing"
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_TOO_LARGE = "manifest_too_large"
    SIGNATURE_MISSING = "signature_missing"
    PUBLIC_KEY_UNUSABLE = "public_key_unusable"
    BAD_SIGNATURE = "bad_signature"
    KEY_FINGERPRINT_MISMATCH = "key_fingerprint_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    MANIFEST_INVALID = "manifest_invalid"
    REQUEST_ID_MISMATCH = "request_id_mismatch"
    SPEC_FINGERPRINT_MISMATCH = "spec_fingerprint_mismatch"
    SPEC_HASH_MISMATCH = "spec_hash_mismatch"
    SANDBOX_MISMATCH = "sandbox_mismatch"
    RULE_PACK_MISMATCH = "rule_pack_mismatch"
    TOO_LARGE = "too_large"
    FILE_MISSING = "file_missing"
    FILE_SIZE_MISMATCH = "file_size_mismatch"
    FILE_HASH_MISMATCH = "file_hash_mismatch"
    UNLISTED_FILE = "unlisted_file"
    UNSAFE_PATH = "unsafe_path"
    RECORD_MISMATCH = "record_mismatch"
    NOT_VERIFIED = "not_verified"


class ResultRejected(Exception):
    """The bundle is not an authentic result of the requested measurement."""

    def __init__(self, reason: RejectReason, detail: str = "") -> None:
        super().__init__(f"result rejected ({reason.value}){': ' + detail if detail else ''}")
        self.reason = reason
        self.detail = detail


class MaterializeError(Exception):
    """The verified files could not be placed without overwriting something; nothing was left behind."""


@dataclass(frozen=True)
class VerifiedBundle:
    """A bundle that passed :func:`verify_bundle`. Only that function can make one that :func:`materialize` accepts."""

    manifest: Mapping[str, Any]
    result: ResultManifest
    manifest_sha256: str
    signing_key_fingerprint: str
    staging_dir: Path
    _seal: object = field(default=None, repr=False, compare=False)

    def read_file(self, relative: str) -> bytes:
        """The bytes of one listed file, read through the manifest's size and hash: tamper after verification is caught."""
        expected = {entry.path: entry for entry in self.result.files}.get(relative)
        if expected is None:
            raise ResultRejected(RejectReason.UNLISTED_FILE, relative)
        root_fd = _open_staging(self.staging_dir)
        try:
            data = _read_checked(root_fd, relative, expected.size, expected.sha256)
        finally:
            os.close(root_fd)
        return data


@dataclass(frozen=True)
class BundleExpectation:
    """What the caller knows about a measurement before it reads the bundle that claims to answer it.

    ``public_key_pem`` must come from the sandbox information the platform reported, never from the bundle.
    ``sandbox_id`` is the sandbox the bundle was read from: a bundle signed by the right key but made in another
    sandbox is a replay and is refused. ``spec_sha256`` is the hash of the exact bytes written to ``/in/spec.json``;
    ``None`` skips that comparison and is for a caller that only holds the fingerprint (a record book entry).
    ``rule_pack_digest`` is the digest of the rule pack the session reads: the runner must state that it read a pack
    with that digest, or what it measured followed facts the session did not record; ``None`` skips the comparison.
    """

    public_key_pem: str
    request_id: str
    spec_fingerprint: str
    spec_sha256: str | None
    sandbox_id: str
    rule_pack_digest: str | None = None


@dataclass(frozen=True)
class RecordedBundle:
    """The two hashes a record book entry holds about the bundle it summarises."""

    manifest_sha256: str
    signing_key_fingerprint: str


def verify_bundle(
    staging_dir: Path, expected: BundleExpectation, *, max_total_bytes: int | None = None
) -> VerifiedBundle:
    """Run the five checks of 4.7 against ``expected``; any failure raises :class:`ResultRejected`."""
    root_fd = _open_staging(Path(staging_dir))
    try:
        manifest_raw = _read_small(root_fd, MANIFEST_NAME, MAX_MANIFEST_BYTES, RejectReason.MANIFEST_MISSING)
        signature = _read_small(root_fd, SIGNATURE_NAME, MAX_SIGNATURE_BYTES, RejectReason.SIGNATURE_MISSING)
        fingerprint = _check_signature(manifest_raw, signature, expected.public_key_pem)
        manifest, document = _parse_manifest(manifest_raw)
        _check_identity(manifest, fingerprint, expected)
        if max_total_bytes is not None and manifest.total_size() > max_total_bytes:
            raise ResultRejected(RejectReason.TOO_LARGE, f"{manifest.total_size()} > {max_total_bytes}")
        _check_files(root_fd, manifest)
    finally:
        os.close(root_fd)
    return VerifiedBundle(
        manifest=document,
        result=manifest,
        manifest_sha256=manifest_sha256(manifest_raw),
        signing_key_fingerprint=fingerprint,
        staging_dir=Path(staging_dir),
        _seal=_SEAL,
    )


def reverify_record(bundle_dir: Path, expected: BundleExpectation, recorded: RecordedBundle) -> VerifiedBundle:
    """Check a record book entry against its archived bundle before a decision rests on it (L26).

    The record book can be edited by the processes that share its container, so an entry is trusted only after the
    archived bundle verifies for the same request, spec and sandbox *and* the bundle is the one the entry names.
    """
    bundle = verify_bundle(bundle_dir, expected)
    if bundle.manifest_sha256 != recorded.manifest_sha256:
        raise ResultRejected(RejectReason.RECORD_MISMATCH, "the manifest hash is not the recorded one")
    if bundle.signing_key_fingerprint != recorded.signing_key_fingerprint:
        raise ResultRejected(RejectReason.RECORD_MISMATCH, "the signing key fingerprint is not the recorded one")
    return bundle


def _open_staging(staging_dir: Path) -> int:
    try:
        return fsafe.open_root(staging_dir)
    except fsafe.MissingEntry:
        raise ResultRejected(RejectReason.STAGING_MISSING, "the staging directory does not exist") from None
    except fsafe.UnsafeEntry as exc:
        raise ResultRejected(RejectReason.UNSAFE_PATH, str(exc)) from None


def _read_small(root_fd: int, name: str, limit: int, missing: RejectReason) -> bytes:
    try:
        fd = fsafe.open_beneath(root_fd, name)
    except fsafe.MissingEntry:
        raise ResultRejected(missing, name) from None
    except fsafe.UnsafeEntry as exc:
        raise ResultRejected(RejectReason.UNSAFE_PATH, str(exc)) from None
    try:
        data = os.read(fd, limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ResultRejected(RejectReason.MANIFEST_TOO_LARGE, name)
    return data


def _check_signature(manifest_raw: bytes, signature: bytes, public_key_pem: str) -> str:
    try:
        text = signature.decode("ascii")
    except UnicodeDecodeError:
        raise ResultRejected(RejectReason.BAD_SIGNATURE, "the signature is not base64 text") from None
    try:
        valid = verify(manifest_raw, text, public_key_pem)
        fingerprint = public_key_fingerprint(public_key_pem)
    except SigningError as exc:
        raise ResultRejected(RejectReason.PUBLIC_KEY_UNUSABLE, str(exc)) from None
    if not valid:
        raise ResultRejected(RejectReason.BAD_SIGNATURE, "the signature does not match the manifest bytes")
    return fingerprint


def _parse_manifest(manifest_raw: bytes) -> tuple[ResultManifest, dict[str, Any]]:
    try:
        document = parse_json_object(manifest_raw, what="manifest")
        if document.get("schema") != RESULT_SCHEMA:
            raise ResultRejected(RejectReason.SCHEMA_MISMATCH, f"expected {RESULT_SCHEMA}")
        return ResultManifest.from_document(document), document
    except DocumentError as exc:
        raise ResultRejected(RejectReason.MANIFEST_INVALID, str(exc)) from None


def _check_identity(manifest: ResultManifest, fingerprint: str, expected: BundleExpectation) -> None:
    if manifest.runner.signing_key_fingerprint != fingerprint:
        raise ResultRejected(RejectReason.KEY_FINGERPRINT_MISMATCH, "the manifest names another signing key")
    if manifest.request_id != expected.request_id:
        raise ResultRejected(RejectReason.REQUEST_ID_MISMATCH, "the manifest is for another request")
    if manifest.spec_fingerprint != expected.spec_fingerprint:
        raise ResultRejected(RejectReason.SPEC_FINGERPRINT_MISMATCH, "the manifest measured another spec")
    if expected.spec_sha256 is not None and manifest.spec_sha256 != expected.spec_sha256:
        raise ResultRejected(RejectReason.SPEC_HASH_MISMATCH, "the manifest measured other spec bytes")
    if manifest.runner.sandbox_id != expected.sandbox_id:
        raise ResultRejected(RejectReason.SANDBOX_MISMATCH, "the manifest was made in another sandbox")
    if expected.rule_pack_digest is not None:
        _check_rule_pack(manifest, expected.rule_pack_digest)


def _check_rule_pack(manifest: ResultManifest, session_digest: str) -> None:
    stated = manifest.runner.rule_pack
    if stated is None:
        raise ResultRejected(RejectReason.RULE_PACK_MISMATCH, "the runner states no rule pack")
    if stated.digest != session_digest:
        raise ResultRejected(
            RejectReason.RULE_PACK_MISMATCH,
            f"the runner read rule pack {stated.version} ({stated.digest}), the session reads {session_digest}",
        )


def _read_checked(root_fd: int, relative: str, size: int, sha256: str) -> bytes:
    """Read a listed file fully and compare it with the manifest; used for the small ``read_file`` path."""
    chunks: list[bytes] = []
    fd = _open_listed(root_fd, relative)
    try:
        for chunk in fsafe.stream(fd):
            chunks.append(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    _compare(relative, len(data), hashlib.sha256(data).hexdigest(), size, sha256)
    return data


def _compare(relative: str, actual_size: int, actual_sha: str, size: int, sha256: str) -> None:
    if actual_size != size:
        raise ResultRejected(RejectReason.FILE_SIZE_MISMATCH, relative)
    if actual_sha != sha256:
        raise ResultRejected(RejectReason.FILE_HASH_MISMATCH, relative)


def _open_listed(root_fd: int, relative: str) -> int:
    try:
        return fsafe.open_beneath(root_fd, relative)
    except fsafe.MissingEntry:
        raise ResultRejected(RejectReason.FILE_MISSING, relative) from None
    except fsafe.UnsafeEntry as exc:
        raise ResultRejected(RejectReason.UNSAFE_PATH, str(exc)) from None


def _check_files(root_fd: int, manifest: ResultManifest) -> None:
    try:
        present = set(fsafe.walk_regular_files(root_fd))
    except fsafe.UnsafeEntry as exc:
        raise ResultRejected(RejectReason.UNSAFE_PATH, str(exc)) from None
    listed = {entry.path for entry in manifest.files}
    extra = sorted(present - listed - set(RESERVED_NAMES))
    if extra:
        raise ResultRejected(RejectReason.UNLISTED_FILE, extra[0])
    for entry in manifest.files:
        fd = _open_listed(root_fd, entry.path)
        try:
            sha256, size = fsafe.sha256_and_size(fd)
        finally:
            os.close(fd)
        _compare(entry.path, size, sha256, entry.size, entry.sha256)


def materialize(bundle: VerifiedBundle, output_dir: Path) -> None:
    """Place the verified files below ``output_dir``; never overwrite, and land all of them or none.

    Each file is re-read from the staging directory through the manifest's size and hash while it is copied to a
    hidden directory inside ``output_dir``; only when every file has passed are they linked into place. A name that
    already exists, a symlink anywhere on a destination path, or a byte that changed since :func:`verify_bundle`
    leaves ``output_dir`` as it was. Every destination is reached through directory descriptors opened without
    following symlinks, so a link planted in ``output_dir`` cannot redirect a write outside it.
    """
    if bundle._seal is not _SEAL:
        raise ResultRejected(RejectReason.NOT_VERIFIED, "the bundle was not produced by verify_bundle")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    out_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    scratch_name = f".hl-materialize-{uuid.uuid4().hex}"
    try:
        _refuse_existing(out_fd, bundle.result.files)
        os.mkdir(scratch_name, 0o700, dir_fd=out_fd)
        try:
            scratch_fd = os.open(
                scratch_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=out_fd
            )
            try:
                _stage_files(bundle, scratch_fd)
                _place_all(bundle.result.files, scratch_fd, out_fd)
            finally:
                os.close(scratch_fd)
        finally:
            shutil.rmtree(output / scratch_name, ignore_errors=True)
    finally:
        os.close(out_fd)


def _refuse_existing(out_fd: int, files: tuple[ManifestFile, ...]) -> None:
    for entry in files:
        try:
            taken = fsafe.exists_beneath(out_fd, entry.path)
        except fsafe.UnsafeEntry:
            raise MaterializeError(f"{entry.path} lies behind a symlink in the output directory") from None
        if taken:
            raise MaterializeError(f"{entry.path} already exists in the output directory")


def _stage_files(bundle: VerifiedBundle, scratch_fd: int) -> None:
    root_fd = _open_staging(bundle.staging_dir)
    try:
        for entry in bundle.result.files:
            *parents, name = entry.path.split("/")
            parent = fsafe.open_dir_beneath(scratch_fd, parents, created=[])
            try:
                _copy_checked(root_fd, entry, parent, name)
            finally:
                os.close(parent)
    finally:
        os.close(root_fd)


def _copy_checked(root_fd: int, entry: ManifestFile, parent: int, name: str) -> None:
    digest, total = hashlib.sha256(), 0
    source = _open_listed(root_fd, entry.path)
    try:
        target = fsafe.create_exclusive(parent, name)
        try:
            for chunk in fsafe.stream(source):
                digest.update(chunk)
                total += len(chunk)
                os.write(target, chunk)
        finally:
            os.close(target)
    finally:
        os.close(source)
    _compare(entry.path, total, digest.hexdigest(), entry.size, entry.sha256)


def _place_all(files: tuple[ManifestFile, ...], scratch_fd: int, out_fd: int) -> None:
    placed: list[str] = []
    created: list[str] = []
    try:
        for entry in files:
            _place(entry.path, scratch_fd, out_fd, created)
            placed.append(entry.path)
    except BaseException:
        _roll_back(out_fd, placed, created)
        raise


def _place(rel: str, scratch_fd: int, out_fd: int, created: list[str]) -> None:
    *parents, name = rel.split("/")
    source_parent = fsafe.open_dir_beneath(scratch_fd, parents)
    try:
        try:
            destination = fsafe.open_dir_beneath(out_fd, parents, created=created)
        except fsafe.UnsafeEntry:
            raise MaterializeError(f"{rel} lies behind a symlink in the output directory") from None
        try:
            _link_or_copy(source_parent, destination, name)
        finally:
            os.close(destination)
    finally:
        os.close(source_parent)


def _link_or_copy(source_parent: int, destination: int, name: str) -> None:
    try:
        os.link(name, name, src_dir_fd=source_parent, dst_dir_fd=destination, follow_symlinks=False)
        return
    except FileExistsError:
        raise MaterializeError(f"{name} appeared in the output directory during placement") from None
    except OSError as exc:
        if exc.errno not in (errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK):
            raise
    _copy_exclusive(source_parent, destination, name)


def _copy_exclusive(source_parent: int, destination: int, name: str) -> None:
    source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=source_parent)
    try:
        try:
            target = fsafe.create_exclusive(destination, name)
        except FileExistsError:
            raise MaterializeError(f"{name} appeared in the output directory during placement") from None
        try:
            for chunk in fsafe.stream(source):
                os.write(target, chunk)
        except BaseException:
            os.unlink(name, dir_fd=destination)
            raise
        finally:
            os.close(target)
    finally:
        os.close(source)


def _roll_back(out_fd: int, placed: list[str], created: list[str]) -> None:
    """Undo a partial placement; each step is best effort so one failure cannot hide the original error."""
    _remove_each(out_fd, reversed(placed), os.unlink)
    _remove_each(out_fd, reversed(created), os.rmdir)


def _remove_each(out_fd: int, rels: Iterable[str], remove: Callable[..., None]) -> None:
    for rel in rels:
        *parents, name = rel.split("/")
        try:
            parent = fsafe.open_dir_beneath(out_fd, parents)
        except (fsafe.MissingEntry, fsafe.UnsafeEntry):
            continue
        try:
            remove(name, dir_fd=parent)
        except OSError:
            pass
        finally:
            os.close(parent)
