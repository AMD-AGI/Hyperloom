# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Vet attachment bytes with the checks the repro path applies, then keep them in the session's blob store."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zlib
from typing import Any, Literal

from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.measure.attachments import AttachmentError, safe_members
from hyperloom.measure.spec import ATTACHMENT_ID_RE
from hyperloom.measure.hosted_spec import MAX_ATTACHMENT_BYTES, SpecRefused, dir_entries
from hyperloom_multibranch.contracts.toolbox import RequestRejected


def store_attachment(blobs: Any, attachment_id: str, kind: Literal["file", "dir"], data: bytes) -> AttachmentRef:
    """Check ``data`` and put it in ``blobs``; refused data is not stored. The same bytes give the same reference."""
    if kind not in ("file", "dir"):
        raise RequestRejected(f"an attachment kind must be 'file' or 'dir', got {kind!r}")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise RequestRejected(
            f"attachment {attachment_id} is {len(data)} bytes, the cap is {MAX_ATTACHMENT_BYTES}",
            "attachment_too_large",
        )
    if not ATTACHMENT_ID_RE.fullmatch(attachment_id):
        raise RequestRejected(f"attachment id {attachment_id!r} must match {ATTACHMENT_ID_RE.pattern}")
    ref = _ref(attachment_id, kind, hashlib.sha256(data).hexdigest())
    if kind == "dir":
        try:
            _vet_members(ref, data)
            dir_entries(ref, data)
        except SpecRefused as exc:
            raise RequestRejected(exc.detail, exc.reason) from None
        except (EOFError, zlib.error, OSError) as exc:  # a truncated or corrupt compressed archive
            raise RequestRejected(
                f"attachment {attachment_id}: {type(exc).__name__}", "attachment_unreadable"
            ) from None
        except tarfile.TarError as exc:
            raise RequestRejected(
                f"attachment {attachment_id}: not an uncompressed tar ({type(exc).__name__})", "attachment_unreadable"
            ) from None
        except AttachmentError as exc:
            raise RequestRejected(str(exc), "attachment_unsafe") from None
    if blobs.put(data) != ref.sha256:
        raise RuntimeError("the blob store returned an address that is not the sha256 of the data")
    return ref


def _ref(attachment_id: str, kind: Literal["file", "dir"], sha256: str) -> AttachmentRef:
    try:
        return AttachmentRef(attachment_id, kind, sha256)
    except ValueError as exc:  # MeasurementError: an empty or malformed id
        raise RequestRejected(str(exc)) from None


def _vet_members(ref: AttachmentRef, data: bytes) -> None:
    """The entry rules the sandbox applies when it unpacks the archive: no escaping path, link or device.

    Only an uncompressed tar is taken, so what is read here is bounded by the size cap.
    """
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        safe_members(archive, ref.id)
