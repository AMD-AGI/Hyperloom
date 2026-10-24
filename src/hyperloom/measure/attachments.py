# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unpack the sheet's attachments into the working directory and name them in the YAML (design 3.3, 4.8.3 step 4).

A file attachment is written to ``attach/<id>``; a directory attachment is a tar archive unpacked into ``attach/<id>/``.
The YAML refers to either as ``${HL_ATTACH:<id>}``.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

from hyperloom.measure.spec import Attachment

__all__ = ["AttachmentError", "place_attachments", "safe_members", "substitute"]

_PLACEHOLDER = re.compile(r"\$\{HL_ATTACH:([A-Za-z0-9][A-Za-z0-9._-]{0,63})\}")
_MAX_MEMBERS = 100_000


class AttachmentError(Exception):
    """An attachment is missing, does not match its address, or would write outside its directory."""


def _safe_members(archive: tarfile.TarFile, attachment_id: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if len(members) > _MAX_MEMBERS:
        raise AttachmentError(f"attachment {attachment_id}: more than {_MAX_MEMBERS} entries")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isreg() or member.isdir()):
            raise AttachmentError(f"attachment {attachment_id}: refused entry {member.name!r}")
    return members


safe_members = _safe_members  # public name for callers that vet an archive before it is stored


def _unpack_dir(data: bytes, destination: Path, attachment_id: str) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = _safe_members(archive, attachment_id)
            archive.extractall(destination, members=members, filter="data")
    except tarfile.TarError as exc:
        raise AttachmentError(f"attachment {attachment_id}: not a tar archive ({exc})") from exc


def place_attachments(
    attachments: Iterable[Attachment], attach_dir: Path, read_blob: Callable[[str], bytes]
) -> dict[str, Path]:
    """Write every attachment under ``attach_dir`` and return the path each id now has."""
    placed: dict[str, Path] = {}
    for attachment in attachments:
        data = read_blob(attachment.blob)
        if f"sha256:{hashlib.sha256(data).hexdigest()}" != attachment.blob:
            raise AttachmentError(f"attachment {attachment.id}: content does not hash to {attachment.blob}")
        target = attach_dir / attachment.id
        if attachment.kind == "dir":
            target.mkdir(parents=True)
            _unpack_dir(data, target, attachment.id)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        placed[attachment.id] = target
    return placed


def substitute(text: str, placed: dict[str, Path]) -> str:
    """Replace each ``${HL_ATTACH:<id>}`` with its path; a placeholder with no attachment is an error."""

    def path_of(match: re.Match[str]) -> str:
        try:
            return str(placed[match.group(1)])
        except KeyError:
            raise AttachmentError(
                f"the YAML refers to attachment {match.group(1)!r}, which the sheet does not carry"
            ) from None

    return _PLACEHOLDER.sub(path_of, text)
