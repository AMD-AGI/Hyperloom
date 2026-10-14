# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from hyperloom.measure.attachments import AttachmentError, place_attachments, substitute
from hyperloom.measure.spec import Attachment


def _address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _tar(members: dict[str, bytes], extra=()) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info in extra:
            archive.addfile(info)
    return buffer.getvalue()


def _attachment(id_: str, kind: str, data: bytes) -> Attachment:
    return Attachment(id=id_, kind=kind, blob=_address(data), manifest_sha256="0" * 64)


def test_a_file_and_a_directory_are_placed_and_substituted(tmp_path):
    blobs = {}
    file_data, dir_data = b"csv", _tar({"k/a.py": b"print(1)"})
    for data in (file_data, dir_data):
        blobs[_address(data)] = data

    placed = place_attachments(
        [_attachment("cfg", "file", file_data), _attachment("ovl", "dir", dir_data)], tmp_path, blobs.__getitem__
    )

    assert (tmp_path / "cfg").read_bytes() == b"csv"
    assert (tmp_path / "ovl" / "k" / "a.py").read_text() == "print(1)"
    assert substitute("x: ${HL_ATTACH:cfg}\ny: ${HL_ATTACH:ovl}", placed) == f"x: {tmp_path}/cfg\ny: {tmp_path}/ovl"


def test_content_that_does_not_hash_to_its_address_is_refused(tmp_path):
    attachment = _attachment("cfg", "file", b"expected")

    with pytest.raises(AttachmentError, match="does not hash"):
        place_attachments([attachment], tmp_path, lambda address: b"tampered")


@pytest.mark.parametrize("name", ["../escape", "/abs"])
def test_a_tar_member_outside_its_directory_is_refused(tmp_path, name):
    data = _tar({name: b"x"})

    with pytest.raises(AttachmentError, match="refused entry"):
        place_attachments([_attachment("ovl", "dir", data)], tmp_path, lambda address: data)
    assert not (tmp_path.parent / "escape").exists()


def test_a_tar_symlink_is_refused(tmp_path):
    link = tarfile.TarInfo("l")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    data = _tar({}, extra=[link])

    with pytest.raises(AttachmentError, match="refused entry 'l'"):
        place_attachments([_attachment("ovl", "dir", data)], tmp_path, lambda address: data)


def test_a_placeholder_without_an_attachment_is_refused():
    with pytest.raises(AttachmentError, match="'gone'"):
        substitute("${HL_ATTACH:gone}", {})
