# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a reuse reads a cached wheel it compares with the one it builds again: as pip installs it on any Python.

pip installs each member of a wheel under the name its interpreter's :mod:`zipfile` reads, and those differ: from Python
3.12 a member's Info-ZIP Unicode Path extra field names it, before that its raw name does, and patch releases of one
Python find zip64 records differently. So the comparison takes each member's raw name, content and executable bit, and
refuses a wheel that another Python could read other names or other members from. Each zip below is made by hand
(:func:`~hyperloom.measure.runner.tests.bootstrap_rig.rewrite_wheel` and byte edits), so the cases are the same on every
Python, and each is one this Python's :mod:`zipfile` opens and lists.
"""

from __future__ import annotations

import stat
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.measure.runner.cache_build import _wheel_files
from hyperloom.measure.runner.tests.bootstrap_rig import build_wheel, rewrite_wheel, unicode_path_name

_MODULE = "pkg.py"
_END = struct.Struct("<4s4H2LH")
_NOT_ONE_END = "it does not end with its central directory and one end record, with no comment and no zip64 records"


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    """A wheel of the module ``pkg`` as a builder writes one."""
    return build_wheel(tmp_path, "pkg", "1.0", "ORIGIN = 'built'\n")[0]


def _edit_module(change: Callable[[zipfile.ZipInfo], None]) -> Callable[[zipfile.ZipInfo], zipfile.ZipInfo]:
    def edit(member: zipfile.ZipInfo) -> zipfile.ZipInfo:
        if member.filename == _MODULE:
            change(member)
        return member

    return edit


def _central_directory(data: bytes) -> tuple[int, int]:
    """The size and offset of the central directory that the end record of ``data`` records."""
    *_, size, offset, _ = _END.unpack_from(data, len(data) - _END.size)
    return size, offset


def _unicode_path(wheel: Path, out: Path) -> bytes:
    return rewrite_wheel(wheel, out, unicode_path_name(_MODULE, "sitecustomize.py")).read_bytes()


def _unicode_path_after_another_field(wheel: Path, out: Path) -> bytes:
    """The same field behind an extended timestamp field (0x5455), as Info-ZIP writes them: zipfile reads every field of
    a member, so the check must too."""
    rename = unicode_path_name(_MODULE, "sitecustomize.py")

    def edit(member: zipfile.ZipInfo) -> zipfile.ZipInfo:
        renamed = rename(member)
        if renamed is not member:
            renamed.extra = struct.pack("<HHBL", 0x5455, 5, 1, 981173106) + renamed.extra
        return renamed

    return rewrite_wheel(wheel, out, edit).read_bytes()


def _zip64_field(wheel: Path, out: Path) -> bytes:
    """An empty zip64 extra field on the module: zipfile reads it only for a size or offset of all ones."""
    edit = _edit_module(lambda member: setattr(member, "extra", b"\x01\x00\x00\x00"))
    return rewrite_wheel(wheel, out, edit).read_bytes()


def _name_cut_at_a_nul(wheel: Path, out: Path) -> bytes:
    """The module stored as ``pkg.py\\0x``, which zipfile reads as ``pkg.py``."""
    data = rewrite_wheel(wheel, out, _edit_module(lambda member: setattr(member, "filename", "pkg.pyQx"))).read_bytes()
    return data.replace(b"pkg.pyQx", b"pkg.py\x00x")


def _prepended(wheel: Path, out: Path) -> bytes:
    """Bytes before the first member: zipfile shifts every offset by them."""
    return b"\x00" * 16 + wheel.read_bytes()


def _record_after_the_end(wheel: Path, out: Path) -> bytes:
    """22 bytes after the end record, which agree with it on where the central directory lies but are not one: zipfile
    looks back for the end record and finds the real one."""
    data = wheel.read_bytes()
    size, offset = _central_directory(data)
    return data + _END.pack(b"PK\x05\x05", 0, 0, 0, 0, size + _END.size, offset, 0)


def _comment_claimed(wheel: Path, out: Path) -> bytes:
    """The end record claims a comment of one byte that is not there: zipfile looks back for it and finds that one."""
    return wheel.read_bytes()[:-2] + b"\x01\x00"


def _zip64_locator(wheel: Path, out: Path) -> bytes:
    """A zip64 end record and its locator as the comment of the last member's central directory entry, pointing at the
    same central directory: zipfile follows them, and where a zip64 record lies is what Python patch releases read
    differently. The plain end record still ends the file right after the central directory."""
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(out, "w") as target:
        infos = source.infolist()
        for info in infos:
            info.comment = b"\x00" * 76 if info is infos[-1] else b""
            target.writestr(info, source.read(info))
    data = bytearray(out.read_bytes())
    end, (_, offset) = len(data) - _END.size, _central_directory(bytes(data))
    count, record = len(infos), end - 76
    zip64 = struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, record - offset, offset)
    data[record:end] = zip64 + struct.pack("<4sLQL", b"PK\x06\x07", 0, record, 1)
    return bytes(data)


@pytest.mark.parametrize(
    ("craft", "refusal"),
    [
        (_unicode_path, "its member 'sitecustomize.py' has a Unicode Path extra field"),
        (_unicode_path_after_another_field, "its member 'sitecustomize.py' has a Unicode Path extra field"),
        (_zip64_field, f"its member '{_MODULE}' has a zip64 extra field"),
        (_name_cut_at_a_nul, f"its member '{_MODULE}\\x00x' is read as '{_MODULE}'"),
        (_prepended, _NOT_ONE_END),
        (_record_after_the_end, _NOT_ONE_END),
        (_comment_claimed, _NOT_ONE_END),
        (_zip64_locator, _NOT_ONE_END),
    ],
    ids=[
        "unicode-path-field",
        "unicode-path-field-after-another-field",
        "zip64-field",
        "name-cut-at-a-nul",
        "bytes-before-the-first-member",
        "a-record-after-the-end-record",
        "a-comment-claimed",
        "a-zip64-locator",
    ],
)
def test_a_wheel_another_python_could_read_other_names_or_members_from_is_refused(
    wheel: Path, tmp_path: Path, craft: Callable[[Path, Path], bytes], refusal: str
) -> None:
    crafted = tmp_path / "crafted.whl"
    crafted.write_bytes(craft(wheel, tmp_path / "rewritten.whl"))
    with zipfile.ZipFile(wheel) as built, zipfile.ZipFile(crafted) as archive:
        assert sorted(archive.read(info) for info in archive.infolist()) == sorted(map(built.read, built.namelist()))
    with pytest.raises(zipfile.BadZipFile) as refused:
        _wheel_files(crafted)
    assert str(refused.value).startswith(refusal)


def test_a_wheel_is_read_by_raw_name_and_content_not_by_its_layout(wheel: Path, tmp_path: Path) -> None:
    files = _wheel_files(wheel)
    with zipfile.ZipFile(wheel) as archive:
        assert [name for name, *_ in files] == sorted(archive.namelist())
    # Written again with other timestamps and the members in another order: the same.
    assert _wheel_files(rewrite_wheel(wheel, tmp_path / "again.whl")) == files


@pytest.mark.parametrize("mode", [0o755, 0o654, 0o645], ids=["owner-group-other", "group-only", "other-only"])
def test_a_member_with_any_execute_bit_is_read_as_executable(wheel: Path, tmp_path: Path, mode: int) -> None:
    """pip makes a member executable when any of its three execute bits is set, not only the owner's."""
    files = _wheel_files(wheel)

    def executable(member: zipfile.ZipInfo) -> None:
        member.external_attr = (stat.S_IFREG | mode) << 16

    made_executable = _wheel_files(rewrite_wheel(wheel, tmp_path / "x.whl", _edit_module(executable)))
    assert {name for name, *_ in set(made_executable) ^ set(files)} == {_MODULE}
    assert [executable for name, _, executable in made_executable if name == _MODULE] == [True]
