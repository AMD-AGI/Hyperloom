# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import io
import os
import tarfile

import pytest

from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest
from hyperloom.measure.attachments import place_attachments
from hyperloom.measure.spec import Attachment
from hyperloom_multibranch.rulegate.fix_case import (
    CASE_SCRIPT,
    FIX_CASE_ID,
    CaseError,
    FixCase,
    case_of,
    pack_case,
    read_case,
)

SCRIPT = '#!/bin/bash\ncurl -s "$HL_BASE_URL/v1/models" | grep -q "$HL_MODEL" || exit 1\nexit 0\n'
DIAGNOSIS = "fp8 scales overflow at 8k context; clamp them before the cast"
OTHER = AttachmentRef("prompts.json", "file", "a" * 64)


def _request(*attachments: AttachmentRef) -> TestRequest:
    return TestRequest(TestKind.TEST, Purpose.EXPLORE, "s1", "c1", attachments=attachments)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_equal_inputs_pack_to_equal_bytes_and_read_back():
    data = pack_case(SCRIPT, DIAGNOSIS)

    assert pack_case(SCRIPT, DIAGNOSIS) == data
    assert _sha256(pack_case(SCRIPT, DIAGNOSIS + " ")) != _sha256(data)
    assert read_case(data) == FixCase(SCRIPT, DIAGNOSIS)


def test_the_script_is_an_executable_at_the_top_owned_by_no_one():
    with tarfile.open(fileobj=io.BytesIO(pack_case(SCRIPT, DIAGNOSIS))) as archive:
        members = {member.name: member for member in archive.getmembers()}

    assert set(members) == {CASE_SCRIPT, "DIAGNOSIS.md"}
    script = members[CASE_SCRIPT]
    assert (script.mode, script.mtime, script.uid, script.gid, script.uname) == (0o755, 0, 0, 0, "")


def _place(attach_dir) -> dict:
    """Unpack the case as the toolbox unpacks a directory attachment."""
    data = pack_case(SCRIPT, DIAGNOSIS)
    blob = f"sha256:{_sha256(data)}"
    return place_attachments([Attachment(FIX_CASE_ID, "dir", blob, "")], attach_dir, lambda _blob: data)


def test_the_toolbox_unpacks_the_case_with_its_script_where_the_runner_looks(tmp_path):
    placed = _place(tmp_path)

    assert (placed[FIX_CASE_ID] / CASE_SCRIPT).read_text(encoding="utf-8") == SCRIPT


@pytest.mark.skipif(os.name == "nt", reason="the repro runner limits resources the POSIX way")
def test_the_runner_finds_the_case_script(tmp_path):
    from hyperloom.measure import repro

    _place(tmp_path)

    assert repro.CASE_SCRIPT == CASE_SCRIPT
    assert repro.find_case(tmp_path, [FIX_CASE_ID, OTHER.id]) == tmp_path / FIX_CASE_ID / CASE_SCRIPT


@pytest.mark.parametrize(("script", "diagnosis"), [(" \n", DIAGNOSIS), (SCRIPT, "")])
def test_a_case_without_a_script_or_a_diagnosis_is_refused(script, diagnosis):
    with pytest.raises(CaseError, match="needs"):
        pack_case(script, diagnosis)


def test_bytes_that_are_no_case_are_refused():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.addfile(tarfile.TarInfo("notes.txt"), io.BytesIO())

    with pytest.raises(CaseError, match="not a tar"):
        read_case(b"plain text")
    with pytest.raises(CaseError, match=CASE_SCRIPT):
        read_case(buffer.getvalue())


def test_a_request_carries_its_case_under_the_case_id():
    case = AttachmentRef(FIX_CASE_ID, "dir", _sha256(pack_case(SCRIPT, DIAGNOSIS)))

    assert case_of(_request(OTHER, case)) == case
    assert case_of(_request(OTHER)) is None
    assert case_of(_request()) is None


def test_a_file_under_the_case_id_is_no_case():
    with pytest.raises(CaseError, match="directory"):
        case_of(_request(AttachmentRef(FIX_CASE_ID, "file", "b" * 64)))
