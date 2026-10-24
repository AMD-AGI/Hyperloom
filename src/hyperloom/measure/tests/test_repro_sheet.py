# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The instruction sheet of a repro test carries its case, by hash, to the runner."""

from __future__ import annotations

import hashlib
import io
import tarfile
import time
from types import SimpleNamespace
from dataclasses import replace

import pytest

from hyperloom.contracts.measurement import AttachmentRef
from hyperloom.measure import hosted_spec
from hyperloom.measure.hosted_spec import SpecRefused, build_spec, dir_manifest_sha256
from hyperloom.measure.tests.hosted_fixtures import make_call, spec_context
from hyperloom.measure.tests.repro_fixtures import case_attachment


def _build(tmp_path, call, ctx):
    return build_spec(call, request_id="req-1", ctx=ctx, deadline_unix=time.time() + 60)


def test_the_sheet_names_the_requests_attachments_with_their_blob_and_directory_hash(tmp_path):
    ctx = spec_context(tmp_path)
    case = case_attachment(ctx.blobs, "exit 0\n")
    call = replace(make_call(tmp_path), kind="repro", attachments=(case,))

    sheet = _build(tmp_path, call, ctx)

    (attached,) = sheet.attachments
    expected = dir_manifest_sha256([("repro.sh", 7, hashlib.sha256(b"exit 0\n").hexdigest())])
    assert (attached.id, attached.kind, attached.blob) == (case.id, "dir", f"sha256:{case.sha256}")
    assert sheet.kind == "repro" and attached.manifest_sha256 == expected


def test_a_test_that_ships_no_attachment_has_none_on_its_sheet(tmp_path):
    ctx = spec_context(tmp_path)

    assert _build(tmp_path, make_call(tmp_path), ctx).attachments == ()


def test_an_attachment_whose_blob_the_store_does_not_hold_is_refused_not_sent(tmp_path):
    ctx = spec_context(tmp_path)
    call = replace(make_call(tmp_path), kind="repro", attachments=(AttachmentRef("case-1", "dir", "e" * 64),))

    with pytest.raises(SpecRefused) as refused:
        _build(tmp_path, call, ctx)

    assert refused.value.reason == "attachment_missing" and "case-1" in refused.value.detail


def test_an_attachment_with_the_id_of_an_environment_attachment_is_refused(tmp_path):
    ctx = spec_context(tmp_path)
    taken = case_attachment(ctx.blobs, "exit 0\n", attachment_id="k-1")
    other = case_attachment(ctx.blobs, "exit 1\n", attachment_id="k-1", name="other.sh")
    environment = SimpleNamespace(attachments=(taken,))
    call = replace(make_call(tmp_path), kind="repro", attachments=(other,), environment=environment)

    with pytest.raises(SpecRefused) as refused:
        _build(tmp_path, call, ctx)

    assert refused.value.reason == "attachment_id_taken"


def _refused(tmp_path, ctx, ref) -> SpecRefused:
    call = replace(make_call(tmp_path), kind="repro", attachments=(ref,))
    with pytest.raises(SpecRefused) as refused:
        _build(tmp_path, call, ctx)
    return refused.value


def test_a_directory_attachment_that_is_not_a_tar_archive_is_refused(tmp_path):
    ctx = spec_context(tmp_path)
    ref = AttachmentRef("case-1", "dir", ctx.blobs.put(b"this is not a tar archive" * 100))

    assert _refused(tmp_path, ctx, ref).reason == "attachment_unreadable"


def test_a_directory_attachment_past_the_size_cap_is_refused_before_it_is_read_into_the_sheet(tmp_path, monkeypatch):
    ctx = spec_context(tmp_path)
    ref = case_attachment(ctx.blobs, "x")  # one byte of file, a tar record of 10 KiB as a blob
    monkeypatch.setattr(hosted_spec, "MAX_ATTACHMENT_BYTES", 2048)

    assert _refused(tmp_path, ctx, ref).reason == "attachment_too_large"


def test_a_directory_attachment_whose_files_expand_past_the_cap_is_refused(tmp_path, monkeypatch):
    ctx = spec_context(tmp_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:  # small as a blob, large once expanded
        data = b"\0" * 100_000
        info = tarfile.TarInfo("big")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    ref = AttachmentRef("case-1", "dir", ctx.blobs.put(buffer.getvalue()))
    monkeypatch.setattr(hosted_spec, "MAX_ATTACHMENT_BYTES", 50_000)

    assert len(buffer.getvalue()) < 50_000
    assert _refused(tmp_path, ctx, ref).reason == "attachment_too_large"


def test_a_file_attachment_past_the_size_cap_is_refused_too(tmp_path, monkeypatch):
    ctx = spec_context(tmp_path)
    ref = AttachmentRef("data", "file", ctx.blobs.put(b"x" * 3000))
    monkeypatch.setattr(hosted_spec, "MAX_ATTACHMENT_BYTES", 2048)

    assert _refused(tmp_path, ctx, ref).reason == "attachment_too_large"


def test_attachments_that_are_small_one_by_one_but_pass_the_cap_together_are_refused(tmp_path, monkeypatch):
    ctx = spec_context(tmp_path)
    refs = tuple(AttachmentRef(f"data-{i}", "file", ctx.blobs.put(b"x" * (1000 + i))) for i in range(3))
    monkeypatch.setattr(hosted_spec, "MAX_ATTACHMENT_BYTES", 2500)
    call = replace(make_call(tmp_path), kind="repro", attachments=refs)

    with pytest.raises(SpecRefused) as refused:
        _build(tmp_path, call, ctx)

    assert refused.value.reason == "attachment_too_large" and "data-2" in refused.value.detail
