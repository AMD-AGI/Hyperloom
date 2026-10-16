# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from hyperloom.measure.tests.bundle_factory import (
    DEFAULT_FILES,
    REQUEST_ID,
    SANDBOX_ID,
    SPEC_FINGERPRINT,
    flip_one_byte,
    make_bundle,
    verify_args,
)

from hyperloom.measure import bundle as bundle_mod
from hyperloom.measure import signing
from hyperloom.measure.bundle import (
    RecordedBundle,
    MaterializeError,
    RejectReason,
    ResultRejected,
    VerifiedBundle,
    materialize,
    reverify_record,
    verify_bundle,
)

REPORT = "benchmark_sglang_20261005_101500/benchmark_report.json"


def _reason(staging: Path, bundle, max_total_bytes: int | None = None, **overrides) -> RejectReason:
    expected = replace(verify_args(bundle), **overrides)
    with pytest.raises(ResultRejected) as raised:
        verify_bundle(staging, expected, max_total_bytes=max_total_bytes)
    return raised.value.reason


@pytest.fixture
def bundle(tmp_path: Path):
    return make_bundle(tmp_path / "staging")


def test_an_authentic_bundle_verifies(bundle) -> None:
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    assert verified.result.request_id == REQUEST_ID
    assert verified.manifest["runner"]["sandbox_id"] == SANDBOX_ID
    assert verified.signing_key_fingerprint == signing.public_key_fingerprint(bundle.public_pem)
    assert verified.read_file(REPORT) == DEFAULT_FILES[REPORT]


def test_every_byte_of_the_manifest_is_covered_by_the_signature(bundle) -> None:
    raw = (bundle.staging / "manifest.json").read_bytes()
    for offset in range(0, len(raw), max(1, len(raw) // 40)):
        flip_one_byte(bundle.staging / "manifest.json", offset)
        assert _reason(bundle.staging, bundle) is RejectReason.BAD_SIGNATURE, offset
        flip_one_byte(bundle.staging / "manifest.json", offset)
    verify_bundle(bundle.staging, verify_args(bundle))


def test_a_flipped_signature_byte_is_a_bad_signature(bundle) -> None:
    flip_one_byte(bundle.staging / "manifest.sig", 3)
    assert _reason(bundle.staging, bundle) is RejectReason.BAD_SIGNATURE


def test_a_manifest_signed_by_a_key_the_sandbox_does_not_hold_is_rejected(bundle) -> None:
    own = signing.generate_private_key()
    forged = bundle.document | {
        "runner": bundle.document["runner"]
        | {"signing_key_fingerprint": signing.public_key_fingerprint(signing.public_key_pem(own))}
    }
    bundle.write_manifest(forged, key=own)
    assert _reason(bundle.staging, bundle) is RejectReason.BAD_SIGNATURE


def test_a_valid_signature_over_a_manifest_naming_another_key_is_rejected(bundle) -> None:
    other = signing.public_key_fingerprint(signing.public_key_pem(signing.generate_private_key()))
    document = bundle.document | {"runner": bundle.document["runner"] | {"signing_key_fingerprint": other}}
    bundle.write_manifest(document)
    assert _reason(bundle.staging, bundle) is RejectReason.KEY_FINGERPRINT_MISMATCH


def test_a_missing_signature_or_manifest_or_directory_is_named(bundle, tmp_path: Path) -> None:
    (bundle.staging / "manifest.sig").unlink()
    assert _reason(bundle.staging, bundle) is RejectReason.SIGNATURE_MISSING
    (bundle.staging / "manifest.json").unlink()
    assert _reason(bundle.staging, bundle) is RejectReason.MANIFEST_MISSING
    assert _reason(tmp_path / "nowhere", bundle) is RejectReason.STAGING_MISSING


def test_an_unusable_public_key_is_not_reported_as_tampering(bundle) -> None:
    assert _reason(bundle.staging, bundle, public_key_pem="not a key") is RejectReason.PUBLIC_KEY_UNUSABLE


def test_a_signed_manifest_of_another_schema_is_rejected(bundle) -> None:
    bundle.write_manifest(bundle.document | {"schema": "hyperloom.measurement-result/v2"})
    assert _reason(bundle.staging, bundle) is RejectReason.SCHEMA_MISMATCH


def test_a_signed_manifest_that_breaks_the_contract_is_rejected(bundle) -> None:
    bundle.write_manifest(bundle.document | {"surprise": 1})
    assert _reason(bundle.staging, bundle) is RejectReason.MANIFEST_INVALID


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape", "a/../b", "a//b", "manifest.json"])
def test_a_signed_manifest_listing_an_unsafe_path_is_rejected(bundle, path: str) -> None:
    files = [{"path": path, "size": 0, "sha256": "0" * 64}]
    bundle.write_manifest(bundle.document | {"files": files})
    assert _reason(bundle.staging, bundle) is RejectReason.MANIFEST_INVALID


def test_a_bundle_for_another_request_spec_or_sandbox_is_rejected(bundle) -> None:
    assert _reason(bundle.staging, bundle, request_id="other") is RejectReason.REQUEST_ID_MISMATCH
    assert _reason(bundle.staging, bundle, spec_fingerprint="0" * 16) is RejectReason.SPEC_FINGERPRINT_MISMATCH
    assert _reason(bundle.staging, bundle, sandbox_id="sbx_other") is RejectReason.SANDBOX_MISMATCH


def test_a_manifest_that_measured_other_spec_bytes_is_rejected_unless_the_caller_has_only_the_fingerprint(
    bundle,
) -> None:
    assert _reason(bundle.staging, bundle, spec_sha256="ff" * 32) is RejectReason.SPEC_HASH_MISMATCH
    verify_bundle(bundle.staging, replace(verify_args(bundle), spec_sha256=None))


def test_a_result_larger_than_the_allowed_total_is_rejected(bundle) -> None:
    total = sum(len(data) for data in DEFAULT_FILES.values())
    verify_bundle(bundle.staging, verify_args(bundle), max_total_bytes=total)
    assert _reason(bundle.staging, bundle, max_total_bytes=total - 1) is RejectReason.TOO_LARGE


def test_one_flipped_byte_in_a_listed_file_is_a_hash_mismatch(bundle) -> None:
    flip_one_byte(bundle.staging / REPORT, 5)
    assert _reason(bundle.staging, bundle) is RejectReason.FILE_HASH_MISMATCH


def test_a_truncated_or_extended_file_is_a_size_mismatch(bundle) -> None:
    path = bundle.staging / "server.log"
    path.write_bytes(path.read_bytes()[:-1])
    assert _reason(bundle.staging, bundle) is RejectReason.FILE_SIZE_MISMATCH
    path.write_bytes(path.read_bytes() + b"xx")
    assert _reason(bundle.staging, bundle) is RejectReason.FILE_SIZE_MISMATCH


def test_a_listed_file_that_was_not_read_is_missing(bundle) -> None:
    (bundle.staging / "server.log").unlink()
    assert _reason(bundle.staging, bundle) is RejectReason.FILE_MISSING


def test_a_file_the_manifest_does_not_list_is_rejected(bundle) -> None:
    (bundle.staging / "extra.txt").write_text("planted")
    assert _reason(bundle.staging, bundle) is RejectReason.UNLISTED_FILE
    (bundle.staging / "extra.txt").unlink()
    (bundle.staging / "benchmark_sglang_20261005_101500" / "planted.json").write_text("{}")
    assert _reason(bundle.staging, bundle) is RejectReason.UNLISTED_FILE


def test_a_symlinked_file_is_rejected_even_when_it_points_at_the_right_bytes(bundle, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.log"
    real.write_bytes(DEFAULT_FILES["server.log"])
    (bundle.staging / "server.log").unlink()
    os.symlink(real, bundle.staging / "server.log")
    assert _reason(bundle.staging, bundle) is RejectReason.UNSAFE_PATH


def test_a_planted_symlink_the_manifest_does_not_list_is_rejected_as_unsafe(bundle, tmp_path: Path) -> None:
    os.symlink(tmp_path, bundle.staging / "planted")
    assert _reason(bundle.staging, bundle) is RejectReason.UNSAFE_PATH


def test_a_symlinked_directory_is_rejected(bundle, tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    (bundle.staging / "benchmark_sglang_20261005_101500").rename(moved)
    os.symlink(moved, bundle.staging / "benchmark_sglang_20261005_101500")
    assert _reason(bundle.staging, bundle) is RejectReason.UNSAFE_PATH


def test_a_symlinked_staging_directory_is_rejected(bundle, tmp_path: Path) -> None:
    link = tmp_path / "link"
    os.symlink(bundle.staging, link)
    assert _reason(link, bundle) is RejectReason.UNSAFE_PATH


def test_a_special_file_is_rejected(bundle) -> None:
    os.mkfifo(bundle.staging / "pipe")
    assert _reason(bundle.staging, bundle) is RejectReason.UNSAFE_PATH


def test_a_hand_built_verified_bundle_cannot_be_materialized(bundle, tmp_path: Path) -> None:
    real = verify_bundle(bundle.staging, verify_args(bundle))
    forged = VerifiedBundle(
        manifest=real.manifest,
        result=real.result,
        manifest_sha256=real.manifest_sha256,
        signing_key_fingerprint=real.signing_key_fingerprint,
        staging_dir=real.staging_dir,
    )
    with pytest.raises(ResultRejected) as raised:
        materialize(forged, tmp_path / "out")
    assert raised.value.reason is RejectReason.NOT_VERIFIED
    assert not (tmp_path / "out").exists()


def test_materialize_places_every_file_with_the_same_bytes_and_leaves_nothing_else(bundle, tmp_path: Path) -> None:
    out = tmp_path / "out"
    materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == sorted(DEFAULT_FILES)
    for relative, data in DEFAULT_FILES.items():
        assert (out / relative).read_bytes() == data
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []


def test_materialize_never_overwrites_and_leaves_the_output_directory_as_it_was(bundle, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "server.log").write_text("mine")
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    with pytest.raises(MaterializeError, match="already exists"):
        materialize(verified, out)
    assert sorted(p.name for p in out.iterdir()) == ["server.log"]
    assert (out / "server.log").read_text() == "mine"


def _outside(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    return outside


def test_a_symlinked_directory_in_the_output_is_not_followed_and_nothing_is_written_through_it(
    bundle, tmp_path: Path
) -> None:
    out, outside = tmp_path / "out", _outside(tmp_path)
    out.mkdir()
    os.symlink(outside, out / "benchmark_sglang_20261005_101500")
    with pytest.raises(MaterializeError, match="symlink"):
        materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    assert list(outside.iterdir()) == []
    assert not (out / "server.log").exists()


def test_a_symlink_planted_after_the_precheck_is_not_followed(bundle, tmp_path: Path, monkeypatch) -> None:
    out, outside = tmp_path / "out", _outside(tmp_path)
    out.mkdir()
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    real_place = bundle_mod._place
    calls = []

    def plant_before_first_file(rel, scratch_fd, out_fd, created):
        calls.append(rel)
        if len(calls) == 1:
            os.symlink(outside, out / "benchmark_sglang_20261005_101500")
        return real_place(rel, scratch_fd, out_fd, created)

    monkeypatch.setattr(bundle_mod, "_place", plant_before_first_file)
    with pytest.raises(MaterializeError, match="symlink"):
        materialize(verified, out)
    assert list(outside.iterdir()) == []
    assert [p.name for p in out.iterdir()] == ["benchmark_sglang_20261005_101500"]  # only the planted link remains


def test_a_copy_that_fails_without_hard_links_leaves_no_partial_file(bundle, tmp_path: Path, monkeypatch) -> None:
    import errno

    def no_link(src, dst, **kwargs):
        raise OSError(errno.EPERM, "no hard links here")

    real_write, writes = os.write, []

    def failing_write(fd, data):
        writes.append(fd)
        if len(writes) == 3:  # two staged files first, then the first placed copy
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(fd, data)

    monkeypatch.setattr(bundle_mod.os, "link", no_link)
    monkeypatch.setattr(bundle_mod.os, "write", failing_write)
    out = tmp_path / "out"
    with pytest.raises(OSError, match="No space"):
        materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    assert [p for p in out.rglob("*") if p.is_file()] == []


def test_a_dangling_symlink_at_the_destination_counts_as_existing(bundle, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    os.symlink(tmp_path / "nowhere", out / "server.log")
    with pytest.raises(MaterializeError):
        materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    assert not (tmp_path / "nowhere").exists()


def test_a_byte_changed_after_verification_never_reaches_the_output_directory(bundle, tmp_path: Path) -> None:
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    flip_one_byte(bundle.staging / "server.log", 7)
    out = tmp_path / "out"
    with pytest.raises(ResultRejected) as raised:
        materialize(verified, out)
    assert raised.value.reason is RejectReason.FILE_HASH_MISMATCH
    assert [p for p in out.rglob("*") if p.is_file()] == []
    assert [p.name for p in out.iterdir()] == []


def test_a_symlink_swapped_in_after_verification_is_not_followed(bundle, tmp_path: Path) -> None:
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    secret = tmp_path / "secret"
    secret.write_bytes(DEFAULT_FILES["server.log"])
    (bundle.staging / "server.log").unlink()
    os.symlink(secret, bundle.staging / "server.log")
    with pytest.raises(ResultRejected) as raised:
        materialize(verified, tmp_path / "out")
    assert raised.value.reason is RejectReason.UNSAFE_PATH


def test_a_failure_while_linking_removes_what_was_already_placed(bundle, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    real_link, calls = os.link, []

    def failing_link(src, dst, **kwargs):
        calls.append(dst)
        if len(calls) == 2:
            raise FileExistsError(dst)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(bundle_mod.os, "link", failing_link)
    with pytest.raises(MaterializeError, match="appeared"):
        materialize(verified, out)
    assert len(calls) == 2
    assert list(out.rglob("*")) == []


def test_without_hard_links_the_files_are_still_placed_exclusively(bundle, tmp_path: Path, monkeypatch) -> None:
    import errno

    def no_link(src, dst, **kwargs):
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(bundle_mod.os, "link", no_link)
    out = tmp_path / "out"
    materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    assert (out / "server.log").read_bytes() == DEFAULT_FILES["server.log"]


def test_read_file_refuses_a_file_changed_after_verification_and_an_unlisted_name(bundle) -> None:
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    flip_one_byte(bundle.staging / REPORT, 2)
    with pytest.raises(ResultRejected) as raised:
        verified.read_file(REPORT)
    assert raised.value.reason is RejectReason.FILE_HASH_MISMATCH
    with pytest.raises(ResultRejected) as raised:
        verified.read_file("nope.txt")
    assert raised.value.reason is RejectReason.UNLISTED_FILE


def _record(verified) -> RecordedBundle:
    return RecordedBundle(verified.manifest_sha256, verified.signing_key_fingerprint)


def test_a_record_is_confirmed_by_its_archived_bundle(bundle) -> None:
    record = _record(verify_bundle(bundle.staging, verify_args(bundle)))
    again = reverify_record(bundle.staging, verify_args(bundle), record)
    assert again.manifest_sha256 == record.manifest_sha256


def test_a_record_naming_another_manifest_or_key_fails_the_recheck(bundle) -> None:
    record = _record(verify_bundle(bundle.staging, verify_args(bundle)))
    for field, value in (("manifest_sha256", "0" * 64), ("signing_key_fingerprint", "sha256:" + "0" * 64)):
        with pytest.raises(ResultRejected) as raised:
            reverify_record(bundle.staging, verify_args(bundle), replace(record, **{field: value}))
        assert raised.value.reason is RejectReason.RECORD_MISMATCH, field


def test_a_record_for_a_bundle_whose_file_changed_fails_the_recheck(bundle) -> None:
    record = _record(verify_bundle(bundle.staging, verify_args(bundle)))
    flip_one_byte(bundle.staging / REPORT, 1)
    with pytest.raises(ResultRejected) as raised:
        reverify_record(bundle.staging, verify_args(bundle), record)
    assert raised.value.reason is RejectReason.FILE_HASH_MISMATCH


def test_the_spec_fingerprint_constant_is_the_one_the_fixture_uses() -> None:
    assert len(SPEC_FINGERPRINT) == 16


def test_placed_files_and_directories_have_fixed_modes_whatever_the_umask(bundle, tmp_path: Path) -> None:
    out = tmp_path / "out"
    old = os.umask(0)
    try:
        materialize(verify_bundle(bundle.staging, verify_args(bundle)), out)
    finally:
        os.umask(old)
    assert (out / "server.log").stat().st_mode & 0o777 == 0o600
    assert (out / "benchmark_sglang_20261005_101500").stat().st_mode & 0o777 == 0o700
