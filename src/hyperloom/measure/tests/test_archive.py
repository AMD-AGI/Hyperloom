# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest
from hyperloom.measure.tests.bundle_factory import REQUEST_ID, flip_one_byte, make_bundle, verify_args

from hyperloom.measure import archive as archive_mod
from hyperloom.measure.archive import ARCHIVE_FAILED, BUNDLE_DIR_MODE, ArchiveError, DirResultArchive
from hyperloom.measure.bundle import verify_bundle


@pytest.fixture
def staged(tmp_path: Path):
    bundle = make_bundle(tmp_path / "staging")
    verified = verify_bundle(bundle.staging, verify_args(bundle))
    return bundle, verified


@pytest.fixture
def archive(tmp_path: Path) -> DirResultArchive:
    return DirResultArchive(tmp_path / "archive")


def test_a_bundle_is_stored_under_its_request_and_manifest_hash_and_verifies_again(staged, archive) -> None:
    bundle, verified = staged
    location = archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    assert Path(location) == archive.bundle_dir(REQUEST_ID, verified.manifest_sha256)
    again = verify_bundle(Path(location), verify_args(bundle))
    assert again.manifest_sha256 == verified.manifest_sha256


def test_storing_the_same_bundle_again_is_a_success_at_the_same_location(staged, archive) -> None:
    bundle, verified = staged
    first = archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    assert archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging) == first


def test_a_stored_bundle_is_never_replaced_by_different_content(staged, archive) -> None:
    bundle, verified = staged
    location = Path(archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging))
    flip_one_byte(bundle.staging / "server.log", 4)
    with pytest.raises(ArchiveError, match="different bundle"):
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    flip_one_byte(bundle.staging / "server.log", 4)
    assert (location / "server.log").read_bytes() == bundle.staging.joinpath("server.log").read_bytes()
    assert not os.access(location / "server.log", os.W_OK) or os.geteuid() == 0


def test_the_archive_key_must_name_the_manifest_that_is_staged(staged, archive, tmp_path: Path) -> None:
    bundle, _ = staged
    with pytest.raises(ArchiveError, match="manifest.json is not the manifest"):
        archive.put_bundle(REQUEST_ID, "0" * 64, bundle.staging)
    assert not (tmp_path / "archive" / REQUEST_ID).exists()


@pytest.mark.parametrize("request_id", ["", ".", "..", "../x", "a/b", "a b"])
def test_an_unusable_request_id_is_refused(staged, archive, request_id: str) -> None:
    bundle, verified = staged
    with pytest.raises(ArchiveError, match="request id"):
        archive.put_bundle(request_id, verified.manifest_sha256, bundle.staging)


def test_a_symlink_in_the_staging_directory_is_refused(staged, archive, tmp_path: Path) -> None:
    bundle, verified = staged
    (bundle.staging / "link").symlink_to(tmp_path)
    with pytest.raises(ArchiveError, match="cannot be archived"):
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)


def test_a_file_changed_while_being_copied_never_becomes_visible(staged, archive, tmp_path: Path, monkeypatch) -> None:
    bundle, verified = staged
    real = archive_mod._copy_listing

    def tampering(staging_dir, incoming, listing):
        flip_one_byte(staging_dir / "server.log", 9)
        return real(staging_dir, incoming, listing)

    monkeypatch.setattr(archive_mod, "_copy_listing", tampering)
    with pytest.raises(ArchiveError, match="changed while"):
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    assert not archive.bundle_dir(REQUEST_ID, verified.manifest_sha256).exists()
    assert [p.name for p in (tmp_path / "archive").iterdir() if p.name.startswith(".incoming")] == []


def test_a_write_that_fails_halfway_leaves_no_partial_bundle(staged, archive, tmp_path: Path, monkeypatch) -> None:
    bundle, verified = staged
    real_fsync, calls = os.fsync, []

    def failing_fsync(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_fsync(fd)

    monkeypatch.setattr(archive_mod.os, "fsync", failing_fsync)
    with pytest.raises(ArchiveError) as raised:
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    assert raised.value.error_class == ARCHIVE_FAILED == "archive_failed"
    assert not archive.bundle_dir(REQUEST_ID, verified.manifest_sha256).exists()
    assert [p for p in (tmp_path / "archive").rglob("*") if p.is_file()] == []
    assert [p.name for p in (tmp_path / "archive").iterdir() if p.name.startswith(".incoming")] == []
    monkeypatch.undo()
    archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)  # the retry lands whole


def test_an_unwritable_archive_root_is_an_archive_failure(staged, tmp_path: Path) -> None:
    bundle, verified = staged
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ArchiveError) as raised:
        DirResultArchive(blocker / "root").put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    assert raised.value.error_class == "archive_failed"


def test_concurrent_writers_of_the_same_bundle_all_succeed_and_one_copy_exists(staged, archive) -> None:
    bundle, verified = staged
    results, errors = [], []

    def put() -> None:
        try:
            results.append(archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging))
        except Exception as exc:  # noqa: BLE001 - the test reports whatever a writer raised
            errors.append(exc)

    threads = [threading.Thread(target=put) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(set(results)) == 1
    assert [p.name for p in archive.bundle_dir(REQUEST_ID, verified.manifest_sha256).parent.iterdir()] == [
        verified.manifest_sha256
    ]


def test_bundle_path_names_the_stored_bundle_and_refuses_a_key_nothing_is_stored_under(staged, archive) -> None:
    bundle, verified = staged
    with pytest.raises(ArchiveError, match="no bundle is archived"):
        archive.bundle_path(REQUEST_ID, verified.manifest_sha256)
    archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    path = archive.bundle_path(REQUEST_ID, verified.manifest_sha256)
    assert verify_bundle(path, verify_args(bundle)).manifest_sha256 == verified.manifest_sha256
    with pytest.raises(ArchiveError, match="not a sha256"):
        archive.bundle_path(REQUEST_ID, "not-a-hash")


def _modes(paths) -> set[int]:
    return {stat.S_IMODE(path.stat().st_mode) for path in paths}


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_on_shared_storage_a_stored_bundle_is_the_groups_to_read_and_the_way_to_it_the_groups_to_write(
    staged, tmp_path: Path, umask: int
) -> None:
    """A later leg of the run may be another account of the group: it reads the bundle, and stores its own beside it."""
    bundle, verified = staged
    shared = tmp_path / "toolbox"
    shared.mkdir()
    shared.chmod(0o2775)
    old = os.umask(umask)
    try:
        location = Path(
            DirResultArchive(shared / "archive").put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
        )
    finally:
        os.umask(old)
    assert _modes((shared / "archive", location.parent)) == {0o2775}
    assert _modes(path for path in (location, *location.rglob("*")) if path.is_dir()) == {BUNDLE_DIR_MODE}
    assert _modes(path for path in location.rglob("*") if path.is_file()) == {0o444}


def test_off_shared_storage_a_stored_bundle_stays_its_writers(staged, archive) -> None:
    bundle, verified = staged
    location = Path(archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging))
    assert stat.S_IMODE(location.stat().st_mode) == 0o700


def _swap_incoming(root: Path, replacement: Path | None, target: Path) -> None:
    """Put ``replacement`` (a link to ``target`` when ``None``) where the writer's incoming directory is, as another
    account of the group can below a root it writes."""
    (incoming,) = [p for p in root.iterdir() if p.name.startswith(".incoming-")]
    incoming.rename(root.parent / "moved-away")
    if replacement is None:
        incoming.symlink_to(target)
    else:
        replacement.rename(incoming)


@pytest.mark.parametrize("taken_by", ["link", "directory open to others"])
def test_an_incoming_directory_taken_over_before_it_is_opened_is_refused_and_nothing_is_written_elsewhere(
    staged, archive, tmp_path: Path, monkeypatch, taken_by: str
) -> None:
    bundle, verified = staged
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    victim.chmod(0o700)
    real_mkdir = os.mkdir

    def mkdir_then_take_over(path, mode=0o777, *, dir_fd=None):
        real_mkdir(path, mode, dir_fd=dir_fd)
        if str(path).startswith(".incoming-"):
            if taken_by == "link":
                _swap_incoming(tmp_path / "archive", None, victim)
            else:
                other = tmp_path / "other"
                other.mkdir()
                other.chmod(0o755)
                _swap_incoming(tmp_path / "archive", other, victim)

    monkeypatch.setattr(archive_mod.os, "mkdir", mkdir_then_take_over)
    with pytest.raises(ArchiveError):
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    monkeypatch.undo()
    assert list(victim.iterdir()) == [] and stat.S_IMODE(victim.stat().st_mode) == 0o700
    assert not archive.bundle_dir(REQUEST_ID, verified.manifest_sha256).exists()


def test_a_failed_store_removes_only_its_own_incoming_directory(staged, archive, tmp_path: Path, monkeypatch) -> None:
    """The name of the writer's incoming directory may name another directory by the time a failure cleans up."""
    bundle, verified = staged
    precious = tmp_path / "precious"
    precious.mkdir(mode=0o700)
    (precious / "kept").write_text("x")
    real = archive_mod._copy_listing

    def take_over_then_fail(staging_dir, incoming, listing):
        real(staging_dir, incoming, listing)
        _swap_incoming(tmp_path / "archive", precious, precious)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(archive_mod, "_copy_listing", take_over_then_fail)
    with pytest.raises(ArchiveError):
        archive.put_bundle(REQUEST_ID, verified.manifest_sha256, bundle.staging)
    kept = [p / "kept" for p in (tmp_path / "archive").iterdir() if p.name.startswith(".incoming-")]
    assert [p.read_text() for p in kept] == ["x"], "the other directory now under the incoming name was removed"
