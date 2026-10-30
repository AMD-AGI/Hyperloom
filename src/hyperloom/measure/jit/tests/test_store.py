# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache as a store: it only adds, and what it gives back has been proved byte for byte (4.21).

Every refusal is asserted by what it says, because a store that "fails somehow" is also the store that serves a tampered
file: the tests below separate a hash that does not match, a name that is not the hash, a list kept under another key,
a symlink, a missing blob and a file the kind does not admit.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from hyperloom.measure.jit import store as jit_store
from hyperloom.measure.jit_kinds import AITER_JIT, TRITON
from hyperloom.measure.jit.store import DirJitStore, JitStoreError, Snapshot, SnapshotInvalid, StoredFile
from hyperloom.measure.jit.tests.triton_cache import PRODUCTS, get_group

KEY = "a" * 64
OTHER_KEY = "b" * 64
MODULE = b"\x7fELF module_gemm"


def _stage(directory: Path, files: dict[str, bytes], *, kind=AITER_JIT, key: str = KEY, parent: str | None = None):
    """A contribution as a runner stages it: ``blobs/<sha>`` and the ``manifest.json`` that lists the whole snapshot."""
    (directory / "blobs").mkdir(parents=True, exist_ok=True)
    entries = {}
    for relative, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        (directory / "blobs" / sha).write_bytes(data)
        entries[relative] = StoredFile(sha, len(data))
    snapshot = Snapshot(kind.name, key, parent, entries)
    (directory / "manifest.json").write_bytes(snapshot.to_bytes())
    return directory, snapshot


def _publish(store: DirJitStore, tmp_path: Path, files: dict[str, bytes], *, name: str = "s", **kwargs) -> Snapshot:
    staged, snapshot = _stage(tmp_path / name, files, **kwargs)
    store.publish(staged, snapshot.sha256())
    return snapshot


def _blob_of(store: DirJitStore, data: bytes) -> Path:
    sha = hashlib.sha256(data).hexdigest()
    return store.root / "blobs" / sha[:2] / sha


@pytest.fixture
def store(tmp_path) -> DirJitStore:
    return DirJitStore(tmp_path / "store")


def test_what_was_published_is_what_a_test_is_given_and_the_bytes_are_the_ones_published(store, tmp_path) -> None:
    snapshot = _publish(store, tmp_path, {"module_a.so": MODULE, "module_b.so": b"two"})

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot == snapshot and got.problems == () and (got.files, got.bytes) == (2, len(MODULE) + 3)
    assert (tmp_path / "private" / "module_a.so").read_bytes() == MODULE
    assert (tmp_path / "private" / "module_b.so").read_bytes() == b"two"
    assert os.stat(tmp_path / "private" / "module_a.so").st_mode & 0o077 == 0, "the copy is the test's own"


def test_a_key_nothing_was_published_under_is_a_miss_not_a_problem(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE})

    got = store.fetch(AITER_JIT, OTHER_KEY, tmp_path / "private")

    assert got.snapshot is None and got.problems == ()
    assert not (tmp_path / "private").exists() or not any((tmp_path / "private").iterdir())


@pytest.mark.parametrize("key", ["", "not-a-hash", "A" * 64, "a" * 63])
def test_a_key_that_is_not_a_sha256_is_refused_before_anything_is_read(store, tmp_path, key) -> None:
    with pytest.raises(JitStoreError, match="the cache key is not usable"):
        store.fetch(AITER_JIT, key, tmp_path / "private")


def test_a_blob_whose_bytes_changed_fails_verification_and_leaves_no_file_behind(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE, "module_b.so": b"two"})
    blob = _blob_of(store, b"two")
    blob.chmod(0o644)
    blob.write_bytes(b"TWO")  # same size, other bytes: only the hash can tell

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None
    assert any("module_b.so: the bytes are not the 3 bytes whose hash is listed" in problem for problem in got.problems)
    assert list((tmp_path / "private").iterdir()) == [], "the file that did verify is taken back too: no half snapshot"


def _damage(store: DirJitStore, data: bytes, damage: bytes) -> Path:
    blob = _blob_of(store, data)
    blob.chmod(0o644)
    blob.write_bytes(damage)
    return blob


def _while_verifying(monkeypatch, meanwhile) -> None:
    """Run ``meanwhile`` after a blob's bytes fail verification and before the reader acts on it: another writer."""
    verify = jit_store._copy_checked

    def racing(*args, **kwargs):
        try:
            verify(*args, **kwargs)
        except SnapshotInvalid:
            meanwhile()
            raise

    monkeypatch.setattr(jit_store, "_copy_checked", racing)


def test_a_blob_found_damaged_is_removed_so_the_next_reader_need_not_copy_it_and_a_publisher_stores_it_anew(
    store, tmp_path
) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE, "module_b.so": b"two"})
    blob = _damage(store, b"two", b"TWO")

    first = store.fetch(AITER_JIT, KEY, tmp_path / "first")
    second = store.fetch(AITER_JIT, KEY, tmp_path / "second")

    assert any("module_b.so: the bytes are not the 3 bytes" in problem for problem in first.problems)
    assert any("module_b.so: the blob is missing from the store" in problem for problem in second.problems)
    assert list(blob.parent.iterdir()) == [], "nothing is left beside it"
    assert _blob_of(store, MODULE).read_bytes() == MODULE, "the sound blob of the same snapshot stays"
    _publish(store, tmp_path, {"module_b.so": b"two"}, name="again")
    assert blob.read_bytes() == b"two"


def test_sound_bytes_a_publisher_put_in_place_of_a_damaged_blob_after_it_was_read_are_kept(
    store, tmp_path, monkeypatch
) -> None:
    _publish(store, tmp_path, {"module_b.so": b"two"})
    blob = _damage(store, b"two", b"TWO")

    def repaired() -> None:
        pending = blob.with_name("pending")
        pending.write_bytes(b"two")
        os.rename(pending, blob)

    _while_verifying(monkeypatch, repaired)
    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the bytes are not the 3 bytes" in problem for problem in got.problems)
    assert blob.read_bytes() == b"two" and [p.name for p in blob.parent.iterdir()] == [blob.name]


def test_a_damaged_blob_another_reader_removed_first_is_no_further_problem(store, tmp_path, monkeypatch) -> None:
    _publish(store, tmp_path, {"module_b.so": b"two"})
    blob = _damage(store, b"two", b"TWO")
    _while_verifying(monkeypatch, blob.unlink)

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the bytes are not the 3 bytes" in problem for problem in got.problems)
    assert list(blob.parent.iterdir()) == []


def test_a_damaged_blob_in_a_store_this_reader_cannot_write_stays_and_the_read_still_fails(
    store, tmp_path, monkeypatch
) -> None:
    _publish(store, tmp_path, {"module_b.so": b"two"})
    blob = _damage(store, b"two", b"TWO")

    def refused(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(jit_store.os, "rename", refused)
    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the bytes are not the 3 bytes" in problem for problem in got.problems)
    assert blob.read_bytes() == b"TWO"


@pytest.mark.parametrize("damage", [b"two plus more", b"tw", b""])
def test_a_blob_of_another_length_fails_verification(store, tmp_path, damage) -> None:
    _publish(store, tmp_path, {"module_b.so": b"two"})
    blob = _blob_of(store, b"two")
    blob.chmod(0o644)
    blob.write_bytes(damage)

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the bytes are not the 3 bytes" in problem for problem in got.problems)


def test_a_blob_that_was_replaced_by_a_symlink_is_not_followed(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE})
    outside = tmp_path / "evil.so"
    outside.write_bytes(MODULE)  # the very same bytes: only the symlink gives it away
    blob = _blob_of(store, MODULE)
    blob.unlink()
    blob.symlink_to(outside)

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the blob is not a plain file" in problem for problem in got.problems)


def test_a_blob_that_is_gone_from_the_store_is_named(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE})
    blob = _blob_of(store, MODULE)
    blob.chmod(0o644)
    blob.unlink()

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("module_a.so: the blob is missing from the store" in p for p in got.problems)


def test_a_snapshot_list_that_was_edited_no_longer_hashes_to_its_name(store, tmp_path) -> None:
    snapshot = _publish(store, tmp_path, {"module_a.so": MODULE})
    listing = store.root / "snapshots" / "aiter_jit" / KEY / f"{snapshot.sha256()}.json"
    listing.chmod(0o644)
    document = json.loads(listing.read_text())
    document["files"]["module_evil.so"] = document["files"]["module_a.so"]
    listing.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("the list does not hash to its name" in problem for problem in got.problems)


def test_a_snapshot_kept_under_another_key_is_not_served_for_this_one(store, tmp_path) -> None:
    snapshot = _publish(store, tmp_path, {"module_a.so": MODULE})
    elsewhere = store.root / "snapshots" / "aiter_jit" / OTHER_KEY
    elsewhere.mkdir(parents=True)
    (store.root / "snapshots" / "aiter_jit" / KEY / f"{snapshot.sha256()}.json").rename(
        elsewhere / f"{snapshot.sha256()}.json"
    )

    got = store.fetch(AITER_JIT, OTHER_KEY, tmp_path / "private")

    assert got.snapshot is None
    assert any("the list is for another kind or key than the directory it is in" in p for p in got.problems)


def test_a_list_that_names_a_file_its_kind_does_not_admit_is_refused_even_when_its_hash_is_right(
    store, tmp_path
) -> None:
    """The name is the hash of the bytes, so an attacker who writes both passes the hash check: the kind's own list of
    admitted paths is what stops ``../x`` and a ``.py`` from being materialised in the test's cache directory."""
    raw = json.dumps(
        {
            "schema": jit_store.SNAPSHOT_SCHEMA,
            "kind": "aiter_jit",
            "key": KEY,
            "parent": None,
            "files": {"evil.py": {"sha256": "c" * 64, "size": 1}},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    directory = store.root / "snapshots" / "aiter_jit" / KEY
    directory.mkdir(parents=True)
    (directory / f"{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot is None and any("evil.py is not a file the aiter_jit cache admits" in p for p in got.problems)


def test_a_list_in_another_spelling_than_the_canonical_one_is_refused(store, tmp_path) -> None:
    snapshot = _publish(store, tmp_path, {"module_a.so": MODULE})
    padded = json.dumps(snapshot.document(), indent=2).encode()
    directory = store.root / "snapshots" / "aiter_jit" / KEY
    (directory / f"{hashlib.sha256(padded).hexdigest()}.json").write_bytes(padded)

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot == snapshot, "the canonical generation is still served"
    assert any("the list is not in its canonical form" in problem for problem in got.problems)


def test_when_the_newest_generation_fails_the_next_one_is_used_and_the_failure_is_reported(store, tmp_path) -> None:
    old = _publish(store, tmp_path, {"module_a.so": MODULE}, name="old")
    new = _publish(store, tmp_path, {"module_a.so": MODULE, "module_b.so": b"two"}, name="new", parent=old.sha256())
    blob = _blob_of(store, b"two")
    blob.chmod(0o644)
    blob.write_bytes(b"TWO")

    got = store.fetch(AITER_JIT, KEY, tmp_path / "private")

    assert got.snapshot == old and got.files == 1
    assert any(new.sha256()[:12] in problem and "the bytes are not the 3 bytes" in problem for problem in got.problems)
    assert sorted(p.name for p in (tmp_path / "private").iterdir()) == ["module_a.so"]


def test_the_generation_with_the_most_files_is_the_one_a_test_is_given(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE}, name="one")
    bigger = _publish(store, tmp_path, {"module_a.so": MODULE, "module_c.so": b"three"}, name="two")

    assert store.fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == bigger


def test_a_contribution_whose_list_is_not_the_one_the_result_signed_is_not_published(store, tmp_path) -> None:
    staged, snapshot = _stage(tmp_path / "s", {"module_a.so": MODULE})

    with pytest.raises(SnapshotInvalid, match="is not the one whose hash the result signed"):
        store.publish(staged, "f" * 64)

    assert not (store.root / "snapshots").exists() and not (store.root / "blobs").exists()
    assert snapshot.sha256() != "f" * 64


def test_a_listed_file_that_is_neither_stored_nor_staged_is_refused_and_nothing_is_added(store, tmp_path) -> None:
    staged, snapshot = _stage(tmp_path / "s", {"module_a.so": MODULE, "module_b.so": b"two"})
    (staged / "blobs" / hashlib.sha256(b"two").hexdigest()).unlink()

    with pytest.raises(SnapshotInvalid, match="module_b.so is neither stored nor staged"):
        store.publish(staged, snapshot.sha256())

    assert not (store.root / "snapshots").exists(), "a list is visible only once every blob it names is stored"


def test_a_staged_blob_that_is_not_the_bytes_its_hash_names_is_refused(store, tmp_path) -> None:
    staged, snapshot = _stage(tmp_path / "s", {"module_b.so": b"two"})
    (staged / "blobs" / hashlib.sha256(b"two").hexdigest()).write_bytes(b"TWO")

    with pytest.raises(SnapshotInvalid, match="module_b.so: the bytes are not the 3 bytes whose hash is listed"):
        store.publish(staged, snapshot.sha256())

    assert not _blob_of(store, b"two").exists() and not (store.root / "snapshots").exists()


def test_a_stored_blob_is_never_replaced_and_publishing_a_list_again_is_a_success(store, tmp_path) -> None:
    snapshot = _publish(store, tmp_path, {"module_a.so": MODULE}, name="first")
    before = _blob_of(store, MODULE).stat()
    staged, again = _stage(tmp_path / "second", {"module_a.so": MODULE})

    published = store.publish(staged, again.sha256())

    after = _blob_of(store, MODULE).stat()
    assert published.existed and published.new_blobs == 0 and again.sha256() == snapshot.sha256()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_a_sound_stored_blob_is_never_rewritten_even_when_a_contributor_stages_it_again(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE}, name="first")
    before = _blob_of(store, MODULE).stat()
    staged, snapshot = _stage(tmp_path / "second", {"module_a.so": MODULE}, parent="d" * 64)

    published = store.publish(staged, snapshot.sha256())

    after = _blob_of(store, MODULE).stat()
    assert published.new_blobs == 0 and (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_a_damaged_stored_blob_is_repaired_by_the_verified_bytes_of_the_same_name_and_only_by_them(
    store, tmp_path
) -> None:
    """A tampered blob would otherwise stay in the cache for good and every later test would pay to find it out. The staged
    bytes are hashed against the name before they are stored, so a contributor cannot replace a blob with other content."""
    _publish(store, tmp_path, {"module_a.so": MODULE}, name="first")
    blob = _blob_of(store, MODULE)
    blob.chmod(0o644)
    blob.write_bytes(bytes(reversed(MODULE)))  # the same length: a size check alone would not see it
    forged, snapshot = _stage(tmp_path / "forged", {"module_a.so": MODULE}, parent="d" * 64)
    (forged / "blobs" / hashlib.sha256(MODULE).hexdigest()).write_bytes(b"\x7fELF forged....")

    with pytest.raises(SnapshotInvalid, match="module_a.so: the bytes are not the 16 bytes whose hash is listed"):
        store.publish(forged, snapshot.sha256())
    assert blob.read_bytes() == bytes(reversed(MODULE)), "forged bytes replace nothing"
    assert store.fetch(AITER_JIT, KEY, tmp_path / "private").snapshot is None

    honest, snapshot = _stage(tmp_path / "honest", {"module_a.so": MODULE}, parent="d" * 64)
    published = store.publish(honest, snapshot.sha256())

    assert published.new_blobs == 1 and blob.read_bytes() == MODULE
    assert store.fetch(AITER_JIT, KEY, tmp_path / "private2").snapshot is not None


def test_a_stored_blob_of_another_size_that_is_not_staged_is_a_refusal_not_a_silent_reuse(store, tmp_path) -> None:
    first = _publish(store, tmp_path, {"module_a.so": MODULE}, name="first")
    blob = _blob_of(store, MODULE)
    blob.chmod(0o644)
    blob.write_bytes(b"short")
    staged, extended = _stage(
        tmp_path / "second", {"module_a.so": MODULE, "module_b.so": b"two"}, parent=first.sha256()
    )
    (staged / "blobs" / hashlib.sha256(MODULE).hexdigest()).unlink()  # taken from the cache, so not staged again

    with pytest.raises(SnapshotInvalid, match=f"module_a.so: the stored blob is not {len(MODULE)} bytes"):
        store.publish(staged, extended.sha256())


def test_a_generation_that_extends_another_needs_only_the_files_the_store_does_not_hold(store, tmp_path) -> None:
    first = _publish(store, tmp_path, {"module_a.so": MODULE}, name="first")
    staged, extended = _stage(
        tmp_path / "second", {"module_a.so": MODULE, "module_b.so": b"two"}, parent=first.sha256()
    )
    (staged / "blobs" / hashlib.sha256(MODULE).hexdigest()).unlink()  # the seed's file is not staged again

    published = store.publish(staged, extended.sha256())

    assert (published.new_blobs, published.new_bytes, published.existed) == (1, 3, False)
    assert store.fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == extended


def test_a_symlinked_staged_list_is_refused(store, tmp_path) -> None:
    staged, snapshot = _stage(tmp_path / "s", {"module_a.so": MODULE})
    real = tmp_path / "elsewhere.json"
    (staged / "manifest.json").rename(real)
    (staged / "manifest.json").symlink_to(real)

    with pytest.raises(SnapshotInvalid, match="manifest.json cannot be read"):
        store.publish(staged, snapshot.sha256())


def test_a_snapshot_over_the_size_a_snapshot_may_have_is_refused(store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jit_store, "MAX_SNAPSHOT_BYTES", 5)
    staged, snapshot = _stage(tmp_path / "s", {"module_a.so": MODULE})

    with pytest.raises(SnapshotInvalid, match="more bytes than a snapshot may hold"):
        store.publish(staged, snapshot.sha256())


def test_kinds_do_not_share_snapshots(store, tmp_path) -> None:
    _publish(store, tmp_path, {"module_a.so": MODULE}, name="jit")
    triton_files = {f"{'e' * 20}/kernel.hsaco": b"hsaco"}
    triton = _publish(store, tmp_path, triton_files, name="triton", kind=TRITON)

    assert store.fetch(TRITON, KEY, tmp_path / "t").snapshot == triton
    assert store.fetch(AITER_JIT, KEY, tmp_path / "j").snapshot.kind == "aiter_jit"
    assert not (tmp_path / "j" / ("e" * 20)).exists()


def test_concurrent_publishers_of_overlapping_snapshots_leave_every_one_readable(store, tmp_path) -> None:
    """Two tests that extend the same snapshot publish at the same time: the shared blob is stored once, each list is its
    own file, and neither writer can corrupt the other's."""
    contributions = []
    for number in range(8):
        files = {"module_shared.so": MODULE, f"module_{number}.so": f"own-{number}".encode()}
        contributions.append(_stage(tmp_path / f"c{number}", files))
    failures: list[Exception] = []

    def publish(staged: Path, snapshot: Snapshot) -> None:
        try:
            store.publish(staged, snapshot.sha256())
        except Exception as exc:  # noqa: BLE001 - the test reports whatever error a publisher raised
            failures.append(exc)

    threads = [threading.Thread(target=publish, args=pair) for pair in contributions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    listed = sorted((store.root / "snapshots" / "aiter_jit" / KEY).iterdir())
    assert len(listed) == 8 and not any(p.name.startswith(".incoming-") for p in store.root.iterdir())
    for number, (_, snapshot) in enumerate(contributions):
        destination = tmp_path / f"read{number}"
        assert store.fetch(AITER_JIT, KEY, destination).snapshot is not None
    assert _blob_of(store, MODULE).read_bytes() == MODULE


HASH_DIR = "0123456789abcdef0123456789abcdef"


def _triton_files(group: dict[str, str]) -> dict[str, bytes]:
    """A Triton kernel as the cache stores it: its products, and its group naming them relative to the cache directory."""
    files = {f"{HASH_DIR}/{name}": f"{name} bytes".encode() for name in PRODUCTS}
    files[f"{HASH_DIR}/__grp__kernel.json"] = json.dumps({"child_paths": group}).encode()
    return files


def test_a_stored_group_names_the_files_in_the_directory_of_the_test_that_is_given_it(store, tmp_path) -> None:
    _publish(
        store,
        tmp_path,
        _triton_files({name: f"{HASH_DIR}/{name}" for name in PRODUCTS}),
        kind=TRITON,
    )
    private = tmp_path / "another" / "sandbox" / "triton"

    got = store.fetch(TRITON, KEY, private)

    assert got.snapshot is not None and got.problems == ()
    found = get_group(private, HASH_DIR)
    assert found == {name: f"{private}/{HASH_DIR}/{name}" for name in PRODUCTS}, "Triton would find the kernel"
    assert (private / HASH_DIR / "kernel.hsaco").read_bytes() == b"kernel.hsaco bytes", (
        "the others are copied as they are"
    )


def test_a_stored_group_is_checked_against_its_hash_before_any_path_is_written_into_it(store, tmp_path) -> None:
    _publish(store, tmp_path, _triton_files({name: f"{HASH_DIR}/{name}" for name in PRODUCTS}), kind=TRITON)
    group = json.dumps({"child_paths": {name: f"{HASH_DIR}/{name}" for name in PRODUCTS}}).encode()
    blob = _blob_of(store, group)
    blob.chmod(0o644)
    blob.write_bytes(group.replace(b"kernel.ttir", b"kernel.TTIR"))  # still a group, same size, other bytes

    got = store.fetch(TRITON, KEY, tmp_path / "private")

    assert got.snapshot is None
    assert any("__grp__kernel.json: the bytes are not the" in problem for problem in got.problems)
    assert [p for p in (tmp_path / "private").rglob("*") if p.is_file()] == [], "nothing of it reaches the test"


@pytest.mark.parametrize("path", ["/etc/passwd", f"../{HASH_DIR}/kernel.json", f"{HASH_DIR}/kernel.pkl"])
def test_a_stored_group_that_names_a_file_outside_what_the_cache_holds_drops_the_snapshot_whole(
    store, tmp_path, path
) -> None:
    _publish(store, tmp_path, _triton_files({"kernel.json": path}), kind=TRITON)

    got = store.fetch(TRITON, KEY, tmp_path / "private")

    assert got.snapshot is None
    assert any(
        "__grp__kernel.json: kernel.json: " in problem and "is not a file of the triton cache" in problem
        for problem in got.problems
    )
    assert [p for p in (tmp_path / "private").rglob("*") if p.is_file()] == [], "the files that did verify go too"


def test_a_stored_group_of_names_is_copied_as_it_is_and_triton_finds_the_kernel_in_the_new_directory(
    store, tmp_path
) -> None:
    names = sorted(PRODUCTS)
    files = {f"{HASH_DIR}/{name}": f"{name} bytes".encode() for name in PRODUCTS}
    files[f"{HASH_DIR}/__grp__kernel.json"] = json.dumps({"child_paths": names}).encode()
    _publish(store, tmp_path, files, kind=TRITON)
    private = tmp_path / "another" / "triton"

    got = store.fetch(TRITON, KEY, private)

    assert got.snapshot is not None and got.problems == ()
    assert get_group(private, HASH_DIR, release="current") == {
        name: f"{private}/{HASH_DIR}/{name}" for name in PRODUCTS
    }
