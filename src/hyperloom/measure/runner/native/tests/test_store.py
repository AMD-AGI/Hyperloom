# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The build store on its own: publish, coalesce, and each refusal of a stored entry, by reason."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure import signing
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import FAILED, READY, RECORD_FILES, BuildRecord, OutputFile
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused, Refusal, Signer, Trust, TrustedKey
from hyperloom.measure.runner.native.service import Staged, ensure_build
from hyperloom.measure.runner.native.tests._native import BUILDER_IMAGE, OTHER_IMAGE
from hyperloom.measure.jit.store import BLOBS_DIR
from hyperloom.measure.tests._recipes import IMAGE

INPUTS = BuildInputs(IMAGE, BUILDER_IMAGE, "fakepkg", "gfx942", "e" * 64, "fake/1")
PAYLOAD = b"\x7fELF payload"
MANIFEST, SIGNATURE = RECORD_FILES[READY]


class FakeBuild:
    """Stages one output (or a failure) and counts its calls in a file, so other processes' calls count too."""

    def __init__(self, counter: Path, *, status: str = READY, pause: float = 0.0) -> None:
        self.counter, self.status, self.pause = counter, status, pause

    def __call__(self, scratch: Path) -> Staged:
        with open(self.counter, "a", encoding="utf-8") as out:
            out.write("x")
        time.sleep(self.pause)
        sha = hashlib.sha256(PAYLOAD).hexdigest()
        (scratch / BLOBS_DIR).mkdir()
        (scratch / BLOBS_DIR / sha).write_bytes(PAYLOAD)
        outputs = (OutputFile("fakepkg/kernel.so", sha, len(PAYLOAD), "0755"),)
        failure = "compile_error: the compile exited 1" if self.status == FAILED else ""
        return Staged(self.status, outputs, {"compiler": "fake"}, 1.5, 2.0, "log", scratch, failure=failure)

    @property
    def calls(self) -> int:
        return len(self.counter.read_text()) if self.counter.exists() else 0


@pytest.fixture
def key():
    return signing.generate_private_key()


@pytest.fixture
def store(tmp_path: Path, key) -> BuildStore:
    return BuildStore(tmp_path / "store", Trust((TrustedKey(signing.public_key_pem(key), 2),), 2))


def _publish(store: BuildStore, key, tmp_path: Path, inputs: BuildInputs = INPUTS, **build) -> BuildRecord:
    return ensure_build(store, inputs, FakeBuild(tmp_path / "count", **build), Signer(key)).record


def _refused(reason: Refusal, call) -> None:
    with pytest.raises(BuildRefused) as caught:
        call()
    assert caught.value.reason is reason


def test_a_published_entry_reads_back_and_its_blob_fetches(store: BuildStore, key, tmp_path: Path) -> None:
    record = _publish(store, key, tmp_path)

    assert store.read(INPUTS) == record
    assert (record.build_seconds, record.wall_seconds) == (1.5, 2.0)
    store.fetch(record, tmp_path / "out")
    assert (tmp_path / "out/fakepkg/kernel.so").read_bytes() == PAYLOAD


def test_concurrent_requests_of_one_key_build_once(store: BuildStore, key, tmp_path: Path) -> None:
    build = FakeBuild(tmp_path / "count", pause=1.0)
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=ensure_build, args=(store, INPUTS, build, Signer(key))) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert build.calls == 1
    assert ensure_build(store, INPUTS, build, Signer(key)).reused is True


def test_an_infra_failure_stores_nothing_and_the_next_request_builds(store: BuildStore, key, tmp_path: Path) -> None:
    def broken(scratch: Path) -> Staged:
        raise BuildInfraError("the node ran out of disk")

    with pytest.raises(BuildInfraError):
        ensure_build(store, INPUTS, broken, Signer(key))
    assert store.read(INPUTS) is None
    assert ensure_build(store, INPUTS, FakeBuild(tmp_path / "count"), Signer(key)).reused is False


def test_a_failed_build_is_stored_as_failed_json_without_outputs(store: BuildStore, key, tmp_path: Path) -> None:
    record = _publish(store, key, tmp_path, status=FAILED)

    assert (record.status, record.outputs) == (FAILED, ())
    assert (store.entry_dir(INPUTS) / RECORD_FILES[FAILED][0]).is_file()
    assert not (store.entry_dir(INPUTS) / MANIFEST).exists()


def test_a_tampered_record_is_refused_as_bad_signature(store: BuildStore, key, tmp_path: Path) -> None:
    _publish(store, key, tmp_path)
    manifest = store.entry_dir(INPUTS) / MANIFEST
    os.chmod(manifest, 0o600)
    manifest.write_bytes(manifest.read_bytes().replace(b'"wall_seconds":2.0', b'"wall_seconds":0.0'))

    _refused(Refusal.BAD_SIGNATURE, lambda: store.read(INPUTS))


def test_a_record_signed_by_an_unlisted_key_is_refused_as_bad_signature(store: BuildStore, tmp_path: Path) -> None:
    _publish(store, signing.generate_private_key(), tmp_path)

    _refused(Refusal.BAD_SIGNATURE, lambda: store.read(INPUTS))


def test_a_key_older_than_the_trust_epoch_is_refused_as_stale(tmp_path: Path, key) -> None:
    old = BuildStore(tmp_path / "store", Trust((TrustedKey(signing.public_key_pem(key), 1),), 1))
    _publish(old, key, tmp_path)
    raised = BuildStore(tmp_path / "store", Trust((TrustedKey(signing.public_key_pem(key), 1),), 2))

    _refused(Refusal.STALE_EPOCH, lambda: raised.read(INPUTS))


def _move_entry(store: BuildStore, source: BuildInputs, target: BuildInputs) -> None:
    store.entry_dir(target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store.entry_dir(source), store.entry_dir(target))


def test_an_entry_under_another_key_is_refused_as_key_mismatch(store: BuildStore, key, tmp_path: Path) -> None:
    _publish(store, key, tmp_path)
    other = replace(INPUTS, native_digest="f" * 64)
    _move_entry(store, INPUTS, other)

    _refused(Refusal.KEY_MISMATCH, lambda: store.read(other))


def test_an_entry_built_for_another_image_is_refused_as_image_mismatch(store: BuildStore, key, tmp_path: Path) -> None:
    _publish(store, key, tmp_path)
    other = replace(INPUTS, image_digest=OTHER_IMAGE)
    _move_entry(store, INPUTS, other)

    _refused(Refusal.IMAGE_MISMATCH, lambda: store.read(other))


def _blob(store_root: Path) -> Path:
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    return store_root / BLOBS_DIR / sha[:2] / sha


def test_a_blob_that_does_not_match_its_hash_is_refused(store: BuildStore, key, tmp_path: Path) -> None:
    record = _publish(store, key, tmp_path)
    blob = _blob(tmp_path / "store")
    os.chmod(blob, 0o600)
    blob.write_bytes(b"\x7fELF payloaX")

    _refused(Refusal.BLOB_MISMATCH, lambda: store.fetch(record, tmp_path / "out"))
    assert not (tmp_path / "out/fakepkg/kernel.so").exists()


def test_a_missing_blob_is_refused(store: BuildStore, key, tmp_path: Path) -> None:
    record = _publish(store, key, tmp_path)
    _blob(tmp_path / "store").unlink()

    _refused(Refusal.BLOB_MISSING, lambda: store.fetch(record, tmp_path / "out"))


def test_an_entry_without_a_record_is_refused(store: BuildStore, tmp_path: Path) -> None:
    store.entry_dir(INPUTS).mkdir(parents=True)

    _refused(Refusal.RECORD_INVALID, lambda: store.read(INPUTS))


def test_a_failed_record_under_the_ready_name_is_refused(store: BuildStore, key, tmp_path: Path) -> None:
    _publish(store, key, tmp_path, status=FAILED)
    entry = store.entry_dir(INPUTS)
    record_name, signature_name = RECORD_FILES[FAILED]
    (entry / record_name).rename(entry / MANIFEST)
    (entry / signature_name).rename(entry / SIGNATURE)

    _refused(Refusal.RECORD_INVALID, lambda: store.read(INPUTS))


def test_a_refused_entry_is_never_rebuilt_over(store: BuildStore, key, tmp_path: Path) -> None:
    _publish(store, key, tmp_path)
    signature = store.entry_dir(INPUTS) / SIGNATURE
    os.chmod(signature, 0o600)
    signature.write_bytes(b"AAAA")
    build = FakeBuild(tmp_path / "count2")

    _refused(Refusal.BAD_SIGNATURE, lambda: ensure_build(store, INPUTS, build, Signer(key)))
    assert (build.calls, signature.read_bytes()) == (0, b"AAAA")
