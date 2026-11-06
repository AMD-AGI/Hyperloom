# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the build store leaves on disk: every file of an entry and every blob is its writer's alone, read-only, and the
writer's account reads a ready and a failed entry back without privilege."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest

from hyperloom.measure import signing
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import EXECUTABLE, FAILED, READY, BuildRecord, OutputFile
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import Signer, Trust, TrustedKey
from hyperloom.measure.jit.store import BLOBS_DIR
from hyperloom.measure.jit.tests._accounts import as_store_owner

INPUTS = BuildInputs("sha256:" + "a" * 64, "sha256:" + "c" * 64, "fakepkg", "gfx942", "e" * 64, "fake/1")
PAYLOAD = b"\x7fELF payload"


def _record(status: str) -> BuildRecord:
    outputs = (OutputFile("fakepkg/kernel.so", hashlib.sha256(PAYLOAD).hexdigest(), len(PAYLOAD), EXECUTABLE),)
    return BuildRecord(
        key=INPUTS.key,
        inputs=INPUTS,
        status=status,
        outputs=outputs if status == READY else (),
        toolchain={"compiler": "fake"},
        built_at="2026-11-02T00:00:00Z",
        build_seconds=1.5,
        wall_seconds=2.0,
        failure="" if status == READY else "compile_error: the compile exited 1",
    )


@pytest.mark.parametrize("status", [READY, FAILED])
def test_a_stored_entry_is_its_owners_alone_and_the_owner_still_reads_it(tmp_path: Path, status: str) -> None:
    key = signing.generate_private_key()
    trust = Trust((TrustedKey(signing.public_key_pem(key), 1),), 1)
    record = _record(status)

    def publish_and_read(base: Path) -> None:
        store = BuildStore(base / "store", trust)
        staging = base / "staging"
        (staging / BLOBS_DIR).mkdir(parents=True)
        for output in record.outputs:
            (staging / BLOBS_DIR / output.sha256).write_bytes(PAYLOAD)
        store.publish(record, staging, "the compiler's tail", Signer(key))
        assert store.read(INPUTS) == record
        store.fetch(record, base / "out")
        for output in record.outputs:
            assert (base / "out" / output.path).read_bytes() == PAYLOAD

    owner = as_store_owner(tmp_path, publish_and_read)

    root = tmp_path / "store"
    stored = {
        path.relative_to(root).as_posix(): (stat.S_IMODE(path.lstat().st_mode), path.lstat().st_uid)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    entry = f"builds/{INPUTS.component}/{INPUTS.key}"
    expected = {
        f"{entry}/{name}"
        for name in (("manifest.json", "manifest.sig") if status == READY else ("failed.json", "failed.sig"))
    }
    expected |= {f"{entry}/log.txt"} | {f"{BLOBS_DIR}/{o.sha256[:2]}/{o.sha256}" for o in record.outputs}
    assert set(stored) == expected
    assert set(stored.values()) == {(0o400, owner)}, "no group or other bit, and written by the account that reads it"
