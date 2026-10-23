# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One test's use of the shared compiled-code cache: seeded before the server starts, harvested after it succeeds (4.21).

The compilers are played by a function that writes into the directories the session names (what aiter and Triton do when
they find a variable set), so a second test that "finds" a module built by the first is found through the real seed, store
and publish, and nothing but the compiler is faked.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from hyperloom.measure import jit_session
from hyperloom.measure.jit_builds import BuildLog, scan_builds
from hyperloom.measure.jit_kinds import AITER_CPP, AITER_JIT, DISABLED, HIT, MISS, TRITON, VERIFY_FAILED
from hyperloom.measure.jit_session import JitCacheSession, KindPlan
from hyperloom.measure.jit_store import DirJitStore

KEY = "a" * 64
TRITON_KEY = "b" * 64
MODULE = b"\x7fELF module_gemm"
HASH_DIR = "0123456789abcdef0123456789abcdef"
NO_BUILDS = BuildLog({}, frozenset())


def _session(store, tmp_path, name="t", plans=None, **kwargs) -> JitCacheSession:
    plans = (
        [KindPlan(AITER_JIT, KEY), KindPlan(AITER_CPP, KEY), KindPlan(TRITON, TRITON_KEY)] if plans is None else plans
    )
    return JitCacheSession(
        store, plans, private_root=tmp_path / name / "private", staging=tmp_path / name / "out", **kwargs
    )


def _write(directory: Path, relative: str, data: bytes) -> Path:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _dirs(session: JitCacheSession) -> dict[str, Path]:
    return {name: Path(path) for name, path in session.environment().items()}


def _publish(store: DirJitStore, session: JitCacheSession, tmp_path: Path, name: str = "t") -> None:
    """What the host does with a signed result: publish each kind the test staged."""
    for kind in session.report(NO_BUILDS).kinds:
        if kind.contribution is not None:
            store.publish(tmp_path / name / "out" / kind.kind, kind.contribution.sha256)


def _listed(tmp_path: Path, name: str, kind: str) -> list[str]:
    """The paths the staged snapshot list of ``kind`` names."""
    return sorted(json.loads((tmp_path / name / "out" / kind / "manifest.json").read_text())["files"])


@pytest.fixture
def store(tmp_path) -> DirJitStore:
    return DirJitStore(tmp_path / "store")


def test_each_kind_gets_a_directory_of_the_tests_own_and_two_tests_never_share_one(store, tmp_path) -> None:
    first, second = _session(store, tmp_path, "one"), _session(store, tmp_path, "two")

    dirs = _dirs(first)

    assert set(dirs) == {"AITER_JIT_DIR", "AITER_ROOT_DIR", "TRITON_CACHE_DIR"}
    assert all(path.is_relative_to(tmp_path / "one" / "private") for path in dirs.values())
    assert not set(dirs.values()) & set(_dirs(second).values())


def test_the_second_test_is_given_what_the_first_compiled_and_has_nothing_left_to_build(store, tmp_path) -> None:
    first = _session(store, tmp_path, "one")
    first.seed()
    dirs = _dirs(first)
    _write(dirs["AITER_JIT_DIR"], "module_gemm.so", MODULE)
    _write(dirs["AITER_ROOT_DIR"], "build/pa_ragged_x/lib.so", b"cpp lib")
    _write(dirs["TRITON_CACHE_DIR"], f"{HASH_DIR}/kernel.hsaco", b"hsaco")
    first.collect()
    cold = first.report(NO_BUILDS)
    _publish(store, first, tmp_path, "one")

    second = _session(store, tmp_path, "two")
    second.seed()

    assert [k.state for k in cold.kinds] == [MISS, MISS, MISS]
    assert [k.state for k in second.report(NO_BUILDS).kinds] == [HIT, HIT, HIT]
    assert (_dirs(second)["AITER_JIT_DIR"] / "module_gemm.so").read_bytes() == MODULE
    assert (_dirs(second)["AITER_ROOT_DIR"] / "build" / "pa_ragged_x" / "lib.so").read_bytes() == b"cpp lib"
    assert (_dirs(second)["TRITON_CACHE_DIR"] / HASH_DIR / "kernel.hsaco").read_bytes() == b"hsaco"
    second.collect()
    assert all(k.contribution is None for k in second.report(NO_BUILDS).kinds), "nothing new: nothing to contribute"
    assert not (tmp_path / "two" / "out").exists() or not any((tmp_path / "two" / "out").iterdir())


def test_a_second_test_that_compiles_one_more_module_extends_the_snapshot_and_names_its_parent(store, tmp_path) -> None:
    first = _session(store, tmp_path, "one", plans=[KindPlan(AITER_JIT, KEY)])
    first.seed()
    _write(_dirs(first)["AITER_JIT_DIR"], "module_a.so", b"a")
    first.collect()
    _publish(store, first, tmp_path, "one")
    second = _session(store, tmp_path, "two", plans=[KindPlan(AITER_JIT, KEY)])
    second.seed()
    _write(_dirs(second)["AITER_JIT_DIR"], "module_b.so", b"b")

    second.collect()

    (kind,) = second.report(NO_BUILDS).kinds
    assert kind.state == HIT and kind.seed_files == 1
    assert kind.contribution is not None and (kind.contribution.files, kind.contribution.bytes) == (1, 1)
    assert _listed(tmp_path, "two", "aiter_jit") == ["module_a.so", "module_b.so"], "the list is the whole snapshot"
    staged = (tmp_path / "two" / "out" / "aiter_jit" / "manifest.json").read_text()
    assert kind.seed_snapshot is not None and kind.seed_snapshot in staged, "and it names the snapshot it extends"
    assert [p.name for p in (tmp_path / "two" / "out" / "aiter_jit" / "blobs").iterdir()] == [
        hashlib.sha256(b"b").hexdigest()
    ]


def test_only_the_files_a_kind_admits_are_contributed(store, tmp_path) -> None:
    session = _session(store, tmp_path)
    session.seed()
    dirs = _dirs(session)
    _write(dirs["AITER_JIT_DIR"], "module_gemm.so", MODULE)
    _write(dirs["AITER_JIT_DIR"], "build/module_gemm/build/x.so", b"object state")
    _write(dirs["AITER_JIT_DIR"], "notes.txt", b"text")
    _write(dirs["AITER_JIT_DIR"], "module_gemm.o", b"object")
    _write(dirs["AITER_ROOT_DIR"], "build/pa_x/lib.so", b"lib")
    _write(dirs["AITER_ROOT_DIR"], "build/pa_x/build.ninja", b"ninja")
    _write(dirs["AITER_ROOT_DIR"], "build/pa_x/lock", b"")
    _write(dirs["TRITON_CACHE_DIR"], f"{HASH_DIR}/kernel.hsaco", b"hsaco")
    _write(dirs["TRITON_CACHE_DIR"], f"{HASH_DIR}/kernel.autotune.json", b"{}")
    _write(dirs["TRITON_CACHE_DIR"], f"{HASH_DIR}/kernel.best_config.json", b"{}")
    _write(dirs["TRITON_CACHE_DIR"], f"{HASH_DIR}/kernel.pkl", b"pickle")
    _write(dirs["TRITON_CACHE_DIR"], "short/kernel.hsaco", b"hsaco")

    session.collect()

    assert {kind: _listed(tmp_path, "t", kind) for kind in ("aiter_jit", "aiter_cpp", "triton")} == {
        "aiter_jit": ["module_gemm.so"],
        "aiter_cpp": ["pa_x/lib.so"],
        "triton": [f"{HASH_DIR}/kernel.hsaco"],
    }


def test_a_symlink_is_neither_followed_nor_contributed(store, tmp_path) -> None:
    session = _session(store, tmp_path, plans=[KindPlan(AITER_JIT, KEY)])
    session.seed()
    outside = _write(tmp_path, "outside/module_secret.so", b"not the test's")
    directory = _dirs(session)["AITER_JIT_DIR"]
    (directory / "module_secret.so").symlink_to(outside)
    _write(directory, "module_real.so", b"real")

    session.collect()

    (kind,) = session.report(NO_BUILDS).kinds
    assert kind.contribution is not None and kind.contribution.files == 1
    assert not (tmp_path / "t" / "out" / "aiter_jit" / "blobs" / hashlib.sha256(b"not the test's").hexdigest()).exists()


def test_the_prebuilt_modules_of_the_image_are_linked_in_and_a_cached_module_of_the_same_name_wins(
    store, tmp_path
) -> None:
    prebuilt = tmp_path / "image" / "jit"
    _write(prebuilt, "module_rmsnorm.so", b"prebuilt rmsnorm")
    _write(prebuilt, "module_gemm.so", b"prebuilt gemm")
    _write(prebuilt, "README", b"not a module")
    seed = _session(store, tmp_path, "seed", plans=[KindPlan(AITER_JIT, KEY)])
    seed.seed()
    _write(_dirs(seed)["AITER_JIT_DIR"], "module_gemm.so", MODULE)
    seed.collect()
    _publish(store, seed, tmp_path, "seed")

    session = _session(store, tmp_path, plans=[KindPlan(AITER_JIT, KEY, prebuilt=prebuilt)])
    session.seed()

    directory = _dirs(session)["AITER_JIT_DIR"]
    assert (directory / "module_rmsnorm.so").is_symlink() and (
        directory / "module_rmsnorm.so"
    ).read_bytes() == b"prebuilt rmsnorm"
    assert not (directory / "module_gemm.so").is_symlink() and (directory / "module_gemm.so").read_bytes() == MODULE
    assert not (directory / "README").exists()
    session.collect()
    assert session.report(NO_BUILDS).kinds[0].contribution is None, "a link to the image's module is never contributed"
    assert (prebuilt / "module_rmsnorm.so").read_bytes() == b"prebuilt rmsnorm", "the image is never written"


def test_a_file_the_test_changed_under_a_name_the_snapshot_holds_is_not_contributed(store, tmp_path) -> None:
    first = _session(store, tmp_path, "one", plans=[KindPlan(AITER_JIT, KEY)])
    first.seed()
    _write(_dirs(first)["AITER_JIT_DIR"], "module_a.so", b"original")
    first.collect()
    _publish(store, first, tmp_path, "one")
    second = _session(store, tmp_path, "two", plans=[KindPlan(AITER_JIT, KEY)])
    second.seed()
    _write(_dirs(second)["AITER_JIT_DIR"], "module_a.so", b"rebuilt differently")
    _write(_dirs(second)["AITER_JIT_DIR"], "module_b.so", b"new")

    second.collect()

    report = second.report(NO_BUILDS)
    assert any("module_a.so was changed by the test; the stored one stays" in note for note in report.notes)
    staged = (tmp_path / "two" / "out" / "aiter_jit" / "manifest.json").read_text()
    assert (
        hashlib.sha256(b"original").hexdigest() in staged
        and hashlib.sha256(b"rebuilt differently").hexdigest() not in staged
    )
    assert report.kinds[0].contribution.files == 1, "only the new module counts as new"


def test_a_tampered_cache_is_dropped_with_a_note_and_the_test_builds_cold_and_still_contributes(
    store, tmp_path
) -> None:
    first = _session(store, tmp_path, "one", plans=[KindPlan(AITER_JIT, KEY)])
    first.seed()
    _write(_dirs(first)["AITER_JIT_DIR"], "module_a.so", MODULE)
    first.collect()
    _publish(store, first, tmp_path, "one")
    blob = store.root / "blobs" / hashlib.sha256(MODULE).hexdigest()[:2] / hashlib.sha256(MODULE).hexdigest()
    blob.chmod(0o644)
    blob.write_bytes(bytes(reversed(MODULE)))

    second = _session(store, tmp_path, "two", plans=[KindPlan(AITER_JIT, KEY)])
    second.seed()

    directory = _dirs(second)["AITER_JIT_DIR"]
    report = second.report(NO_BUILDS)
    assert report.kinds[0].state == VERIFY_FAILED and report.kinds[0].seed_files == 0
    assert any(
        "module_a.so: the bytes are not the 16 bytes whose hash is listed; ignored" in note for note in report.notes
    )
    assert list(directory.iterdir()) == [], "no file of the tampered cache reaches the server"
    _write(directory, "module_a.so", MODULE)  # the test builds it cold
    second.collect()
    assert second.report(NO_BUILDS).kinds[0].contribution is not None


def test_a_store_that_cannot_be_read_is_a_note_and_a_cold_start_not_a_failed_test(tmp_path) -> None:
    class Unreadable(DirJitStore):
        def fetch(self, kind, key, dest):
            raise PermissionError("denied")

    session = _session(Unreadable(tmp_path / "store"), tmp_path, plans=[KindPlan(AITER_JIT, KEY)])

    session.seed()

    report = session.report(NO_BUILDS)
    assert report.kinds[0].state == VERIFY_FAILED
    assert report.notes == ("jit cache: aiter_jit could not be read (PermissionError); it builds cold",)


def test_a_kind_that_is_switched_off_is_reported_with_its_reason_and_gets_no_directory(store, tmp_path) -> None:
    session = _session(
        store, tmp_path, plans=[KindPlan(AITER_JIT, KEY)], disabled={"triton": "the sheet sets TRITON_CACHE_DIR itself"}
    )

    session.seed()

    report = session.report(NO_BUILDS)
    assert [(k.kind, k.state, k.key) for k in report.kinds] == [("aiter_jit", MISS, KEY), ("triton", DISABLED, None)]
    assert "TRITON_CACHE_DIR" not in session.environment()
    assert "jit cache: triton is not used: the sheet sets TRITON_CACHE_DIR itself" in report.notes


def test_a_snapshot_over_the_size_a_snapshot_may_have_is_not_contributed(store, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jit_session, "MAX_SNAPSHOT_BYTES", 3)
    session = _session(store, tmp_path, plans=[KindPlan(AITER_JIT, KEY)])
    session.seed()
    _write(_dirs(session)["AITER_JIT_DIR"], "module_a.so", b"four")

    session.collect()

    report = session.report(NO_BUILDS)
    assert report.kinds[0].contribution is None
    assert "jit cache: aiter_jit is over the size a snapshot may have; not contributed" in report.notes
    assert not (tmp_path / "t" / "out" / "aiter_jit").exists()


def test_seeding_again_starts_from_an_empty_directory(store, tmp_path) -> None:
    session = _session(store, tmp_path, plans=[KindPlan(AITER_JIT, KEY)])
    session.seed()
    _write(_dirs(session)["AITER_JIT_DIR"], "module_left.so", b"left by an earlier run")

    session.seed()

    assert list(_dirs(session)["AITER_JIT_DIR"].iterdir()) == []


def test_tests_that_run_at_once_each_work_in_their_own_directories_and_contribute_only_their_own(
    store, tmp_path
) -> None:
    sessions = [_session(store, tmp_path, f"t{n}", plans=[KindPlan(AITER_JIT, KEY)]) for n in range(6)]

    def play(number: int, session: JitCacheSession) -> None:
        session.seed()
        _write(_dirs(session)["AITER_JIT_DIR"], f"module_{number}.so", f"own {number}".encode())
        session.collect()

    threads = [threading.Thread(target=play, args=(n, s)) for n, s in enumerate(sessions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for number, session in enumerate(sessions):
        assert sorted(p.name for p in _dirs(session)["AITER_JIT_DIR"].iterdir()) == [f"module_{number}.so"]
        (kind,) = session.report(NO_BUILDS).kinds
        assert kind.contribution is not None and kind.contribution.files == 1


def test_the_report_carries_what_the_logs_say_aiter_built(store, tmp_path) -> None:
    log = tmp_path / "files" / "variant_0" / "server.log"
    _write(
        log.parent,
        "server.log",
        b"[aiter] start build [module_x] under /b\n\x1b[32mfinish build [module_x], cost 12.5s \x1b[0m\n",
    )
    session = _session(store, tmp_path, plans=[KindPlan(AITER_JIT, KEY)])

    report = session.report(scan_builds([log]))

    assert report.builds_seen == 1 and report.compile_seconds == 12.5
    assert report.document()["jit_compile_sec"] == 12.5 and report.document()["builds_seen"] == 1


def test_a_kind_the_key_does_not_cover_is_given_the_cache_and_stages_nothing(store, tmp_path) -> None:
    first = _session(store, tmp_path, "one", plans=[KindPlan(AITER_JIT, KEY)])
    first.seed()
    _write(_dirs(first)["AITER_JIT_DIR"], "module_a.so", b"a")
    first.collect()
    _publish(store, first, tmp_path, "one")
    reason = "the test carries patched code that the key does not cover"
    second = _session(store, tmp_path, "two", plans=[KindPlan(AITER_JIT, KEY, read_only=reason)])
    second.seed()
    _write(_dirs(second)["AITER_JIT_DIR"], "module_b.so", b"b")

    second.collect()

    report = second.report(NO_BUILDS)
    assert report.kinds[0].state == HIT and (_dirs(second)["AITER_JIT_DIR"] / "module_a.so").read_bytes() == b"a"
    assert report.kinds[0].contribution is None and not (tmp_path / "two" / "out").exists()
    assert f"jit cache: aiter_jit is read only in this test: {reason}" in report.notes
