# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serving the compiled-code cache to a sandboxed runner by a handshake: what the host reads from the sandbox, what it
packs from its store, and what the runner unpacks, each treated as untrusted by the side that reads it."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hyperloom.measure.jit import serve
from hyperloom.measure.jit.serve import WantRefused
from hyperloom.measure.jit.store import DirJitStore, Snapshot, StoredFile
from hyperloom.measure.jit_kinds import AITER_JIT, FLYDSL, TRITON
from hyperloom.measure.jit.tests.dir_sandbox import PLATFORM_WRITE_LIMIT, DirSandbox

KEY = "b" * 64
MODULE = b"\x7fELF a module"


def _store(tmp_path: Path, files: dict[str, bytes], kind=AITER_JIT, key: str = KEY) -> tuple[DirJitStore, Snapshot]:
    staging = tmp_path / f"staging-{kind.name}"
    (staging / "blobs").mkdir(parents=True)
    listed = {}
    for relative, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        (staging / "blobs" / sha).write_bytes(data)
        listed[relative] = StoredFile(sha, len(data))
    snapshot = Snapshot(kind.name, key, None, listed)
    (staging / "manifest.json").write_bytes(snapshot.to_bytes())
    store = DirJitStore(tmp_path / "store")
    store.publish(staging, snapshot.sha256())
    return store, snapshot


def _tar(tmp_path: Path, *members: tarfile.TarInfo, data: bytes = b"x") -> Path:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for member in members:
            archive.addfile(member, io.BytesIO(data) if member.isreg() else None)
    path = tmp_path / "served.tar"
    path.write_bytes(buffer.getvalue())
    return path


def _member(name: str, kind: bytes = tarfile.REGTYPE, size: int = 1, link: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type, info.size, info.linkname = kind, size if kind == tarfile.REGTYPE else 0, link
    return info


def _blob_name(data: bytes = b"x") -> str:
    sha = hashlib.sha256(data).hexdigest()
    return f"blobs/{sha[:2]}/{sha}"


# -- the want ------------------------------------------------------------------------------------------------------


def test_the_want_the_runner_writes_is_the_one_the_host_reads() -> None:
    raw = serve.want_document([(AITER_JIT, KEY), (TRITON, "c" * 64)])

    assert serve.parse_want(raw) == [(AITER_JIT, KEY), (TRITON, "c" * 64)]


@pytest.mark.parametrize(
    "document",
    [
        {"schema": "other", "want": []},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "nope", "key": KEY}]},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "aiter_jit", "key": "../../etc/passwd"}]},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "aiter_jit", "key": KEY.upper()}]},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "aiter_jit", "key": KEY, "path": "/"}]},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "aiter_jit", "key": KEY}] * 2},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [{"kind": "triton", "key": KEY}] * 5},
        {"schema": "hyperloom.jit-cache-want/v1", "want": {"kind": "aiter_jit"}},
        {"schema": "hyperloom.jit-cache-want/v1", "want": [], "more": 1},
    ],
)
def test_a_want_that_is_not_known_kinds_and_sha256_keys_is_refused(document) -> None:
    with pytest.raises(WantRefused):
        serve.parse_want(json.dumps(document).encode())


def test_a_want_over_its_bound_is_refused_before_it_is_parsed() -> None:
    with pytest.raises(WantRefused, match="larger"):
        serve.parse_want(b" " * (serve.MAX_WANT_BYTES + 1))


# -- packing on the host and unpacking in the sandbox ---------------------------------------------------------------


def test_a_packed_snapshot_unpacks_into_a_store_the_session_reads_and_proves(tmp_path) -> None:
    store, snapshot = _store(tmp_path, {"module_a.so": MODULE, "module_b.so": b"other"})
    packed = serve.pack(store, AITER_JIT, KEY)
    assert packed is not None and packed.snapshot == snapshot.sha256() and packed.files == 2
    (tmp_path / "a.tar").write_bytes(packed.data)

    assert serve.unpack(tmp_path / "a.tar", tmp_path / "served", AITER_JIT, KEY) == 3

    fetched = DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private")
    assert fetched.snapshot == snapshot and (tmp_path / "private" / "module_a.so").read_bytes() == MODULE


def test_the_host_serves_no_stored_blob_that_does_not_prove(tmp_path) -> None:
    store, snapshot = _store(tmp_path, {"module_a.so": MODULE})
    blob = store.root / "blobs" / snapshot.files["module_a.so"].sha256[:2] / snapshot.files["module_a.so"].sha256
    blob.chmod(0o600)
    blob.write_bytes(b"\x7fELF a modulE")

    assert serve.pack(store, AITER_JIT, KEY) is None


def test_the_host_follows_no_link_in_its_store(tmp_path) -> None:
    store, snapshot = _store(tmp_path, {"module_a.so": MODULE})
    sha = snapshot.files["module_a.so"].sha256
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(MODULE)
    blob = store.root / "blobs" / sha[:2] / sha
    blob.unlink()
    blob.symlink_to(elsewhere)

    assert serve.pack(store, AITER_JIT, KEY) is None


def test_a_snapshot_over_the_bound_of_one_upload_is_not_served_and_said_once_at_warning(
    tmp_path, monkeypatch, caplog
) -> None:
    store, _ = _store(tmp_path, {"module_a.so": MODULE})
    monkeypatch.setattr(serve, "MAX_TAR_BYTES", len(MODULE) - 1)
    monkeypatch.setattr(serve, "_TOO_LARGE", set())

    assert serve.pack(store, AITER_JIT, KEY) is None
    assert serve.pack(store, AITER_JIT, KEY) is None
    assert serve.pack(store, AITER_JIT, "c" * 64) is None, "nothing stored under that key"
    (warning,) = [record for record in caplog.records if "too large to serve" in record.getMessage()]
    assert warning.levelname == "WARNING" and f"aiter_jit key {KEY[:12]}" in warning.getMessage()


def test_a_tar_at_its_bound_is_served_and_one_byte_over_is_not(tmp_path, monkeypatch) -> None:
    store, _ = _store(tmp_path, {"module_a.so": MODULE})
    packed = serve.pack(store, AITER_JIT, KEY)
    assert packed is not None and serve.MAX_TAR_BYTES == 1024**3
    monkeypatch.setattr(serve, "MAX_TAR_BYTES", len(packed.data))
    assert serve.pack(store, AITER_JIT, KEY) == packed
    (tmp_path / "a.tar").write_bytes(packed.data)
    assert serve.unpack(tmp_path / "a.tar", tmp_path / "served", AITER_JIT, KEY) == 2

    monkeypatch.setattr(serve, "MAX_TAR_BYTES", len(packed.data) - 1)
    assert serve.pack(store, AITER_JIT, KEY) is None
    with pytest.raises(WantRefused, match="larger"):
        serve.unpack(tmp_path / "a.tar", tmp_path / "served", AITER_JIT, KEY)


def test_the_tar_is_built_in_a_temp_file_that_is_gone_on_every_path(tmp_path, monkeypatch) -> None:
    store, snapshot = _store(tmp_path, {"module_a.so": MODULE})
    spools, temp_dir, made = [], tmp_path / "tmp", tempfile.TemporaryFile

    def tracked(*args, **kwargs):
        spools.append(made(*args, **kwargs))
        return spools[-1]

    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    monkeypatch.setattr(tempfile, "TemporaryFile", tracked)
    bound = serve.MAX_TAR_BYTES

    assert serve.pack(store, AITER_JIT, KEY) is not None
    monkeypatch.setattr(serve, "MAX_TAR_BYTES", len(MODULE))
    assert serve.pack(store, AITER_JIT, KEY) is None, "over the bound once built"
    monkeypatch.setattr(serve, "MAX_TAR_BYTES", bound)
    blob = store.root / "blobs" / snapshot.files["module_a.so"].sha256[:2] / snapshot.files["module_a.so"].sha256
    blob.chmod(0o600)
    blob.write_bytes(b"\x7fELF a modulE")
    assert serve.pack(store, AITER_JIT, KEY) is None, "a blob that does not prove"

    assert len(spools) == 3 and all(spool.closed for spool in spools)
    assert not any(temp_dir.iterdir())


@pytest.mark.parametrize(
    "member",
    [
        _member("../escape"),
        _member("/abs/blobs/x"),
        _member(f"{_blob_name()}/../../escape"),
        _member("snapshots/aiter_jit/" + "c" * 64 + "/" + "d" * 64 + ".json"),  # another key's list
        _member("snapshots/triton/" + KEY + "/" + "d" * 64 + ".json"),  # another kind's list
        _member(_blob_name(), tarfile.SYMTYPE, link="/etc/passwd"),
        _member(_blob_name(), tarfile.LNKTYPE, link="blobs/x"),
        _member(_blob_name(), tarfile.CHRTYPE),
        _member(_blob_name(), tarfile.FIFOTYPE),
        _member("blobs", tarfile.DIRTYPE),
    ],
    ids=["dotdot", "absolute", "nested", "other-key", "other-kind", "symlink", "hardlink", "device", "fifo", "dir"],
)
def test_a_served_member_that_is_not_a_plain_file_of_the_stores_layout_is_refused(tmp_path, member) -> None:
    tar = _tar(tmp_path, member)
    dest = tmp_path / "in" / "served"

    with pytest.raises(WantRefused):
        serve.unpack(tar, dest, AITER_JIT, KEY)

    assert not (tmp_path / "escape").exists() and not (tmp_path / "in" / "escape").exists()
    assert not dest.exists() or not any(path.is_symlink() for path in dest.rglob("*"))


def test_a_served_tar_over_its_bounds_is_refused(tmp_path, monkeypatch) -> None:
    tar = _tar(tmp_path, _member(_blob_name(b"x" * 8), size=8), data=b"x" * 8)
    monkeypatch.setattr(serve, "MAX_TAR_BYTES", 4)
    with pytest.raises(WantRefused, match="larger"):
        serve.unpack(tar, tmp_path / "served", AITER_JIT, KEY)

    monkeypatch.setattr(serve, "MAX_TAR_BYTES", len(tar.read_bytes()))
    monkeypatch.setattr(serve, "MAX_SNAPSHOT_BYTES", 4)
    with pytest.raises(WantRefused, match="more bytes"):
        serve.unpack(tar, tmp_path / "served", AITER_JIT, KEY)


def test_a_served_tar_with_too_many_files_is_refused(tmp_path, monkeypatch) -> None:
    names = [_blob_name(bytes([n])) for n in range(3)]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for n, name in enumerate(names):
            archive.addfile(_member(name), io.BytesIO(bytes([n])))
    (tmp_path / "many.tar").write_bytes(buffer.getvalue())
    monkeypatch.setattr(serve, "MAX_TAR_FILES", 2)

    with pytest.raises(WantRefused, match="more than 2 files"):
        serve.unpack(tmp_path / "many.tar", tmp_path / "served", AITER_JIT, KEY)


def test_a_served_tar_that_is_a_link_is_not_read(tmp_path) -> None:
    real = _tar(tmp_path, _member(_blob_name()))
    (tmp_path / "link.tar").symlink_to(real)

    with pytest.raises(WantRefused, match="not a regular file"):
        serve.unpack(tmp_path / "link.tar", tmp_path / "served", AITER_JIT, KEY)


# -- the handshake from both sides ---------------------------------------------------------------------------------


def _asked(tmp_path: Path, wanted) -> tuple[list[str], list[str]]:
    phases: list[str] = []
    notes = serve.ask(tmp_path / "in", tmp_path / "out", wanted, tmp_path / "served", phases.append)
    return phases, notes


def _sandbox(tmp_path: Path, **kwargs) -> DirSandbox:
    for directory in ("in", "out"):
        (tmp_path / directory).mkdir(exist_ok=True)
    return DirSandbox(tmp_path, **kwargs)


def test_the_host_serves_what_the_runner_asked_for_and_the_runner_reads_it(tmp_path) -> None:
    store, snapshot = _store(tmp_path / "host", {"module_a.so": MODULE})
    sandbox = _sandbox(tmp_path)
    (tmp_path / "out" / "jit-want.json").write_bytes(serve.want_document([(AITER_JIT, KEY), (TRITON, "c" * 64)]))

    assert serve.serve(store, sandbox, "req") == 1
    assert sandbox.writes == [
        "/in/jit-cache/aiter_jit.tar.000",
        "/in/jit-cache/aiter_jit.tar.json",
        "/in/jit-cache/DONE",
    ], "triton has nothing stored"

    phases, notes = _asked(tmp_path, [(AITER_JIT, KEY), (TRITON, "c" * 64)])
    assert phases == [serve.WAIT_PHASE] and "aiter_jit 2 files" in notes[-1]
    assert sorted(path.name for path in (tmp_path / "in" / "jit-cache").iterdir()) == ["DONE"], "removed once unpacked"
    served = DirJitStore(tmp_path / "served")
    assert served.fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == snapshot
    assert served.fetch(TRITON, "c" * 64, tmp_path / "private-triton").snapshot is None, "missing: it builds cold"


def test_a_served_tar_the_runner_cannot_remove_from_its_input_directory_does_not_fail_the_test(
    tmp_path, monkeypatch
) -> None:
    """An input directory the runner may only read: the tar stays, the cache is still read."""
    store, snapshot = _store(tmp_path / "host", {"module_a.so": MODULE})
    sandbox = _sandbox(tmp_path)
    (tmp_path / "out" / "jit-want.json").write_bytes(serve.want_document([(AITER_JIT, KEY)]))
    assert serve.serve(store, sandbox, "req") == 1
    unlink = Path.unlink

    def read_only(path: Path, missing_ok: bool = False) -> None:
        if path.parent == tmp_path / "in" / "jit-cache":
            raise PermissionError(13, "Permission denied")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", read_only)

    _, notes = _asked(tmp_path, [(AITER_JIT, KEY)])

    assert "aiter_jit 2 files" in notes[-1] and (tmp_path / "in" / "jit-cache" / "aiter_jit.tar.json").exists()
    assert DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == snapshot


@pytest.mark.parametrize("kind", ["oversize", "link", "missing", "garbage"])
def test_a_want_the_host_cannot_take_serves_nothing_and_still_says_done(tmp_path, kind) -> None:
    store, _ = _store(tmp_path / "host", {"module_a.so": MODULE})
    sandbox = _sandbox(tmp_path)
    want = tmp_path / "out" / "jit-want.json"
    if kind == "oversize":
        want.write_bytes(b" " * (serve.MAX_WANT_BYTES + 1))
    elif kind == "link":
        (tmp_path / "secret").write_bytes(serve.want_document([(AITER_JIT, KEY)]))
        want.symlink_to(tmp_path / "secret")
    elif kind == "garbage":
        want.write_bytes(b"{not json")

    assert serve.serve(store, sandbox, "req") == 0
    assert sandbox.writes == ["/in/jit-cache/DONE"]


def test_a_sandbox_that_fails_while_served_is_still_told_done(tmp_path) -> None:
    store, _ = _store(tmp_path / "host", {"module_a.so": MODULE})
    sandbox = _sandbox(tmp_path, fail_writes=True)
    (tmp_path / "out" / "jit-want.json").write_bytes(serve.want_document([(AITER_JIT, KEY)]))

    assert serve.serve(store, sandbox, "req") == 0
    assert sandbox.writes[-1] == "/in/jit-cache/DONE"


def test_a_runner_the_host_never_answers_goes_on_cold(tmp_path, monkeypatch) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(serve, "WAIT_SEC", 0.1)

    phases, notes = _asked(tmp_path, [(AITER_JIT, KEY)])

    assert phases == [serve.WAIT_PHASE] and "did not serve" in notes[-1]
    assert DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private").snapshot is None


def test_a_served_snapshot_missing_a_blob_falls_back_to_cold(tmp_path) -> None:
    """A tar cut short (or a host that lost a blob) leaves a list whose files are not all there: the read proves it and
    the kind builds cold."""
    _, snapshot = _store(tmp_path / "host", {"module_a.so": MODULE})
    (tmp_path / "in" / "jit-cache").mkdir(parents=True)
    (tmp_path / "out").mkdir()
    name = f"snapshots/aiter_jit/{KEY}/{snapshot.sha256()}.json"
    raw = snapshot.to_bytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        archive.addfile(_member(name, size=len(raw)), io.BytesIO(raw))
    _serve_raw(tmp_path / "in" / "jit-cache", AITER_JIT, buffer.getvalue())
    (tmp_path / "in" / "jit-cache" / "DONE").write_bytes(b"")

    _asked(tmp_path, [(AITER_JIT, KEY)])

    fetched = DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private")
    assert fetched.snapshot is None and any("missing" in problem for problem in fetched.problems)


def _serve_raw(inbox: Path, kind, data: bytes) -> list[str]:
    """Write ``data`` as the host writes a kind's tar (its parts, then its index); returns the names written."""
    names = []
    for name, chunk in serve.served_files(kind, serve.Packed(data, "", 0)):
        (inbox / name).write_bytes(chunk)
        names.append(name)
    return names


def _two_kinds(tmp_path: Path) -> tuple[DirJitStore, Snapshot, Snapshot]:
    """A store with an aiter snapshot of two modules and a Triton one of one kernel, under :data:`KEY` both."""
    host = tmp_path / "host"
    store, aiter = _store(host, {"module_a.so": MODULE, "module_b.so": b"m" * 3000})
    _, triton = _store(host, {"a" * 32 + "/kernel.hsaco": b"k" * 9000}, kind=TRITON)
    return store, aiter, triton


def _want(tmp_path: Path, *kinds) -> list:
    wanted = [(kind, KEY) for kind in kinds]
    (tmp_path / "out" / "jit-want.json").write_bytes(serve.want_document(wanted))
    return wanted


def test_a_tar_larger_than_a_part_is_written_in_numbered_parts_and_joined_by_the_runner(tmp_path, monkeypatch) -> None:
    store, aiter, _ = _two_kinds(tmp_path)
    sandbox = _sandbox(tmp_path)
    wanted = _want(tmp_path, AITER_JIT)
    monkeypatch.setattr(serve, "PART_BYTES", 2048)

    assert serve.serve(store, sandbox, "req") == 1

    names = [path.rsplit("/", 1)[1] for path in sandbox.writes]
    parts = [name for name in names if name.startswith("aiter_jit.tar.0")]
    assert len(parts) > 2 and parts == [f"aiter_jit.tar.{index:03d}" for index in range(len(parts))]
    assert names == [*parts, "aiter_jit.tar.json", "DONE"], "the index after every part, DONE last"
    assert all((tmp_path / "in" / "jit-cache" / part).stat().st_size <= 2048 for part in parts)
    _, notes = _asked(tmp_path, wanted)
    assert "aiter_jit 3 files" in notes[-1]
    assert DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == aiter
    assert sorted(path.name for path in (tmp_path / "in" / "jit-cache").iterdir()) == ["DONE"]


@pytest.mark.parametrize("damage", ["missing", "short", "swapped", "link", "extra-bytes", "index-lies"])
def test_a_kind_whose_parts_do_not_add_up_to_its_index_builds_cold(tmp_path, monkeypatch, damage) -> None:
    store, _, triton = _two_kinds(tmp_path)
    sandbox = _sandbox(tmp_path)
    wanted = _want(tmp_path, AITER_JIT, TRITON)
    monkeypatch.setattr(serve, "PART_BYTES", 2048)
    assert serve.serve(store, sandbox, "req") == 2
    inbox = tmp_path / "in" / "jit-cache"
    second = inbox / "aiter_jit.tar.001"
    if damage == "missing":
        second.unlink()
    elif damage == "short":
        second.write_bytes(second.read_bytes()[:-1])
    elif damage == "swapped":
        first = (inbox / "aiter_jit.tar.000").read_bytes()
        (inbox / "aiter_jit.tar.000").write_bytes(second.read_bytes())
        second.write_bytes(first)
    elif damage == "link":
        (tmp_path / "elsewhere").write_bytes(second.read_bytes())
        second.unlink()
        second.symlink_to(tmp_path / "elsewhere")
    elif damage == "extra-bytes":
        second.write_bytes(second.read_bytes() + b"x")
    else:
        index = json.loads((inbox / "aiter_jit.tar.json").read_bytes())
        (inbox / "aiter_jit.tar.json").write_text(json.dumps({**index, "parts": 100}))

    _, notes = _asked(tmp_path, wanted)

    assert any(note.startswith("jit cache: aiter_jit: what the host served was not used") for note in notes)
    assert "triton 2 files" in notes[-1] and "aiter_jit" not in notes[-1]
    served = DirJitStore(tmp_path / "served")
    assert served.fetch(AITER_JIT, KEY, tmp_path / "private").snapshot is None, "it builds cold"
    assert served.fetch(TRITON, KEY, tmp_path / "private-triton").snapshot == triton
    assert not any(path.name.startswith(".") for path in tmp_path.iterdir()), "no joined tar is left"


@pytest.mark.parametrize(
    "index",
    [
        {"schema": "other", "parts": 1, "size": 1, "sha256": KEY},
        {"schema": "hyperloom.jit-cache-parts/v1", "parts": 1, "size": 0, "sha256": KEY},
        {"schema": "hyperloom.jit-cache-parts/v1", "parts": 1, "size": 1024**3 + 1, "sha256": KEY},
        {"schema": "hyperloom.jit-cache-parts/v1", "parts": 2, "size": 1, "sha256": KEY},
        {"schema": "hyperloom.jit-cache-parts/v1", "parts": 1, "size": 1, "sha256": "../x"},
        {"schema": "hyperloom.jit-cache-parts/v1", "parts": 1, "size": 1, "sha256": KEY, "path": "/"},
    ],
)
def test_an_index_that_is_not_the_one_the_host_writes_is_refused(tmp_path, index) -> None:
    (tmp_path / "aiter_jit.tar.json").write_text(json.dumps(index))

    with pytest.raises(WantRefused):
        serve._read_index(tmp_path / "aiter_jit.tar.json")


def test_a_kind_the_platform_refuses_to_take_builds_cold_and_the_next_kind_is_still_served(
    tmp_path, monkeypatch, caplog
) -> None:
    """A part over the platform's limit of one write is refused (413): that kind is logged and builds cold, the next kind
    is served all the same, and the runner is told DONE."""
    store, aiter, _ = _two_kinds(tmp_path)
    sandbox = _sandbox(tmp_path, write_limit=16384)
    wanted = _want(tmp_path, TRITON, AITER_JIT)
    monkeypatch.setattr(serve, "PART_BYTES", 32768)  # one part each: Triton's 20 KiB, aiter's 10 KiB
    caplog.set_level("INFO", logger=serve.__name__)

    assert serve.serve(store, sandbox, "req") == 1

    assert sandbox.writes[-1] == "/in/jit-cache/DONE" and "/in/jit-cache/aiter_jit.tar.json" in sandbox.writes
    assert "/in/jit-cache/triton.tar.json" not in sandbox.writes
    (refused,) = [record for record in caplog.records if "triton not served" in record.getMessage()]
    assert refused.levelname == "WARNING" and "SandboxDenied: 413" in refused.getMessage()
    assert any("aiter_jit served: 2 files, 10240 bytes in 1 parts" in record.getMessage() for record in caplog.records)
    _, notes = _asked(tmp_path, wanted)
    assert "aiter_jit 3 files" in notes[-1] and "triton" not in notes[-1]
    assert DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private").snapshot == aiter


def test_a_kind_larger_than_the_platforms_limit_of_one_write_is_served_in_parts_under_it(tmp_path) -> None:
    """The case the platform refused when a kind went as one tar: a snapshot past 32 MB, and a small kind beside it."""
    big = bytes(range(256)) * (36 * 1024**2 // 256)
    host = tmp_path / "host"
    store, triton = _store(host, {"a" * 32 + "/kernel.hsaco": big}, kind=TRITON)
    _, flydsl = _store(host, {"add_" + "0" * 32 + "/" + "1" * 16 + ".pkl": MODULE}, kind=FLYDSL)
    sandbox = _sandbox(tmp_path, write_limit=PLATFORM_WRITE_LIMIT)
    wanted = _want(tmp_path, TRITON, FLYDSL)

    assert serve.PART_BYTES < PLATFORM_WRITE_LIMIT
    assert serve.serve(store, sandbox, "req") == 2

    assert sum(path.startswith("/in/jit-cache/triton.tar.0") for path in sandbox.writes) == 2
    _, notes = _asked(tmp_path, wanted)
    assert "triton 2 files" in notes[-1] and "flydsl 2 files" in notes[-1]
    served = DirJitStore(tmp_path / "served")
    assert served.fetch(TRITON, KEY, tmp_path / "private").snapshot == triton
    assert served.fetch(FLYDSL, KEY, tmp_path / "private-flydsl").snapshot == flydsl


def test_a_runner_keeps_waiting_while_parts_keep_coming(tmp_path, monkeypatch) -> None:
    inbox = tmp_path / "in" / "jit-cache"
    inbox.mkdir(parents=True)
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(serve, "WAIT_SEC", 0.6)

    def host() -> None:
        for index in range(5):
            time.sleep(0.3)
            (inbox / f"aiter_jit.tar.{index:03d}").write_bytes(b"")
        (inbox / "DONE").write_bytes(b"")

    writer = threading.Thread(target=host)
    writer.start()
    _, notes = _asked(tmp_path, [(AITER_JIT, KEY)])
    writer.join()

    assert "served by the host" in notes[-1], "1.5s in all, never 0.6s without a new part"


def test_a_runner_stops_waiting_at_the_bound_of_the_whole_wait(tmp_path, monkeypatch) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(serve, "MAX_WAIT_SEC", 0.2)

    _, notes = _asked(tmp_path, [(AITER_JIT, KEY)])

    assert "did not serve" in notes[-1]


def test_a_runner_is_offered_the_cache_only_when_the_host_said_so(tmp_path) -> None:
    assert not serve.offered(tmp_path)
    (tmp_path / "jit-cache").mkdir()
    (tmp_path / "jit-cache" / "SERVING").write_bytes(b"")
    assert serve.offered(tmp_path)
