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
from pathlib import Path

import pytest

from hyperloom.measure.jit import serve
from hyperloom.measure.jit.serve import WantRefused
from hyperloom.measure.jit.store import DirJitStore, Snapshot, StoredFile
from hyperloom.measure.jit_kinds import AITER_JIT, TRITON
from hyperloom.sandbox.socket import DirEntry, SandboxGone

KEY = "b" * 64
MODULE = b"\x7fELF a module"


class _DirSandbox:
    """A sandbox whose files are a directory: the file API reads through links, the listing does not."""

    sandbox_id = "sbx"

    def __init__(self, root: Path, *, fail_writes: bool = False) -> None:
        self.root, self.fail_writes, self.writes = root, fail_writes, []

    def _at(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def write_file(self, path: str, data: bytes) -> None:
        self.writes.append(path)
        if self.fail_writes and not path.endswith("DONE"):
            raise SandboxGone("gone")
        self._at(path).parent.mkdir(parents=True, exist_ok=True)
        self._at(path).write_bytes(data)

    def read_file(self, path: str) -> bytes:
        return self._at(path).read_bytes()

    def copy_out(self, path: str, dest: Path) -> int:
        data = self._at(path).read_bytes()
        dest.write_bytes(data)
        return len(data)

    def list_entries(self, path: str) -> list[DirEntry]:
        found = []
        for child in sorted(self._at(path).iterdir()):
            kind = "link" if child.is_symlink() else "dir" if child.is_dir() else "file"
            found.append(DirEntry(child.name, kind, child.lstat().st_size))
        return found


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


def _sandbox(tmp_path: Path, **kwargs) -> _DirSandbox:
    for directory in ("in", "out"):
        (tmp_path / directory).mkdir(exist_ok=True)
    return _DirSandbox(tmp_path, **kwargs)


def test_the_host_serves_what_the_runner_asked_for_and_the_runner_reads_it(tmp_path) -> None:
    store, snapshot = _store(tmp_path / "host", {"module_a.so": MODULE})
    sandbox = _sandbox(tmp_path)
    (tmp_path / "out" / "jit-want.json").write_bytes(serve.want_document([(AITER_JIT, KEY), (TRITON, "c" * 64)]))

    assert serve.serve(store, sandbox, "req") == 1
    assert sandbox.writes == ["/in/jit-cache/aiter_jit.tar", "/in/jit-cache/DONE"], "triton has nothing stored"

    phases, notes = _asked(tmp_path, [(AITER_JIT, KEY), (TRITON, "c" * 64)])
    assert phases == [serve.WAIT_PHASE] and "aiter_jit 2 files" in notes[-1]
    assert not (tmp_path / "in" / "jit-cache" / "aiter_jit.tar").exists(), "the served tar is removed once unpacked"
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

    assert "aiter_jit 2 files" in notes[-1] and (tmp_path / "in" / "jit-cache" / "aiter_jit.tar").exists()
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
    (tmp_path / "in" / "jit-cache" / "aiter_jit.tar").write_bytes(buffer.getvalue())
    (tmp_path / "in" / "jit-cache" / "DONE").write_bytes(b"")

    _asked(tmp_path, [(AITER_JIT, KEY)])

    fetched = DirJitStore(tmp_path / "served").fetch(AITER_JIT, KEY, tmp_path / "private")
    assert fetched.snapshot is None and any("missing" in problem for problem in fetched.problems)


def test_a_runner_is_offered_the_cache_only_when_the_host_said_so(tmp_path) -> None:
    assert not serve.offered(tmp_path)
    (tmp_path / "jit-cache").mkdir()
    (tmp_path / "jit-cache" / "SERVING").write_bytes(b"")
    assert serve.offered(tmp_path)
