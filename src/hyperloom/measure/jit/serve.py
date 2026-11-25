# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serving the shared compiled-code cache to a sandboxed runner, so that no sandbox ever sees the store.

The store lies on the host, where no measurement sandbox can read or write it. A cache key is only known inside the
sandbox (it depends on the image's aiter sources and the server's environment), so the host serves a test by a
handshake through the files the two already exchange:

1. The host, when it has a store, writes ``/in/jit-cache/SERVING`` before the sheet's ``READY`` marker.
2. The runner computes its keys and, when it was offered the cache, writes ``/out/jit-want.json`` (the kinds and keys it
   wants), sets its progress phase to :data:`WAIT_PHASE` and waits for ``/in/jit-cache/DONE`` (:data:`WAIT_SEC`).
3. The host sees that phase at a poll, reads the want strictly (:func:`parse_want`), picks for each kind the snapshot a
   reader would pick, proves every blob it lists (a plain file of the store, the listed size and hash) while it packs the
   snapshot list and its blobs, in the store's own layout, into one tar per kind, writes ``/in/jit-cache/<kind>.tar`` with
   the sandbox's ``write_file`` and then ``DONE``, also when it served nothing.
4. The runner unpacks each tar into a directory of its own (:func:`unpack`) that the test's session then reads as its
   store: :meth:`~hyperloom.measure.jit.store.DirJitStore.fetch` proves every byte again before anything is used.

Whatever the sandbox writes is untrusted on the host: the want is bounded in size, its kinds and keys must be known
kinds and sha256 keys, and nothing it names is a path. What the host sends is untrusted on the runner the same way: a
tar member must be a regular file under a name of the store's layout for the kind and key asked for (no link, no device,
no directory, no path of its own), within the size and count bounds. Anything that does not hold is dropped and the test
builds what it lacks itself: the cache costs time when it fails, never a result.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import stat
import tarfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from hyperloom.measure import fsafe
from hyperloom.measure.docs import SHA256_RE, DocumentError, Reader, canonical_json, parse_json_object
from hyperloom.measure.jit.store import (
    BLOBS_DIR,
    MAX_SNAPSHOT_BYTES,
    SNAPSHOTS_DIR,
    DirJitStore,
    Snapshot,
    SnapshotInvalid,
)
from hyperloom.measure.jit_kinds import KINDS, CacheKind, kind_named
from hyperloom.sandbox.socket import Sandbox, SandboxError

__all__ = [
    "DONE_PATH",
    "MAX_TAR_BYTES",
    "MAX_TAR_FILES",
    "MAX_WANT_BYTES",
    "Packed",
    "SERVE_DIR",
    "SERVING_PATH",
    "WAIT_PHASE",
    "WAIT_SEC",
    "WANT_PATH",
    "WantRefused",
    "ask",
    "offered",
    "pack",
    "parse_want",
    "serve",
    "unpack",
    "want_document",
]

log = logging.getLogger(__name__)

#: The sandbox's directory the host serves into, and its name below the runner's input directory.
SERVE_DIR = "/in/jit-cache"
_SERVE_NAME = "jit-cache"
_SERVING, _DONE, _TAR = "SERVING", "DONE", ".tar"
SERVING_PATH, DONE_PATH = f"{SERVE_DIR}/{_SERVING}", f"{SERVE_DIR}/{_DONE}"
#: The runner's list of the kinds and keys it wants, below its output directory.
_WANT_NAME = "jit-want.json"
WANT_PATH = f"/out/{_WANT_NAME}"
_WANT_SCHEMA = "hyperloom.jit-cache-want/v1"
_WANT = "want"
#: The runner's progress phase while it waits to be served: the host serves when it reads it.
WAIT_PHASE = "jit_wait"
#: How long a runner waits to be served before its test builds cold: a host polls every few seconds, and an upload of a
#: few hundred MB takes under a minute per kind.
WAIT_SEC = 300.0
#: The bounds of what crosses: a want is a few hundred bytes; one tar is one ``write_file`` request, which the platform
#: bounds in time, so a kind whose snapshot is larger is not served (it builds cold); the most files one tar holds.
MAX_WANT_BYTES = 16 * 1024
MAX_TAR_BYTES = 256 * 1024**2
MAX_TAR_FILES = 20_000
_BLOB_NAME = re.compile(r"blobs/(?P<prefix>[0-9a-f]{2})/(?P=prefix)[0-9a-f]{62}")
_TAR_MODE = 0o400
#: The only member types a served tar may hold: plain regular files (no link, device, directory, fifo or sparse file).
_REGULAR = (tarfile.REGTYPE, tarfile.AREGTYPE)
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class WantRefused(ValueError):
    """The runner's want is not one the host serves: the message says why."""


# -- the runner's side ---------------------------------------------------------------------------------------------


def offered(in_dir: Path) -> bool:
    """Whether the host offered this runner the cache (``SERVING`` in its input directory)."""
    return (in_dir / _SERVE_NAME / _SERVING).is_file()


def want_document(wanted: Sequence[tuple[CacheKind, str]]) -> bytes:
    """The want the runner writes for ``wanted`` (kind, key) pairs."""
    items = [{"kind": kind.name, "key": key} for kind, key in wanted]
    return canonical_json({"schema": _WANT_SCHEMA, _WANT: items}).encode("utf-8")


def ask(
    in_dir: Path, out_dir: Path, wanted: Sequence[tuple[CacheKind, str]], dest: Path, phase: Callable[[str], None]
) -> list[str]:
    """Ask the host for ``wanted`` and unpack what it serves into ``dest``, the directory the session reads as its store;
    returns the notes for the result.

    Nothing here fails the test: a host that does not answer in :data:`WAIT_SEC`, or a tar that does not unpack, leaves
    the kinds it would have given empty (a note says why), and their tests build cold.
    """
    served, notes, wait_sec = in_dir / _SERVE_NAME, list[str](), WAIT_SEC
    _write_atomic(out_dir / _WANT_NAME, want_document(wanted))
    phase(WAIT_PHASE)
    started = time.monotonic()
    if not _wait_for(served / _DONE, wait_sec):
        notes.append(f"jit cache: the host did not serve the cache within {wait_sec:.0f}s; the test builds cold")
        return notes
    given = []
    for kind, key in wanted:
        tar = served / f"{kind.name}{_TAR}"
        if not os.path.lexists(tar):
            continue
        try:
            files = unpack(tar, dest, kind, key)
        except (WantRefused, tarfile.TarError, OSError, fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
            notes.append(f"jit cache: {kind.name}: what the host served was not used ({exc}); it builds cold")
            continue
        finally:
            tar.unlink(missing_ok=True)  # the copy below the test's directory is what is read from here on
        given.append(f"{kind.name} {files} files")
    notes.append(f"jit cache: served by the host in {time.monotonic() - started:.1f}s: {', '.join(given) or 'nothing'}")
    return notes


def unpack(tar: Path, dest: Path, kind: CacheKind, key: str) -> int:
    """Unpack the tar the host served for ``kind`` under ``key`` into ``dest``; returns the files it held.

    Every member must be a regular file named as the store lays out a blob, or the list of a snapshot of this kind and
    key; the tar, each file and all of them together within the bounds; a file already in ``dest`` is kept (a blob
    another kind brought: its name is its hash). What was written before a refusal stays: the read that follows proves
    every file it uses.

    Raises:
        WantRefused: A member that is not such a file, or a bound that is passed.
        tarfile.TarError: The tar cannot be read.
    """
    facts = os.lstat(tar)
    if not stat.S_ISREG(facts.st_mode):
        raise WantRefused("the tar is not a regular file")
    if facts.st_size > MAX_TAR_BYTES:
        raise WantRefused(f"the tar is larger than {MAX_TAR_BYTES} bytes")
    dest.mkdir(parents=True, exist_ok=True, mode=fsafe.DIR_MODE)
    snapshot_name = re.compile(rf"{SNAPSHOTS_DIR}/{re.escape(kind.name)}/{re.escape(key)}/[0-9a-f]{{64}}\.json")
    count = total = 0
    with tarfile.open(tar, mode="r:") as archive:
        for member in archive:
            count += 1
            total += member.size
            if count > MAX_TAR_FILES:
                raise WantRefused(f"the tar holds more than {MAX_TAR_FILES} files")
            if member.type not in _REGULAR:
                raise WantRefused(f"{member.name!r} is not a regular file")
            if not (_BLOB_NAME.fullmatch(member.name) or snapshot_name.fullmatch(member.name)):
                raise WantRefused(f"{member.name!r} is not a name of the store's layout")
            if member.size > MAX_SNAPSHOT_BYTES or total > MAX_TAR_BYTES:
                raise WantRefused("the tar holds more bytes than it may")
            source = archive.extractfile(member)
            if source is None:
                raise WantRefused(f"{member.name!r} cannot be read")
            with source:
                _write_member(dest, member.name, source, member.size)
    return count


def _write_member(dest: Path, name: str, source: IO[bytes], size: int) -> None:
    *parents, leaf = name.split("/")
    created: list[str] = []
    root = fsafe.open_root(dest)
    try:
        directory = fsafe.open_dir_beneath(root, parents, created=created)
    finally:
        os.close(root)
    try:
        try:
            out = os.open(leaf, _WRITE_FLAGS, fsafe.FILE_MODE, dir_fd=directory)
        except FileExistsError:
            return
        try:
            left = size
            while left > 0 and (chunk := source.read(min(fsafe.CHUNK, left))):
                fsafe.write_all(out, chunk)
                left -= len(chunk)
        finally:
            os.close(out)
    finally:
        os.close(directory)


def _wait_for(path: Path, timeout_sec: float, poll_sec: float = 0.5) -> bool:
    """Wait until ``path`` exists; False when ``timeout_sec`` passes first."""
    deadline = time.monotonic() + timeout_sec
    while not path.exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_sec)
    return True


def _write_atomic(path: Path, data: bytes) -> None:
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_bytes(data)
    os.replace(pending, path)


# -- the host's side -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Packed:
    """One kind's tar, and the snapshot and files it holds."""

    data: bytes
    snapshot: str
    files: int


def parse_want(raw: bytes) -> list[tuple[CacheKind, str]]:
    """The (kind, key) pairs of a runner's want, read strictly: at most one per known kind, every key a sha256.

    Raises:
        WantRefused: The want is too large, not the document this version reads, or names an unknown kind, a key that is
            not a sha256, or a kind twice.
    """
    if len(raw) > MAX_WANT_BYTES:
        raise WantRefused(f"the want is larger than {MAX_WANT_BYTES} bytes")
    try:
        r = Reader(parse_json_object(raw, what=_WANT), _WANT)
        if r.text("schema") != _WANT_SCHEMA:
            raise DocumentError(f"want.schema: expected {_WANT_SCHEMA}")
        items = r.raw(_WANT)
        r.done()
        if not isinstance(items, list) or len(items) > len(KINDS):
            raise DocumentError(f"want.want: expected a list of at most {len(KINDS)} items")
        wanted: dict[str, tuple[CacheKind, str]] = {}
        for index, item in enumerate(items):
            e = Reader(item, f"want.want[{index}]")
            kind = kind_named(e.text("kind") or "")
            key = e.text("key", pattern=SHA256_RE) or ""
            e.done()
            if kind.name in wanted:
                raise DocumentError(f"want.want: {kind.name} is named twice")
            wanted[kind.name] = (kind, key)
    except KeyError as exc:
        raise WantRefused(f"want: no kind {exc}") from None
    except DocumentError as exc:
        raise WantRefused(str(exc)) from None
    return list(wanted.values())


def pack(store: DirJitStore, kind: CacheKind, key: str) -> Packed | None:
    """The tar of the snapshot of ``kind`` under ``key`` a reader would take, every blob proved as it is packed; the next
    generation when one does not prove; ``None`` when none does or the store has none within the bounds."""
    for snapshot in store.generations(kind, key):
        if len(snapshot.files) >= MAX_TAR_FILES or snapshot.total_bytes > MAX_TAR_BYTES:
            log.info("jit cache: %s snapshot %s is too large to serve", kind.name, snapshot.sha256()[:12])
            continue
        try:
            return _pack_snapshot(store, snapshot)
        except (SnapshotInvalid, OSError, fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
            log.info("jit cache: %s snapshot %s not served: %s", kind.name, snapshot.sha256()[:12], exc)
    return None


def _pack_snapshot(store: DirJitStore, snapshot: Snapshot) -> Packed | None:
    buffer = io.BytesIO()
    name = snapshot.sha256()
    root = fsafe.open_root(store.root)
    try:
        with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.PAX_FORMAT) as archive:
            _add(archive, f"{SNAPSHOTS_DIR}/{snapshot.kind}/{snapshot.key}/{name}.json", snapshot.to_bytes())
            for sha256, size in sorted({(f.sha256, f.size) for f in snapshot.files.values()}):
                _add(archive, f"{BLOBS_DIR}/{sha256[:2]}/{sha256}", _proved_blob(root, sha256, size))
    finally:
        os.close(root)
    data = buffer.getvalue()
    if len(data) > MAX_TAR_BYTES:
        return None
    return Packed(data, name, len(snapshot.files))


def _proved_blob(root: int, sha256: str, size: int) -> bytes:
    """The bytes of the stored blob, read through a descriptor that refuses links and non-regular files, and proved."""
    fd = fsafe.open_beneath(root, f"{BLOBS_DIR}/{sha256[:2]}/{sha256}")
    try:
        data = b"".join(fsafe.stream(fd, size))
    finally:
        os.close(fd)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise SnapshotInvalid(f"the stored blob {sha256[:12]} is not the {size} bytes its hash names")
    return data


def _add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime, info.type = len(data), _TAR_MODE, 0, tarfile.REGTYPE
    archive.addfile(info, io.BytesIO(data))


def serve(store: DirJitStore | None, handle: Sandbox, shown: str) -> int:
    """Serve the want the runner of ``handle`` wrote from ``store`` (``None``: serve nothing); returns the kinds served.
    Always ends with ``DONE`` (also when the want is refused or nothing is stored), so the runner waits no longer than it
    takes; never raises."""
    started, served = time.monotonic(), list[str]()
    try:
        if store is not None:
            _serve_wanted(store, handle, served)
    except WantRefused as exc:
        log.warning("jit cache: test %s: its want is refused (%s); nothing is served", shown, exc)
    except (SandboxError, OSError) as exc:
        log.warning("jit cache: test %s: serving stopped (%s: %s)", shown, type(exc).__name__, exc)
    try:
        handle.write_file(DONE_PATH, b"")
    except (SandboxError, OSError) as exc:
        log.warning("jit cache: test %s: DONE could not be written (%s); its runner builds cold", shown, exc)
    log.info(
        "jit cache: test %s served in %.1fs: %s", shown, time.monotonic() - started, ", ".join(served) or "nothing"
    )
    return len(served)


def _serve_wanted(store: DirJitStore, handle: Sandbox, served: list[str]) -> None:
    """Write a tar of every kind the want names that the store holds a snapshot of; each is named in ``served``."""
    for kind, key in parse_want(_read_want(handle)):
        packed = pack(store, kind, key)
        if packed is None:
            continue
        handle.write_file(f"{SERVE_DIR}/{kind.name}{_TAR}", packed.data)
        served.append(f"{kind.name} {packed.files} files {len(packed.data)} bytes")


def _read_want(handle: Sandbox) -> bytes:
    """The want, read only when the listing shows a regular file within the bound, and checked again once read."""
    listed = {entry.name: entry for entry in handle.list_entries(WANT_PATH.rsplit("/", 1)[0])}
    entry = listed.get(_WANT_NAME)
    if entry is None or entry.kind != "file" or entry.size > MAX_WANT_BYTES:
        raise WantRefused("there is no want within the bound as a regular file")
    raw = handle.read_file(WANT_PATH)
    if len(raw) > MAX_WANT_BYTES:
        raise WantRefused(f"the want is larger than {MAX_WANT_BYTES} bytes")
    return raw
