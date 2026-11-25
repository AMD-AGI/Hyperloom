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
   snapshot list and its blobs, in the store's own layout, into one tar per kind, and writes it with the sandbox's
   ``write_file`` as numbered parts of at most :data:`PART_BYTES` (``/in/jit-cache/<kind>.tar.000``, ``.001``, ...)
   followed by ``<kind>.tar.json``, the index that gives the number of parts and the size and sha256 of the whole tar.
   Each kind is served on its own: one that fails is logged and the next is still served. ``DONE`` comes last, also
   when nothing was served.
4. The runner joins each kind's parts in order, checks the size and sha256 the index names, and unpacks the tar into a
   directory of its own (:func:`unpack`) that the test's session then reads as its store:
   :meth:`~hyperloom.measure.jit.store.DirJitStore.fetch` proves every byte again before anything is used.

Whatever the sandbox writes is untrusted on the host: the want is bounded in size, its kinds and keys must be known
kinds and sha256 keys, and nothing it names is a path. What the host sends is untrusted on the runner the same way:
the runner reads only the file names it expects for the kinds it asked for, an index within its bounds, and parts that
are regular files adding up to the size and hash the index names; a tar member must be a regular file under a name of the store's layout for the kind and key asked for (no link, no device,
no directory, no path of its own), within the size and count bounds. Anything that does not hold is dropped and the test
builds what it lacks itself: the cache costs time when it fails, never a result.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import logging
import os
import re
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
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
    "MAX_WAIT_SEC",
    "MAX_WANT_BYTES",
    "PART_BYTES",
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
    "served_files",
    "unpack",
    "want_document",
]

log = logging.getLogger(__name__)

#: The sandbox's directory the host serves into, and its name below the runner's input directory.
SERVE_DIR = "/in/jit-cache"
_SERVE_NAME = "jit-cache"
_SERVING, _DONE, _TAR = "SERVING", "DONE", ".tar"
#: The index written after a kind's parts: ``<kind>.tar.json``; a part is ``<kind>.tar.<index, three digits>``.
_INDEX = f"{_TAR}.json"
_INDEX_SCHEMA = "hyperloom.jit-cache-parts/v1"
_MAX_INDEX_BYTES = 1024
SERVING_PATH, DONE_PATH = f"{SERVE_DIR}/{_SERVING}", f"{SERVE_DIR}/{_DONE}"
#: The runner's list of the kinds and keys it wants, below its output directory.
_WANT_NAME = "jit-want.json"
WANT_PATH = f"/out/{_WANT_NAME}"
_WANT_SCHEMA = "hyperloom.jit-cache-want/v1"
_WANT = "want"
#: The runner's progress phase while it waits to be served: the host serves when it reads it.
WAIT_PHASE = "jit_wait"
#: How long a runner waits for the host without seeing a new file served before its test builds cold: a host polls every
#: few seconds and writes one part at a time, so the wait starts again with every part that comes, up to
#: :data:`MAX_WAIT_SEC` in all.
WAIT_SEC = 300.0
MAX_WAIT_SEC = 1800.0
#: The bounds of what crosses: a want is a few hundred bytes; one tar is one ``write_file`` request, which the platform
#: bounds in time, so a kind whose snapshot is larger is not served (it builds cold); the most files one tar holds. A
#: DeepSeek aiter JIT cache passes 256 MiB, so a kind may send up to 1 GiB.
MAX_WANT_BYTES = 16 * 1024
MAX_TAR_BYTES = 1024**3
MAX_TAR_FILES = 20_000
#: The largest part of a kind's tar the host writes. The sandbox's file API refuses a single write over 32 MB (HTTP 413,
#: "file exceeds 32MB"), so a tar is written as parts of at most 30 MiB, safely under that limit.
PART_BYTES = 30 * 1024**2
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

    Nothing here fails the test: a host that does not answer in time (:func:`_wait_for_done`), or a kind whose parts do
    not join into the tar its index names or do not unpack, leaves the kinds it would have given empty (a note says
    why), and their tests build cold.
    """
    served, notes = in_dir / _SERVE_NAME, list[str]()
    _write_atomic(out_dir / _WANT_NAME, want_document(wanted))
    phase(WAIT_PHASE)
    started = time.monotonic()
    if not _wait_for_done(served):
        notes.append(
            f"jit cache: the host did not serve the cache within {time.monotonic() - started:.0f}s; the test builds cold"
        )
        return notes
    given = []
    for kind, key in wanted:
        if not os.path.lexists(served / f"{kind.name}{_INDEX}"):
            continue
        try:
            files = _receive(served, dest, kind, key)
        except (WantRefused, tarfile.TarError, ValueError, OSError, fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
            notes.append(f"jit cache: {kind.name}: what the host served was not used ({exc}); it builds cold")
            files = None
        _remove_served(served, kind)
        if files is not None:
            given.append(f"{kind.name} {files} files")
    notes.append(f"jit cache: served by the host in {time.monotonic() - started:.1f}s: {', '.join(given) or 'nothing'}")
    return notes


def _max_parts() -> int:
    return max(1, -(-MAX_TAR_BYTES // PART_BYTES))


def _part_name(kind: CacheKind, index: int) -> str:
    return f"{kind.name}{_TAR}.{index:03d}"


def _receive(served: Path, dest: Path, kind: CacheKind, key: str) -> int:
    """Join the parts the host served for ``kind`` into one tar beside ``dest``, proved against its index, and unpack it;
    returns the files it held. The joined tar is gone afterwards on every path."""
    index = _read_index(served / f"{kind.name}{_INDEX}")
    dest.parent.mkdir(parents=True, exist_ok=True, mode=fsafe.DIR_MODE)
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{kind.name}-", suffix=_TAR) as joined:
        digest, total = hashlib.sha256(), 0
        for number in range(index.parts):
            for chunk in _read_part(served / _part_name(kind, number)):
                total += len(chunk)
                if total > index.size:
                    raise WantRefused(f"the parts hold more than the {index.size} bytes the index names")
                digest.update(chunk)
                joined.write(chunk)
        joined.flush()
        if total != index.size or digest.hexdigest() != index.sha256:
            raise WantRefused(f"the parts are {total} bytes, not the {index.size} bytes and hash the index names")
        return unpack(Path(joined.name), dest, kind, key)


@dataclass(frozen=True)
class _Index:
    """What the host's index says of a kind's tar: the parts it is written as, and its size and sha256."""

    parts: int
    size: int
    sha256: str


def _read_index(path: Path) -> _Index:
    """The number of parts, the size and the sha256 of a kind's tar, read strictly from the index the host wrote."""
    raw = b"".join(_read_part(path, _MAX_INDEX_BYTES))
    try:
        r = Reader(parse_json_object(raw, what="index"), "index")
        if r.text("schema") != _INDEX_SCHEMA:
            raise DocumentError(f"index.schema: expected {_INDEX_SCHEMA}")
        parts, size = r.integer("parts", minimum=1) or 0, r.integer("size", minimum=1) or 0
        sha256 = r.text("sha256", pattern=SHA256_RE) or ""
        r.done()
    except DocumentError as exc:
        raise WantRefused(str(exc)) from None
    if size > MAX_TAR_BYTES:
        raise WantRefused(f"index.size: expected a size of at most {MAX_TAR_BYTES} bytes")
    if parts != -(-size // PART_BYTES) or parts > _max_parts():
        raise WantRefused(f"index.parts: expected the {-(-size // PART_BYTES)} parts of {size} bytes")
    return _Index(parts, size, sha256)


def _read_part(path: Path, bound: int | None = None) -> Iterator[bytes]:
    """The bytes of a served file, opened without following a link and only when it is a regular file within ``bound``
    (by default a part's)."""
    bound = PART_BYTES if bound is None else bound
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        facts = os.fstat(fd)
        if not stat.S_ISREG(facts.st_mode):
            raise WantRefused(f"{path.name} is not a regular file")
        if facts.st_size > bound:
            raise WantRefused(f"{path.name} is larger than {bound} bytes")
        left = facts.st_size
        while left > 0 and (chunk := os.read(fd, min(fsafe.CHUNK, left))):
            left -= len(chunk)
            yield chunk
    finally:
        os.close(fd)


def _remove_served(served: Path, kind: CacheKind) -> None:
    """Remove what the host served for ``kind``, which is read from its unpacked copy from here on; an input directory
    this runner cannot write keeps it, which costs only its space."""
    names = itertools.chain([f"{kind.name}{_INDEX}"], (_part_name(kind, index) for index in range(_max_parts())))
    for name in names:
        try:
            (served / name).unlink()
        except FileNotFoundError:
            if name != f"{kind.name}{_INDEX}":
                return  # the parts are numbered from 000 on: the first missing one is past the last
        except OSError as exc:
            log.info("jit cache: the served %s was not removed (%s)", name, type(exc).__name__)
            return


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


def _wait_for_done(served: Path, poll_sec: float = 0.5) -> bool:
    """Wait until the host writes ``DONE``; False when :data:`WAIT_SEC` passes with no new file served, or
    :data:`MAX_WAIT_SEC` in all."""
    started = time.monotonic()
    deadline, seen = started + WAIT_SEC, -1
    while not (served / _DONE).exists():
        now = time.monotonic()
        try:
            count = len(os.listdir(served))
        except OSError:
            count = 0
        if count != seen:
            seen, deadline = count, min(now + WAIT_SEC, started + MAX_WAIT_SEC)
        if now >= deadline:
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

    @property
    def parts(self) -> int:
        """The number of parts of at most :data:`PART_BYTES` the tar is written as."""
        return max(1, -(-len(self.data) // PART_BYTES))


def served_files(kind: CacheKind, packed: Packed) -> Iterator[tuple[str, bytes]]:
    """The names below ``/in/jit-cache`` and the bytes the host writes for ``kind``, in the order it writes them: the
    tar's parts of at most :data:`PART_BYTES` each, then the index that names their number and the tar's size and hash,
    so that a runner never finds an index whose parts are not all written."""
    view = memoryview(packed.data)
    for index in range(packed.parts):
        yield _part_name(kind, index), bytes(view[index * PART_BYTES : (index + 1) * PART_BYTES])
    index_doc = {
        "schema": _INDEX_SCHEMA,
        "parts": packed.parts,
        "size": len(packed.data),
        "sha256": hashlib.sha256(packed.data).hexdigest(),
    }
    yield f"{kind.name}{_INDEX}", canonical_json(index_doc).encode("utf-8")


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


#: The keys whose snapshot was found too large to serve, said once each.
_TOO_LARGE: set[tuple[str, str]] = set()


def pack(store: DirJitStore, kind: CacheKind, key: str) -> Packed | None:
    """The tar of the snapshot of ``kind`` under ``key`` a reader would take, every blob proved as it is packed; the next
    generation when one does not prove; ``None`` when none does or the store has none within the bounds."""
    for snapshot in store.generations(kind, key):
        if len(snapshot.files) >= MAX_TAR_FILES or snapshot.total_bytes > MAX_TAR_BYTES:
            if (kind.name, key) not in _TOO_LARGE:  # once per key: every test of it builds cold
                _TOO_LARGE.add((kind.name, key))
                log.warning(
                    "jit cache: %s key %s: snapshot %s (%d files, %d bytes) is too large to serve; its tests build cold",
                    kind.name,
                    key[:12],
                    snapshot.sha256()[:12],
                    len(snapshot.files),
                    snapshot.total_bytes,
                )
            continue
        try:
            return _pack_snapshot(store, snapshot)
        except (SnapshotInvalid, OSError, fsafe.MissingEntry, fsafe.UnsafeEntry) as exc:
            log.info("jit cache: %s snapshot %s not served: %s", kind.name, snapshot.sha256()[:12], exc)
    return None


def _pack_snapshot(store: DirJitStore, snapshot: Snapshot) -> Packed | None:
    """The tar is built in an unnamed file in the system temp dir (gone once closed, on every path), and read back once:
    ``write_file`` takes bytes, so the upload holds one copy of it in memory, not a growing buffer and its copy."""
    name = snapshot.sha256()
    root = fsafe.open_root(store.root)
    try:
        with tempfile.TemporaryFile() as spool:
            with tarfile.open(fileobj=spool, mode="w:", format=tarfile.PAX_FORMAT) as archive:
                _add(archive, f"{SNAPSHOTS_DIR}/{snapshot.kind}/{snapshot.key}/{name}.json", snapshot.to_bytes())
                for sha256, size in sorted({(f.sha256, f.size) for f in snapshot.files.values()}):
                    _add(archive, f"{BLOBS_DIR}/{sha256[:2]}/{sha256}", _proved_blob(root, sha256, size))
            if spool.tell() > MAX_TAR_BYTES:
                return None
            spool.seek(0)
            data = spool.read()
    finally:
        os.close(root)
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
            _serve_wanted(store, handle, served, shown)
    except WantRefused as exc:
        log.warning("jit cache: test %s: its want is refused (%s); nothing is served", shown, exc)
    except (SandboxError, OSError) as exc:
        log.warning("jit cache: test %s: its want could not be read (%s: %s)", shown, type(exc).__name__, exc)
    try:
        handle.write_file(DONE_PATH, b"")
    except (SandboxError, OSError) as exc:
        log.warning("jit cache: test %s: DONE could not be written (%s); its runner builds cold", shown, exc)
    log.info(
        "jit cache: test %s served in %.1fs: %s", shown, time.monotonic() - started, ", ".join(served) or "nothing"
    )
    return len(served)


def _serve_wanted(store: DirJitStore, handle: Sandbox, served: list[str], shown: str) -> None:
    """Write the parts of a tar of every kind the want names that the store holds a snapshot of, each kind on its own: one
    that fails (the platform refuses a write, say) is logged and builds cold, the next is still served. Each kind served
    is named in ``served``."""
    for kind, key in parse_want(_read_want(handle)):
        started = time.monotonic()
        try:
            packed = pack(store, kind, key)
            if packed is None:
                continue
            for name, data in served_files(kind, packed):
                handle.write_file(f"{SERVE_DIR}/{name}", data)
        except (SandboxError, OSError) as exc:
            log.warning(
                "jit cache: test %s: %s not served (%s: %s); it builds cold", shown, kind.name, type(exc).__name__, exc
            )
            continue
        log.info(
            "jit cache: test %s: %s served: %d files, %d bytes in %d parts, %.1fs",
            shown,
            kind.name,
            packed.files,
            len(packed.data),
            packed.parts,
            time.monotonic() - started,
        )
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
