# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One test's use of the shared compiled-code cache (design 4.21).

Before the server starts, :meth:`JitCacheSession.seed` copies the verified snapshot of each kind into a directory of the
test's own and :meth:`~JitCacheSession.environment` names those directories for the server; no two tests ever share a
writable cache directory. After a successful test, :meth:`~JitCacheSession.collect` stages what the compilers added: the
files the kind admits that the snapshot did not have, as content-addressed blobs and a list of the whole snapshot they
extend. A file the test changed under a name the snapshot already holds, judged by its content and not by its time, is not
contributed (a stored name never changes), and a symlink is not followed or taken (the test's own directory holds links to
the image's prebuilt modules). A compiler's index of the files it made names the directory it was written in, so it is
staged without that directory and given the test's own at seed time (:mod:`hyperloom.measure.jit.locations`).

The session never fails a test: whatever goes wrong with the cache is a note (:mod:`hyperloom.measure.jit.notes`) and a
state in the report, and the test builds what it lacks itself. The staged contribution is published by the host after it
has verified the signed result (:meth:`hyperloom.measure.jit.store.DirJitStore.publish`), never by the code that runs next
to the server.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.jit.builds import BuildLog
from hyperloom.measure.jit.locations import NotPortable, portable, read_located
from hyperloom.measure.jit.notes import LISTED, SessionNotes
from hyperloom.measure.jit_kinds import DISABLED, HIT, MISS, VERIFY_FAILED, CacheKind
from hyperloom.measure.jit.store import (
    BLOBS_DIR,
    MANIFEST_NAME,
    MAX_SNAPSHOT_BYTES,
    DirJitStore,
    JitStoreError,
    Snapshot,
    StoredFile,
)

__all__ = [
    "Contribution",
    "JitCacheReport",
    "JitCacheSession",
    "KindPlan",
    "KindReport",
]

_TMP = ".tmp-"
_MODE = 0o600
_DIR_MODE = 0o700
_PREBUILT_SUFFIX = ".so"
_CHANGED = "changed by the test; the stored bytes stay"
_NOT_PORTABLE = "not contributed: they name files the cache does not hold"
#: How the cache stores the bytes of a file: as they are (``None``), or with the directory they name taken out.
_Form = Callable[[bytes], bytes]


@dataclass(frozen=True)
class KindPlan:
    """A kind this test uses and the key it is stored under; ``prebuilt`` is a directory of compiled modules the image
    ships, which the test's own directory links to so the redirected cache does not hide them.

    ``read_only`` is the reason a test may be given the cache and not add to it: it runs code the key does not cover, and
    code in the server can write any file into the directory the compiler uses, which no test sharing the key may load.

    ``touched`` are the modules a patch to aiter's sources makes stale: the test's directory does not link the image's copy
    of them, so the server builds each when it first calls it."""

    kind: CacheKind
    key: str
    prebuilt: Path | None = None
    read_only: str | None = None
    touched: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Contribution:
    """What a test staged for a kind: the hash of its snapshot list, and the files and bytes that were new."""

    sha256: str
    files: int
    bytes: int


@dataclass(frozen=True)
class KindReport:
    kind: str
    key: str | None
    state: str
    seed_files: int = 0
    seed_bytes: int = 0
    seed_snapshot: str | None = None
    contribution: Contribution | None = None

    def document(self) -> dict[str, object]:
        gave = self.contribution
        return {
            "kind": self.kind,
            "key": self.key,
            "state": self.state,
            "seed_files": self.seed_files,
            "seed_bytes": self.seed_bytes,
            "seed_snapshot": self.seed_snapshot,
            "contribution": None if gave is None else {"sha256": gave.sha256, "files": gave.files, "bytes": gave.bytes},
        }


@dataclass(frozen=True)
class JitCacheReport:
    """What the test did with the cache: for each kind its state, and how much aiter built while the server ran."""

    kinds: tuple[KindReport, ...]
    builds_seen: int
    compile_seconds: float
    notes: tuple[str, ...] = ()

    def document(self) -> dict[str, object]:
        return {
            "kinds": [kind.document() for kind in self.kinds],
            "builds_seen": self.builds_seen,
            "jit_compile_sec": self.compile_seconds,
        }


@dataclass
class _Seeded:
    """A kind as it stood after seeding: the snapshot it was given, if any."""

    plan: KindPlan
    state: str = MISS
    snapshot: Snapshot | None = None
    seed_bytes: int = 0
    contribution: Contribution | None = None

    @property
    def files(self) -> Mapping[str, StoredFile]:
        return {} if self.snapshot is None else self.snapshot.files


class JitCacheSession:
    """The cache as one test uses it: a private directory per kind, seeded before and harvested after."""

    def __init__(
        self,
        store: DirJitStore,
        plans: Sequence[KindPlan],
        *,
        private_root: Path,
        staging: Path,
        disabled: Mapping[str, str] | None = None,
    ) -> None:
        self._store = store
        self._kinds = {plan.kind.name: _Seeded(plan) for plan in plans}
        self._private = Path(private_root)
        self._staging = Path(staging)
        self._disabled = dict(disabled or {})
        self._notes = SessionNotes()
        for name, why in sorted(self._disabled.items()):
            self._notes.fact(f"jit cache: {name} is not used: {why}")
        for plan in plans:
            if plan.read_only is not None:
                self._notes.problem(f"jit cache: {plan.kind.name} is read only in this test: {plan.read_only}")
            if plan.touched:
                self._notes.fact(_touched_note(plan))

    @property
    def touched(self) -> frozenset[str]:
        """The aiter modules this test's patch makes stale, which the server builds itself."""
        return frozenset().union(*(seeded.plan.touched for seeded in self._kinds.values()))

    def environment(self) -> dict[str, str]:
        """The variables that point each compiler at this test's own directory for its kind."""
        return {seeded.plan.kind.env_var: str(self._root(seeded.plan.kind)) for seeded in self._kinds.values()}

    def _root(self, kind: CacheKind) -> Path:
        return self._private / kind.name

    def _content(self, kind: CacheKind) -> Path:
        return self._root(kind) / kind.content if kind.content else self._root(kind)

    # -- before the server starts ------------------------------------------------------------------------------------

    def seed(self) -> None:
        """Copy the verified snapshot of every kind into the test's directories; a snapshot that fails is dropped."""
        shutil.rmtree(self._private, ignore_errors=True)
        for seeded in self._kinds.values():
            content = self._content(seeded.plan.kind)
            os.makedirs(content, mode=_DIR_MODE, exist_ok=True)
            self._seed_kind(seeded, content)
            if seeded.plan.prebuilt is not None:
                _link_prebuilt(seeded.plan.prebuilt, content, seeded.plan.touched)

    def _seed_kind(self, seeded: _Seeded, content: Path) -> None:
        kind = seeded.plan.kind
        try:
            fetched = self._store.fetch(kind, seeded.plan.key, content)
        except (JitStoreError, OSError) as exc:
            _clear(content)
            seeded.state = VERIFY_FAILED
            self._notes.problem(f"jit cache: {kind.name} could not be read ({type(exc).__name__}); it builds cold")
            return
        for problem in fetched.problems[:LISTED]:
            self._notes.problem(f"jit cache: {problem}; ignored")
        if len(fetched.problems) > LISTED:
            self._notes.problem(f"jit cache: {kind.name}: {len(fetched.problems) - LISTED} more snapshots were ignored")
        if fetched.snapshot is None:
            seeded.state = VERIFY_FAILED if fetched.problems else MISS
            return
        seeded.state, seeded.snapshot, seeded.seed_bytes = HIT, fetched.snapshot, fetched.bytes

    # -- after a successful test ------------------------------------------------------------------------------------

    def collect(self) -> None:
        """Stage what the compilers added in each kind's directory; call it once the server has stopped."""
        for seeded in self._kinds.values():
            if seeded.plan.read_only is not None:
                continue
            try:
                seeded.contribution = self._stage(seeded)
            except (JitStoreError, OSError, fsafe.UnsafeEntry, fsafe.MissingEntry) as exc:
                self._notes.problem(
                    f"jit cache: {seeded.plan.kind.name} was not contributed ({type(exc).__name__}: {exc})"
                )
                shutil.rmtree(self._staging / seeded.plan.kind.name, ignore_errors=True)

    def _stage(self, seeded: _Seeded) -> Contribution | None:
        kind = seeded.plan.kind
        directory = self._staging / kind.name
        new = self._stage_new_files(seeded, directory)
        if not new:
            shutil.rmtree(directory, ignore_errors=True)
            return None
        parent = None if seeded.snapshot is None else seeded.snapshot.sha256()
        snapshot = Snapshot(kind.name, seeded.plan.key, parent, {**seeded.files, **new})
        if snapshot.total_bytes > MAX_SNAPSHOT_BYTES:
            shutil.rmtree(directory, ignore_errors=True)
            self._notes.fact(f"jit cache: {kind.name} is over the size a snapshot may have; not contributed")
            return None
        raw = snapshot.to_bytes()
        with open(directory / MANIFEST_NAME, "xb") as out:
            out.write(raw)
        return Contribution(snapshot.sha256(), len(new), sum(entry.size for entry in new.values()))

    def _stage_new_files(self, seeded: _Seeded, directory: Path) -> dict[str, StoredFile]:
        kind, content = seeded.plan.kind, self._content(seeded.plan.kind)
        new: dict[str, StoredFile] = {}
        if not content.is_dir():
            return new
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory / BLOBS_DIR, mode=_DIR_MODE)
        root_fd = fsafe.open_root(content)
        try:
            for relative in _candidates(content, kind):
                form = _stored_form(kind, content, relative)
                stored = seeded.files.get(relative)
                if stored is not None:
                    if _differs(root_fd, relative, stored, form):
                        self._notes.tally(kind.name, _CHANGED, relative)
                    continue
                try:
                    new[relative] = _stage_file(root_fd, relative, directory / BLOBS_DIR, form)
                except NotPortable:
                    self._notes.tally(kind.name, _NOT_PORTABLE, relative)
        finally:
            os.close(root_fd)
        return new

    def stale_modules(self, built: BuildLog) -> list[str]:
        """The touched modules the test ran from a stale copy; call it once the server has stopped.

        A touched module has no copy of the image's in the directory, so one the server called is there as a file it built
        (the log names it) or one the cache gave (the snapshot names it). A module that is there as anything else, or that
        is there when the log of what was built is not, ran as the image built it, or cannot be told from that: either way
        its numbers are not the patch's. A module the server never called is not there, and nothing of it was measured.
        """
        stale: list[str] = []
        for seeded in self._kinds.values():
            content = self._content(seeded.plan.kind)
            for name in sorted(seeded.plan.touched):
                path = content / f"{name}.so"
                if not os.path.lexists(path):
                    continue
                fresh = not path.is_symlink() and (name in built.started or f"{name}.so" in seeded.files)
                if not fresh:
                    stale.append(name)
        return stale

    # -- the report --------------------------------------------------------------------------------------------------

    def report(self, built: BuildLog) -> JitCacheReport:
        """The states, the contributions and what the server's logs say aiter built (``built``)."""
        kinds = [
            KindReport(
                seeded.plan.kind.name,
                seeded.plan.key,
                seeded.state,
                len(seeded.files),
                seeded.seed_bytes,
                None if seeded.snapshot is None else seeded.snapshot.sha256(),
                seeded.contribution,
            )
            for seeded in self._kinds.values()
        ]
        kinds += [KindReport(name, None, DISABLED) for name in sorted(self._disabled)]
        return JitCacheReport(tuple(kinds), built.builds_seen, round(built.compile_seconds, 3), self._notes.lines())


def _clear(content: Path) -> None:
    shutil.rmtree(content, ignore_errors=True)
    os.makedirs(content, mode=_DIR_MODE, exist_ok=True)


def _touched_note(plan: KindPlan) -> str:
    names = sorted(plan.touched)
    listed = ", ".join(names[:LISTED]) + (f", and {len(names) - LISTED} more" if len(names) > LISTED else "")
    return (
        f"jit cache: {plan.kind.name}: the patch touches {len(names)} aiter modules, built when first called ({listed})"
    )


def _link_prebuilt(prebuilt: Path, content: Path, skip: frozenset[str]) -> None:
    """Link each compiled module the image ships into ``content`` unless the cache already gave one of that name, or the
    module is one in ``skip`` (stale: the patch changed what it is built from)."""
    try:
        names = sorted(
            path.name
            for path in Path(prebuilt).iterdir()
            if path.name.endswith(_PREBUILT_SUFFIX) and path.name[: -len(_PREBUILT_SUFFIX)] not in skip
        )
    except OSError:
        return
    for name in names:
        source = Path(os.path.realpath(Path(prebuilt) / name))
        if source.is_file() and not (content / name).exists():
            os.symlink(source, content / name)


def _candidates(content: Path, kind: CacheKind) -> list[str]:
    """The regular files below ``content`` the kind admits, symlinks left out, in a stable order."""
    found = []
    for directory, subdirs, names in os.walk(content, followlinks=False):
        subdirs.sort()
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(content).as_posix()
            if kind.admits(relative) and not path.is_symlink() and path.is_file():
                found.append(relative)
    return found


def _stored_form(kind: CacheKind, content: Path, relative: str) -> _Form | None:
    """How the cache stores ``relative``: as it is, or without the directory a compiler's index of its files names."""
    return partial(portable, root=content, kind=kind, relative=relative) if kind.locates(relative) else None


def _stored_chunks(source: int, form: _Form | None) -> Iterator[bytes]:
    """The bytes of an open file as the cache stores them."""
    return fsafe.stream(source) if form is None else iter((form(read_located(source)),))


def _differs(root_fd: int, relative: str, stored: StoredFile, form: _Form | None) -> bool:
    """Whether the file holds other bytes than the stored one: a compiler that writes the same bytes again, as Triton does
    for what it found, has changed nothing."""
    source = fsafe.open_beneath(root_fd, relative)
    try:
        if form is None and os.fstat(source).st_size != stored.size:
            return True
        digest = hashlib.sha256()
        for chunk in _stored_chunks(source, form):
            digest.update(chunk)
        return digest.hexdigest() != stored.sha256
    except NotPortable:
        return True
    finally:
        os.close(source)


def _stage_file(root_fd: int, relative: str, blobs: Path, form: _Form | None) -> StoredFile:
    """Copy one file into ``blobs`` in the form the cache stores it, under the hash of the bytes that were copied, so a
    byte that changes meanwhile is not taken for the one that was hashed."""
    source = fsafe.open_beneath(root_fd, relative)
    digest, size = hashlib.sha256(), 0
    pending = blobs / f"{_TMP}{uuid.uuid4().hex}"
    try:
        chunks = _stored_chunks(source, form)
        with os.fdopen(os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _MODE), "wb") as out:
            for chunk in chunks:
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    finally:
        os.close(source)
    sha = digest.hexdigest()
    final = blobs / sha
    if final.exists():
        pending.unlink()
    else:
        os.rename(pending, final)
    return StoredFile(sha, size)
