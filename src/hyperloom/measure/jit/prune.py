# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded clean-up of the shared compiled-code cache, run by the host that publishes to it.

The store only adds (:mod:`hyperloom.measure.jit.store`), so without this it grows with every image, card and patched
aiter tree a deployment ever measured. A key's directory (``snapshots/<kind>/<key>``) is its unit: the host stamps its
time each time an accepted result names the key (:func:`note_used`), and a publication stamps it too, so that time is
when the key was last used.

:func:`prune` works in two steps, so a test that is reading an entry never loses it:

1. A key unused for :data:`MAX_IDLE_SEC`, or, while the blobs the live keys list come to more than
   :data:`MAX_STORE_BYTES`, the least recently used keys unused for at least :data:`RETIRE_GRACE_SEC`, are *retired*:
   the key's directory is moved below ``retired/`` in one rename. No new test finds it from then on; a test that
   already read its list keeps finding the files it names, since no blob is removed in this step.
2. A retired key is removed only :data:`RETIRE_GRACE_SEC` after it was retired, longer than any test runs, and a blob
   only when no snapshot of a live or a still-retired key lists it and it was stored more than that long ago (so a
   publisher that stored it and has not yet renamed its list into place keeps it). A publisher's leftover
   (``.incoming-*``) of that age is removed too.

The size bound is on what the live keys list, each blob counted once at the size its list gives, not on every stored
blob: a retired key's blobs, and blobs nothing lists, stay stored uncounted for the grace, and counting them would make
every later start inside it retire another batch of live keys. The directory can therefore hold more than the bound for
up to two days.

A prune runs as a session starts, so it stops once it has taken :data:`PRUNE_BUDGET_SEC`, and every start does part of
the work whatever the store's size. It first removes the retired keys past their grace and the stale leftovers, which
stay removed. It then reads the live keys, retires the idle ones among those it read, and, once it has read them all,
applies the size bound and removes the blobs nothing lists, shard by shard. Where a prune stopped while reading the keys
or the blobs is kept in :data:`PRUNE_CURSOR` (written by one rename), and the next start goes on from there, so
successive starts cover the whole store. The size bound and the removal of blobs need every key read within one start:
a store whose keys take longer than the budget to read is never bounded by size, and each such start says so at
WARNING. It always logs one line saying what it did (:func:`prune_logged`).

A test that read a key just before it was retired and publishes after it extends it under the same key again: the files
it took from the cache are still stored, as the retired list names them. Two hosts may prune one store at once: every
step is a rename or a removal that the other's finding the name gone leaves correct. A directory that cannot be listed
(for any reason but its not being there) stops the prune: it is never taken as empty, which would remove the blobs of
the keys it did not see. Nothing here fails a session; the
host logs what it could not do and the store is pruned at the next start.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import logging
import os
import shutil
import stat
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from hyperloom.measure.docs import SHA256_RE
from hyperloom.measure.jit.store import (
    BLOBS_DIR,
    INCOMING_PREFIX,
    SNAPSHOTS_DIR,
    DirJitStore,
    Snapshot,
    SnapshotInvalid,
)
from hyperloom.measure.jit_kinds import KINDS, CacheKind

__all__ = [
    "MAX_IDLE_SEC",
    "MAX_STORE_BYTES",
    "PRUNE_BUDGET_SEC",
    "PRUNE_CURSOR",
    "RETIRED_DIR",
    "RETIRE_GRACE_SEC",
    "PruneLimits",
    "Pruned",
    "note_used",
    "prune",
    "prune_logged",
]

log = logging.getLogger(__name__)

_DAY = 24 * 3600
#: A key no accepted result has named for this long is retired: two weeks covers a campaign that pauses over a weekend
#: and comes back to the same image.
MAX_IDLE_SEC = 14 * _DAY
#: Past this many bytes listed by the live keys the least recently used keys are retired until the rest fit. An aiter tree's modules
#: are a few GiB at most, so this keeps dozens of image and tree combinations.
MAX_STORE_BYTES = 200 * 1024**3
#: How long a retired key, and a blob nothing lists, stay before they are removed: longer than any test runs (the boot
#: budget is at most six hours and the measured window a few more), so no test still reading one loses it.
RETIRE_GRACE_SEC = 2 * _DAY
#: How long one prune may take before it stops: it runs on the session's start, on a store that may be on a network
#: file system, with one lstat per stored blob and one read per snapshot list.
PRUNE_BUDGET_SEC = 60.0
#: The file below the store's root where a prune that stopped says where the next goes on (see the module).
PRUNE_CURSOR = "prune-cursor.json"
#: Where retired keys wait, below the store's root, as ``<retired at>-<kind>-<key>-<nonce>``.
RETIRED_DIR = "retired"
_KINDS_BY_NAME = {kind.name: kind for kind in KINDS}


@dataclass(frozen=True)
class PruneLimits:
    """The bounds :func:`prune` keeps the store to (see the module)."""

    max_idle_sec: float = MAX_IDLE_SEC
    max_bytes: int = MAX_STORE_BYTES
    grace_sec: float = RETIRE_GRACE_SEC
    budget_sec: float = PRUNE_BUDGET_SEC


@dataclass
class Pruned:
    """What one :func:`prune` did: keys retired and removed, blobs and bytes removed, what it could not do, and the
    step it stopped at when its time budget ran out (``None`` when it finished)."""

    retired: int = 0
    removed: int = 0
    blobs: int = 0
    bytes: int = 0
    problems: list[str] = field(default_factory=list)
    stopped_at: str | None = None


@dataclass(frozen=True)
class _Key:
    path: Path
    used: float
    blobs: Mapping[str, int]


def note_used(store: DirJitStore, kind: str, key: str | None, now: float | None = None) -> None:
    """Stamp the key ``key`` of ``kind`` as used now, when the store holds it; a name that is not a stored key is left
    alone, and a stamp that cannot be written is only a key retired sooner."""
    if key is None or kind not in _KINDS_BY_NAME or not SHA256_RE.fullmatch(key):
        return
    stamp = time.time() if now is None else now
    try:
        os.utime(store.root / SNAPSHOTS_DIR / kind / key, (stamp, stamp), follow_symlinks=False)
    except OSError:
        return


def prune(store: DirJitStore, limits: PruneLimits = PruneLimits(), now: float | None = None) -> Pruned:
    """Remove what was retired long enough ago, retire the keys unused too long or past the size bound, then remove the
    blobs nothing lists any more, from where the last stopped prune left off (see the module)."""
    run = _Run(store.root, limits, time.time() if now is None else now, _clock() + limits.budget_sec)
    if (store.root / SNAPSHOTS_DIR).is_dir():
        run.prune()
    return run.done


def prune_logged(store: DirJitStore, limits: PruneLimits = PruneLimits()) -> Pruned | None:
    """:func:`prune`, said in one line of the log even when it removed nothing; ``None`` when the store could not be
    read at all, which is logged too. A prune that stopped before it read every key is said at WARNING: the size bound
    and the removal of blobs wait until a start reads them all within its budget."""
    started = time.monotonic()
    try:
        done = prune(store, limits)
    except OSError as exc:
        log.warning("jit cache: %s was not pruned (%s)", store.root, type(exc).__name__)
        return None
    stopped = (
        ""
        if done.stopped_at is None
        else f"; stopped {done.stopped_at} after its {limits.budget_sec:.0f} s budget, the next start goes on from there"
    )
    log.info(
        "jit cache: %s pruned in %.1f s: %d key(s) retired, %d removed, %d blob(s) of %d bytes removed%s",
        store.root,
        time.monotonic() - started,
        done.retired,
        done.removed,
        done.blobs,
        done.bytes,
        stopped,
    )
    if done.stopped_at in _BEFORE_THE_BOUND:
        log.warning(
            "jit cache: %s was not brought under its size bound: the prune stopped %s, and the bound and the removal "
            "of blobs need every key read within one start's %.0f s; a store whose keys take longer than that to read "
            "is never bounded by size",
            store.root,
            done.stopped_at,
            limits.budget_sec,
        )
    for problem in done.problems:
        log.warning("jit cache: %s", problem)
    return done


#: The steps a prune can stop at before it applied the size bound.
_BEFORE_THE_BOUND = frozenset({"reading the live keys", "reading the retired keys"})
#: The clock of the budget; a test replaces it.
_clock = time.monotonic


@dataclass
class _Run:
    """One pruning of the store at ``root`` at the time ``at``."""

    root: Path
    limits: PruneLimits
    at: float
    deadline: float
    done: Pruned = field(default_factory=Pruned)

    def prune(self) -> None:
        read = _read_cursor(self.root)
        cursor = dict(read)
        self._steps(cursor)
        if cursor != read:
            self._write_cursor(cursor)

    def _steps(self, cursor: dict[str, str]) -> None:
        if not self._remove_expired():
            return
        live, last = self._read_live(cursor.get(_KEY))
        kept = self._retire_idle(live)  # a key's idleness is its own; the size bound and the blobs need every key
        if last is not None:
            cursor[_KEY] = last
            return
        cursor.pop(_KEY, None)
        self._retire_for_size(kept)
        waiting = []
        for blobs in self._waiting_blobs():
            waiting.append(blobs)
            if self._out_of_time("reading the retired keys"):
                return
        listed = set().union(*(key.blobs for key in live), *waiting)
        finished, last = self._sweep(listed, cursor.get(_BLOB))
        if not finished and last is not None:
            cursor[_BLOB] = last
        elif finished:
            cursor.pop(_BLOB, None)

    def _out_of_time(self, step: str) -> bool:
        """Whether the budget has run out, noting ``step`` as where the prune stopped the first time it has."""
        if self.done.stopped_at is None and _clock() >= self.deadline:
            self.done.stopped_at = step
        return self.done.stopped_at is not None

    def _remove_expired(self) -> bool:
        """Remove each retired key whose grace has passed and each publisher's leftover of that age; whether that was
        done within the budget. What is removed stays removed, so a start that stops here still shortens the next."""
        for path in _children(self.root / RETIRED_DIR):
            named = _retired_name(path.name)
            if named is None or self.at - named.retired_at <= self.limits.grace_sec:
                continue
            if self._out_of_time("removing retired keys"):
                return False
            if self._remove(path):
                self.done.removed += 1
        for path in _children(self.root):
            if not path.name.startswith(INCOMING_PREFIX):
                continue
            try:
                stale = self.at - path.lstat().st_mtime > self.limits.grace_sec
            except OSError:
                continue
            if stale:
                if self._out_of_time("removing leftovers"):
                    return False
                self._remove(path)
        return True

    def _read_live(self, after: str | None) -> tuple[list[_Key], str | None]:
        """The live keys, read from the one after ``after`` round to it; with the last one read when the budget ran out
        first, ``None`` once every key was read."""
        names = _from(sorted(_live_names(self.root)), after)
        live = []
        for name in names:
            key = _live_key(self.root, name)
            if key is not None:
                live.append(key)
            if self._out_of_time("reading the live keys"):
                return live, name
        return live, None

    def _retire_idle(self, live: list[_Key]) -> list[_Key]:
        """Retire the keys unused for longer than the idle bound; the keys left live."""
        return [key for key in live if self.at - key.used <= self.limits.max_idle_sec or not self._retired(key)]

    def _retire_for_size(self, kept: list[_Key]) -> list[_Key]:
        """Retire the least recently used keys past the grace while the blobs the live keys list come to more than the
        size bound, each counted once at the size its list gives; the keys left live. A retired key's blobs stay stored
        for the grace, so the bound counts only the live keys' blobs: a later start inside the grace finds the rest
        within it and retires nothing more."""
        kept = list(kept)
        sizes: dict[str, int] = {}
        for key in kept:
            sizes.update(key.blobs)
        total = sum(sizes.values())
        for key in sorted(kept, key=lambda key: key.used):
            if total <= self.limits.max_bytes or self.at - key.used <= self.limits.grace_sec:
                break
            others = set().union(*(other.blobs for other in kept if other is not key))
            if self._retired(key):
                kept.remove(key)
                total -= sum(size for sha, size in key.blobs.items() if sha not in others)
        return kept

    def _retired(self, key: _Key) -> bool:
        """Move ``key`` below ``retired/`` in one rename; whether it is gone from where readers look."""
        kind = key.path.parent.name
        target = self.root / RETIRED_DIR / f"{int(self.at)}-{kind}-{key.path.name}-{uuid.uuid4().hex[:12]}"
        try:
            target.parent.mkdir(mode=0o700, exist_ok=True)
            os.rename(key.path, target)
        except FileNotFoundError:
            return True  # another host retired it first
        except OSError as exc:
            self.done.problems.append(f"{kind} key {key.path.name[:12]} was not retired ({type(exc).__name__})")
            return False
        self.done.retired += 1
        return True

    def _waiting_blobs(self) -> Iterator[Mapping[str, int]]:
        """The blobs each retired key still in its grace lists."""
        for path in _children(self.root / RETIRED_DIR):
            named = _retired_name(path.name)
            if named is not None and self.at - named.retired_at <= self.limits.grace_sec:
                yield _listed(path, named.kind, named.key)

    def _sweep(self, listed: set[str], after: str | None) -> tuple[bool, str | None]:
        """Remove the stored blobs nothing in ``listed`` names that were stored longer ago than the grace, from the one
        after ``after`` round to it; whether every one was looked at, and the last one that was."""
        last = after
        for path in _blobs_from(self.root, after):
            if self._out_of_time("removing blobs"):
                return False, last
            last = path.name
            try:
                info = path.lstat()
            except OSError:
                continue
            if path.name in listed or self.at - info.st_mtime <= self.limits.grace_sec:
                continue
            if self._remove(path):
                self.done.blobs, self.done.bytes = self.done.blobs + 1, self.done.bytes + info.st_size
        return True, last

    def _write_cursor(self, cursor: dict[str, str]) -> None:
        """Keep where this prune stopped for the next, in one rename; one that cannot be written only starts the next
        from the beginning."""
        pending = self.root / f"{INCOMING_PREFIX}{uuid.uuid4().hex}.json"
        try:
            descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as out:
                out.write(json.dumps(cursor, sort_keys=True))
            os.replace(pending, self.root / PRUNE_CURSOR)
        except OSError as exc:
            self.done.problems.append(f"{PRUNE_CURSOR} was not written ({type(exc).__name__})")
            with contextlib.suppress(OSError):
                pending.unlink()

    def _remove(self, path: Path) -> bool:
        """Remove ``path`` (a tree, never through a link); whether this call removed it."""
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return False  # another host removed it first
        except OSError as exc:
            self.done.problems.append(f"{path.name[:24]} was not removed ({type(exc).__name__})")
            return False
        return True


_KEY, _BLOB = "key", "blob"


def _read_cursor(root: Path) -> dict[str, str]:
    """Where the last stopped prune left off: the last live key (``<kind>/<key>``) and the last blob it looked at; none
    when nothing stopped, and when the file cannot be read or says anything else."""
    try:
        cursor = json.loads((root / PRUNE_CURSOR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(cursor, dict):
        return {}
    return {name: value for name, value in cursor.items() if name in (_KEY, _BLOB) and isinstance(value, str)}


def _from(names: list[str], after: str | None) -> list[str]:
    """``names`` (sorted) from the first after ``after``, round to it."""
    if after is None:
        return names
    split = bisect.bisect_right(names, after)
    return names[split:] + names[:split]


def _live_names(root: Path) -> Iterator[str]:
    for kind in KINDS:
        for path in _children(root / SNAPSHOTS_DIR / kind.name):
            if SHA256_RE.fullmatch(path.name):
                yield f"{kind.name}/{path.name}"


def _live_key(root: Path, name: str) -> _Key | None:
    """The live key ``<kind>/<key>``; ``None`` when it is no directory (any more)."""
    kind_name, key = name.split("/")
    path = root / SNAPSHOTS_DIR / kind_name / key
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None  # another host retired it
    if not stat.S_ISDIR(info.st_mode):
        return None
    used = info.st_mtime
    return _Key(path, used, _listed(path, _KINDS_BY_NAME[kind_name], key))


def _listed(directory: Path, kind: CacheKind, key: str) -> Mapping[str, int]:
    """The blobs the valid snapshots in ``directory`` list, with the size each list gives; a list that does not verify is
    no reader's, so it keeps nothing."""
    found: dict[str, int] = {}
    for path in _children(directory):
        try:
            snapshot: Snapshot = DirJitStore.read_snapshot(path, kind, key)
        except SnapshotInvalid:
            continue
        found.update((entry.sha256, entry.size) for entry in snapshot.files.values())
    return found


def _blobs_from(root: Path, after: str | None) -> Iterator[Path]:
    """The stored blobs, shard by shard, from the first after the blob ``after`` round to it."""
    shards = _children(root / BLOBS_DIR)
    first = "" if after is None else after[:2]
    split = bisect.bisect_left([shard.name for shard in shards], first)
    shards = shards[split:] + shards[:split]
    resumed = after is not None and bool(shards) and shards[0].name == first
    for index, shard in enumerate(shards):
        children = _children(shard)
        if index == 0 and resumed:
            children = [path for path in children if path.name > after]
        yield from children
    if resumed:
        yield from (path for path in _children(shards[0]) if path.name <= after)


class _RetiredName(NamedTuple):
    """What a retired key's directory name says: when it was retired, and the kind and key it was."""

    retired_at: float
    kind: CacheKind
    key: str


def _retired_name(name: str) -> _RetiredName | None:
    """The parts of a retired key's directory name, ``None`` for any other name."""
    parts = name.split("-")
    if len(parts) != 4 or not parts[0].isdecimal() or parts[1] not in _KINDS_BY_NAME:
        return None
    if not SHA256_RE.fullmatch(parts[2]):
        return None
    return _RetiredName(float(parts[0]), _KINDS_BY_NAME[parts[1]], parts[2])


def _children(directory: Path) -> list[Path]:
    """What ``directory`` holds; nothing when it is not there. Any other error leaves the prune: a listing that failed is
    not an empty one, and taking it as one would remove the blobs of keys it did not see."""
    try:
        return sorted(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
