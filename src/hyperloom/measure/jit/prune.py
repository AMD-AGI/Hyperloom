# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bounded clean-up of the shared compiled-code cache, run by the host that publishes to it.

The store only adds (:mod:`hyperloom.measure.jit.store`), so without this it grows with every image, card and patched
aiter tree a deployment ever measured. A key's directory (``snapshots/<kind>/<key>``) is its unit: the host stamps its
time each time an accepted result names the key (:func:`note_used`), and a publication stamps it too, so that time is
when the key was last used.

:func:`prune` works in two steps, so a test that is reading an entry never loses it:

1. A key unused for :data:`MAX_IDLE_SEC`, or, while the store holds more than :data:`MAX_STORE_BYTES`, the least
   recently used keys unused for at least :data:`RETIRE_GRACE_SEC`, are *retired*: the key's directory is moved below
   ``retired/`` in one rename. No new test finds it from then on; a test that already read its list keeps finding the
   files it names, since no blob is removed in this step.
2. A retired key is removed only :data:`RETIRE_GRACE_SEC` after it was retired, longer than any test runs, and a blob
   only when no snapshot of a live or a still-retired key lists it and it was stored more than that long ago (so a
   publisher that stored it and has not yet renamed its list into place keeps it). A publisher's leftover
   (``.incoming-*``) of that age is removed too.

A test that read a key just before it was retired and publishes after it extends it under the same key again: the files
it took from the cache are still stored, as the retired list names them. Two hosts may prune one store at once: every
step is a rename or a removal that the other's finding the name gone leaves correct. Nothing here fails a session; the
host logs what it could not do and the store is pruned at the next start.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import uuid
from collections.abc import Iterator
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
#: Past this many stored bytes the least recently used keys are retired until the rest fit. An aiter tree's modules
#: are a few GiB at most, so this keeps dozens of image and tree combinations.
MAX_STORE_BYTES = 200 * 1024**3
#: How long a retired key, and a blob nothing lists, stay before they are removed: longer than any test runs (the boot
#: budget is at most six hours and the measured window a few more), so no test still reading one loses it.
RETIRE_GRACE_SEC = 2 * _DAY
#: Where retired keys wait, below the store's root, as ``<retired at>-<kind>-<key>-<nonce>``.
RETIRED_DIR = "retired"
_KINDS_BY_NAME = {kind.name: kind for kind in KINDS}


@dataclass(frozen=True)
class PruneLimits:
    """The bounds :func:`prune` keeps the store to (see the module)."""

    max_idle_sec: float = MAX_IDLE_SEC
    max_bytes: int = MAX_STORE_BYTES
    grace_sec: float = RETIRE_GRACE_SEC


@dataclass
class Pruned:
    """What one :func:`prune` did: keys retired and removed, blobs and bytes removed, and what it could not do."""

    retired: int = 0
    removed: int = 0
    blobs: int = 0
    bytes: int = 0
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Key:
    path: Path
    used: float
    blobs: frozenset[str]


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
    """Retire the keys unused too long or past the size bound, then remove what was retired long enough ago and the blobs
    nothing lists any more (see the module)."""
    run = _Run(store.root, limits, time.time() if now is None else now)
    if (store.root / SNAPSHOTS_DIR).is_dir():
        run.prune()
    return run.done


def prune_logged(store: DirJitStore, limits: PruneLimits = PruneLimits()) -> Pruned | None:
    """:func:`prune`, said in the log; ``None`` when the store could not be read at all, which is logged too."""
    try:
        done = prune(store, limits)
    except OSError as exc:
        log.warning("jit cache: %s was not pruned (%s)", store.root, type(exc).__name__)
        return None
    if done.retired or done.removed or done.blobs:
        log.info(
            "jit cache: %s pruned: %d key(s) retired, %d removed, %d blob(s) of %d bytes removed",
            store.root,
            done.retired,
            done.removed,
            done.blobs,
            done.bytes,
        )
    for problem in done.problems:
        log.warning("jit cache: %s", problem)
    return done


@dataclass
class _Run:
    """One pruning of the store at ``root`` at the time ``at``."""

    root: Path
    limits: PruneLimits
    at: float
    done: Pruned = field(default_factory=Pruned)

    def prune(self) -> None:
        stored = _stored_blobs(self.root)
        kept = self._retire(list(_live_keys(self.root)), stored)
        listed = set().union(*(key.blobs for key in kept), *self._waiting_blobs())
        for sha256, (path, size, stored_at) in stored.items():
            if sha256 in listed or self.at - stored_at <= self.limits.grace_sec:
                continue
            if self._remove(path):
                self.done.blobs, self.done.bytes = self.done.blobs + 1, self.done.bytes + size
        self._remove_leftovers()

    def _retire(self, live: list[_Key], stored: dict[str, tuple[Path, int, float]]) -> list[_Key]:
        """Retire the idle keys, then the least recently used ones past the grace while the store is over its size; the
        keys left live."""
        kept = []
        for key in live:
            if self.at - key.used <= self.limits.max_idle_sec or not self._retired(key):
                kept.append(key)
        total = sum(size for _, size, _ in stored.values())
        for key in sorted(kept, key=lambda key: key.used):
            if total <= self.limits.max_bytes or self.at - key.used <= self.limits.grace_sec:
                break
            others = set().union(*(other.blobs for other in kept if other is not key))
            if self._retired(key):
                kept.remove(key)
                total -= sum(stored[sha][1] for sha in key.blobs - others if sha in stored)
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

    def _waiting_blobs(self) -> Iterator[frozenset[str]]:
        """Remove each retired key whose grace has passed; the blobs each one still waiting lists."""
        for path in _children(self.root / RETIRED_DIR):
            named = _retired_name(path.name)
            if named is None:
                continue
            if self.at - named.retired_at <= self.limits.grace_sec:
                yield _listed(path, named.kind, named.key)
            elif self._remove(path):
                self.done.removed += 1

    def _remove_leftovers(self) -> None:
        for path in _children(self.root):
            if not path.name.startswith(INCOMING_PREFIX):
                continue
            try:
                stale = self.at - path.lstat().st_mtime > self.limits.grace_sec
            except OSError:
                continue
            if stale:
                self._remove(path)

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


def _live_keys(root: Path) -> Iterator[_Key]:
    for kind in KINDS:
        directory = root / SNAPSHOTS_DIR / kind.name
        for path in _children(directory):
            if SHA256_RE.fullmatch(path.name) and path.is_dir() and not path.is_symlink():
                yield _Key(path, path.stat().st_mtime, _listed(path, kind, path.name))


def _listed(directory: Path, kind: CacheKind, key: str) -> frozenset[str]:
    """The blobs the valid snapshots in ``directory`` list; a list that does not verify is no reader's, so it keeps
    nothing."""
    found: set[str] = set()
    for path in _children(directory):
        try:
            snapshot: Snapshot = DirJitStore.read_snapshot(path, kind, key)
        except SnapshotInvalid:
            continue
        found.update(entry.sha256 for entry in snapshot.files.values())
    return frozenset(found)


def _stored_blobs(root: Path) -> dict[str, tuple[Path, int, float]]:
    """Each stored blob by its hash: its path, size and when it was stored."""
    blobs: dict[str, tuple[Path, int, float]] = {}
    for shard in _children(root / BLOBS_DIR):
        for path in _children(shard):
            try:
                info = path.lstat()
            except OSError:
                continue
            blobs[path.name] = (path, info.st_size, info.st_mtime)
    return blobs


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
    try:
        return sorted(directory.iterdir())
    except OSError:
        return []
