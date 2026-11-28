# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The host's side of the shared compiled-code cache: publishing what a signed result says its test compiled.

The runner stages a contribution under ``/out/jit-cache/<kind>/`` and signs the hash of each kind's snapshot list into its
result. Only the host, after it has verified that signature, brings the staged files over and adds them to the store: the
process that runs next to the server (and the AI that can reach it) never writes the cache. A contribution that does not
fetch, or is not the list the result signed, is skipped with a note; publishing never fails a result that was accepted.

What the sandbox left there is untrusted: a staged list or blob is fetched only when the sandbox's listing shows a regular
file (no link) of the size the signed list allows, and a copy that brings another byte count is refused; the store then
proves every blob against its hash before it keeps it.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

from hyperloom.measure.jit_kinds import kind_named
from hyperloom.measure.jit.store import (
    BLOBS_DIR,
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    DirJitStore,
    JitStoreError,
    Snapshot,
    read_staged_snapshot,
)
from hyperloom.measure.manifest import JitCacheFacts, JitContribution
from hyperloom.sandbox.socket import DirEntry, Sandbox, SandboxError

__all__ = ["STAGING_PATH", "PublishTally", "publish_contributions"]

log = logging.getLogger(__name__)

#: Where the runner stages its contributions, below the sandbox's output directory.
STAGING_PATH = "/out/jit-cache"
#: The state a signed result states for a kind its test found in the cache.
_HIT = "hit"


class PublishTally:
    """How many contributions a session published and how many of its tests found the cache warm, so that a session that
    only ever published says, as it closes, that its runners may not have been given what it publishes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._published = 0
        self._hits = 0

    def note(self, facts: JitCacheFacts, published: int) -> None:
        with self._lock:
            self._published += published
            self._hits += sum(1 for kind in facts.kinds if kind.state == _HIT)

    def warn_if_never_hit(self, store: DirJitStore) -> None:
        """Log one warning when contributions were published and no test found any kind warm; once per tally."""
        with self._lock:
            published, hits = self._published, self._hits
            self._published = 0
        if published and not hits:
            log.warning(
                "jit cache: this session published %d contribution(s) to %s and none of its tests found the cache warm; "
                "its runners may not have been given this store (a runner on this host reads it only where "
                "HYPERLOOM_JIT_CACHE names it, a runner in a sandbox only what this host serves it, and a measurement "
                "template must not set HYPERLOOM_JIT_CACHE; each test's notes say what it found)",
                published,
                store.root,
            )


def publish_contributions(store: DirJitStore, handle: Sandbox, facts: JitCacheFacts, scratch: Path) -> int:
    """Add the contribution of each kind the signed result lists; the number of kinds that were added to the store."""
    added = 0
    for kind in facts.kinds:
        if kind.contribution is None:
            continue
        try:
            kind_named(kind.kind)
            _publish_kind(store, handle, kind.kind, kind.contribution, scratch)
        except (JitStoreError, SandboxError, OSError, KeyError) as exc:
            log.warning("jit cache: a contribution was not published (%s)", type(exc).__name__)
            continue
        added += 1
    return added


def _publish_kind(store: DirJitStore, handle: Sandbox, kind: str, signed: JitContribution, scratch: Path) -> None:
    """Fetch the staged list, check it is the signed one, fetch the blobs the store does not have, and publish."""
    staged = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=scratch))
    source = f"{STAGING_PATH}/{kind}"
    (staged / BLOBS_DIR).mkdir()
    _sized(handle, source, MANIFEST_NAME, None)
    raw = handle.read_file(f"{source}/{MANIFEST_NAME}")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise JitStoreError("the staged list is larger than a snapshot list may be")
    (staged / MANIFEST_NAME).write_bytes(raw)
    snapshot = _verified_snapshot(staged, signed.sha256)
    staged_by_runner = _staged_by_runner(store, snapshot)
    wanted = {
        entry.sha256: entry.size
        for relative, entry in snapshot.files.items()
        if relative in staged_by_runner or not store.holds(entry.sha256)
    }
    if wanted:
        blobs = _listing(handle, f"{source}/{BLOBS_DIR}")
        for sha256, size in sorted(wanted.items()):
            _sized_entry(blobs.get(sha256), sha256, size)
            if handle.copy_out(f"{source}/{BLOBS_DIR}/{sha256}", staged / BLOBS_DIR / sha256) != size:
                raise JitStoreError(f"the staged blob {sha256[:12]} did not come over as {size} bytes")
    store.publish(staged, signed.sha256)


def _listing(handle: Sandbox, directory: str) -> dict[str, DirEntry]:
    return {entry.name: entry for entry in handle.list_entries(directory)}


def _sized(handle: Sandbox, directory: str, name: str, size: int | None) -> None:
    """Refuse ``name`` below ``directory`` unless the sandbox lists it as a regular file (of ``size`` bytes, when given;
    else within the bound of a snapshot list)."""
    _sized_entry(_listing(handle, directory).get(name), name, size)


def _sized_entry(entry: DirEntry | None, shown: str, size: int | None) -> None:
    if entry is None or entry.kind != "file":
        raise JitStoreError(f"the staged {shown[:12]} is not a regular file")
    wrong = entry.size > MAX_MANIFEST_BYTES if size is None else entry.size != size
    if wrong:
        raise JitStoreError(f"the staged {shown[:12]} is not of the size its list allows")


def _staged_by_runner(store: DirJitStore, snapshot: Snapshot) -> frozenset[str]:
    """The files the runner staged itself: those the snapshot it extended did not have. Those are brought even when the
    store holds a blob of that name, so a stored blob that is damaged is repaired by the bytes the test built; a file taken
    from the cache was not staged, and is not fetched back. The parent's listing is read once for the whole snapshot: a
    snapshot holds thousands of files and each read parses the whole listing."""
    if snapshot.parent is None:
        return frozenset(snapshot.files)
    parent = store.listing(kind_named(snapshot.kind), snapshot.key, snapshot.parent)
    if parent is None:
        return frozenset()
    return frozenset(relative for relative, entry in snapshot.files.items() if parent.get(relative) != entry)


def _verified_snapshot(staged: Path, signed_sha256: str) -> Snapshot:
    snapshot, actual = read_staged_snapshot(staged)
    if actual != signed_sha256:
        raise JitStoreError("the staged list is not the one the result signed")
    return snapshot
