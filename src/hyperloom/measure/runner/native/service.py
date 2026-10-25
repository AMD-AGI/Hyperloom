# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one driver of native builds: equal keys are coalesced, each key is built at most once, and the service signs.

:func:`ensure_build` holds the key's exclusive ``flock`` (:func:`hyperloom.measure.runner.cache_build.exclusive`, the
runner cache's own) around looking the key up and building it, so a second request of a key waits for the first and
gets its entry with ``reused`` set: build seconds are charged once, to the request that built. The private key stays
here; the build itself runs candidate-controlled code and never sees it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.timeutil import now_iso
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import READY, BuildRecord, OutputFile
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import Signer
from hyperloom.measure.runner.cache_build import CacheBuildError, exclusive

__all__ = ["DEFAULT_LOCK_TIMEOUT_SEC", "Ensured", "Staged", "ensure_build"]

#: Longer than the longest recipe budget, so a waiter outlives the build it waits for.
DEFAULT_LOCK_TIMEOUT_SEC = 3 * 3600.0
#: Seconds are recorded to the millisecond.
_SECOND_DIGITS = 3


@dataclass(frozen=True)
class Staged:
    """What a build hands the service: each output's bytes at ``<staging>/blobs/<sha256>``, or the failure of a failed
    build. ``cpu_seconds`` is the compile's CPU time, the unit build accounting is kept in."""

    status: str
    outputs: tuple[OutputFile, ...]
    toolchain: dict[str, str]
    cpu_seconds: float
    wall_seconds: float
    log: str
    staging: Path
    failure: str = ""


@dataclass(frozen=True)
class Ensured:
    """The entry of a key; ``reused`` when another request built it, so its ``build_seconds`` are not charged again."""

    record: BuildRecord
    reused: bool


Build = Callable[[Path], Staged]


def ensure_build(
    store: BuildStore,
    inputs: BuildInputs,
    build: Build,
    signer: Signer,
    *,
    lock_timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
) -> Ensured:
    """The entry of ``inputs``, built by ``build`` in a fresh scratch directory when there is none.

    Raises:
        BuildRefused: When an entry is there and fails a check (it is not rebuilt over).
        BuildInfraError: When the build failed for machine reasons or the lock was not had in time; nothing is stored.
    """
    lock = store.lock_path(inputs)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive(lock, lock_timeout_sec):
            store.clear_leftovers(inputs)
            found = store.read(inputs)
            if found is not None:
                return Ensured(found, reused=True)
            return Ensured(_build(store, inputs, build, signer), reused=False)
    except CacheBuildError as exc:
        raise BuildInfraError(str(exc)) from None


def _build(store: BuildStore, inputs: BuildInputs, build: Build, signer: Signer) -> BuildRecord:
    with tempfile.TemporaryDirectory(prefix="hl-native-") as scratch:
        staged = build(Path(scratch))
        record = BuildRecord(
            key=inputs.key,
            inputs=inputs,
            status=staged.status,
            outputs=staged.outputs if staged.status == READY else (),
            toolchain=staged.toolchain,
            built_at=now_iso(z_suffix=True),
            build_seconds=round(staged.cpu_seconds, _SECOND_DIGITS),
            wall_seconds=round(staged.wall_seconds, _SECOND_DIGITS),
            failure=staged.failure,
        )
        store.publish(record, staged.staging, staged.log, signer)
    return record
