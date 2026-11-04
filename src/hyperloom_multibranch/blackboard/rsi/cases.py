# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The case pool: the reproduction cases of the correctness fixes every session registered, kept beside the archive
for the sessions after it on the same image and model.

CLOSE keeps each case the session registered once (:func:`pool_cases`), with where it came from: the session, the fix
commit, the image digest, the model, the framework and its version, and the hardware. A case runs against the
framework source of the image it was found on, so a later session is handed the cases of its own image digest and
model only (:class:`SessionCases`), for its regression set to run on its root before its first check
(:class:`~hyperloom_multibranch.rulegate.regression.RegressionSet`). A case that reproduced there is a defect the root
carries (:func:`known_defects`), which the briefs name.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.contracts.measurement import REPRODUCED, TestKind
from hyperloom.measure.compile import BlobError, BlobStore
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.blackboard.interfaces import SessionRecords
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot
from hyperloom_multibranch.blackboard.codec import decode, encode
from hyperloom_multibranch.blackboard.local.files import encode_json, publish_once
from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive
from hyperloom_multibranch.rulegate.fix_case import CaseError, case_of, pack_case, read_case
from hyperloom_multibranch.rulegate.regression import PooledCase

log = logging.getLogger(__name__)

#: The digest of the image the session runs, which the cli requires.
IMAGE_DIGEST_ENV = "HYPERLOOM_ACTIVE_IMAGE_DIGEST"
CASES_DIR = "cases"
CASE_FORMAT = "multibranch.regression-case"
NO_IMAGE_DIGEST = "no_image_digest"
NO_PROFILE = "no_session_profile"


@dataclass(frozen=True)
class PoolEntry:
    """One case as the pool keeps it: its ``repro.sh`` and diagnosis, and where it came from; ``written_at`` is the
    time of the snapshot CLOSE wrote it from."""

    repro_sh: str
    diagnosis: str
    session_id: str
    fix_commit: str
    image_digest: str
    model: str
    framework: str
    framework_version: str
    hardware: str
    written_at: float

    @property
    def data(self) -> bytes:
        return pack_case(self.repro_sh, self.diagnosis)

    @property
    def case_id(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def origin(self) -> str:
        return f"{self.session_id} {self.fix_commit[:12]}"


@dataclass(frozen=True)
class CasePool:
    """``<archive>/cases``, a file a case for each image and model it was found on, written once; with ``as_of``,
    read as it stood then."""

    root: Path
    as_of: float | None = None

    @classmethod
    def beside(cls, archive: TreeArchive) -> CasePool:
        """The pool beside ``archive``, read as of the time it is."""

        return cls(archive.root.parent / CASES_DIR, archive.as_of)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> CasePool | None:
        """The pool beside the archive a running session reads; ``None`` when it reads none
        (:meth:`TreeArchive.read_from_env`)."""

        archive = TreeArchive.read_from_env(env)
        return None if archive is None else cls.beside(archive)

    def write(self, entry: PoolEntry) -> bool:
        """Keep ``entry`` unless the pool holds its case for its image and model already."""

        name = hashlib.sha256(f"{entry.case_id}:{entry.image_digest}:{entry.model}".encode()).hexdigest()
        return publish_once(self.root / f"{name}.json", encode_json({"format": CASE_FORMAT, "case": encode(entry)}))

    def entries(self) -> tuple[PoolEntry, ...]:
        """Every readable case, by file name; a file of another format, or one that does not parse, is left out."""

        found = []
        for path in sorted(self.root.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if document.get("format") == CASE_FORMAT:
                    entry = decode(PoolEntry, document.get("case"))
                    if self.as_of is None or entry.written_at <= self.as_of:
                        found.append(entry)
            except (OSError, AttributeError, KeyError, TypeError, ValueError):
                continue
        return tuple(found)

    def cases_for(self, image_digest: str, model: str) -> tuple[PoolEntry, ...]:
        return tuple(entry for entry in self.entries() if (entry.image_digest, entry.model) == (image_digest, model))


@dataclass(frozen=True)
class CasesPooled:
    """What CLOSE did with the session's cases: how many it wrote, how many it could not read, and why."""

    written: int = 0
    skipped: int = 0
    reasons: tuple[str, ...] = ()


def pool_cases(snapshot: BlackboardSnapshot, blobs: BlobStore, pool: CasePool, image_digest: str) -> CasesPooled:
    """Keep in ``pool`` each case the session registered, read from the session's ``blobs``, as a case of the image
    ``image_digest`` and the session's model; none without either."""

    profile = snapshot.profile
    if not image_digest:
        return CasesPooled(reasons=(NO_IMAGE_DIGEST,))
    if profile is None:
        return CasesPooled(reasons=(NO_PROFILE,))
    records = {record.request_id: record for record in snapshot.records}
    written, reasons = 0, []
    for case in snapshot.regressions:
        record = records.get(case.request_id)
        try:
            reference = case_of(record.request) if record is not None else None
            if reference is None:
                raise CaseError(f"{case.request_id} carries no case")
            found = read_case(blobs.get(reference.sha256))
        except (BlobError, CaseError) as exc:
            reasons.append(f"{case.case_ref}: {exc}")
            continue
        entry = PoolEntry(
            found.repro_sh,
            found.diagnosis,
            profile.session_id,
            case.fix_commit,
            image_digest,
            profile.model,
            profile.framework,
            profile.framework_version,
            profile.hardware,
            snapshot.taken_at,
        )
        written += int(pool.write(entry))
    return CasesPooled(written, len(reasons), tuple(reasons))


def known_defects(snapshot: BlackboardSnapshot, pool: CasePool) -> tuple[PoolEntry, ...]:
    """The pooled cases that reproduced on the session's root: defects the root carries, which no merge is held to."""

    root = snapshot.baseline.root_commit if snapshot.baseline is not None else None
    reproduced = {
        item.sha256
        for record in snapshot.records
        if record.request.kind is TestKind.REPRO
        and (record.request.branch_id, record.request.parent_commit) == (INTEGRATION_BRANCH, root)
        and record.result is not None
        and record.result.repro_verdict == REPRODUCED
        for item in record.request.attachments
    }
    found: dict[str, PoolEntry] = {}
    for entry in pool.entries() if reproduced else ():
        if entry.case_id in reproduced:
            found.setdefault(entry.case_id, entry)
    return tuple(found.values())


def image_digest_from_env(env: Mapping[str, str] | None = None) -> str:
    return str((os.environ if env is None else env).get(IMAGE_DIGEST_ENV) or "").strip()


@dataclass(frozen=True)
class SessionCases:
    """The ``pool``'s cases of the session's image ``image_digest`` and model, other sessions' only, as the regression
    set takes them; none before the session has a profile, or when the pool cannot be read."""

    pool: CasePool
    image_digest: str
    blackboard: SessionRecords

    async def cases(self) -> tuple[PooledCase, ...]:
        try:
            profile = await self.blackboard.session_profile()
        except NotRecorded:
            return ()
        try:
            entries = await asyncio.to_thread(self.pool.cases_for, self.image_digest, profile.model)
        except OSError as exc:
            log.warning("could not read the case pool %s: %s", self.pool.root, exc)
            return ()
        others = [entry for entry in entries if entry.session_id != profile.session_id]
        return tuple(PooledCase(entry.case_id, entry.data, entry.origin) for entry in others)


def session_cases(
    image_digest: str, blackboard: SessionRecords, archive: TreeArchive | None = None
) -> SessionCases | None:
    """The pool for ``RegressionSet(pool=...)``: beside ``archive``, by default the one a running session reads;
    ``None`` without one or without the session's image digest."""

    target = archive if archive is not None else TreeArchive.read_from_env()
    if target is None or not image_digest:
        return None
    return SessionCases(CasePool.beside(target), image_digest, blackboard)


__all__ = [
    "CASES_DIR",
    "CASE_FORMAT",
    "IMAGE_DIGEST_ENV",
    "NO_IMAGE_DIGEST",
    "NO_PROFILE",
    "CasePool",
    "CasesPooled",
    "PoolEntry",
    "SessionCases",
    "image_digest_from_env",
    "known_defects",
    "pool_cases",
    "session_cases",
]
