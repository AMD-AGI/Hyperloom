# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The case pool: CLOSE keeps the cases a session registered, and a later session on the same image and model is handed
them for its regression set."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import REPRODUCED, AttachmentRef, Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.pins import IMAGE_DIGEST_VAR, IMAGE_REF_VAR
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, RegressionCase, TestRecord
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV, TreeArchive
from hyperloom_multibranch.blackboard.rsi.briefing import KNOWN_DEFECT, CrossSessionBriefing, RecallCache
from hyperloom_multibranch.blackboard.rsi.cases import (
    CASE_FORMAT,
    CASES_DIR,
    NO_IMAGE_DIGEST,
    CasePool,
    SessionCases,
    image_digest_from_env,
    pool_cases,
    session_cases,
)
from hyperloom_multibranch.blackboard.tests.factories import FakeClock, result
from hyperloom_multibranch.blackboard.tests.sessions import scripted_session
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, pack_case

REPRO_SH = "#!/bin/sh\n[ \"$(./ask 'What is 6 x 7?')\" = 42 ]\n"
DIAGNOSIS = "rope_theta is read as a quarter of its value, so greedy answers drift past 256 tokens"
IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64


def _registered(
    snapshot: BlackboardSnapshot, blobs: LocalBlobStore, request_id: str, diagnosis: str = DIAGNOSIS
) -> BlackboardSnapshot:
    """``snapshot`` with the case (:data:`REPRO_SH`, ``diagnosis``) in ``blobs``, carried by the repro test
    ``request_id`` and registered as the regression case of ``c1``."""

    reference = AttachmentRef(FIX_CASE_ID, "dir", blobs.put(pack_case(REPRO_SH, diagnosis)))
    request = TestRequest(
        TestKind.REPRO, Purpose.REPRO, "s1", "c0", RecipeDelta(), attachments=(reference,), request_id=request_id
    )
    case = RegressionCase(FIX_CASE_ID, "c1", request_id)
    return dataclasses.replace(
        snapshot, records=(*snapshot.records, TestRecord(request)), regressions=(*snapshot.regressions, case)
    )


def _of_session(snapshot: BlackboardSnapshot, session_id: str) -> BlackboardSnapshot:
    assert snapshot.profile is not None
    return dataclasses.replace(snapshot, profile=dataclasses.replace(snapshot.profile, session_id=session_id))


def test_close_keeps_each_case_the_session_registered_once_with_where_it_came_from(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool(tmp_path / "cases")
    registered = _registered(snapshot, blobs, "p1")

    first = pool_cases(registered, blobs, pool, IMAGE)
    again = pool_cases(registered, blobs, pool, IMAGE)

    (entry,) = pool.entries()
    profile = snapshot.profile
    assert profile is not None
    assert (entry.repro_sh, entry.diagnosis) == (REPRO_SH, DIAGNOSIS)
    assert (entry.session_id, entry.fix_commit, entry.image_digest) == ("sess-1", "c1", IMAGE)
    identity = (entry.model, entry.framework, entry.framework_version, entry.hardware)
    assert identity == (profile.model, profile.framework, profile.framework_version, profile.hardware)
    assert entry.case_id == hashlib.sha256(pack_case(REPRO_SH, DIAGNOSIS)).hexdigest()
    assert (first.written, again.written) == (1, 0)


def test_a_session_is_offered_the_pooled_cases_of_its_image_and_its_model_only(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool(tmp_path / "cases")
    assert snapshot.profile is not None
    other_model = dataclasses.replace(snapshot, profile=dataclasses.replace(snapshot.profile, model="Qwen/Qwen3-8B"))

    pool_cases(_registered(snapshot, blobs, "p1"), blobs, pool, IMAGE)
    pool_cases(_registered(snapshot, blobs, "p2"), blobs, pool, OTHER_IMAGE)
    pool_cases(_registered(other_model, blobs, "p3", "a second defect of the same build"), blobs, pool, IMAGE)

    (offered,) = pool.cases_for(IMAGE, snapshot.profile.model)
    assert (offered.image_digest, offered.diagnosis) == (IMAGE, DIAGNOSIS)
    # The same case on another image is a case of that image too.
    assert len(pool.entries()) == 3


def test_close_pools_nothing_without_an_image_digest_and_leaves_out_a_case_it_cannot_read(
    snapshot: BlackboardSnapshot, tmp_path: Path
) -> None:
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool(tmp_path / "cases")
    registered = _registered(snapshot, blobs, "p1")
    # r1 is an explore test, which carries no case.
    unreadable = dataclasses.replace(
        registered, regressions=(*registered.regressions, RegressionCase("r1-case", "c1", "r1"))
    )

    no_image = pool_cases(registered, blobs, pool, "")
    partly = pool_cases(unreadable, blobs, pool, IMAGE)

    assert (no_image.written, no_image.reasons) == (0, (NO_IMAGE_DIGEST,))
    assert (partly.written, partly.skipped) == (1, 1)
    assert partly.reasons[0].startswith("r1-case: ")


def test_close_writes_each_case_beside_the_archive_in_a_file_named_for_its_case_image_and_model(
    snapshot: BlackboardSnapshot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    monkeypatch.setenv(IMAGE_REF_VAR, "rocm/vllm:v0.11")
    monkeypatch.setenv(IMAGE_DIGEST_VAR, IMAGE)
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool.from_env()
    assert pool is not None

    pooled = pool_cases(_registered(snapshot, blobs, "p1"), blobs, pool, image_digest_from_env())

    (entry,) = pool.entries()
    name = hashlib.sha256(f"{entry.case_id}:{IMAGE}:{entry.model}".encode()).hexdigest()
    document = json.loads((tmp_path / "archive" / CASES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    assert (pooled.written, entry.image_digest, document["format"]) == (1, IMAGE, CASE_FORMAT)
    then, before = (TreeArchive(tmp_path / "archive", at) for at in (entry.written_at, entry.written_at - 1.0))
    assert (CasePool.beside(then).entries(), CasePool.beside(before).entries()) == ((entry,), ())
    monkeypatch.setenv(IMAGE_DIGEST_VAR, "sha256:abc")
    assert image_digest_from_env() == ""


async def test_a_session_is_handed_other_sessions_cases_of_its_image_and_model_as_they_were_packed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())
    snapshot = await blackboard.snapshot()
    blobs, archive = LocalBlobStore(tmp_path / "blobs"), TreeArchive(tmp_path / "archive")
    pool = CasePool.beside(archive)
    pool_cases(_registered(_of_session(snapshot, "past-1"), blobs, "p1"), blobs, pool, IMAGE)
    pool_cases(_registered(snapshot, blobs, "p2", "a defect this very session found"), blobs, pool, IMAGE)

    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    offered = session_cases(IMAGE, blackboard)
    assert isinstance(offered, SessionCases)
    (case,) = await offered.cases()

    assert case.data == pack_case(REPRO_SH, DIAGNOSIS) and case.case_id == hashlib.sha256(case.data).hexdigest()
    assert case.origin == "past-1 c1"
    assert await SessionCases(pool, OTHER_IMAGE, blackboard).cases() == ()
    assert session_cases("", blackboard) is None


async def test_a_brief_names_a_pooled_case_that_reproduced_on_the_root_as_a_defect_the_root_carries(
    tmp_path: Path,
) -> None:
    blackboard = await scripted_session(tmp_path / "blackboard", FakeClock())
    snapshot = await blackboard.snapshot()
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool(tmp_path / "cases")
    pool_cases(_registered(_of_session(snapshot, "past-1"), blobs, "p1"), blobs, pool, IMAGE)
    (entry,) = pool.entries()
    # The regression set ran the pooled case on the root c0, where it reproduced.
    toolbox = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    case = (AttachmentRef(FIX_CASE_ID, "dir", entry.case_id),)
    repro = TestRequest(
        TestKind.REPRO, Purpose.REPRO, INTEGRATION_BRANCH, "c0", RecipeDelta(), attachments=case, request_id="p9"
    )
    await toolbox.write_request(repro)
    await toolbox.write_result(dataclasses.replace(result("p9"), repro_verdict=REPRODUCED))

    brief = await CrossSessionBriefing(
        blackboard, recalls=RecallCache(), champions={}, refused={}, cases=pool
    ).startup_brief("c1")
    without = await CrossSessionBriefing(
        blackboard, recalls=RecallCache(), champions={}, refused={}, cases=None
    ).startup_brief("c1")

    assert brief.lessons[0] == f"{KNOWN_DEFECT} past-1 c1: {DIAGNOSIS}"
    assert not any(line.startswith(KNOWN_DEFECT) for line in without.lessons)


def test_a_pooled_case_this_account_cannot_open_is_named_with_its_owner_and_mode(
    snapshot: BlackboardSnapshot, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blobs, pool = LocalBlobStore(tmp_path / "blobs"), CasePool(tmp_path / "cases")
    pool_cases(_registered(snapshot, blobs, "p1"), blobs, pool, IMAGE)
    unopenable = pool.root / "other-account.json"
    unopenable.mkdir()  # read_text fails with an OSError as a file another account wrote owner-only does
    unopenable.chmod(0o700)

    with caplog.at_level(logging.WARNING, logger="hyperloom_multibranch.blackboard.rsi.cases"):
        (entry,) = pool.entries()

    info = unopenable.stat()
    assert entry.session_id == "sess-1"
    assert f"{unopenable} (owner {info.st_uid}:{info.st_gid} 0700)" in caplog.text
