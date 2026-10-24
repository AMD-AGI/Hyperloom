# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recorded result is verified again from its archived bundle and what acceptance journalled (L26)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.archive import ArchiveError
from hyperloom.measure.bundle import RecordedBundle, RejectReason, ResultRejected
from hyperloom.measure.hosted_log import STATE_FINISHED, Record
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.helpers import build
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import POLICY, PointPlanner, _run

THROUGHPUT = "output_throughput"


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


async def _accepted(tmp_path: Path, r: Rig, **fake: object) -> tuple[HostedExecutor, TestResult]:
    point = point_yaml(tmp_path, "planned.yaml", **{"fake_accuracy": "0.64", **fake})  # the default request asks for it
    executor = HostedExecutor(r.backend, tmp_path / "session")
    tb = build(tmp_path, executor=executor, planner=PointPlanner(point), policy=POLICY)
    _, result = await _run(tb)
    assert result.manifest_sha256, f"the test must have been accepted: {result.failure_reason}"
    return HostedExecutor(r.backend, tmp_path / "session"), result


def _rewrite_journal(r: Rig, edit) -> None:  # type: ignore[no-untyped-def]
    records = [edit(rec) for rec in r.backend.journal.records()]
    r.backend.journal.path.write_text("".join(rec.to_line() + "\n" for rec in records), encoding="utf-8")


def _without(rec: Record, key: str) -> Record:
    if rec.state != STATE_FINISHED:
        return rec
    return replace(rec, extra={k: v for k, v in rec.extra.items() if k != key})


async def test_an_untouched_result_reverifies_without_a_live_sandbox(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_throughput=4321.5)
    assert rig.provider.find({}, role="measurement") == [], "no runner is alive: the key comes from the journal"
    assert executor.difference(result) is None


async def test_the_key_is_kept_beside_the_journal_entry_from_the_platform_report(tmp_path: Path, rig) -> None:
    _, result = await _accepted(tmp_path, rig)
    record = rig.backend.journal.latest(result.request_id)
    assert str(record.extra["public_key_pem"]).startswith("-----BEGIN PUBLIC KEY-----")
    assert record.extra["spec_fingerprint"] == result.spec_fingerprint


async def test_an_edited_throughput_with_untouched_hashes_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_throughput=4321.5)
    first = result.rounds[0]
    edited = replace(result, rounds=(replace(first, metrics={**first.metrics, THROUGHPUT: 9999.0}),))
    assert (edited.manifest_sha256, edited.signing_key_fingerprint) == (
        result.manifest_sha256,
        result.signing_key_fingerprint,
    )
    assert executor.difference(edited) == (
        f"rounds[0].metrics.{THROUGHPUT}: recorded 9999.0, the bundle has {first.metrics[THROUGHPUT]!r}"
    )


async def test_a_dropped_round_and_a_dropped_score_are_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert "rounds: recorded 0, the bundle has" in executor.difference(replace(result, rounds=()))
    assert executor.difference(replace(result, accuracy=())) == "accuracy: recorded 0 scores, the bundle has 1"


async def test_an_edited_accuracy_score_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    edited = replace(result, accuracy=(replace(result.accuracy[0], score=0.99),))
    assert executor.difference(edited) == "accuracy[0].score: recorded 0.99, the bundle has 0.64"


async def test_a_graded_result_whose_one_evaluation_ran_in_the_warmup_round_reverifies_from_its_bundle(
    tmp_path: Path, rig
) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_eval_in_warmup=True)
    record = rig.backend.journal.latest(result.request_id)
    derived = rig.backend.reverify(
        result.request_id,
        RecordedBundle(result.manifest_sha256, result.signing_key_fingerprint),
        tmp_path / "again",
    )

    assert [r.phase for r in derived.rounds] == ["warmup", "accuracy", "measure"]
    assert derived.rounds[1].output_dir == derived.rounds[0].output_dir != derived.rounds[2].output_dir
    assert [value.score for value in result.accuracy] == [0.64] and record.extra["verified"] is True
    assert [r.phase for r in result.rounds] == ["warmup", "measure"], "the gate reads the same rounds and score"
    assert (derived.rounds[1].output_dir / "eval_output" / "results_2026-10-14T10-00-00.json").is_file(), "the evidence"
    assert executor.difference(result) is None
    edited = replace(result, accuracy=(replace(result.accuracy[0], score=0.99),))
    assert executor.difference(edited) == "accuracy[0].score: recorded 0.99, the bundle has 0.64"


async def test_an_edited_spec_fingerprint_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    other = "f" * len(result.spec_fingerprint)
    assert executor.difference(replace(result, spec_fingerprint=other)).startswith("spec_fingerprint: recorded")


async def test_a_record_without_a_bundle_hash_names_no_bundle(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert executor.difference(replace(result, manifest_sha256=None)) == "manifest_sha256: the record names no bundle"


async def test_a_key_that_was_not_kept_is_refused(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    _rewrite_journal(rig, lambda rec: _without(rec, "public_key_pem"))
    with pytest.raises(ResultRejected) as raised:
        executor.difference(result)
    assert raised.value.reason is RejectReason.PUBLIC_KEY_UNUSABLE


async def test_a_swapped_key_does_not_verify_the_bundle(tmp_path: Path, rig) -> None:
    from hyperloom.measure import signing

    executor, result = await _accepted(tmp_path, rig)
    other = signing.public_key_pem(signing.generate_private_key())
    _rewrite_journal(
        rig,
        lambda rec: replace(rec, extra={**rec.extra, "public_key_pem": other}) if rec.state == STATE_FINISHED else rec,
    )
    with pytest.raises(ResultRejected) as raised:
        executor.difference(result)
    assert raised.value.reason is RejectReason.BAD_SIGNATURE


async def test_a_bundle_the_journal_never_accepted_is_refused(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    with pytest.raises(ResultRejected) as raised:
        executor.difference(replace(result, manifest_sha256="0" * 64))
    assert raised.value.reason is RejectReason.RECORD_MISMATCH


async def test_a_bundle_signed_by_another_key_than_recorded_is_refused(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    with pytest.raises(ResultRejected) as raised:
        executor.difference(replace(result, signing_key_fingerprint="not-the-key"))
    assert raised.value.reason is RejectReason.RECORD_MISMATCH


async def test_an_edited_archived_file_is_refused(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    stored = rig.archive.bundle_path(result.request_id, result.manifest_sha256)
    report = next(stored.rglob("benchmark_report.json"))
    report.chmod(0o600)
    report.write_text(json.dumps({"throughput": {"output_throughput": 1.0}}), encoding="utf-8")
    with pytest.raises(ResultRejected) as raised:
        executor.difference(result)
    assert raised.value.reason in (RejectReason.FILE_HASH_MISMATCH, RejectReason.FILE_SIZE_MISMATCH)


async def test_an_archive_that_cannot_be_reached_raises_the_archive_error(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)

    def unreachable(request_id: str, manifest_sha256: str) -> Path:
        raise ArchiveError("unreachable")

    rig.archive.bundle_path = unreachable  # type: ignore[method-assign]
    with pytest.raises(ArchiveError, match="unreachable"):
        executor.difference(result)


async def test_backend_reverify_lands_the_files_it_verified(tmp_path: Path, rig) -> None:
    _, result = await _accepted(tmp_path, rig)
    out = tmp_path / "again"
    derived = rig.backend.reverify(
        result.request_id, RecordedBundle(result.manifest_sha256, result.signing_key_fingerprint), out
    )
    assert [r.phase for r in derived.rounds] == ["measure", "accuracy"] and derived.rounds[0].output_dir.is_relative_to(
        out
    )
    assert next(out.rglob("benchmark_report.json")).is_file()


# -- Executor.reverify: what the rule gate sees --------------------------------------------------------------------


async def _reason(executor: HostedExecutor, result: TestResult) -> str:
    with pytest.raises(ResultUnverified) as raised:
        await executor.reverify(result)
    assert raised.value.request_id == result.request_id
    return raised.value.reason


async def test_an_untouched_result_passes_executor_reverify(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert await executor.reverify(result) is None


async def test_a_difference_is_the_reason(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    edited = replace(result, accuracy=(replace(result.accuracy[0], score=0.99),))
    assert await _reason(executor, edited) == "accuracy[0].score: recorded 0.99, the bundle has 0.64"


async def test_a_rejected_bundle_names_the_reject_reason(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    reason = await _reason(executor, replace(result, manifest_sha256="0" * 64))
    assert reason.startswith(f"the bundle was rejected ({RejectReason.RECORD_MISMATCH.value}): ")


async def test_an_unreachable_archive_names_the_archive(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)

    def unreachable(request_id: str, manifest_sha256: str) -> Path:
        raise ArchiveError("unreachable")

    rig.archive.bundle_path = unreachable  # type: ignore[method-assign]
    assert await _reason(executor, result) == "the archive cannot be read: unreachable"


async def test_an_unreadable_file_names_the_os_error(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)

    def broken(request_id: str, manifest_sha256: str) -> Path:
        raise PermissionError("secret path")

    rig.archive.bundle_path = broken  # type: ignore[method-assign]
    assert await _reason(executor, result) == "the archive cannot be read: PermissionError"


async def test_a_record_without_a_bundle_is_unverified_by_executor_reverify(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert await _reason(executor, replace(result, manifest_sha256=None)) == (
        "manifest_sha256: the record names no bundle"
    )


# -- a result that did not succeed: the signature path still runs, no values are re-derived ------------------------


async def test_a_failed_result_still_has_its_bundle_signature_checked(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    failed = replace(result, validity=Validity.CANDIDATE_FAILED, rounds=(), accuracy=())
    assert await executor.reverify(failed) is None
    reason = await _reason(executor, replace(failed, manifest_sha256="0" * 64))
    assert reason.startswith(f"the bundle was rejected ({RejectReason.RECORD_MISMATCH.value}): ")


async def test_a_failed_result_that_records_measured_rounds_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    failed = replace(result, validity=Validity.CANDIDATE_FAILED, accuracy=())
    assert await _reason(executor, failed) == "rounds: a result that did not succeed records measured rounds"


WARNINGS = '["warmup_round_seed:1", "other"]'


async def test_the_warnings_of_the_rounds_reach_the_result_in_order(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_warnings=WARNINGS)
    assert result.nonfatal_warnings == ("warmup_round_seed:1", "other")
    assert executor.difference(result) is None


async def test_a_result_of_a_run_without_warnings_records_none(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert result.nonfatal_warnings == ()
    assert executor.difference(result) is None


@pytest.mark.parametrize(
    "edit",
    [
        lambda have: (),
        lambda have: (*have, "invented"),
        lambda have: have[:1],
        lambda have: have[::-1],
    ],
    ids=["dropped", "added", "shortened", "reordered"],
)
async def test_an_edited_warning_list_is_named(tmp_path: Path, rig, edit) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_warnings=WARNINGS)
    difference = executor.difference(replace(result, nonfatal_warnings=edit(result.nonfatal_warnings)))
    assert difference is not None and difference.startswith("nonfatal_warnings: recorded ")
    assert "the bundle has ['warmup_round_seed:1', 'other']" in difference


async def test_warnings_added_to_a_run_that_had_none_are_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig)
    assert await _reason(executor, replace(result, nonfatal_warnings=("x",))) == (
        "nonfatal_warnings: recorded ['x'], the bundle has []"
    )


async def test_a_failed_result_that_records_warnings_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_warnings=WARNINGS)
    failed = replace(result, validity=Validity.CANDIDATE_FAILED, rounds=(), accuracy=(), nonfatal_warnings=())
    assert await executor.reverify(failed) is None
    reason = await _reason(executor, replace(failed, nonfatal_warnings=("invented",)))
    assert reason == "nonfatal_warnings: recorded ['invented'], the bundle has []"


FOREIGN_REASON = (
    "1 held the test's cards: pid 900 on card 0, first seen 2026-10-24T04:05:06Z during the test: python3 x.py"
)
KEPT_BESIDE_A_PROCESS = {"state": "foreign", "reason": FOREIGN_REASON, "checks": 5}


async def test_a_result_kept_beside_a_process_by_a_guard_that_observes_reverifies_with_its_warning(
    tmp_path: Path, rig
) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_card_occupancy=json.dumps(KEPT_BESIDE_A_PROCESS))

    assert result.nonfatal_warnings == (
        f"foreign_process_on_card, kept because the card guard only observes: {FOREIGN_REASON}",
    )
    assert executor.difference(result) is None


async def test_a_result_that_dropped_the_warning_of_a_process_the_bundle_signed_is_named(tmp_path: Path, rig) -> None:
    executor, result = await _accepted(tmp_path, rig, fake_card_occupancy=json.dumps(KEPT_BESIDE_A_PROCESS))

    difference = executor.difference(replace(result, nonfatal_warnings=()))

    assert difference is not None and difference.startswith("nonfatal_warnings")


async def test_a_clean_result_that_claims_the_warning_of_a_process_is_named(tmp_path: Path, rig) -> None:
    clean = {"state": "clean", "reason": None, "checks": 5}
    executor, result = await _accepted(tmp_path, rig, fake_card_occupancy=json.dumps(clean))

    forged = replace(
        result,
        nonfatal_warnings=(f"foreign_process_on_card, kept because the card guard only observes: {FOREIGN_REASON}",),
    )

    difference = executor.difference(forged)
    assert difference is not None and difference.startswith("nonfatal_warnings")
