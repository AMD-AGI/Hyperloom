# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom_multibranch.specialist.keep_requeue import REQUEUE_FAILURES, Ruling, rule

BUSY = "4 attempts failed for infrastructure reasons: foreign_process_on_card: pid 4242 on card 3"


def _ended(validity: Validity, reason: str = BUSY, failure: FailureClass | None = FailureClass.INFRA) -> TestResult:
    return TestResult(request_id="r1", validity=validity, failure_class=failure, failure_reason=reason)


@pytest.mark.parametrize(
    "reason, ruling",
    [
        (BUSY, Ruling.REQUEUE),
        ("sandbox_acquire_timeout: waited 1800 s for a slot of the pool", Ruling.REQUEUE),
        ("inert_patch: process 1 imported aiter from the image", Ruling.SETTLED),
        ("hf_cache_permission: /x/hf/datasets/x.lock", Ruling.SETTLED),
        ("measurement refused: the spec was rejected", Ruling.SETTLED),
        ("executor error: RuntimeError('boom')", Ruling.SETTLED),
    ],
    ids=["cards_busy", "no_sandbox_slot", "inert_patch", "hf_cache", "refused", "executor_error"],
)
def test_only_an_infrastructure_failure_a_later_attempt_may_not_meet_is_requeued(reason: str, ruling: Ruling) -> None:
    assert rule(_ended(Validity.MEASUREMENT_FAILED, reason), again=False, failures=0) is ruling


def test_a_candidate_failure_a_success_and_a_withdrawal_are_never_requeued() -> None:
    failed = _ended(Validity.CANDIDATE_FAILED, "server_boot_timeout", FailureClass.CANDIDATE)
    assert rule(failed, again=False, failures=0) is Ruling.SETTLED
    assert rule(_ended(Validity.SUCCEEDED, "", None), again=False, failures=0) is Ruling.SETTLED
    assert rule(None, again=False, failures=0) is Ruling.SETTLED


def test_a_keep_goes_once_more_only_once_and_none_after_the_cap() -> None:
    busy = _ended(Validity.MEASUREMENT_FAILED)
    assert rule(busy, again=True, failures=0) is Ruling.SPENT
    assert rule(busy, again=False, failures=REQUEUE_FAILURES - 1) is Ruling.REQUEUE
    assert rule(busy, again=False, failures=REQUEUE_FAILURES) is Ruling.CAPPED
    assert rule(_ended(Validity.SUCCEEDED, "", None), again=True, failures=1) is Ruling.MEASURED
    failed = _ended(Validity.CANDIDATE_FAILED, "boom", FailureClass.CANDIDATE)
    assert rule(failed, again=True, failures=1) is Ruling.MEASURED, "a requeue that measured the candidate measured"
