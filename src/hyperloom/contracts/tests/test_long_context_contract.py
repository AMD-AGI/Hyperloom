# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context check as the measurement contract carries it: the request's flag and the result's record."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyperloom.contracts.long_context import (
    LONG_CONTEXT_COMPLETE,
    LONG_CONTEXT_INCOMPLETE,
    PROMPT_ERROR,
    PROMPT_NOT_RUN,
    PROMPT_OK,
    LongContextPrompt,
    LongContextResult,
    LongContextTask,
)
from hyperloom.contracts.measurement import MeasurementError, Purpose, TestKind, TestRequest, TestResult, Validity

SHA = "a" * 64


def _prompt(prompt_id: str, task: str, *, passed: bool = True, status: str = PROMPT_OK) -> LongContextPrompt:
    return LongContextPrompt(prompt_id, task, status, passed and status == PROMPT_OK, 2000, 1.5)


def _check(**overrides: object) -> LongContextResult:
    fields: dict[str, object] = {
        "status": LONG_CONTEXT_COMPLETE,
        "reason": "",
        "prompt_set_sha256": SHA,
        "length": 2048,
        "max_new_tokens": 128,
        "concurrency": 4,
        "served_context": 4096,
        "n": 4,
        "passed": 3,
        "score": 0.75,
        "tasks": (LongContextTask("niah_single", 2, 2, 1.0), LongContextTask("common_words", 2, 1, 0.5)),
        "prompts": (
            _prompt("a-0", "niah_single"),
            _prompt("a-1", "niah_single"),
            _prompt("b-0", "common_words"),
            _prompt("b-1", "common_words", passed=False),
        ),
        "seconds": 12.5,
        "prefill_tokens": 8000,
        "trigger": ("request long_context_check",),
        "outputs": "long_context/outputs.json",
    }
    return LongContextResult(**{**fields, **overrides})  # type: ignore[arg-type]


def _result(**overrides: object) -> TestResult:
    return TestResult(request_id="r1", validity=Validity.SUCCEEDED, **overrides)  # type: ignore[arg-type]


def _request(**overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.INTEGRATION,
        "branch_id": "b1",
        "parent_commit": "c0",
    }
    return TestRequest(**{**fields, **overrides})  # type: ignore[arg-type]


def test_a_result_with_a_check_survives_json_and_a_result_without_one_has_no_key() -> None:
    result = _result(long_context=_check())

    again = TestResult.from_document(json.loads(json.dumps(result.to_document())))

    assert again == result and again.long_context.score == 0.75  # type: ignore[union-attr]
    assert "long_context" not in _result().to_document(), "every document written before it is unchanged"
    assert TestResult.from_document(_result().to_document()).long_context is None


def test_an_incomplete_check_states_its_reason_and_an_unread_one_has_no_score() -> None:
    refused = _check(
        status=LONG_CONTEXT_INCOMPLETE,
        reason="served context too short",
        passed=0,
        score=0.0,
        tasks=(LongContextTask("niah_single", 2, 0, 0.0), LongContextTask("common_words", 2, 0, 0.0)),
        prompts=tuple(
            _prompt(f"{t}-{i}", t, status=PROMPT_NOT_RUN) for t in ("niah_single", "common_words") for i in "01"
        ),
    )
    unread = _check(
        status=LONG_CONTEXT_INCOMPLETE, reason="no outputs", n=0, passed=0, score=None, tasks=(), prompts=()
    )

    for check in (refused, unread):
        assert LongContextResult.from_document(check.to_document()) == check


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"status": "done"}, "status must be one of"),
        ({"reason": "why"}, "a reason exactly when it is incomplete"),
        ({"status": LONG_CONTEXT_INCOMPLETE}, "a reason exactly when it is incomplete"),
        ({"score": 0.5}, "score must be passed over n"),
        ({"score": None}, "score must be passed over n, and absent exactly"),
        ({"passed": 4}, "n and passed must be what the prompts and the tasks add up to"),
        ({"prompt_set_sha256": "x"}, "lowercase hex sha256"),
        ({"length": 0}, "length must be an integer of at least 1"),
        ({"outputs": "/etc/passwd"}, "relative path"),
        ({"prefill_tokens": -1}, "prefill_tokens"),
    ],
)
def test_a_record_that_contradicts_itself_is_refused_naming_what(overrides, match) -> None:
    with pytest.raises(MeasurementError, match=match):
        _check(**overrides)


def test_a_check_is_complete_exactly_when_every_prompt_was_answered() -> None:
    errored = (*_check().prompts[:3], _prompt("b-1", "common_words", status=PROMPT_ERROR))

    with pytest.raises(MeasurementError, match="complete exactly when every prompt was answered"):
        replace(_check(), prompts=errored, passed=3)
    with pytest.raises(MeasurementError, match="complete exactly when every prompt was answered"):
        _check(status=LONG_CONTEXT_INCOMPLETE, reason="r")


def test_a_prompt_that_was_not_answered_cannot_have_passed_and_a_task_cannot_pass_more_than_it_has() -> None:
    with pytest.raises(MeasurementError, match="cannot have passed"):
        LongContextPrompt("a", "t", PROMPT_ERROR, True, 0, 0.0)
    with pytest.raises(MeasurementError, match="more prompts than it has"):
        LongContextTask("t", 2, 3, 1.5)
    with pytest.raises(MeasurementError, match="score must be passed over n"):
        LongContextTask("t", 2, 1, 0.9)


def test_the_record_is_replaceable_but_revalidated() -> None:
    with pytest.raises(MeasurementError, match="must be a LongContextResult"):
        replace(_result(long_context=_check()), long_context=3)


@pytest.mark.parametrize(("value", "forces"), [(True, True), (False, False), (None, False)])
def test_only_true_forces_the_check_and_false_is_what_none_is(value, forces) -> None:
    request = _request(long_context_check=value)

    assert request.forces_long_context_check is forces
    assert TestRequest.from_document(request.to_document()).long_context_check is value


def test_a_request_with_no_opinion_has_no_key_and_the_flag_is_a_boolean_or_nothing() -> None:
    assert "long_context_check" not in _request().to_document()
    assert _request(long_context_check=False).to_document()["long_context_check"] is False
    for bad in ("yes", 1, []):
        with pytest.raises(MeasurementError, match="long_context_check must be true or false"):
            _request(long_context_check=bad)
        with pytest.raises(MeasurementError, match="long_context_check must be true or false"):
            TestRequest.from_document({**_request().to_document(), "long_context_check": bad})
