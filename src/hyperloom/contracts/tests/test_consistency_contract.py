# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency check as a result carries it, and the request field that forces it."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hyperloom.contracts.consistency import (
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    OutputCounts,
    OutputDifference,
    ShortPromptConsistency,
)
from hyperloom.contracts.measurement import MeasurementError, Purpose, TestKind, TestRequest, TestResult, Validity

DIFFERENCE = OutputDifference(1, "s05", "sequential", 7, 1, "abc", "aXY")
CLEAN = OutputCounts((0, 0, 0))
FOUND = OutputCounts((0, 2, 0), DIFFERENCE)


def _check(**overrides: object) -> ShortPromptConsistency:
    fields: dict[str, object] = {
        "status": SHORT_PROMPTS_COMPLETE,
        "reason": "",
        "prompt_set_sha256": "a" * 64,
        "prompts": 64,
        "max_new_tokens": 32,
        "trigger": ("purpose final",),
        "processes_planned": 3,
        "processes": (0, 1, 2),
        "miss_probability_one_third": 0.296,
        "compared_by": "tokens",
        "self_consistency": FOUND,
        "outputs": "short_prompts/outputs.json",
    }
    return ShortPromptConsistency(**{**fields, **overrides})  # type: ignore[arg-type]


def _result(**overrides: object) -> TestResult:
    return TestResult(request_id="r1", validity=Validity.SUCCEEDED, **overrides)  # type: ignore[arg-type]


def _roundtrip(value):
    again = type(value).from_document(json.loads(json.dumps(value.to_document())))
    assert again == value and again.to_document() == value.to_document()
    return again


def test_a_check_round_trips_through_json_in_a_result():
    result = _result(short_prompt_consistency=_check())

    assert _roundtrip(result).short_prompt_consistency == _check()
    document = result.to_document()["short_prompt_consistency"]
    assert document["self_consistency"]["first_difference"]["position"] == 1
    assert "reference" not in document, "the check compares its own processes only"


def test_a_check_recorded_while_it_compared_with_the_parent_still_decodes_and_its_reference_is_left_out():
    old = {
        **_check().to_document(),
        "self_consistency": {**FOUND.to_document(), "unstable_prompts": 0},
        "reference": {
            "status": "compared",
            "reason": "",
            "request_id": "r0",
            "counts": {"mismatches": [0, 0, 0], "first_difference": None, "unstable_prompts": 1},
        },
    }

    decoded = TestResult.from_document({**_result().to_document(), "short_prompt_consistency": old})

    assert decoded.short_prompt_consistency == _check()
    assert "reference" not in decoded.to_document()["short_prompt_consistency"]


@pytest.mark.parametrize(
    ("counts", "inconsistency"),
    [(FOUND, 2), (CLEAN, 0), (None, None)],
    ids=["one-process-differs", "all-agree", "not-compared"],
)
def test_the_inconsistency_is_the_most_prompts_one_process_answers_differently_and_none_when_not_compared(
    counts, inconsistency
):
    single = {"processes_planned": 3, "processes": (0,), "status": SHORT_PROMPTS_INCOMPLETE, "reason": "two failed"}
    check = _check(self_consistency=counts) if counts is not None else _check(**single, self_consistency=None)

    assert check.inconsistency == inconsistency, "nothing compared is None, never a count of zero"


def test_a_result_with_no_check_has_no_key_and_a_document_written_before_it_loads_with_none():
    document = _result().to_document()

    assert "short_prompt_consistency" not in document
    assert TestResult.from_document(document).short_prompt_consistency is None


def test_an_incomplete_check_with_no_process_and_an_absent_reference_round_trips():
    nothing = _check(
        status=SHORT_PROMPTS_INCOMPLETE,
        reason="no process answered",
        processes=(),
        compared_by="",
        self_consistency=None,
        miss_probability_one_third=1.0,
        outputs="",
    )

    assert _roundtrip(nothing).self_consistency is None and nothing.inconsistency is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"status": "passed"}, "status must be one of"),
        ({"status": SHORT_PROMPTS_INCOMPLETE, "reason": ""}, "incomplete consistency check says why"),
        ({"processes_planned": 4}, "complete exactly when every planned process completed"),
        ({"processes": (0, 0, 1)}, "distinct and no more than planned"),
        ({"processes": (0, 1, 2, 3)}, "distinct and no more than planned"),
        ({"prompt_set_sha256": "x"}, "lowercase hex sha256"),
        ({"miss_probability_one_third": 1.5}, "is a probability"),
        ({"compared_by": "vibes"}, "compared_by must be one of"),
        ({"self_consistency": OutputCounts((0, 0))}, "counts every completed process"),
        (
            {"processes_planned": 1, "processes": (0,), "self_consistency": OutputCounts((0,))},
            "needs two processes",
        ),
        ({"self_consistency": "clean"}, "must be OutputCounts"),
        ({"outputs": "/abs/path"}, "relative path"),
    ],
)
def test_a_check_that_breaks_the_contract_is_refused_naming_what(change, message):
    with pytest.raises(MeasurementError, match=message):
        _check(**change)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: OutputCounts((0, 1)), "first difference is given exactly when"),
        (lambda: OutputCounts((0, 0), DIFFERENCE), "first difference is given exactly when"),
        (lambda: OutputCounts((0, -1)), "integer of at least 0"),
        (lambda: OutputDifference(0, "", "sequential", 1, 0, "a", "b"), "prompt_id"),
        (lambda: OutputDifference(0, "s", "sequential", 1, -1, "a", "b"), "position"),
    ],
)
def test_the_parts_of_a_check_refuse_what_breaks_them(build, message):
    with pytest.raises(MeasurementError, match=message):
        build()


def test_a_check_document_is_read_strictly():
    document = _check().to_document()

    for broken in ({**document, "self_consistency": 3}, {**document, "processes": "012"}, {**document, "status": None}):
        with pytest.raises(MeasurementError):
            ShortPromptConsistency.from_document(broken)


def test_a_result_refuses_a_check_that_is_not_one():
    with pytest.raises(MeasurementError, match="must be a ShortPromptConsistency"):
        _result(short_prompt_consistency={"status": "complete"})


# --- the request field ----------------------------------------------------------------------------------------------


def _request(**overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.INTEGRATION,
        "branch_id": "b1",
        "parent_commit": "c0",
    }
    return TestRequest(**{**fields, **overrides})  # type: ignore[arg-type]


@pytest.mark.parametrize(("value", "forces"), [(True, True), (False, False), (None, False)])
def test_only_true_forces_the_check_and_false_is_what_none_is(value, forces):
    request = _request(consistency_check=value)

    assert request.forces_consistency_check is forces
    assert _roundtrip(request).consistency_check is value


def test_a_request_with_no_opinion_has_no_key_so_every_document_written_before_it_is_unchanged():
    assert "consistency_check" not in _request().to_document()
    assert _request(consistency_check=False).to_document()["consistency_check"] is False
    assert TestRequest.from_document(_request().to_document()).consistency_check is None


@pytest.mark.parametrize("bad", ["yes", 1, 0, []])
def test_the_flag_is_a_boolean_or_nothing(bad):
    with pytest.raises(MeasurementError, match="consistency_check must be true or false"):
        _request(consistency_check=bad)
    with pytest.raises(MeasurementError, match="consistency_check must be true or false"):
        TestRequest.from_document({**_request().to_document(), "consistency_check": bad})


def test_the_check_is_replaceable_but_revalidated():
    with pytest.raises(MeasurementError):
        replace(_result(short_prompt_consistency=_check()), short_prompt_consistency=3)
