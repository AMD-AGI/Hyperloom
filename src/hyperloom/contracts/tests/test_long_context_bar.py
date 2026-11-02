# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context bar the board decided (#142 c3219, bar (b)): the root's reading with its 95% interval, and a candidate
refused when its drop from the lower end of that interval exceeds the check's own sampling noise, 3 of 64 items.

The interval bounds below were computed by hand from the Wilson formula, not by the code under test.
"""

from __future__ import annotations

import json

import pytest

from hyperloom.contracts.long_context import (
    NOISE_SHARE,
    LongContextResult,
    LongContextRoot,
    refuses,
    root_reading,
    wilson_interval,
)
from hyperloom.contracts.measurement import MeasurementError, TestResult, Validity
from hyperloom.contracts.tests.long_context_checks import long_context_check

SHA = "a" * 64
OTHER_SHA = "b" * 64
TASK = "niah_single"


def _check(n: int, passed: int, *, errored: int = 0, sha: str = SHA) -> LongContextResult:
    return long_context_check(n, passed, errored=errored, sha=sha)


def _root(*passed: int, n: int = 64) -> LongContextRoot:
    reading = root_reading((f"root-{index}", _check(n, value)) for index, value in enumerate(passed))
    assert reading is not None
    return reading


@pytest.mark.parametrize(
    ("passed", "n", "low", "high"),
    [(192, 192, 0.98038, 1.0), (186, 192, 0.93351, 0.98560), (60, 64, 0.85003, 0.97543), (0, 64, 0.0, 0.05662)],
)
def test_the_roots_interval_is_the_wilson_95_percent_interval(passed, n, low, high) -> None:
    found = wilson_interval(passed, n)

    assert found == (pytest.approx(low, abs=1e-5), pytest.approx(high, abs=1e-5))


def test_the_roots_reading_pools_its_complete_checks_and_names_them() -> None:
    reading = root_reading([("r1", _check(64, 64)), ("r2", _check(64, 61, errored=3)), ("r3", _check(64, 60))])

    assert reading is not None
    assert (reading.tests, reading.n, reading.passed, reading.score) == (("r1", "r3"), 128, 124, 124 / 128)
    assert (reading.ci_low, reading.ci_high) == wilson_interval(124, 128), "an incomplete check is no reading"


def test_the_root_reads_the_prompt_set_of_its_last_check_and_none_without_a_complete_one() -> None:
    reading = root_reading([("old", _check(64, 10, sha=OTHER_SHA)), ("new", _check(64, 64))])

    assert reading is not None and (reading.tests, reading.prompt_set_sha256) == (("new",), SHA)
    assert root_reading([("r1", _check(64, 60, errored=4))]) is None
    assert root_reading([]) is None


@pytest.mark.parametrize(
    ("root", "passed", "refused"),
    [
        ((64, 64, 64), 60, False),
        ((64, 64, 64), 59, True),
        ((64,), 58, False),
        ((64,), 57, True),
        ((62, 62, 62), 57, False),
        ((62, 62, 62), 56, True),
        ((64, 64, 64), 64, False),
    ],
    ids=[
        "root-192-of-192-drop-under-3",
        "root-192-of-192-drop-over-3",
        "root-64-of-64-wider-interval-holds-58",
        "root-64-of-64-refuses-57",
        "root-186-of-192-holds-57",
        "root-186-of-192-refuses-56",
        "no-drop",
    ],
)
def test_a_drop_beyond_3_of_64_items_after_the_roots_interval_refuses(root, passed, refused) -> None:
    assert refuses(_check(64, passed), _root(*root)) is refused


def test_a_drop_of_exactly_the_noise_does_not_refuse_and_one_item_more_does() -> None:
    root = LongContextRoot(SHA, ("r1",), 64, 50, 50 / 64, 0.75, 0.8)  # the lower end is 48 of 64 items

    assert NOISE_SHARE * 64 == 3
    assert refuses(_check(64, 45), root) is False, "48 - 45 is the noise itself"
    assert refuses(_check(64, 44), root) is True


def test_the_noise_scales_with_the_candidates_items() -> None:
    root = _root(64, 64, 64)  # lower end 0.98038: 125.49 of 128 items, and the noise is 6 of them

    assert refuses(_check(128, 120), root) is False
    assert refuses(_check(128, 119), root) is True


def test_an_incomplete_check_never_blocks_whatever_its_score() -> None:
    assert refuses(_check(64, 0, errored=64), _root(64, 64, 64)) is False
    assert refuses(_check(64, 30, errored=1), _root(64, 64, 64)) is False


def test_nothing_is_refused_without_a_root_or_a_check_or_across_prompt_sets() -> None:
    root = _root(64, 64, 64)

    assert refuses(_check(64, 0), None) is False
    assert refuses(None, root) is False
    assert refuses(_check(64, 0, sha=OTHER_SHA), root) is False


def test_a_result_carries_the_roots_reading_through_json_and_one_without_has_no_key() -> None:
    root = _root(64, 62)
    check = _check(64, 58)

    with_root = TestResult("r9", Validity.SUCCEEDED, long_context=LongContextResult(**{**vars(check), "root": root}))
    document = json.loads(json.dumps(with_root.to_document()))

    assert TestResult.from_document(document) == with_root and document["long_context"]["root"]["tests"] == [
        "root-0",
        "root-1",
    ]
    assert "root" not in check.to_document() and LongContextResult.from_document(check.to_document()) == check


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"tests": ()}, "names the tests it pools"),
        ({"passed": 65}, "more prompts than it has"),
        ({"score": 0.5}, "passed over n"),
        ({"ci_low": 0.99}, "interval must hold its score"),
        ({"ci_high": 1.5}, "interval must hold its score"),
    ],
)
def test_a_root_reading_that_contradicts_itself_is_refused_naming_what(fields, match) -> None:
    good = {"prompt_set_sha256": SHA, "tests": ("r1",), "n": 64, "passed": 60, "score": 60 / 64}
    good.update(ci_low=0.85, ci_high=0.97)

    with pytest.raises(MeasurementError, match=match):
        LongContextRoot(**{**good, **fields})  # type: ignore[arg-type]
