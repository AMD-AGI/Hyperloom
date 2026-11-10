# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context bar the gate applies: the root's reading with its 95% interval, and a candidate
refused when its drop from the lower end of that interval exceeds the check's own sampling noise: 0.5 + 0.3125 * sqrt(n)
items, 3 of 64 and 1.75 of 16.

The interval bounds below were computed by hand from the Wilson formula, not by the code under test.
"""

from __future__ import annotations

import json
import math

import pytest

from hyperloom.contracts.long_context import (
    LongContextResult,
    LongContextRoot,
    noise_items,
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

    assert noise_items(64) == 3
    assert refuses(_check(64, 45), root) is False, "48 - 45 is the noise itself"
    assert refuses(_check(64, 44), root) is True


def test_the_noise_is_3_items_of_64_and_1_75_of_16() -> None:
    assert (noise_items(64), noise_items(16), noise_items(8)) == (3.0, 1.75, pytest.approx(1.38388, abs=1e-5))


def test_a_drop_of_exactly_the_noise_on_16_items_does_not_refuse_and_one_item_more_does() -> None:
    root = LongContextRoot(SHA, ("r1",), 16, 13, 13 / 16, 0.796875, 0.9)  # the lower end is 12.75 of 16 items

    assert refuses(_check(16, 11), root) is False, "12.75 - 11 is the noise itself"
    assert refuses(_check(16, 10), root) is True


@pytest.mark.parametrize(
    ("root", "passed", "refused"),
    [
        ((16, 16, 16), 14, False),
        ((16, 16, 16), 13, True),
        ((15, 15, 15), 12, False),
        ((15, 15, 15), 11, True),
        ((16,), 12, False),
        ((16,), 11, True),
        ((16, 16, 16), 16, False),
    ],
    ids=[
        "root-48-of-48-two-misses-held",
        "root-48-of-48-three-misses-refused",
        "root-45-of-48-holds-12",
        "root-45-of-48-refuses-11",
        "root-16-of-16-wider-interval-holds-12",
        "root-16-of-16-refuses-11",
        "no-drop",
    ],
)
def test_on_16_items_a_drop_beyond_1_75_items_after_the_roots_interval_refuses(root, passed, refused) -> None:
    assert refuses(_check(16, passed), _root(*root, n=16)) is refused


def test_the_noise_grows_as_the_square_root_of_the_candidates_items() -> None:
    root = _root(64, 64, 64)  # lower end 0.98038: 125.49 of 128 items, and the noise is 0.5 + 0.3125 * sqrt(128) = 4.04

    assert refuses(_check(128, 122), root) is False
    assert refuses(_check(128, 121), root) is True


def _false_refusals(n: int, share: float, root_checks: int) -> float:
    """How often an unchanged candidate is refused: its root's checks and its own pass each prompt with ``share``."""

    def binomial(count: int, k: int) -> float:
        return math.comb(count, k) * share**k * (1 - share) ** (count - k)

    total = 0.0
    for root_passed in range(n * root_checks + 1):
        root = LongContextRoot(
            SHA,
            ("r",),
            n * root_checks,
            root_passed,
            root_passed / (n * root_checks),
            *wilson_interval(root_passed, n * root_checks),
        )
        refused = sum(binomial(n, passed) for passed in range(n + 1) if refuses(_check(n, passed), root))
        total += binomial(n * root_checks, root_passed) * refused
    return total


@pytest.mark.parametrize("share", [0.95, 0.9, 0.8])
def test_an_unchanged_candidate_on_16_items_is_refused_no_more_often_than_on_64(share) -> None:
    at_64, at_16 = _false_refusals(64, share, 3), _false_refusals(16, share, 3)

    assert at_16 <= at_64, f"{at_16:.4f} on 16 items against {at_64:.4f} on 64"
    assert at_64 < 0.04


def test_an_unchanged_candidate_at_a_95_percent_pass_rate_is_refused_under_1_percent_of_the_time() -> None:
    assert _false_refusals(64, 0.95, 3) == pytest.approx(0.0084, abs=5e-4)
    assert _false_refusals(16, 0.95, 3) == pytest.approx(0.0073, abs=5e-4)


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
