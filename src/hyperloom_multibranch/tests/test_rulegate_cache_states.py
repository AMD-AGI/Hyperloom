# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which kinds of the shared compiled-code cache a candidate's tests met otherwise than its parent's."""

from __future__ import annotations

from hyperloom.contracts.measurement import JitCacheState, TestResult, Validity
from hyperloom_multibranch.rulegate.cache_states import NOT_USED, cache_states_differ


def _result(request_id: str, **states: str) -> TestResult:
    return TestResult(
        request_id=request_id,
        validity=Validity.SUCCEEDED,
        jit_cache=tuple(JitCacheState(kind, state) for kind, state in states.items()),
    )


def test_a_kind_met_in_another_state_is_named_with_both_sides() -> None:
    differ = cache_states_differ(
        [_result("c", aiter_jit="miss", triton="hit")], [_result("p", aiter_jit="hit", triton="hit")]
    )

    assert differ == ("aiter_jit: miss vs hit",)


def test_the_same_states_on_both_sides_are_no_difference() -> None:
    same = {"aiter_jit": "hit", "aiter_cpp": "hit", "triton": "hit"}

    assert cache_states_differ([_result("c", **same), _result("r", **same)], [_result("p", **same)]) == ()


def test_a_retest_or_a_baseline_in_another_state_than_the_others_is_a_difference() -> None:
    differ = cache_states_differ(
        [_result("c", aiter_jit="hit"), _result("r", aiter_jit="hit")],
        [_result("p1", aiter_jit="hit"), _result("p2", aiter_jit="miss")],
    )

    assert differ == ("aiter_jit: hit vs hit/miss",)


def test_a_kind_one_side_did_not_use_is_named_as_not_used() -> None:
    differ = cache_states_differ([_result("c", aiter_jit="hit")], [_result("p", aiter_jit="hit", triton="hit")])

    assert differ == (f"triton: {NOT_USED} vs hit",)


def test_a_side_that_states_no_cache_is_not_called_different() -> None:
    assert cache_states_differ([_result("c", aiter_jit="miss")], [_result("p")]) == ()
    assert cache_states_differ([_result("c")], [_result("p", aiter_jit="hit")]) == ()
    assert cache_states_differ([], [_result("p", aiter_jit="hit")]) == ()


def test_the_states_round_trip_through_the_result_document_and_an_older_document_has_none() -> None:
    result = _result("c", aiter_jit="miss", triton="hit")

    assert TestResult.from_document(result.to_document()) == result
    older = _result("o").to_document()
    assert "jit_cache" not in older and TestResult.from_document(older).jit_cache == ()
