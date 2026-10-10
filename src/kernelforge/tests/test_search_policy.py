"""Tests for the search policy: what is accepted and where the next iteration starts."""

from __future__ import annotations

import pytest

from kernelforge.loop.search_policy import (
    DEFAULT_SEARCH_POLICY,
    SEQUENTIAL_DEFAULT_LANES,
    SearchCandidate,
    SearchPolicy,
    StartingVersion,
    accepts,
    parse_search_policy,
    resolve_lanes,
    select_starting_version,
)

BEST = StartingVersion(iteration=3, commit_hash="best", mean_case_speedup=1.4, case_times={"a": 7.0})


def _record(iteration: int, decision: str, *, commit: str = "", score: float | None = None) -> SearchCandidate:
    valid = decision in {"KEEP", "ACCEPT", "REVERT_PERF"}
    return SearchCandidate(
        iteration=iteration,
        parent_iteration=iteration - 1,
        parent_commit=f"parent-{iteration}",
        decision=decision,
        commit_hash=commit,
        mean_case_speedup=score if valid else None,
        case_times={"a": 10.0 / (score or 1.0)} if valid else {},
    )


def test_sequential_is_the_default():
    assert DEFAULT_SEARCH_POLICY is SearchPolicy.SEQUENTIAL


@pytest.mark.parametrize(
    ("policy", "valid", "improves_best", "expected"),
    [
        (SearchPolicy.SEQUENTIAL, True, True, True),
        (SearchPolicy.SEQUENTIAL, True, False, False),
        (SearchPolicy.SEQUENTIAL, False, False, False),
        (SearchPolicy.SEQANY, True, True, True),
        (SearchPolicy.SEQANY, True, False, True),
        (SearchPolicy.SEQANY, False, False, False),
    ],
)
def test_acceptance_rule(policy, valid, improves_best, expected):
    assert accepts(policy, valid=valid, improves_best=improves_best) is expected


def test_sequential_always_starts_from_the_best_version():
    records = [_record(4, "ACCEPT", commit="c4", score=1.2)]

    assert select_starting_version(SearchPolicy.SEQUENTIAL, candidates=records, best=BEST) == BEST


def test_seqany_starts_from_the_most_recent_committed_candidate():
    records = [
        _record(4, "KEEP", commit="c4", score=1.5),
        _record(5, "ACCEPT", commit="c5", score=1.3),
        _record(6, "REVERT_VALIDATION"),
        _record(7, "REVERT_INTEGRITY"),
    ]

    start = select_starting_version(SearchPolicy.SEQANY, candidates=records, best=BEST)

    assert start == StartingVersion(iteration=5, commit_hash="c5", mean_case_speedup=1.3, case_times={"a": 10.0 / 1.3})


def test_seqany_starts_from_the_best_version_before_anything_is_committed():
    records = [_record(1, "REVERT_VALIDATION")]

    assert select_starting_version(SearchPolicy.SEQANY, candidates=records, best=BEST) == BEST


@pytest.mark.parametrize(
    ("policy", "requested", "expected"),
    [
        (SearchPolicy.SEQUENTIAL, None, SEQUENTIAL_DEFAULT_LANES),
        (SearchPolicy.SEQUENTIAL, 5, 5),
        (SearchPolicy.SEQANY, None, 1),
        (SearchPolicy.SEQANY, 1, 1),
    ],
)
def test_lane_count_follows_the_policy(policy, requested, expected):
    assert resolve_lanes(policy, requested) == expected


def test_seqany_refuses_more_than_one_lane():
    with pytest.raises(ValueError, match="single lane"):
        resolve_lanes(SearchPolicy.SEQANY, 2)


def test_policy_names_are_parsed_case_insensitively_and_unknown_ones_refused():
    assert parse_search_policy(" SeqAny ") is SearchPolicy.SEQANY
    with pytest.raises(ValueError, match="unsupported search policy 'mcts'"):
        parse_search_policy("mcts")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"decision": "ACCEPT", "commit_hash": ""}, "requires a commit"),
        ({"decision": "REVERT_PERF", "commit_hash": "c1"}, "must not carry a commit"),
        ({"parent_iteration": 2}, "parent iteration must precede it"),
        ({"parent_commit": " "}, "no parent commit"),
        ({"mean_case_speedup": float("nan")}, "positive finite number"),
    ],
)
def test_a_record_that_cannot_describe_a_candidate_is_refused(change, message):
    fields = {
        "iteration": 2,
        "parent_iteration": 1,
        "parent_commit": "p1",
        "decision": "REVERT_PERF",
        "commit_hash": "",
        "mean_case_speedup": 1.1,
        "case_times": {"a": 9.0},
        **change,
    }
    with pytest.raises(ValueError, match=message):
        SearchCandidate(**fields)
