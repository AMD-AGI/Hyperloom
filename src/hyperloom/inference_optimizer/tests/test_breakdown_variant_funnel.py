# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The variant funnel: how many framework variants were benched and how each ended."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6 import OutcomeSources, collect_v6_outcome
from hyperloom.inference_optimizer.breakdown.variant_funnel import variants_from_timeline

_BUCKETS = ("kept", "reverted", "failed")


def _attempt(outcome: str, fingerprint: str = "", *, arm: str = "config", attempt_id: str = "") -> dict[str, Any]:
    """One attempt row as the framework recorder writes it."""
    return {
        "attempt_id": attempt_id or f"task:round:{fingerprint}",
        "arm": arm,
        "fingerprint": fingerprint,
        "outcome": outcome,
    }


def _framework_event(*attempts: dict[str, Any]) -> dict[str, Any]:
    """A ``framework_agent`` timeline event holding ``attempts``."""
    return {"type": "framework_agent", "kind": "agent", "ext": {"attempts": list(attempts)}}


@pytest.mark.parametrize(
    ("outcome", "bucket"),
    [
        ("KEEP", "kept"),
        ("kept", "kept"),
        ("REVERT", "reverted"),
        ("reverted", "reverted"),
        ("KEEP_UNSTABLE", "reverted"),
        ("FAILED", "failed"),
        ("failed", "failed"),
        ("KILLED_OVERTIME", "failed"),
    ],
)
def test_each_arms_spelling_of_a_verdict_lands_in_its_bucket(outcome: str, bucket: str) -> None:
    funnel = variants_from_timeline([_framework_event(_attempt(outcome, "fp"))])

    assert funnel["bench_runs"] == 1
    assert {name: funnel[name] for name in _BUCKETS} == {name: int(name == bucket) for name in _BUCKETS}


@pytest.mark.parametrize("outcome", ["cancelled", "SKIPPED_DEDUP", ""])
def test_an_attempt_that_never_reached_the_bench_is_not_counted(outcome: str) -> None:
    funnel = variants_from_timeline([_framework_event(_attempt(outcome, "fp-a"), _attempt("KEEP", "fp-b"))])

    assert (funnel["bench_runs"], funnel["distinct"], funnel["kept"]) == (1, 1, 1)


def test_a_change_benched_again_is_one_distinct_change() -> None:
    """The fingerprint names the change, not the stack it was measured on; the arm keeps the two lanes apart."""
    funnel = variants_from_timeline(
        [
            _framework_event(
                _attempt("KEEP", "fp-a", attempt_id="t1:r1:fp-a"),
                _attempt("REVERT", "fp-a", attempt_id="t2:r2:fp-a"),
                _attempt("reverted", "fp-a", arm="source", attempt_id="t3"),
            )
        ]
    )

    assert (funnel["bench_runs"], funnel["distinct"]) == (3, 2)


def test_an_attempt_without_a_fingerprint_is_its_own_change() -> None:
    funnel = variants_from_timeline(
        [_framework_event(_attempt("REVERT", attempt_id="t1"), _attempt("REVERT", attempt_id="t2"))]
    )

    assert (funnel["bench_runs"], funnel["distinct"]) == (2, 2)


def test_every_framework_event_is_counted_and_no_other() -> None:
    """Each macro cycle's framework event holds its own attempts; a kernel event's belong to the kernel lane."""
    kernel = {"type": "kernel", "ext": {"attempts": [_attempt("KEEP", "k1")]}}

    funnel = variants_from_timeline(
        [_framework_event(_attempt("KEEP", "fp-a")), kernel, _framework_event(_attempt("FAILED", "fp-b"))]
    )

    assert (funnel["bench_runs"], funnel["kept"], funnel["failed"]) == (2, 1, 1)


def test_the_keep_rate_is_kept_per_bench_run() -> None:
    """Twelve bench runs over seven changes with two kept."""
    attempts = [
        _attempt("KEEP" if index < 2 else "REVERT", fingerprint, attempt_id=f"t:{index}")
        for index, fingerprint in enumerate("abcdefgabcde")
    ]

    funnel = variants_from_timeline([_framework_event(*attempts)])

    assert funnel == {
        "bench_runs": 12,
        "distinct": 7,
        "kept": 2,
        "reverted": 10,
        "failed": 0,
        "keep_rate_pct": 16.67,
    }


def test_nothing_benched_has_no_keep_rate() -> None:
    assert variants_from_timeline([]) == {
        "bench_runs": 0,
        "distinct": 0,
        "kept": 0,
        "reverted": 0,
        "failed": 0,
        "keep_rate_pct": None,
    }


def test_the_outcome_block_carries_the_funnel() -> None:
    timeline = [_framework_event(_attempt("KEEP", "fp-a"), _attempt("REVERT", "fp-b"))]

    outcome = collect_v6_outcome(
        OutcomeSources(session={"stop_reason": "signal"}, close={}, state={}, timeline=timeline)
    )

    assert outcome["variants"] == variants_from_timeline(timeline)
    assert outcome["variants"]["bench_runs"] == 2
