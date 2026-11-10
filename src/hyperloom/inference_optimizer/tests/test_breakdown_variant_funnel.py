# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The variant funnel: how many framework variants were benched and how each ended."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.v6 import OutcomeSources, collect_v6_outcome
from hyperloom.inference_optimizer.breakdown.variant_funnel import variants_from_timeline

_BUCKETS = ("kept", "reverted", "failed")


def _attempt(
    fingerprint: str = "fp",
    *,
    adopted: bool = False,
    after_tput: float | None = 100.0,
    attribution: str = "",
    **fields: Any,
) -> dict[str, Any]:
    """One attempt row as the framework recorders write it: the settled facts beside the executor's outcome."""
    row: dict[str, Any] = {
        "attempt_id": f"task:round:{fingerprint}",
        "arm": "config",
        "fingerprint": fingerprint,
        "outcome": "",
        "benched": True,
        "adopted": adopted,
        "measurement": {"after_tput": after_tput},
        "failure": {"attribution": attribution},
    }
    row.update(fields)
    return row


def _framework_event(*attempts: dict[str, Any]) -> dict[str, Any]:
    """A ``framework_agent`` timeline event holding ``attempts``."""
    return {"type": "framework_agent", "kind": "agent", "ext": {"attempts": list(attempts)}}


def _bucket(row: dict[str, Any]) -> str:
    """The bucket ``row`` is counted in, or ``""`` when it is not a bench run."""
    funnel = variants_from_timeline([_framework_event(row)])
    return next((name for name in _BUCKETS if funnel[name]), "")


@pytest.mark.parametrize(
    ("row", "bucket"),
    [
        pytest.param(_attempt(adopted=True, outcome="KEEP"), "kept", id="adopted"),
        pytest.param(_attempt(outcome="KEEP"), "reverted", id="keep-the-adoption-rule-refused"),
        pytest.param(_attempt("", outcome="kept_inert", arm="source"), "reverted", id="patch-left-dormant"),
        pytest.param(
            _attempt("", outcome="accuracy_unavailable_reject", arm="source"), "reverted", id="measured-reject"
        ),
        pytest.param(
            _attempt(outcome="FAILED", after_tput=None, attribution="environment"), "failed", id="server-failed"
        ),
        pytest.param(
            _attempt(outcome="FAILED", attribution="candidate_caused"), "failed", id="failure-with-a-throughput"
        ),
        pytest.param(_attempt("", outcome="reverted", arm="source", after_tput=None), "failed", id="bench-raised"),
        pytest.param(
            _attempt("", outcome="failed", arm="source", after_tput=None, attribution="unknown", benched=False),
            "",
            id="failed-before-the-bench",
        ),
        pytest.param(
            _attempt("", outcome="apply_failed", arm="source", after_tput=None, benched=False), "", id="did-not-apply"
        ),
    ],
)
def test_a_row_is_counted_by_the_facts_its_recorder_settled(row: dict[str, Any], bucket: str) -> None:
    assert _bucket(row) == bucket


@pytest.mark.parametrize(
    ("facts", "bucket"),
    [
        pytest.param({"adopted": True}, "kept", id="adopted"),
        pytest.param({}, "reverted", id="measured"),
        pytest.param({"after_tput": None, "attribution": "unknown"}, "failed", id="attributed-failure"),
        pytest.param({"after_tput": None}, "", id="nothing-settled"),
    ],
)
def test_a_row_recorded_before_benched_existed_is_read_by_the_other_facts(facts: dict[str, Any], bucket: str) -> None:
    row = _attempt(**facts)
    del row["benched"]

    assert _bucket(row) == bucket


def test_a_change_benched_again_is_one_distinct_change() -> None:
    """The fingerprint names the change, not the stack it was measured on."""
    funnel = variants_from_timeline(
        [_framework_event(_attempt("fp-a", adopted=True, attempt_id="t1:r1:fp-a"), _attempt("fp-a", attempt_id="t2"))]
    )

    assert (funnel["bench_runs"], funnel["distinct"]) == (2, 1)


def test_an_attempt_without_a_fingerprint_is_its_own_change() -> None:
    """A source row carries no fingerprint, so each attempt counts as its own change."""
    funnel = variants_from_timeline(
        [_framework_event(_attempt("", arm="source", attempt_id="t1"), _attempt("", arm="source", attempt_id="t2"))]
    )

    assert (funnel["bench_runs"], funnel["distinct"]) == (2, 2)


def test_every_framework_event_is_counted_and_no_other() -> None:
    """Each macro cycle's framework event holds its own attempts; a kernel event's belong to the kernel lane."""
    kernel = {"type": "kernel", "ext": {"attempts": [_attempt("k1", adopted=True)]}}

    funnel = variants_from_timeline(
        [
            _framework_event(_attempt("fp-a", adopted=True)),
            kernel,
            _framework_event(_attempt("fp-b", after_tput=None, attribution="unknown")),
        ]
    )

    assert (funnel["bench_runs"], funnel["kept"], funnel["failed"]) == (2, 1, 1)


def test_the_keep_rate_is_kept_per_bench_run() -> None:
    """Twelve bench runs over seven changes with two kept."""
    attempts = [
        _attempt(fingerprint, adopted=index < 2, attempt_id=f"t:{index}")
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
    timeline = [_framework_event(_attempt("fp-a", adopted=True), _attempt("fp-b"))]

    outcome = collect_v6_outcome(
        OutcomeSources(session={"stop_reason": "signal"}, close={}, state={}, timeline=timeline)
    )

    assert outcome["variants"] == variants_from_timeline(timeline)
    assert outcome["variants"]["bench_runs"] == 2
