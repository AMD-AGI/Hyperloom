# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Failures in three tiers, who covers each direction, and the hypothesis ledger, read from the scripted session."""

from __future__ import annotations

import dataclasses

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, ClaimKey
from hyperloom_multibranch.contracts.notebook.peers import BranchLogEntry, BranchLogType, HypothesisStatus
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord, Claim, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus
from hyperloom_multibranch.notebook.rsi.analysis import (
    FailedTest,
    FailureTier,
    direction_occupancy,
    failures_at,
    hypothesis_ledger,
    repeated_failures,
)
from hyperloom_multibranch.notebook.tests.sessions import FP8_KV, GEMM


def _failures(snapshot: NotebookSnapshot, branch: str) -> tuple[FailedTest, ...]:
    head = next((node.head_commit for node in snapshot.lineage.branches if node.branch == branch), "")
    return failures_at(snapshot, head)


def _tiers(snapshot: NotebookSnapshot, branch: str) -> list[tuple[str, FailureTier]]:
    return [(item.request_id, item.tier) for item in _failures(snapshot, branch)]


def test_failures_are_tiered_by_how_close_their_state_is_to_the_branch_head(snapshot: NotebookSnapshot) -> None:
    assert _tiers(snapshot, "s1") == [
        ("r4", FailureTier.SAME_STATE),
        ("r2", FailureTier.ANCESTOR),
        ("r3", FailureTier.ANCESTOR),
        ("r5", FailureTier.ANCESTOR),
    ]
    assert _tiers(snapshot, "s2") == [
        ("r2", FailureTier.SAME_STATE),
        ("r3", FailureTier.SAME_STATE),
        ("r5", FailureTier.SAME_STATE),
        ("r4", FailureTier.ELSEWHERE),
    ]
    assert {tier for _, tier in _tiers(snapshot, "unknown")} == {FailureTier.ELSEWHERE}
    (flat, *_) = _failures(snapshot, "s1")
    assert flat.render() == "- [same_state] moe gemm / tuned kernel on c1: flat at +0.30% (r4)"


def test_a_direction_that_keeps_failing_is_named_once_with_its_count() -> None:
    failed = [
        FailedTest(FailureTier.SAME_STATE, f"r{index}", "s1", "c0", "attention / split kv", "flat", 0.1)
        for index in range(3)
    ]
    other = FailedTest(FailureTier.ELSEWHERE, "r9", "s2", "c0", "sampler / fused topk", "flat", 0.0)

    assert repeated_failures([*failed, other]) == (("attention / split kv", 3),)
    assert repeated_failures([other], at_least=1) == (("sampler / fused topk", 1),)


def test_each_direction_shows_where_it_was_tried_who_carries_it_and_who_holds_it(snapshot: NotebookSnapshot) -> None:
    held = Claim(
        ClaimKey("c0", FP8_KV.fingerprint),
        "s3",
        dataclasses.replace(FP8_KV, branch="s3"),
        "a",
        60.0,
        0.0,
        0.0,
        60.0,
        ClaimStatus.HELD,
    )

    coverage = {item.name: item for item in direction_occupancy(dataclasses.replace(snapshot, claims=(held,)))}

    assert list(coverage) == [
        "moe gemm / tuned kernel",
        "attention / split kv",
        "scheduler / chunked prefill",
        "kv cache / fp8 kv cache",
    ]
    gemm = coverage["moe gemm / tuned kernel"]
    assert (gemm.tried_on, gemm.committed_on, gemm.best_gain_pct) == (("c0", "c1"), (INTEGRATION_BRANCH, "s1"), 4.0)
    assert gemm.outcomes[0] == "moe gemm / tuned kernel on c0: +4.00%, committed as c1"
    assert coverage["kv cache / fp8 kv cache"].claimed_by == ("s3",)
    assert coverage["attention / split kv"].committed_on == ()
    assert coverage["scheduler / chunked prefill"].best_gain_pct is None
    assert gemm.render() == (
        f"- moe gemm / tuned kernel (kernel): tried on 2 states, best +4.00%; on the heads of {INTEGRATION_BRANCH}, s1"
    )
    assert coverage["kv cache / fp8 kv cache"].render().endswith("best +6.00%; claimed by s3")
    assert coverage["scheduler / chunked prefill"].render().endswith("best n/a")
    assert GEMM.fingerprint in snapshot.intents


def test_the_ledger_keeps_each_hypothesis_at_its_latest_version(snapshot: NotebookSnapshot) -> None:
    posted = BranchLogRecord("h1", 1.0, BranchLogEntry("s3", BranchLogType.HYPOTHESIS, "decode is bound by attention"))
    settled = BranchLogRecord(
        "h2",
        2.0,
        BranchLogEntry(
            "s3", BranchLogType.HYPOTHESIS, "decode is bound by attention", ("r7",), "h1", HypothesisStatus.CONFIRMED
        ),
    )
    plan = BranchLogRecord("p1", 3.0, BranchLogEntry("s3", BranchLogType.PLAN, "profile decode next"))
    logs = {**snapshot.branch_logs, "s3": (posted, settled, plan)}

    rows = hypothesis_ledger(dataclasses.replace(snapshot, branch_logs=logs))

    assert [(row.branch, row.status) for row in rows] == [
        ("s1", HypothesisStatus.CONFIRMED),
        ("s2", HypothesisStatus.FALSIFIED),
        ("s3", HypothesisStatus.CONFIRMED),
    ]
    assert (rows[-1].first_entry, rows[-1].latest_entry, rows[-1].updated_at) == ("h1", "h2", 2.0)
    assert rows[-1].render() == "- confirmed: decode is bound by attention (s3) [r7]"
    open_row = hypothesis_ledger(dataclasses.replace(snapshot, branch_logs={"s3": (posted,)}))[0]
    assert open_row.status is HypothesisStatus.OPEN and open_row.render().endswith("(s3)")
