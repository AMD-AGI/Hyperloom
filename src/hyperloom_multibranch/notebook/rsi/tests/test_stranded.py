# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch ahead of integration by more than the merge bar while its commits fail to merge is a stranded lead."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import (
    BranchProgress,
    CommittedCommit,
    MergeState,
    NotebookSnapshot,
)
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.notebook.rsi.stranded import REASON_CHARS, short_reason, stranded_leads
from hyperloom_multibranch.notebook.tests.sessions import METRIC

CONFLICT = "unresolved patch conflict"


def _row(branch: str, value: float, *, running: bool = True) -> BranchProgress:
    return BranchProgress(branch, running, "c0", f"{branch}-head", 4, 2, 600.0, {METRIC: value})


def _commit(
    commit_id: str, branch: str, outcome: MergeOutcome | None, reason: str = "", tries: int = 1
) -> CommittedCommit:
    last = None if outcome is None else MergeState(outcome, "i1", reason=reason, at=100.0)
    return CommittedCommit(
        commit_id, branch, "c0", f"r-{commit_id}", 5.0, False, False, last_merge=last, merge_attempts=tries
    )


def _session(
    snapshot: NotebookSnapshot, rows: Sequence[BranchProgress], commits: Sequence[CommittedCommit]
) -> NotebookSnapshot:
    return dataclasses.replace(snapshot, progress=tuple(rows), committed=tuple(commits))


def test_a_lead_over_the_bar_behind_a_commit_that_failed_to_merge_is_stranded_and_one_still_waiting_is_not(
    snapshot: NotebookSnapshot,
) -> None:
    rows = [
        _row(INTEGRATION_BRANCH, 1000.0),
        _row("b01", 1100.0),
        _row("b02", 1080.0, running=False),
        _row("b03", 1200.0),
        _row("b04", 1010.0),
    ]
    commits = [
        _commit("aaaaaaaa1", "b01", MergeOutcome.CONFLICT_SKIPPED, CONFLICT, tries=3),
        _commit("bbbbbbbb2", "b02", MergeOutcome.NOT_STACKABLE, "-0.99% on the integration head"),
        _commit("cccccccc3", "b03", MergeOutcome.PENDING),
        _commit("dddddddd4", "b03", None, tries=0),
        _commit("eeeeeeee5", "b04", MergeOutcome.CONFLICT_SKIPPED, CONFLICT),
    ]

    leads = stranded_leads(_session(snapshot, rows, commits), METRIC)

    assert [(lead.branch, lead.running, round(lead.lead_pct, 1)) for lead in leads] == [
        ("b01", True, 10.0),
        ("b02", False, 8.0),
    ]
    assert (leads[0].bar_pct, leads[0].head_value, leads[0].integration_value) == (1.6, 1100.0, 1000.0)
    assert leads[0].describe() == (
        "b01 leads integration by +10.0% on output_tps (1100 against 1000, over the 1.6% bar), and 1 of its commits "
        "did not merge: aaaaaaaa (+5.0%): unresolved patch conflict, 3 tries"
    )
    assert leads[1].describe().startswith("b02 (stopped) leads integration by +8.0%")


def test_a_reason_shows_its_first_line_only_cut_to_a_line() -> None:
    patch_error = "unresolved patch conflict: patch/sglang/aa.patch\nerror: patch failed: srt/models/qwen2.py:60\n"

    assert short_reason(patch_error) == "unresolved patch conflict: patch/sglang/aa.patch"
    assert short_reason("x" * 500) == "x" * (REASON_CHARS - 3) + "..."
    assert short_reason("") == short_reason("  \n ") == ""


def test_a_merged_commit_strands_nothing_and_nothing_is_stranded_without_integrations_value(
    snapshot: NotebookSnapshot,
) -> None:
    rows = [_row(INTEGRATION_BRANCH, 1000.0), _row("b01", 1100.0)]
    failed = _commit("aaaaaaaa1", "b01", MergeOutcome.CONFLICT_SKIPPED, CONFLICT)
    merged = dataclasses.replace(failed, merged_into="i2")

    assert stranded_leads(_session(snapshot, rows, [merged]), METRIC) == ()
    assert stranded_leads(_session(snapshot, rows[1:], [failed]), METRIC) == ()
    assert stranded_leads(_session(snapshot, rows, [failed]), "another_metric") == ()
