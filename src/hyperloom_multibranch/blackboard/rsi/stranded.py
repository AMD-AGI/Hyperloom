# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stranded leads: branches whose head leads integration by more than the merge bar while their commits fail to merge.

A branch ahead of integration is usual for a while, its commit waiting for its merge test. A lead is stranded when it
holds over the bar on the session's metric and a commit behind it has already failed to merge: skipped untested
because it conflicted, or tested and not stackable. Integration, and every branch opened from it, then goes on from the
lower base. In session 4aef3318 three branches held 7.8 to 11.1% over integration that way, for up to four hours (#126).

:func:`stranded_leads` reads them off a snapshot, the largest lead first, each with the commits that did not merge, how
many merges each took part in, and why its last one failed.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import SHORT_COMMIT, CommittedCommit, BlackboardSnapshot
from hyperloom_multibranch.contracts.blackboard.vocab import FAILED_MERGES, MergeOutcome

#: The outcomes of a merge that did not carry its commits onto integration.
FAILED = FAILED_MERGES
#: How much of a merge's reason a line keeps: its first line, a patch tool's errors under it left out.
REASON_CHARS = 120


def short_reason(reason: str) -> str:
    """The first line of ``reason``, at most :data:`REASON_CHARS` long."""

    first = " ".join(reason.strip().splitlines()[0].split()) if reason.strip() else ""
    return first if len(first) <= REASON_CHARS else first[: REASON_CHARS - 3] + "..."


@dataclass(frozen=True)
class StrandedCommit:
    """A commit behind a stranded lead: its own gain over its parent, its last merge's outcome and why, and how many
    merges it took part in."""

    commit_id: str
    gain_pct: float
    outcome: MergeOutcome
    reason: str
    attempts: int

    def describe(self) -> str:
        tries = f", {self.attempts} tries" if self.attempts > 1 else ""
        why = short_reason(self.reason) or self.outcome.value
        return f"{self.commit_id[:SHORT_COMMIT]} ({self.gain_pct:+.1f}%): {why}{tries}"


@dataclass(frozen=True)
class StrandedLead:
    """A branch whose head value leads integration's by ``lead_pct`` on ``metric``, over the merge bar ``bar_pct``,
    with ``commits`` of its own that did not merge."""

    branch: str
    running: bool
    head_commit: str
    metric: str
    head_value: float
    integration_value: float
    lead_pct: float
    bar_pct: float
    commits: tuple[StrandedCommit, ...]

    def describe(self) -> str:
        stopped = "" if self.running else " (stopped)"
        failed = "; ".join(commit.describe() for commit in self.commits)
        return (
            f"{self.branch}{stopped} leads integration by {self.lead_pct:+.1f}% on {self.metric} "
            f"({self.head_value:.0f} against {self.integration_value:.0f}, over the {self.bar_pct:.1f}% bar), and "
            f"{len(self.commits)} of its commits did not merge: {failed}"
        )


def stranded_leads(snapshot: BlackboardSnapshot, metric: str) -> tuple[StrandedLead, ...]:
    """The stranded leads of ``snapshot`` on ``metric``, the largest first; none without the baseline's bar on
    ``metric`` or integration's value on it."""

    rows = {row.branch: row for row in snapshot.progress}
    integration = rows.get(INTEGRATION_BRANCH)
    base = integration.head_value.get(metric) if integration is not None else None
    bar = snapshot.baseline.threshold_pct.get(metric) if snapshot.baseline is not None else None
    if not base or bar is None:
        return ()
    failed: dict[str, list[CommittedCommit]] = {}
    for commit in snapshot.committed:
        last = commit.last_merge
        if commit.merged_into is None and last is not None and last.outcome in FAILED:
            failed.setdefault(commit.branch, []).append(commit)
    leads = []
    for branch, commits in failed.items():
        row = rows.get(branch)
        value = row.head_value.get(metric) if row is not None else None
        if row is None or not value or (value / base - 1.0) * 100.0 <= bar:
            continue
        stranded = tuple(_stranded(commit) for commit in commits)
        lead_pct = (value / base - 1.0) * 100.0
        leads.append(StrandedLead(branch, row.running, row.head_commit, metric, value, base, lead_pct, bar, stranded))
    return tuple(sorted(leads, key=lambda lead: -lead.lead_pct))


def _stranded(commit: CommittedCommit) -> StrandedCommit:
    last = commit.last_merge
    outcome, reason = (last.outcome, last.reason) if last is not None else (MergeOutcome.PENDING, "")
    return StrandedCommit(commit.commit_id, commit.gain_pct, outcome, reason, commit.merge_attempts)


__all__ = ["FAILED", "REASON_CHARS", "StrandedCommit", "StrandedLead", "short_reason", "stranded_leads"]
