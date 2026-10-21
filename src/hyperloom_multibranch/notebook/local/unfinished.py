# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What this session left unfinished, for the startup brief of a branch opened later in it.

:func:`unfinished_leads` reads the session's notebook for results worth finishing:

* a commit of a stopped branch whose last merge onto the integration head was not stackable: that answer stands until
  the head moves in a way that offers the commit again, while a commit not merge-tested yet, or skipped on a conflict,
  is the integrator's to offer again, not a lead;
* a measured explore that kept accuracy, gained, and fell short of its bar by less than :data:`NEAR_BAR_PCT`;
* a blocked candidate whose repair never finished;
* an explore withdrawn before its result, a branch stop's withdrawal included.

A lead drops out once its direction is tested again on any branch, its own confirm aside, or a branch holds a claim on
it, so two later branches do not chase the same one; a commit drops out once the integrator offers it again. The leads
come in that order, unstackable commits and near misses by gain, unfinished repairs and withdrawn explores newest
first, at most :data:`MAX_UNFINISHED` of them.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from hyperloom_multibranch.contracts.notebook.peers import Purpose, TestKind
from hyperloom_multibranch.contracts.notebook.records import LineageView, TestRecord, UnfinishedLead
from hyperloom_multibranch.contracts.notebook.vocab import (
    INTEGRATION_BRANCH,
    Conclusion,
    MergeOutcome,
    UnfinishedReason,
)
from hyperloom_multibranch.notebook.changes import describe_change
from hyperloom_multibranch.notebook.local.state import NotebookState
from hyperloom_multibranch.notebook.local.views import committed

#: How far short of its bar a measured explore may fall and still be worth finishing, in points.
NEAR_BAR_PCT = 0.5
MAX_UNFINISHED = 8
_REPAIRING = frozenset({Conclusion.DIAGNOSIS_PENDING, Conclusion.REPAIR_REQUESTED})


def _lead(state: NotebookState, record: TestRecord, reason: UnfinishedReason, detail: str = "") -> UnfinishedLead:
    intent = state.intents.get(record.request.intent_fingerprint or "")
    judgement = record.judgement
    return UnfinishedLead(
        request_id=record.request_id,
        branch=record.request.branch,
        target=intent.target if intent is not None else "",
        mechanism=intent.mechanism if intent is not None else "",
        change=describe_change(record.request.delta),
        measured_on=record.request.parent_commit,
        reason=reason,
        gain_pct=judgement.gain_pct if judgement is not None else None,
        detail=detail,
    )


def _explore(record: TestRecord) -> bool:
    request = record.request
    return request.kind is TestKind.TEST and request.purpose is Purpose.EXPLORE and request.branch != INTEGRATION_BRANCH


def _near_bar(record: TestRecord) -> bool:
    judgement = record.judgement
    return (
        judgement is not None
        and record.conclusion is Conclusion.RECORDED_ONLY
        and judgement.accuracy_ok
        and 0.0 < judgement.gain_pct < judgement.threshold_pct
        and judgement.threshold_pct - judgement.gain_pct < NEAR_BAR_PCT
    )


def _merge_found(state: NotebookState, request_id: str | None) -> str:
    """What the merge test ``request_id`` found, as its judgement records it."""

    write = state.latest_judgement(request_id) if request_id is not None else None
    if write is None:
        return ""
    found = write.judgement
    return f" {found.gain_pct:+.2f}%, bar {found.threshold_pct:.2f}%"


def _not_stackable(state: NotebookState, view: LineageView) -> list[UnfinishedLead]:
    """The commits of stopped branches whose last merge was not stackable, each led by the test it came from."""

    stopped = {node.branch for node in view.branches if node.stop_reason is not None}
    integration = next((node for node in view.branches if node.branch == INTEGRATION_BRANCH), None)
    on_head = set(view.ancestry(integration.head_commit)) if integration is not None else set()
    records = {record.request_id: record for record in state.records()}
    leads = []
    for commit in committed(state, view):
        last = commit.last_merge
        record = records.get(commit.request_id)
        if last is None or last.outcome is not MergeOutcome.NOT_STACKABLE or record is None:
            continue
        if commit.branch not in stopped or commit.commit_id in on_head:
            continue
        found = _merge_found(state, last.request_id)
        detail = f"commit {commit.commit_id}; merge {last.request_id} on {last.integration_head}{found}"
        leads.append(_lead(state, record, UnfinishedReason.NOT_STACKABLE, detail))
    return leads


def _taken_up(record: TestRecord, records: Sequence[TestRecord], held: Collection[str]) -> bool:
    """Whether a later test on any branch, not ``record``'s own confirm, tried ``record``'s direction again, or a
    branch holds a claim on it."""

    fingerprint = record.request.intent_fingerprint
    if fingerprint is None:
        return False
    if fingerprint in held:
        return True
    return any(
        other.request.intent_fingerprint == fingerprint
        and other.requested_at > record.requested_at
        and other.request.retest_of != record.request_id
        and other.withdrawn_at is None
        and other.request.kind is TestKind.TEST
        for other in records
    )


def _by_gain(leads: Sequence[UnfinishedLead]) -> list[UnfinishedLead]:
    return sorted(leads, key=lambda lead: -(lead.gain_pct or 0.0))


def unfinished_leads(state: NotebookState, view: LineageView, held: Collection[str] = ()) -> tuple[UnfinishedLead, ...]:
    """The session's unfinished leads, in the module's order; ``held`` are the intent fingerprints branches claim."""

    records = state.records()
    newest = sorted(records, key=lambda item: -item.requested_at)
    untaken = {record.request_id for record in records if not _taken_up(record, records, held)}
    explores = [record for record in newest if _explore(record) and record.request_id in untaken]
    near = [
        _lead(state, record, UnfinishedReason.NEAR_BAR, f"bar {record.judgement.threshold_pct:.2f}%")
        for record in explores
        if _near_bar(record) and record.judgement is not None
    ]
    repairs = [
        _lead(state, record, UnfinishedReason.REPAIR_UNFINISHED)
        for record in explores
        if record.conclusion in _REPAIRING
    ]
    withdrawn = [
        _lead(state, record, UnfinishedReason.WITHDRAWN, record.withdrawal_reason)
        for record in explores
        if record.withdrawn_at is not None and record.result is None
    ]
    unstackable = [lead for lead in _not_stackable(state, view) if lead.request_id in untaken]
    fingerprints = {record.request_id: record.request.intent_fingerprint or record.request_id for record in records}
    leads: list[UnfinishedLead] = []
    directions: set[str] = set()
    for lead in (*_by_gain(unstackable), *_by_gain(near), *repairs, *withdrawn):
        direction = fingerprints[lead.request_id]
        if direction not in directions:
            directions.add(direction)
            leads.append(lead)
    return tuple(leads[:MAX_UNFINISHED])


__all__ = ["MAX_UNFINISHED", "NEAR_BAR_PCT", "unfinished_leads"]
