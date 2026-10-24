# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What this session left unfinished, for the startup brief of a branch opened later in it.

:func:`unfinished_leads` reads the session's notebook for results worth finishing:

* a commit of a stopped branch whose last merge onto the integration head was not stackable: that answer stands until
  the head moves in a way that offers the commit again, while a commit not merge-tested yet, or skipped on a conflict,
  is the integrator's to offer again, not a lead;
* a measured explore that kept accuracy, gained, and fell short of its bar by less than :data:`NEAR_BAR_PCT`;
* a blocked candidate whose repair never finished;
* an explore withdrawn before its result, a branch stop's withdrawal included, unless another explore measured its
  change.

A lead drops out once its direction or its change is tested again on any branch, its own confirm aside, or a branch
holds a claim on it, so two later branches do not chase the same one: a replay claims an intent of its own, but tests
the lead's change, with its delta or, made with ``notebook_replay``, with one made for its own head and an intent
naming the lead. It drops out of the brief of a start commit too once a replay of it was claimed on that commit or one
it builds on and released with no test (:func:`refused_replays`): the change could not be made for that head, the
toolbox refusing it or the change no longer applying. In session 4aef3318 one lead was claimed and released that way
23 times on 20 branches. A commit drops out once the integrator offers it again. The leads
come in that order, unstackable commits and near misses by gain, unfinished repairs and withdrawn explores newest
first, one for each direction and each change, at most :data:`MAX_UNFINISHED` of them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook.peers import Purpose, TestKind
from hyperloom_multibranch.contracts.notebook.records import (
    REPLAY_TARGET,
    Claim,
    LineageView,
    TestRecord,
    UnfinishedLead,
)
from hyperloom_multibranch.contracts.notebook.vocab import (
    INTEGRATION_BRANCH,
    ClaimStatus,
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
#: How soon after its branch stopped a claim's release is the stop's, not the answer of a replay that could not be made.
STOP_RELEASE_SECONDS = 60.0
_REPAIRING = frozenset({Conclusion.DIAGNOSIS_PENDING, Conclusion.REPAIR_REQUESTED})
_NO_CHANGE = RecipeDelta().fingerprint()


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


def _replayed(state: NotebookState, records: Sequence[TestRecord]) -> dict[str, str]:
    """What each replay made with ``notebook_replay`` replays, by its request id: its intent names it."""

    replayed = {}
    for record in records:
        intent = state.intents.get(record.request.intent_fingerprint or "")
        if intent is not None and intent.target == REPLAY_TARGET:
            replayed[record.request_id] = intent.mechanism
    return replayed


def _taken_up(
    record: TestRecord, records: Sequence[TestRecord], held: Collection[str], replayed: Mapping[str, str]
) -> bool:
    """Whether a later test on any branch, not ``record``'s own confirm, tried ``record``'s direction or its change
    again, or a branch holds a claim on its direction."""

    fingerprint = record.request.intent_fingerprint
    if fingerprint is not None and fingerprint in held:
        return True
    return any(
        (
            (fingerprint is not None and other.request.intent_fingerprint == fingerprint)
            or _tests_the_change(other, record, replayed)
        )
        and other.requested_at > record.requested_at
        and other.request.retest_of != record.request_id
        and other.withdrawn_at is None
        and other.request.kind is TestKind.TEST
        for other in records
    )


def _tests_the_change(other: TestRecord, record: TestRecord, replayed: Mapping[str, str]) -> bool:
    """Whether the explore ``other`` tests ``record``'s change: its delta, or a replay of ``record`` made for another
    head. The merge of a commit and its confirm test the change too, but finish nothing of it."""

    if other.request.purpose is not Purpose.EXPLORE:
        return False
    change = record.request.action_fingerprint
    return replayed.get(other.request_id) == record.request_id or (
        change != _NO_CHANGE and other.request.action_fingerprint == change
    )


def _measured_elsewhere(record: TestRecord, records: Sequence[TestRecord], replayed: Mapping[str, str]) -> bool:
    """Whether another explore measured ``record``'s change, before it or after."""

    return any(
        other.request_id != record.request_id and other.measured and _tests_the_change(other, record, replayed)
        for other in records
    )


def _by_gain(leads: Sequence[UnfinishedLead]) -> list[UnfinishedLead]:
    return sorted(leads, key=lambda lead: -(lead.gain_pct or 0.0))


def refused_replays(claims: Iterable[Claim], view: LineageView, start_commit: str) -> frozenset[str]:
    """The leads whose replay was claimed on ``start_commit`` or a commit it builds on and released with no test, by
    request id; a release its branch's stop made is left out, since it says nothing of the replay."""

    builds_on = set(view.ancestry(start_commit))
    stopped = {node.branch: node.stopped_at for node in view.branches if node.stopped_at is not None}
    return frozenset(
        claim.intent.mechanism
        for claim in claims
        if claim.intent.target == REPLAY_TARGET
        and claim.status is ClaimStatus.RELEASED
        and claim.request_id is None
        and claim.key.parent_commit in builds_on
        and not _released_by_its_stop(claim, stopped)
    )


def _released_by_its_stop(claim: Claim, stopped: Mapping[str, float]) -> bool:
    at = stopped.get(claim.holder)
    return at is not None and 0.0 <= claim.heartbeat_at - at <= STOP_RELEASE_SECONDS


def unfinished_leads(
    state: NotebookState, view: LineageView, held: Collection[str] = (), refused: Collection[str] = ()
) -> tuple[UnfinishedLead, ...]:
    """The session's unfinished leads, in the module's order; ``held`` are the intent fingerprints branches claim, and
    ``refused`` the leads :func:`refused_replays` found for the brief's start commit."""

    records = state.records()
    replayed = _replayed(state, records)
    newest = sorted(records, key=lambda item: -item.requested_at)
    untaken = {
        record.request_id
        for record in records
        if record.request_id not in refused and not _taken_up(record, records, held, replayed)
    }
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
        if record.withdrawn_at is not None
        and record.result is None
        and not _measured_elsewhere(record, records, replayed)
    ]
    unstackable = [lead for lead in _not_stackable(state, view) if lead.request_id in untaken]
    leads = _distinct((*_by_gain(unstackable), *_by_gain(near), *repairs, *withdrawn), records, replayed)
    return tuple(leads[:MAX_UNFINISHED])


def _distinct(
    leads: Sequence[UnfinishedLead], records: Sequence[TestRecord], replayed: Mapping[str, str]
) -> list[UnfinishedLead]:
    """The first of ``leads`` for each direction and for each change, a replay's change being the one it replays."""

    requests = {record.request_id: record.request for record in records}
    kept: list[UnfinishedLead] = []
    seen: set[str] = set()
    for lead in leads:
        request = requests[lead.request_id]
        keys = {request.intent_fingerprint or lead.request_id, replayed.get(lead.request_id, lead.request_id)}
        if request.action_fingerprint != _NO_CHANGE:
            keys.add(request.action_fingerprint)
        if not keys & seen:
            seen |= keys
            kept.append(lead)
    return kept


__all__ = ["MAX_UNFINISHED", "NEAR_BAR_PCT", "STOP_RELEASE_SECONDS", "refused_replays", "unfinished_leads"]
