# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What this session left unfinished, for the startup brief of a branch opened later in it.

:func:`unfinished_leads` reads the session's blackboard for results worth finishing:

* a blocked candidate whose repair never finished;
* an explore withdrawn before its result, a branch stop's withdrawal included, unless another explore measured its
  change.

A commit is no lead: offering it again on a later integration head is the integrator's.

A lead drops out once its direction or its change is tested again on any branch, its own confirm aside, or a branch
holds a claim on it, so two later branches do not chase the same one: a replay claims an intent of its own, but tests
the lead's change, with its delta or, made with ``blackboard_replay``, with one made for its own head and an intent
naming the lead. It drops out of the brief of a start commit too once a replay of it was claimed on that commit or one
it builds on and released with no test (:func:`refused_replays`): the change could not be made for that head, the
toolbox refusing it or the change no longer applying; otherwise one lead would be claimed and released that way again
and again, on branch after branch. The leads come in that order, newest first, one for each direction and each change,
at most :data:`MAX_UNFINISHED` of them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict
from hyperloom_multibranch.contracts.blackboard.peers import Purpose, TestKind
from hyperloom_multibranch.contracts.blackboard.records import (
    REPLAY_TARGET,
    Claim,
    LineageView,
    TestRecord,
    UnfinishedLead,
)
from hyperloom_multibranch.contracts.blackboard.vocab import (
    INTEGRATION_BRANCH,
    ClaimStatus,
    Conclusion,
    UnfinishedReason,
)
from hyperloom_multibranch.blackboard.changes import describe_change
from hyperloom_multibranch.blackboard.local.state import BlackboardState

MAX_UNFINISHED = 8
#: How soon after its branch stopped a claim's release is the stop's, not the answer of a replay that could not be made.
STOP_RELEASE_SECONDS = 60.0
_REPAIRING = frozenset({Conclusion.DIAGNOSIS_PENDING, Conclusion.REPAIR_REQUESTED})
_NO_CHANGE = RecipeDelta().fingerprint()


def _lead(state: BlackboardState, record: TestRecord, reason: UnfinishedReason, detail: str = "") -> UnfinishedLead:
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


def hinted_gain(record: TestRecord) -> float | None:
    """The gain ``record``'s judgement lets a hint rank by, a direction's best gain: one whose accuracy passed, or
    one below its bar whose eval did not run, since an explore short of its bar skips it; above the bar only the
    confirm's passed eval counts, so a fast but wrong change ranks nothing. ``None`` otherwise."""

    judgement = record.judgement
    if judgement is None or record.conclusion is Conclusion.ACCURACY_FAILED:
        return None
    if judgement.accuracy is AccuracyVerdict.PASSED:
        return judgement.gain_pct
    below = judgement.accuracy is AccuracyVerdict.NOT_MEASURED and judgement.gain_pct < judgement.threshold_pct
    return judgement.gain_pct if below else None


def _replayed(state: BlackboardState, records: Sequence[TestRecord]) -> dict[str, str]:
    """What each replay made with ``blackboard_replay`` replays, by its request id: its intent names it."""

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
    state: BlackboardState, held: Collection[str] = (), refused: Collection[str] = ()
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
    leads = _distinct((*repairs, *withdrawn), records, replayed)
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


__all__ = [
    "MAX_UNFINISHED",
    "STOP_RELEASE_SECONDS",
    "hinted_gain",
    "refused_replays",
    "unfinished_leads",
]
