# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist pulls from the blackboard: its branch's inbox, record search, and profile records.

The inbox holds pointers only. A cursor is the last sequence number seen of every writer, encoded opaquely, so a poll
returns each entry once however the writers' clocks interleave.

Given :class:`BriefTriggers`, the inbox also holds ``brief`` items, pointing at the branch's head and asking its
specialist to read its brief again before the next claim. They are derived from the entries as the other items are,
so a poll returns each once: when a judgement brings the branch's tests judged since its last commit to
:data:`STALL_TESTS`, once a stall, and when another branch's pitfall learning is about the direction family of a claim
the branch holds, or, where either names no family, names its direction.

A branch whose claim was refused because another branch was testing the same intent gets a ``peer_outcome`` item when
that test ends, judged, withdrawn or not measured, pointing at the test so its change can be replayed. The refused claim
decision is the registration, so a restarted blackboard still knows who waits.

A branch gets a ``merge`` item once the integrator settles a merge that carried commits of its own, which it writes on
the integration branch: merged, not stackable on the integration head, conflicted or already on it.

Search: ``,`` separates alternatives and ``&`` joins terms that must all appear, so ``fp8&kv,attention`` matches an
entry mentioning both ``fp8`` and ``kv``, or one mentioning ``attention``. Matching is case-insensitive substring search
over an entry's kind, branch, actor, summary and stored detail.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.blackboard.peers import Intent, TestKind, TestResult, Validity
from hyperloom_multibranch.contracts.blackboard.records import (
    JudgementWrite,
    SHORT_COMMIT,
    Claim,
    InboxItem,
    InboxPage,
    Learning,
    LineageEvent,
    LineageView,
    MergeAttempt,
    SearchHit,
    TestRecord,
    Withdrawal,
)
from hyperloom_multibranch.contracts.blackboard.vocab import (
    INTEGRATION_CONCLUSIONS,
    DedupStatus,
    InboxKind,
    LearningKind,
)
from hyperloom_multibranch.blackboard.families import note_family
from hyperloom_multibranch.blackboard.learning_search import SAME_FAMILY, family_of_learning, kinship, names
from hyperloom_multibranch.blackboard.local.segments import Entry, EntryKind
from hyperloom_multibranch.blackboard.local.state import ClaimDecision, BlackboardState
from hyperloom_multibranch.blackboard.local.views import judged_since_commit, lineage_view, merge_state

#: How many tests judged since the last commit make a stall. The run entry has no setting for it, and the judge's
#: prompt quotes the same number, so a session's specialists and its judge count a stall alike.
STALL_TESTS = 4
_REREAD = "read blackboard_brief from {head} and reconsider the direction before the next claim"


@dataclass(frozen=True)
class BriefTriggers:
    """Turns ``brief`` items on in a running specialist's inbox: when its branch has :data:`STALL_TESTS` tests judged
    since its last commit, and when another branch learns a pitfall in a direction it holds a claim on."""


@dataclass(frozen=True)
class _Briefing:
    held: tuple[Claim, ...]
    records: tuple[TestRecord, ...]
    view: LineageView
    intents: Mapping[str, Intent] = field(default_factory=dict)

    def head(self, branch: str) -> str:
        node = next((item for item in self.view.branches if item.branch == branch), None)
        return node.head_commit if node is not None else ""

    def falls_in(self, learning: Learning, intent: Intent) -> bool:
        """Whether ``learning`` is about the family of the direction ``intent`` claims; where either names no family,
        whether it names the direction."""

        about = family_of_learning(learning, self.records, self.intents)
        claimed = note_family(intent.target, intent.mechanism, intent.layer)
        if about is None or claimed is None:
            return names(learning, intent)
        return kinship(claimed, about) == SAME_FAMILY


_INBOX_KINDS = {
    EntryKind.RESULT: InboxKind.RESULT,
    EntryKind.JUDGEMENT: InboxKind.JUDGEMENT,
    EntryKind.TOOL_RESULT: InboxKind.TOOL_RESULT,
    EntryKind.LINEAGE: InboxKind.LINEAGE,
    EntryKind.BUILD: InboxKind.BUILD,
}


def _read_cursor(cursor: str) -> dict[str, int]:
    if not cursor:
        return {}
    try:
        seen = json.loads(cursor)
    except ValueError:
        seen = None
    if not isinstance(seen, dict) or not all(isinstance(seq, int) for seq in seen.values()):
        raise ValueError(f"not an inbox cursor: {cursor!r}")
    return seen


def poll(
    state: BlackboardState,
    branch: str,
    cursor: str,
    triggers: BriefTriggers | None = None,
    held: Sequence[Claim] = (),
) -> InboxPage:
    """The items for ``branch`` written since ``cursor``, and the cursor past everything seen now.

    With ``triggers``, ``held`` are the claims the branch holds now, which a pitfall learning is matched against.
    """

    seen = _read_cursor(cursor)
    briefing = (
        _Briefing(tuple(held), tuple(state.records()), lineage_view(state), dict(state.intents))
        if triggers is not None
        else None
    )
    refusals = _refusals(state, branch)
    new = [(entry, value) for entry, value in state.decoded if entry.seq > seen.get(entry.writer, 0)]
    merges = {entry.entry_id: found for entry, value in new if (found := _settled_merge(value, state)) is not None}
    own = _own_commits(state, branch) if merges else frozenset()
    items = tuple(
        item
        for entry, value in new
        for item in (
            _inbox_item(entry, value, branch),
            _brief_item(entry, value, branch, briefing) if briefing is not None else None,
            _peer_item(entry, value, branch, refusals, state) if refusals else None,
            _merge_item(entry, merges[entry.entry_id], branch, own, state) if entry.entry_id in merges else None,
        )
        if item is not None
    )
    latest = dict(seen)
    for entry in state.entries:
        latest[entry.writer] = max(latest.get(entry.writer, 0), entry.seq)
    return InboxPage(items, json.dumps(dict(sorted(latest.items())), separators=(",", ":")))


def _inbox_item(entry: Entry, value: object, branch: str) -> InboxItem | None:
    kind = _INBOX_KINDS.get(entry.kind)
    split_off = isinstance(value, LineageEvent) and value.from_branch == branch
    if kind is None or (entry.branch != branch and not split_off):
        return None
    pointer = getattr(value, "request_id", None) or getattr(value, "job_id", None) or entry.entry_id
    return InboxItem(kind, branch, str(pointer), entry.summary, entry.at)


def _settled_merge(value: object, state: BlackboardState) -> MergeAttempt | None:
    """The merge attempt the entry ``value`` settles: a merge skipped untested settles itself; a merge test's attempt is
    settled by the integrator's judgement, or by a result the candidate failed, which no judgement follows."""

    if isinstance(value, MergeAttempt):
        return value if value.request_id is None else None
    if isinstance(value, JudgementWrite) and value.conclusion in INTEGRATION_CONCLUSIONS:
        request_id = value.request_id
    elif isinstance(value, TestResult) and value.validity is Validity.CANDIDATE_FAILED:
        request_id = value.request_id
    else:
        return None
    return next((attempt for attempt in state.merge_attempts if attempt.request_id == request_id), None)


def _own_commits(state: BlackboardState, branch: str) -> frozenset[str]:
    return frozenset(node.commit_id for node in lineage_view(state).commits if node.branch == branch)


def _merge_item(
    entry: Entry, attempt: MergeAttempt, branch: str, own: frozenset[str], state: BlackboardState
) -> InboxItem | None:
    """How the merge ``attempt``, which the entry settled, ended for the commits of ``branch`` it carried (``own``)."""

    mine = [commit[:SHORT_COMMIT] for commit in attempt.source_commits if commit in own]
    if not mine:
        return None
    settled = merge_state(state, attempt, entry.at)
    found = settled.outcome.value
    if settled.merged_commit:
        found += f" as {settled.merged_commit[:SHORT_COMMIT]}"
    if settled.reason:
        found += f" ({settled.reason})"
    summary = f"the merge of your {', '.join(mine)} onto {attempt.integration_head[:SHORT_COMMIT]}: {found}"
    return InboxItem(InboxKind.MERGE, branch, attempt.request_id or entry.entry_id, summary, entry.at)


def _brief_item(entry: Entry, value: object, branch: str, briefing: _Briefing) -> InboxItem | None:
    head = briefing.head(branch)
    reread = _REREAD.format(head=head)
    if isinstance(value, JudgementWrite) and entry.branch == branch and value.commit_id is None:
        stalled = judged_since_commit(briefing.records, briefing.view, branch, until=entry.at)
        if stalled == STALL_TESTS:
            summary = f"stall: {stalled} tests judged since this branch's last commit, none committed; {reread}"
            return InboxItem(InboxKind.BRIEF, branch, head, summary, entry.at)
    if (
        isinstance(value, Learning)
        and value.kind is LearningKind.PITFALL
        and value.branch != branch
        and any(briefing.falls_in(value, claim.intent) for claim in briefing.held)
    ):
        summary = f"pitfall: {value.branch} learnt, in a direction this branch holds: {value.content}; {reread}"
        return InboxItem(InboxKind.BRIEF, branch, head, summary, entry.at)
    return None


def _refusals(state: BlackboardState, branch: str) -> tuple[tuple[float, ClaimDecision], ...]:
    """The claims of ``branch`` refused while another branch tested the same intent, with when each was refused."""

    return tuple(
        (entry.at, value)
        for entry, value in state.decoded
        if isinstance(value, ClaimDecision)
        and value.found is DedupStatus.TESTING_ELSEWHERE
        and not value.acquired
        and value.intent.branch == branch
    )


def _ended(value: object) -> str | None:
    """The test an entry ends: judged, withdrawn, or not measured, which no judgement follows."""

    if isinstance(value, (JudgementWrite, Withdrawal)):
        return value.request_id
    if isinstance(value, TestResult) and value.validity is Validity.MEASUREMENT_FAILED:
        return value.request_id
    return None


def _peer_item(
    entry: Entry,
    value: object,
    branch: str,
    refusals: Sequence[tuple[float, ClaimDecision]],
    state: BlackboardState,
) -> InboxItem | None:
    request_id = _ended(value)
    if request_id is None or entry.branch == branch:
        return None
    ended = state.record(request_id).request
    refused = next(
        (
            decision
            for at, decision in refusals
            if at <= entry.at
            and (
                request_id in decision.matched
                or (decision.intent.fingerprint == ended.intent_fingerprint and entry.branch in decision.holders)
            )
        ),
        None,
    )
    if refused is None:
        return None
    if ended.parent_commit == refused.intent.parent_commit:
        summary = (
            f"{entry.branch}'s test of the same change on {ended.parent_commit}, which refused your claim, ended: "
            f"{entry.summary}; it measured the state you claimed on, so there is nothing to replay"
        )
    else:
        summary = (
            f"{entry.branch}'s test of {refused.intent.target} / {refused.intent.mechanism}, which refused your claim, "
            f"ended: {entry.summary}; on a gain, blackboard_replay with request_id {request_id} replays its change on "
            "your head, or record why not"
        )
    return InboxItem(InboxKind.PEER_OUTCOME, branch, request_id, summary, entry.at)


def _alternatives(query: str) -> list[list[str]]:
    groups = [[term.strip().lower() for term in group.split("&") if term.strip()] for group in query.split(",")]
    alternatives = [terms for terms in groups if terms]
    if not alternatives:
        raise ValueError("a query needs at least one term")
    return alternatives


def _searchable(entry: Entry) -> str:
    detail = json.dumps(entry.detail, sort_keys=True, ensure_ascii=False)
    return " ".join((entry.kind.value, entry.branch, entry.actor, entry.summary, detail)).lower()


def search(state: BlackboardState, query: str, limit: int) -> tuple[SearchHit, ...]:
    """The newest ``limit`` entries matching ``query``."""

    alternatives = _alternatives(query)
    hits = []
    for entry in reversed(state.entries):
        text = _searchable(entry)
        if any(all(term in text for term in terms) for terms in alternatives):
            hits.append(SearchHit(entry.entry_id, entry.kind.value, entry.branch, entry.summary))
        if len(hits) >= limit:
            break
    return tuple(hits)


def profiles(state: BlackboardState, commit_id: str) -> tuple[TestRecord, ...]:
    """The profile tests of ``commit_id``'s recipe whose result points at an artifact, oldest first: the commit's own,
    and those of another commit of its recipe digest that changed nothing, whose recipe the blackboard knows."""

    digests = state.recipe_digests()
    recipe = digests.get(commit_id)
    return tuple(
        record
        for record in state.records()
        if record.request.kind is TestKind.PROFILE
        and (
            record.request.parent_commit == commit_id
            or (
                recipe is not None
                and digests.get(record.request.parent_commit) == recipe
                and record.request.delta == RecipeDelta()
            )
        )
        and record.result is not None
        and record.result.profile_location
    )


__all__ = ["STALL_TESTS", "BriefTriggers", "poll", "profiles", "search"]
